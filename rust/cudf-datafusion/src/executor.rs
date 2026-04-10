// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU query executor: parses SQL with DataFusion and executes on GPU with cudf.

use std::sync::Arc;

use cudf::data_type::TypeId;
use cudf::groupby::AggregationKind;
use cudf::sorting::{NullOrder, Order};
use cudf::stream::GpuOp;
use cudf::table::Table;
use datafusion::arrow::array::{
    ArrayRef as DfArrayRef, BooleanArray as DfBooleanArray, Date32Array as DfDate32Array,
    Float32Array as DfFloat32Array, Float64Array as DfFloat64Array, Int8Array as DfInt8Array,
    Int16Array as DfInt16Array, Int32Array as DfInt32Array, Int64Array as DfInt64Array,
    RecordBatch as DfRecordBatch, StringArray as DfStringArray, UInt8Array as DfUint8Array,
    UInt16Array as DfUint16Array, UInt32Array as DfUint32Array, UInt64Array as DfUint64Array,
};
use datafusion::arrow::datatypes::{Field as DfField, Schema as DfSchema};
use datafusion::logical_expr::{Expr, JoinType, LogicalPlan};
use datafusion::prelude::SessionContext;

use crate::catalog::{GpuCatalog, type_id_to_df_arrow};
use crate::expr::{eval_expr, find_column};

type BoxError = Box<dyn std::error::Error>;

/// A GPU-accelerated SQL executor backed by DataFusion's query planner.
pub struct GpuExecutor<'a> {
    catalog: &'a GpuCatalog,
}

/// The result of executing a plan node: a GPU table plus its column names.
struct PlanResult {
    table: Table,
    col_names: Vec<String>,
}

impl<'a> GpuExecutor<'a> {
    /// Creates a new executor backed by the given catalog.
    pub fn new(catalog: &'a GpuCatalog) -> Self {
        Self { catalog }
    }

    /// Parses SQL, optimizes, and executes the plan on GPU, returning an Arrow
    /// `RecordBatch` (using DataFusion's arrow version).
    pub async fn run_sql(&self, sql: &str) -> Result<DfRecordBatch, BoxError> {
        // 1. Parse SQL with DataFusion.
        let ctx = SessionContext::new();
        // Register table schemas so DataFusion can type-check and resolve
        // columns. We build empty RecordBatches using DataFusion's arrow types.
        for (name, gpu_table) in self.catalog.iter() {
            let df_schema = Arc::new(catalog_schema_to_df(&gpu_table.schema)?);
            let empty_batch = DfRecordBatch::new_empty(df_schema);
            ctx.register_batch(name, empty_batch)?;
        }
        let plan = ctx.sql(sql).await?.into_optimized_plan()?;

        // 2. Execute on GPU.
        let result = self.execute_plan(&plan)?;

        // 3. Convert to DataFusion Arrow RecordBatch.
        gpu_table_to_df_record_batch(&result.table, &result.col_names)
    }

    /// Recursively executes a DataFusion `LogicalPlan` on GPU.
    fn execute_plan(&self, plan: &LogicalPlan) -> Result<PlanResult, BoxError> {
        match plan {
            LogicalPlan::TableScan(scan) => self.exec_table_scan(scan),
            LogicalPlan::Filter(filter) => self.exec_filter(filter),
            LogicalPlan::Projection(proj) => self.exec_projection(proj),
            LogicalPlan::Aggregate(agg) => self.exec_aggregate(agg),
            LogicalPlan::Sort(sort) => self.exec_sort(sort),
            LogicalPlan::Limit(limit) => self.exec_limit(limit),
            LogicalPlan::Join(join) => self.exec_join(join),
            LogicalPlan::SubqueryAlias(alias) => self.execute_plan(alias.input.as_ref()),
            LogicalPlan::Repartition(r) => self.execute_plan(r.input.as_ref()),
            _ => Err(format!("Unsupported plan node: {}", plan.display()).into()),
        }
    }

    fn exec_table_scan(
        &self,
        scan: &datafusion::logical_expr::TableScan,
    ) -> Result<PlanResult, BoxError> {
        let gpu_table = self
            .catalog
            .get(&scan.table_name.to_string())
            .ok_or_else(|| format!("table not found: {}", scan.table_name))?;

        if let Some(ref proj_indices) = scan.projection {
            let mut columns = Vec::with_capacity(proj_indices.len());
            let mut names = Vec::with_capacity(proj_indices.len());
            for &idx in proj_indices {
                let col = gpu_table.table.column(idx)?.to_owned_column().call()?;
                columns.push(col);
                names.push(gpu_table.column_names[idx].clone());
            }
            let table = Table::from_columns(columns)?;
            Ok(PlanResult {
                table,
                col_names: names,
            })
        } else {
            let ncols = gpu_table.table.columns_len();
            let mut columns = Vec::with_capacity(ncols);
            for i in 0..ncols {
                columns.push(gpu_table.table.column(i)?.to_owned_column().call()?);
            }
            let table = Table::from_columns(columns)?;
            Ok(PlanResult {
                table,
                col_names: gpu_table.column_names.clone(),
            })
        }
    }

    fn exec_filter(
        &self,
        filter: &datafusion::logical_expr::Filter,
    ) -> Result<PlanResult, BoxError> {
        let child = self.execute_plan(filter.input.as_ref())?;
        let mask = eval_expr(&filter.predicate, &child.table, &child.col_names)?;
        let filtered = child.table.filter(&mask.view()).call()?;
        Ok(PlanResult {
            table: filtered,
            col_names: child.col_names,
        })
    }

    fn exec_projection(
        &self,
        proj: &datafusion::logical_expr::Projection,
    ) -> Result<PlanResult, BoxError> {
        let child = self.execute_plan(proj.input.as_ref())?;
        let mut out_columns = Vec::with_capacity(proj.expr.len());
        let mut out_names = Vec::with_capacity(proj.expr.len());

        for expr in &proj.expr {
            let col = eval_expr(expr, &child.table, &child.col_names)?;
            out_columns.push(col);
            out_names.push(expr_output_name(expr));
        }

        let table = Table::from_columns(out_columns)?;
        Ok(PlanResult {
            table,
            col_names: out_names,
        })
    }

    fn exec_aggregate(
        &self,
        agg: &datafusion::logical_expr::Aggregate,
    ) -> Result<PlanResult, BoxError> {
        let child = self.execute_plan(agg.input.as_ref())?;

        // Identify group-by key columns.
        let mut key_indices: Vec<i32> = Vec::new();
        let mut key_names: Vec<String> = Vec::new();

        for group_expr in &agg.group_expr {
            let name = expr_output_name(group_expr);
            let idx = find_column(&name, &child.col_names)?;
            #[allow(clippy::cast_possible_truncation)]
            key_indices.push(idx as i32);
            key_names.push(name);
        }

        // Pre-materialize computed aggregate arguments. If SUM(a * b), we
        // evaluate `a * b` and append it as a new column to the child table.
        let mut extra_columns: Vec<cudf::column::Column> = Vec::new();
        let mut extra_names: Vec<String> = Vec::new();

        for agg_expr in &agg.aggr_expr {
            let inner = match agg_expr {
                Expr::Alias(a) => a.expr.as_ref(),
                other => other,
            };
            if let Expr::AggregateFunction(agg_fn) = inner {
                if !agg_fn.params.args.is_empty() {
                    let arg = &agg_fn.params.args[0];
                    if !matches!(arg, Expr::Column(_) | Expr::Literal(_)) {
                        let computed = eval_expr(arg, &child.table, &child.col_names)?;
                        let name = format!("__agg_arg_{}", extra_columns.len());
                        extra_columns.push(computed);
                        extra_names.push(name);
                    }
                }
            }
        }

        // Rebuild table with extra computed columns if needed.
        let (work_table, work_names) = if extra_columns.is_empty() {
            (child.table, child.col_names)
        } else {
            let ncols = child.table.columns_len();
            let mut all_cols = Vec::with_capacity(ncols + extra_columns.len());
            for i in 0..ncols {
                all_cols.push(child.table.column(i)?.to_owned_column().call()?);
            }
            all_cols.append(&mut extra_columns);
            let t = Table::from_columns(all_cols)?;
            let mut names = child.col_names;
            names.extend(extra_names);
            (t, names)
        };

        // Build aggregation requests.
        let mut value_indices: Vec<i32> = Vec::new();
        let mut agg_kinds: Vec<AggregationKind> = Vec::new();
        let mut agg_names: Vec<String> = Vec::new();
        let mut computed_counter = 0usize;

        for agg_expr in &agg.aggr_expr {
            let (val_idx, kind, name) =
                resolve_aggregate(agg_expr, &work_names, &mut computed_counter)?;
            value_indices.push(val_idx);
            agg_kinds.push(kind);
            agg_names.push(name);
        }

        if value_indices.is_empty() {
            return Err("No aggregate functions found".into());
        }

        // Scalar aggregation (no GROUP BY) — use column-level reductions.
        if key_indices.is_empty() {
            let mut out_columns = Vec::with_capacity(value_indices.len());
            for (i, kind) in agg_kinds.iter().enumerate() {
                #[allow(clippy::cast_sign_loss)]
                let col_view = work_table.column(value_indices[i] as usize)?;
                let out_tid = col_view.type_id();
                let scalar = match *kind {
                    AggregationKind::SUM => col_view.sum(out_tid).call()?,
                    AggregationKind::MIN => col_view.min(out_tid).call()?,
                    AggregationKind::MAX => col_view.max(out_tid).call()?,
                    AggregationKind::COUNT =>
                    {
                        #[allow(clippy::cast_possible_wrap)]
                        cudf::scalar::Scalar::from_i64(col_view.len() as i64)
                    }
                    AggregationKind::MEAN => col_view.mean(TypeId::FLOAT64).call()?,
                    _ => return Err(format!("Unsupported scalar agg: {kind:?}").into()),
                };
                out_columns.push(cudf::column::Column::from_scalar(&scalar, 1).call()?);
            }
            let result = Table::from_columns(out_columns)?;
            return Ok(PlanResult {
                table: result,
                col_names: agg_names,
            });
        }

        let result = work_table
            .groupby_multi(&key_indices, &value_indices, &agg_kinds)
            .call()?;

        let mut col_names = key_names;
        col_names.extend(agg_names);

        Ok(PlanResult {
            table: result,
            col_names,
        })
    }

    fn exec_sort(&self, sort: &datafusion::logical_expr::Sort) -> Result<PlanResult, BoxError> {
        let child = self.execute_plan(sort.input.as_ref())?;

        let ncols = child.table.columns_len();
        let mut orders = vec![Order::ASCENDING; ncols];
        let mut null_orders = vec![NullOrder::BEFORE; ncols];

        for sort_expr in &sort.expr {
            let name = expr_output_name(&sort_expr.expr);
            if let Ok(idx) = find_column(&name, &child.col_names) {
                orders[idx] = if sort_expr.asc {
                    Order::ASCENDING
                } else {
                    Order::DESCENDING
                };
                null_orders[idx] = if sort_expr.nulls_first {
                    NullOrder::BEFORE
                } else {
                    NullOrder::AFTER
                };
            }
        }

        let sorted = child.table.sort(&orders, &null_orders).call()?;
        Ok(PlanResult {
            table: sorted,
            col_names: child.col_names,
        })
    }

    fn exec_limit(&self, limit: &datafusion::logical_expr::Limit) -> Result<PlanResult, BoxError> {
        use datafusion::logical_expr::{FetchType, SkipType};

        let child = self.execute_plan(limit.input.as_ref())?;

        let skip = match limit.get_skip_type()? {
            SkipType::Literal(n) => n,
            SkipType::UnsupportedExpr => {
                return Err("Unsupported OFFSET expression in LIMIT".into());
            }
        };

        let fetch = match limit.get_fetch_type()? {
            FetchType::Literal(Some(n)) => n,
            FetchType::Literal(None) => child.table.len(),
            FetchType::UnsupportedExpr => {
                return Err("Unsupported FETCH expression in LIMIT".into());
            }
        };

        let end = (skip + fetch).min(child.table.len());
        let sliced = child.table.slice(skip, end).call()?;
        Ok(PlanResult {
            table: sliced,
            col_names: child.col_names,
        })
    }

    fn exec_join(&self, join: &datafusion::logical_expr::Join) -> Result<PlanResult, BoxError> {
        let left = self.execute_plan(join.left.as_ref())?;
        let right = self.execute_plan(join.right.as_ref())?;

        let mut left_on: Vec<i32> = Vec::new();
        let mut right_on: Vec<i32> = Vec::new();

        for (l_expr, r_expr) in &join.on {
            let l_name = expr_output_name(l_expr);
            let r_name = expr_output_name(r_expr);
            let l_idx = find_column(&l_name, &left.col_names)?;
            let r_idx = find_column(&r_name, &right.col_names)?;
            #[allow(clippy::cast_possible_truncation)]
            {
                left_on.push(l_idx as i32);
                right_on.push(r_idx as i32);
            }
        }

        let joined_table = match join.join_type {
            JoinType::Inner => left
                .table
                .inner_join(&right.table, &left_on, &right_on)
                .call()?,
            JoinType::Left => left
                .table
                .left_join(&right.table, &left_on, &right_on)
                .call()?,
            JoinType::Full => left
                .table
                .full_join(&right.table, &left_on, &right_on)
                .call()?,
            JoinType::LeftSemi => left
                .table
                .left_semi_join(&right.table, &left_on, &right_on)
                .call()?,
            JoinType::LeftAnti => left
                .table
                .left_anti_join(&right.table, &left_on, &right_on)
                .call()?,
            other => return Err(format!("Unsupported join type: {other:?}").into()),
        };

        let col_names = match join.join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => left.col_names,
            _ => {
                let mut names = left.col_names;
                names.extend(right.col_names);
                names
            }
        };

        Ok(PlanResult {
            table: joined_table,
            col_names,
        })
    }
}

/// Resolves an aggregate expression to (value_column_index, AggregationKind,
/// output_name).
///
/// Computed aggregate arguments are pre-materialized as columns named
/// `__agg_arg_0`, `__agg_arg_1`, etc. The `computed_counter` tracks which
/// one to use next.
fn resolve_aggregate(
    expr: &Expr,
    col_names: &[String],
    computed_counter: &mut usize,
) -> Result<(i32, AggregationKind, String), BoxError> {
    let (inner, alias_name) = match expr {
        Expr::Alias(a) => (a.expr.as_ref(), Some(a.name.clone())),
        other => (other, None),
    };

    match inner {
        Expr::AggregateFunction(agg_fn) => {
            let kind = map_agg_function(agg_fn.func.name())?;
            let out_name = alias_name.unwrap_or_else(|| format!("{inner}"));

            // For COUNT(*) there may be no args — use column 0 as a dummy.
            if agg_fn.params.args.is_empty() {
                return Ok((0, kind, out_name));
            }

            let arg = &agg_fn.params.args[0];

            // Simple column ref.
            if let Expr::Column(col) = arg {
                if let Ok(idx) = find_column(&col.name, col_names) {
                    #[allow(clippy::cast_possible_truncation)]
                    return Ok((idx as i32, kind, out_name));
                }
            }

            // Literal argument (e.g. COUNT(1)) — use column 0 as dummy.
            if matches!(arg, Expr::Literal(_)) {
                return Ok((0, kind, out_name));
            }

            // Computed argument — find the pre-materialized column.
            let computed_name = format!("__agg_arg_{computed_counter}");
            *computed_counter += 1;
            let idx = find_column(&computed_name, col_names)?;
            #[allow(clippy::cast_possible_truncation)]
            Ok((idx as i32, kind, out_name))
        }
        _ => Err(format!("Expected aggregate function, got: {inner}").into()),
    }
}

/// Maps a DataFusion aggregate function name to a cudf `AggregationKind`.
fn map_agg_function(name: &str) -> Result<AggregationKind, BoxError> {
    match name.to_lowercase().as_str() {
        "sum" => Ok(AggregationKind::SUM),
        "count" => Ok(AggregationKind::COUNT),
        "min" => Ok(AggregationKind::MIN),
        "max" => Ok(AggregationKind::MAX),
        "avg" | "mean" => Ok(AggregationKind::MEAN),
        "nunique" | "count_distinct" => Ok(AggregationKind::NUNIQUE),
        "median" => Ok(AggregationKind::MEDIAN),
        "stddev" | "stddev_samp" => Ok(AggregationKind::STD),
        "variance" | "var_samp" => Ok(AggregationKind::VAR),
        _ => Err(format!("Unsupported aggregate function: {name}").into()),
    }
}

/// Extracts the output name from an expression.
fn expr_output_name(expr: &Expr) -> String {
    match expr {
        Expr::Column(col) => col.name.clone(),
        Expr::Alias(alias) => alias.name.clone(),
        _ => format!("{expr}"),
    }
}

/// Converts a DataFusion catalog schema (using DataFusion's arrow types)
/// to a DataFusion-compatible Arrow Schema.
///
/// This is a pass-through since GpuCatalog already stores schemas using
/// DataFusion arrow types via the `catalog::type_id_to_df_arrow` function.
fn catalog_schema_to_df(
    schema: &datafusion::arrow::datatypes::Schema,
) -> Result<DfSchema, BoxError> {
    // The catalog schema is already built with DataFusion's arrow types,
    // so no conversion is needed.
    Ok(schema.clone())
}

/// Converts a GPU table to a DataFusion Arrow RecordBatch (v54).
///
/// We manually extract data from each column using cudf's `to_vec_*` methods
/// and build DataFusion-compatible arrow arrays.
pub fn gpu_table_to_df_record_batch(
    table: &Table,
    col_names: &[String],
) -> Result<DfRecordBatch, BoxError> {
    let ncols = table.columns_len();
    let mut fields = Vec::with_capacity(ncols);
    let mut arrays: Vec<DfArrayRef> = Vec::with_capacity(ncols);

    for i in 0..ncols {
        let view = table.column(i)?;
        let tid = view.type_id();
        let arrow_dt = type_id_to_df_arrow(tid)?;
        let name = col_names.get(i).cloned().unwrap_or_else(|| format!("c{i}"));
        let nullable = view.has_nulls();
        fields.push(DfField::new(name, arrow_dt, nullable));
        arrays.push(gpu_column_to_df_array(&view, tid)?);
    }

    let schema = Arc::new(DfSchema::new(fields));
    Ok(DfRecordBatch::try_new(schema, arrays)?)
}

/// Converts a single cudf ColumnView to a DataFusion arrow array (v54).
fn gpu_column_to_df_array(
    view: &cudf::column::ColumnView<'_>,
    tid: TypeId,
) -> Result<DfArrayRef, BoxError> {
    match tid {
        TypeId::INT8 => {
            let vals = view.to_vec_i8().call()?;
            Ok(Arc::new(DfInt8Array::from(vals)))
        }
        TypeId::INT16 => {
            let vals = view.to_vec_i16().call()?;
            Ok(Arc::new(DfInt16Array::from(vals)))
        }
        TypeId::INT32 => {
            let vals = view.to_vec_i32().call()?;
            Ok(Arc::new(DfInt32Array::from(vals)))
        }
        TypeId::INT64 => {
            let vals = view.to_vec_i64().call()?;
            Ok(Arc::new(DfInt64Array::from(vals)))
        }
        TypeId::UINT8 => {
            let vals = view.to_vec_u8().call()?;
            Ok(Arc::new(DfUint8Array::from(vals)))
        }
        TypeId::UINT16 => {
            let vals = view.to_vec_u16().call()?;
            Ok(Arc::new(DfUint16Array::from(vals)))
        }
        TypeId::UINT32 => {
            let vals = view.to_vec_u32().call()?;
            Ok(Arc::new(DfUint32Array::from(vals)))
        }
        TypeId::UINT64 => {
            let vals = view.to_vec_u64().call()?;
            Ok(Arc::new(DfUint64Array::from(vals)))
        }
        TypeId::FLOAT32 => {
            let vals = view.to_vec_f32().call()?;
            Ok(Arc::new(DfFloat32Array::from(vals)))
        }
        TypeId::FLOAT64 => {
            let vals = view.to_vec_f64().call()?;
            Ok(Arc::new(DfFloat64Array::from(vals)))
        }
        TypeId::BOOL8 => {
            let vals = view.to_vec_bool().call()?;
            Ok(Arc::new(DfBooleanArray::from(vals)))
        }
        TypeId::STRING => {
            // Need to get owned column for to_vec_string
            let owned = view.to_owned_column().call()?;
            let vals = owned.to_vec_string().call()?;
            let str_refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
            Ok(Arc::new(DfStringArray::from(str_refs)))
        }
        TypeId::TIMESTAMP_DAYS => {
            // TIMESTAMP_DAYS stores days-since-epoch as i32, same as Arrow Date32.
            let vals = view.to_vec_i32().call()?;
            Ok(Arc::new(DfDate32Array::from(vals)))
        }
        _ => Err(format!("Unsupported TypeId for arrow conversion: {tid}").into()),
    }
}
