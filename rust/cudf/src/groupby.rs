// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Group-by aggregation operations on GPU tables.
//!
//! Group-by partitions a table into groups defined by one or more key columns,
//! then applies an aggregation function to a value column within each group.
//! The output table contains the unique key combinations followed by the
//! aggregated result column(s).
//!
//! Two entry points are available on [`Table`](crate::table::Table):
//!
//! - [`groupby`](crate::table::Table::groupby) -- single aggregation on one value column.
//! - [`groupby_multi`](crate::table::Table::groupby_multi) -- multiple aggregations on
//!   (potentially different) value columns in a single pass.
//!
//! The output row order is **not** guaranteed to match the input order; sort the
//! result if a deterministic order is required.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::column::Column;
//! use cudf::groupby::AggregationKind;
//! use cudf::stream::GpuOp;
//! use cudf::table::TableBuilder;
//!
//! let keys = Column::from_slice_i32(&[1, 1, 2, 2]).call()?;
//! let vals = Column::from_slice_i32(&[10, 20, 30, 40]).call()?;
//! let mut b = TableBuilder::new();
//! b.push_column(keys);
//! b.push_column(vals);
//! let table = b.build()?;
//!
//! // Sum values grouped by column 0
//! let result = table.groupby(&[0], 1, AggregationKind::SUM).call()?;
//! assert_eq!(result.columns_len(), 2); // key column + aggregated column
//! # Ok::<(), cudf::error::Error>(())
//! ```

/// Aggregation operation kind for groupby.
///
/// Selects the aggregation function applied to each group's values.
/// Common variants include:
///
/// | Variant | Description |
/// |---|---|
/// | `SUM` | Sum of values in the group |
/// | `MIN` | Minimum value in the group |
/// | `MAX` | Maximum value in the group |
/// | `MEAN` | Arithmetic mean (result is `FLOAT64`) |
/// | `COUNT` | Number of non-null values |
/// | `NUNIQUE` | Number of distinct values |
/// | `MEDIAN` | Median value (result is `FLOAT64`) |
/// | `STD` | Sample standard deviation |
/// | `VAR` | Sample variance |
/// | `PRODUCT` | Product of values |
/// | `ANY` | Logical OR (any true) |
/// | `ALL` | Logical AND (all true) |
/// | `ARGMAX` | Index of maximum value |
/// | `ARGMIN` | Index of minimum value |
/// | `COLLECT_LIST` | Collect values into a list |
/// | `COLLECT_SET` | Collect distinct values into a list |
/// | `SUM_OF_SQUARES` | Sum of squared values |
#[doc(alias = "aggregation")]
pub use cudf_sys::ffi::AggregationKind;

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::data_type::TypeId;
    use crate::groupby::AggregationKind;
    use crate::stream::GpuOp;
    use crate::table::{TableBuilder, UnboundTable};

    /// Helper to build a two-column table: keys (i32) and values (i32).
    fn make_kv_table(keys: &[i32], values: &[i32]) -> UnboundTable {
        let key_col = Column::from_slice_i32(keys).call().unwrap();
        let val_col = Column::from_slice_i32(values).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(key_col);
        builder.push_column(val_col);
        builder.build().unwrap()
    }

    fn col_to_host_i32(tbl: &UnboundTable, col_idx: usize) -> Vec<i32> {
        let view = tbl.column(col_idx).unwrap();
        view.cast(TypeId::INT32)
            .call()
            .unwrap()
            .to_vec_i32()
            .call()
            .unwrap()
    }

    fn col_to_host_i64(tbl: &UnboundTable, col_idx: usize) -> Vec<i64> {
        let view = tbl.column(col_idx).unwrap();
        view.cast(TypeId::INT64)
            .call()
            .unwrap()
            .to_vec_i64()
            .call()
            .unwrap()
    }

    fn col_to_host_f64(tbl: &UnboundTable, col_idx: usize) -> Vec<f64> {
        let view = tbl.column(col_idx).unwrap();
        view.cast(TypeId::FLOAT64)
            .call()
            .unwrap()
            .to_vec_f64()
            .call()
            .unwrap()
    }

    fn sorted_kv_i32(result: &UnboundTable) -> (Vec<i32>, Vec<i32>) {
        let sorted = result.sort(&[], &[]).call().unwrap();
        (col_to_host_i32(&sorted, 0), col_to_host_i32(&sorted, 1))
    }

    fn sorted_kv_i64(result: &UnboundTable) -> (Vec<i32>, Vec<i64>) {
        let sorted = result.sort(&[], &[]).call().unwrap();
        (col_to_host_i32(&sorted, 0), col_to_host_i64(&sorted, 1))
    }

    fn sorted_kv_f64(result: &UnboundTable) -> (Vec<i32>, Vec<f64>) {
        let sorted = result.sort(&[], &[]).call().unwrap();
        (col_to_host_i32(&sorted, 0), col_to_host_f64(&sorted, 1))
    }

    #[test]
    fn groupby_sum() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[1, 2, 3, 4]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::SUM)
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        let (keys, vals) = sorted_kv_i64(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![3, 7]);
    }

    #[test]
    fn groupby_min() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[5, 2, 8, 3]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::MIN)
            .call()
            .unwrap();
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn groupby_max() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[5, 2, 8, 3]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::MAX)
            .call()
            .unwrap();
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![5, 8]);
    }

    #[test]
    fn groupby_mean() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[10, 20, 30, 50]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::MEAN)
            .call()
            .unwrap();
        let (keys, vals) = sorted_kv_f64(&result);
        assert_eq!(keys, vec![1, 2]);
        assert!((vals[0] - 15.0).abs() < 1e-9);
        assert!((vals[1] - 40.0).abs() < 1e-9);
    }

    #[test]
    fn groupby_count() {
        let tbl = make_kv_table(&[1, 1, 1, 2, 2], &[10, 20, 30, 40, 50]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::COUNT)
            .call()
            .unwrap();
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![3, 2]);
    }

    #[test]
    fn groupby_nunique() {
        let tbl = make_kv_table(&[1, 1, 1, 2, 2], &[10, 10, 20, 30, 30]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::NUNIQUE)
            .call()
            .unwrap();
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![2, 1]);
    }

    #[test]
    fn groupby_multi_agg() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[10, 20, 30, 40]);
        let result = tbl
            .groupby_multi(
                &[0i32],
                &[1i32, 1i32],
                &[AggregationKind::SUM, AggregationKind::MIN],
            )
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 3);

        let sorted = result.sort(&[], &[]).call().unwrap();
        let keys = col_to_host_i32(&sorted, 0);
        let sums = col_to_host_i64(&sorted, 1);
        let mins = col_to_host_i32(&sorted, 2);

        assert_eq!(keys, vec![1, 2]);
        assert_eq!(sums, vec![30, 70]);
        assert_eq!(mins, vec![10, 30]);
    }

    #[test]
    fn groupby_single_group() {
        let tbl = make_kv_table(&[1, 1, 1, 1], &[10, 20, 30, 40]);
        let result = tbl
            .groupby(&[0i32], 1, AggregationKind::SUM)
            .call()
            .unwrap();
        assert_eq!(result.len(), 1);
        let (keys, vals) = sorted_kv_i64(&result);
        assert_eq!(keys, vec![1]);
        assert_eq!(vals, vec![100]);
    }
}
