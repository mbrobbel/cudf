// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Expression evaluation: converts DataFusion `Expr` to cudf column operations.

use cudf::column::Column;
use cudf::data_type::TypeId;
use cudf::ops::BinaryOperator;
use cudf::scalar::Scalar;
use cudf::stream::GpuOp;
use cudf::table::Table;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, Operator};

type BoxError = Box<dyn std::error::Error>;

/// Evaluates a DataFusion `Expr` against a cudf `Table`, producing a `Column`.
///
/// The `col_names` slice maps positional column indices to their names so that
/// `Expr::Column` references can be resolved.
pub fn eval_expr(expr: &Expr, table: &Table, col_names: &[String]) -> Result<Column, BoxError> {
    match expr {
        Expr::Column(col) => {
            let idx = find_column(&col.name, col_names)?;
            Ok(table.column(idx)?.to_owned_column().call()?)
        }
        Expr::Literal(scalar_value) => {
            let scalar = scalar_to_cudf(scalar_value)?;
            Ok(Column::from_scalar(&scalar, table.len()).call()?)
        }
        Expr::BinaryExpr(binary) => {
            let left = eval_expr(&binary.left, table, col_names)?;
            let right = eval_expr(&binary.right, table, col_names)?;
            apply_binary_op(&binary.op, &left, &right)
        }
        Expr::Alias(alias) => eval_expr(&alias.expr, table, col_names),
        Expr::Cast(cast) => {
            let inner = eval_expr(&cast.expr, table, col_names)?;
            let target = df_arrow_type_to_cudf(&cast.data_type)?;
            Ok(inner.view().cast(target).call()?)
        }
        Expr::TryCast(cast) => {
            let inner = eval_expr(&cast.expr, table, col_names)?;
            let target = df_arrow_type_to_cudf(&cast.data_type)?;
            Ok(inner.view().cast(target).call()?)
        }
        Expr::AggregateFunction(_) => {
            Err("Aggregate functions should be evaluated at the Aggregate plan node".into())
        }
        #[allow(deprecated)]
        Expr::Wildcard { .. } => {
            Err("Wildcard expressions should be resolved before evaluation".into())
        }
        _ => Err(format!("Unsupported expression: {expr}").into()),
    }
}

/// Resolves a column name to its positional index.
pub fn find_column(name: &str, col_names: &[String]) -> Result<usize, BoxError> {
    col_names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| format!("column not found: {name}").into())
}

/// Converts a DataFusion `ScalarValue` to a cudf `Scalar`.
pub fn scalar_to_cudf(sv: &ScalarValue) -> Result<Scalar, BoxError> {
    match sv {
        ScalarValue::Int8(Some(v)) => Ok(Scalar::from_i8(*v)),
        ScalarValue::Int16(Some(v)) => Ok(Scalar::from_i16(*v)),
        ScalarValue::Int32(Some(v)) => Ok(Scalar::from_i32(*v)),
        ScalarValue::Int64(Some(v)) => Ok(Scalar::from_i64(*v)),
        ScalarValue::UInt8(Some(v)) => Ok(Scalar::from_u8(*v)),
        ScalarValue::UInt16(Some(v)) => Ok(Scalar::from_u16(*v)),
        ScalarValue::UInt32(Some(v)) => Ok(Scalar::from_u32(*v)),
        ScalarValue::UInt64(Some(v)) => Ok(Scalar::from_u64(*v)),
        ScalarValue::Float32(Some(v)) => Ok(Scalar::from_f32(*v)),
        ScalarValue::Float64(Some(v)) => Ok(Scalar::from_f64(*v)),
        ScalarValue::Boolean(Some(v)) => Ok(Scalar::from_bool(*v)),
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => Ok(Scalar::from_string(v)),
        ScalarValue::Date32(Some(v)) => Ok(Scalar::from_i32(*v)), // days since epoch
        ScalarValue::Null => Ok(Scalar::Null(TypeId::INT32)),
        _ => Err(format!("Unsupported scalar value: {sv}").into()),
    }
}

/// Converts a DataFusion Arrow `DataType` to a cudf `TypeId`.
///
/// This uses DataFusion's re-exported arrow types (v54), not the cudf arrow
/// types (v56).
pub fn df_arrow_type_to_cudf(
    dt: &datafusion::arrow::datatypes::DataType,
) -> Result<TypeId, BoxError> {
    use datafusion::arrow::datatypes::DataType;
    match dt {
        DataType::Int8 => Ok(TypeId::INT8),
        DataType::Int16 => Ok(TypeId::INT16),
        DataType::Int32 => Ok(TypeId::INT32),
        DataType::Int64 => Ok(TypeId::INT64),
        DataType::UInt8 => Ok(TypeId::UINT8),
        DataType::UInt16 => Ok(TypeId::UINT16),
        DataType::UInt32 => Ok(TypeId::UINT32),
        DataType::UInt64 => Ok(TypeId::UINT64),
        DataType::Float32 => Ok(TypeId::FLOAT32),
        DataType::Float64 => Ok(TypeId::FLOAT64),
        DataType::Boolean => Ok(TypeId::BOOL8),
        DataType::Utf8 | DataType::LargeUtf8 => Ok(TypeId::STRING),
        DataType::Date32 => Ok(TypeId::TIMESTAMP_DAYS),
        other => Err(format!("Unsupported DataFusion arrow DataType: {other}").into()),
    }
}

/// Applies a DataFusion binary operator to two cudf columns.
///
/// If the columns have mismatched types, a common type is chosen and both
/// sides are cast. For date types (TIMESTAMP_DAYS) compared against INT32,
/// the date column is cast to INT32 since the underlying representation is
/// the same (days-since-epoch).
pub fn apply_binary_op(op: &Operator, left: &Column, right: &Column) -> Result<Column, BoxError> {
    // Reconcile types when they differ.
    let (left_col, right_col);
    if left.type_id() != right.type_id() {
        let common = common_type(left.type_id(), right.type_id());
        left_col = if left.type_id() == common {
            None
        } else {
            Some(left.view().cast(common).call()?)
        };
        right_col = if right.type_id() == common {
            None
        } else {
            Some(right.view().cast(common).call()?)
        };
    } else {
        left_col = None;
        right_col = None;
    }
    let lv = left_col.as_ref().map_or_else(|| left.view(), |c| c.view());
    let rv = right_col.as_ref().map_or_else(|| right.view(), |c| c.view());

    match op {
        Operator::Plus => {
            let out = output_type_for_arithmetic(left.type_id(), right.type_id());
            Ok(lv.add(&rv, out).call()?)
        }
        Operator::Minus => {
            let out = output_type_for_arithmetic(left.type_id(), right.type_id());
            Ok(lv.sub(&rv, out).call()?)
        }
        Operator::Multiply => {
            let out = output_type_for_arithmetic(left.type_id(), right.type_id());
            Ok(lv.mul(&rv, out).call()?)
        }
        Operator::Divide => Ok(lv.div(&rv, TypeId::FLOAT64).call()?),
        Operator::Gt => Ok(lv.gt(&rv).call()?),
        Operator::Lt => Ok(lv.lt(&rv).call()?),
        Operator::GtEq => Ok(lv.ge(&rv).call()?),
        Operator::LtEq => Ok(lv.le(&rv).call()?),
        Operator::Eq => Ok(lv.eq(&rv).call()?),
        Operator::NotEq => Ok(lv.ne(&rv).call()?),
        Operator::And => Ok(lv
            .binary_op(&rv, BinaryOperator::LOGICAL_AND, TypeId::BOOL8)
            .call()?),
        Operator::Or => Ok(lv
            .binary_op(&rv, BinaryOperator::LOGICAL_OR, TypeId::BOOL8)
            .call()?),
        Operator::Modulo => {
            let out = output_type_for_arithmetic(left.type_id(), right.type_id());
            Ok(lv.binary_op(&rv, BinaryOperator::MOD, out).call()?)
        }
        _ => Err(format!("Unsupported binary operator: {op}").into()),
    }
}

/// Finds a common type that both sides can be cast to.
///
/// For date types (TIMESTAMP_DAYS) mixed with integers, we use INT32 since
/// the underlying representation is identical (days-since-epoch as i32).
fn common_type(left: TypeId, right: TypeId) -> TypeId {
    if left == right {
        return left;
    }
    // Date vs integer: use the integer type.
    if left == TypeId::TIMESTAMP_DAYS {
        return right;
    }
    if right == TypeId::TIMESTAMP_DAYS {
        return left;
    }
    // Float promotion.
    if left == TypeId::FLOAT64 || right == TypeId::FLOAT64 {
        return TypeId::FLOAT64;
    }
    if left == TypeId::FLOAT32 || right == TypeId::FLOAT32 {
        return TypeId::FLOAT64;
    }
    // Integer promotion.
    if left == TypeId::INT64 || right == TypeId::INT64 {
        return TypeId::INT64;
    }
    // Default to left.
    left
}

/// Determines the output type for an arithmetic operation between two types.
fn output_type_for_arithmetic(left: TypeId, right: TypeId) -> TypeId {
    if left == TypeId::FLOAT64 || right == TypeId::FLOAT64 {
        return TypeId::FLOAT64;
    }
    if left == TypeId::FLOAT32 || right == TypeId::FLOAT32 {
        return TypeId::FLOAT32;
    }
    if left == TypeId::INT64 || right == TypeId::INT64 {
        return TypeId::INT64;
    }
    TypeId::INT32
}
