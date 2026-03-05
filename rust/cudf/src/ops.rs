// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Binary and unary operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;

pub use cudf_sys::ffi::BinaryOperator;

/// Binary operations between columns and scalars.
pub mod binary {
    use super::*;

    /// Applies a binary operation between two columns.
    pub fn binary_op(
        lhs: &ColumnView<'_>,
        rhs: &ColumnView<'_>,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> Result<Column> {
        let col =
            cudf_sys::ffi::binary_operation_columns(lhs.0, rhs.0, op, output_type.repr)?;
        Ok(Column(col))
    }

    /// Applies a binary operation between a column and a scalar.
    pub fn binary_op_column_scalar(
        lhs: &ColumnView<'_>,
        rhs: &Scalar,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_column_scalar(
            lhs.0,
            &rhs.0,
            op,
            output_type.repr,
        )?;
        Ok(Column(col))
    }

    /// Applies a binary operation between a scalar and a column.
    pub fn binary_op_scalar_column(
        lhs: &Scalar,
        rhs: &ColumnView<'_>,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_scalar_column(
            &lhs.0,
            rhs.0,
            op,
            output_type.repr,
        )?;
        Ok(Column(col))
    }

    /// Element-wise addition of two columns.
    pub fn add(
        lhs: &ColumnView<'_>,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::ADD, output_type)
    }

    /// Element-wise subtraction of two columns.
    pub fn sub(
        lhs: &ColumnView<'_>,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::SUB, output_type)
    }

    /// Element-wise multiplication of two columns.
    pub fn mul(
        lhs: &ColumnView<'_>,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::MUL, output_type)
    }

    /// Element-wise division of two columns.
    pub fn div(
        lhs: &ColumnView<'_>,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::DIV, output_type)
    }

    /// Element-wise equality comparison.
    pub fn eq(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::EQUAL, TypeId::BOOL8)
    }

    /// Element-wise not-equal comparison.
    pub fn ne(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::NOT_EQUAL, TypeId::BOOL8)
    }

    /// Element-wise less-than comparison.
    pub fn lt(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::LESS, TypeId::BOOL8)
    }

    /// Element-wise greater-than comparison.
    pub fn gt(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::GREATER, TypeId::BOOL8)
    }

    /// Element-wise less-than-or-equal comparison.
    pub fn le(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::LESS_EQUAL, TypeId::BOOL8)
    }

    /// Element-wise greater-than-or-equal comparison.
    pub fn ge(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
        binary_op(lhs, rhs, BinaryOperator::GREATER_EQUAL, TypeId::BOOL8)
    }

    /// Adds a scalar to every element of a column.
    pub fn add_scalar(
        lhs: &ColumnView<'_>,
        rhs: &Scalar,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op_column_scalar(lhs, rhs, BinaryOperator::ADD, output_type)
    }

    /// Subtracts a scalar from every element of a column.
    pub fn sub_scalar(
        lhs: &ColumnView<'_>,
        rhs: &Scalar,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op_column_scalar(lhs, rhs, BinaryOperator::SUB, output_type)
    }

    /// Multiplies every element of a column by a scalar.
    pub fn mul_scalar(
        lhs: &ColumnView<'_>,
        rhs: &Scalar,
        output_type: TypeId,
    ) -> Result<Column> {
        binary_op_column_scalar(lhs, rhs, BinaryOperator::MUL, output_type)
    }
}

/// Unary operations on columns.
pub mod unary {
    use super::*;

    /// Casts a column to a different type.
    pub fn cast(col: &ColumnView<'_>, target: TypeId) -> Result<Column> {
        let c = cudf_sys::ffi::unary_cast(col.0, target.repr)?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates a null value.
    pub fn is_null(col: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_null(col.0)?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates a valid value.
    pub fn is_valid(col: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_valid(col.0)?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates NaN.
    pub fn is_nan(col: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_nan(col.0)?;
        Ok(Column(c))
    }

    /// Negates every element of the column.
    pub fn negate(col: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::unary_negate(col.0)?;
        Ok(Column(c))
    }

    /// Returns the absolute value of every element.
    pub fn abs(col: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::unary_abs(col.0)?;
        Ok(Column(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Binary operation tests --

    #[test]
    fn add_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(4), 3);
        let result = binary::add(&c1.view(), &c2.view(), TypeId::INT32).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![5, 5, 5]);
    }

    #[test]
    fn sub_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 3);
        let result = binary::sub(&c1.view(), &c2.view(), TypeId::INT32).unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7, 7]);
    }

    #[test]
    fn mul_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(2.0), 3);
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 3);
        let result = binary::mul(&c1.view(), &c2.view(), TypeId::FLOAT64).unwrap();
        assert_eq!(result.to_vec_f64(), vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn div_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(10.0), 2);
        let c2 = Column::from_scalar(&Scalar::from_f64(4.0), 2);
        let result = binary::div(&c1.view(), &c2.view(), TypeId::FLOAT64).unwrap();
        assert_eq!(result.to_vec_f64(), vec![2.5, 2.5]);
    }

    #[test]
    fn eq_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let result = binary::eq(&c1.view(), &c2.view()).unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![true, true, true]);
    }

    #[test]
    fn lt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = binary::lt(&c1.view(), &c2.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn add_column_scalar() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let s = Scalar::from_i32(10);
        let result = binary::add_scalar(&col.view(), &s, TypeId::INT32).unwrap();
        assert_eq!(result.to_vec_i32(), vec![11, 11, 11]);
    }

    #[test]
    fn binary_op_generic() {
        let c1 = Column::from_scalar(&Scalar::from_i32(2), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(10), 2);
        let result =
            binary::binary_op(&c1.view(), &c2.view(), BinaryOperator::POW, TypeId::FLOAT64)
                .unwrap();
        let vals = result.to_vec_f64();
        assert!((vals[0] - 1024.0).abs() < 1e-6);
    }

    // -- Unary operation tests --

    #[test]
    fn cast_i32_to_f64() {
        let col = Column::from_scalar(&Scalar::from_i32(42), 2);
        let result = unary::cast(&col.view(), TypeId::FLOAT64).unwrap();
        assert_eq!(result.type_id(), TypeId::FLOAT64);
        assert_eq!(result.to_vec_f64(), vec![42.0, 42.0]);
    }

    #[test]
    fn cast_f64_to_i32() {
        let col = Column::from_scalar(&Scalar::from_f64(3.7), 2);
        let result = unary::cast(&col.view(), TypeId::INT32).unwrap();
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![3, 3]);
    }

    #[test]
    fn is_null_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let result = unary::is_null(&col.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![false, false, false]);
    }

    #[test]
    fn is_valid_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let result = unary::is_valid(&col.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true, true]);
    }

    #[test]
    fn negate_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = unary::negate(&col.view()).unwrap();
        assert_eq!(result.to_vec_i32(), vec![-5, -5]);
    }

    #[test]
    fn abs_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(-7), 2);
        let result = unary::abs(&col.view()).unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7]);
    }
}
