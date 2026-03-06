// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Binary and unary operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

pub use cudf_sys::ffi::BinaryOperator;
pub use cudf_sys::ffi::UnaryOperator;
pub use cudf_sys::ffi::RoundingMethod;

/// Extension trait for generic binary operations.
pub trait BinaryOp<Rhs> {
    /// Applies a binary operation.
    fn binary_op(&self, rhs: &Rhs, op: BinaryOperator, output_type: TypeId) -> Result<Column>;
    /// binary_op on a custom CUDA stream.
    fn binary_op_on(
        &self,
        rhs: &Rhs,
        op: BinaryOperator,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column>;
}

impl BinaryOp<ColumnView<'_>> for ColumnView<'_> {
    fn binary_op(
        &self,
        rhs: &ColumnView<'_>,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> Result<Column> {
        self.binary_op_on(rhs, op, output_type, Stream::default_stream())
    }
    fn binary_op_on(
        &self,
        rhs: &ColumnView<'_>,
        op: BinaryOperator,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            op,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

impl BinaryOp<Scalar> for ColumnView<'_> {
    fn binary_op(
        &self,
        rhs: &Scalar,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> Result<Column> {
        self.binary_op_on(rhs, op, output_type, Stream::default_stream())
    }
    fn binary_op_on(
        &self,
        rhs: &Scalar,
        op: BinaryOperator,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let rhs_ffi = crate::scalar::scalar_to_ffi(rhs);
        let col = cudf_sys::ffi::binary_operation_column_scalar(
            self.0,
            &rhs_ffi,
            op,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

impl BinaryOp<ColumnView<'_>> for Scalar {
    fn binary_op(
        &self,
        rhs: &ColumnView<'_>,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> Result<Column> {
        self.binary_op_on(rhs, op, output_type, Stream::default_stream())
    }
    fn binary_op_on(
        &self,
        rhs: &ColumnView<'_>,
        op: BinaryOperator,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let lhs_ffi = crate::scalar::scalar_to_ffi(self);
        let col = cudf_sys::ffi::binary_operation_scalar_column(
            &lhs_ffi,
            rhs.0,
            op,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
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
        let result = c1.view().add(&c2.view(), TypeId::INT32).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![5, 5, 5]);
    }

    #[test]
    fn sub_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 3);
        let result = c1.view().sub(&c2.view(), TypeId::INT32).unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7, 7]);
    }

    #[test]
    fn mul_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(2.0), 3);
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 3);
        let result = c1.view().mul(&c2.view(), TypeId::FLOAT64).unwrap();
        assert_eq!(result.to_vec_f64(), vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn div_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(10.0), 2);
        let c2 = Column::from_scalar(&Scalar::from_f64(4.0), 2);
        let result = c1.view().div(&c2.view(), TypeId::FLOAT64).unwrap();
        assert_eq!(result.to_vec_f64(), vec![2.5, 2.5]);
    }

    #[test]
    fn eq_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let result = c1.view().eq(&c2.view()).unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![true, true, true]);
    }

    #[test]
    fn lt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = c1.view().lt(&c2.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn add_column_scalar() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let s = Scalar::from_i32(10);
        let result = col
            .view()
            .binary_op(&s, BinaryOperator::ADD, TypeId::INT32)
            .unwrap();
        assert_eq!(result.to_vec_i32(), vec![11, 11, 11]);
    }

    #[test]
    fn binary_op_generic() {
        let c1 = Column::from_scalar(&Scalar::from_i32(2), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(10), 2);
        let result = c1
            .view()
            .binary_op(&c2.view(), BinaryOperator::POW, TypeId::FLOAT64)
            .unwrap();
        let vals = result.to_vec_f64();
        assert!((vals[0] - 1024.0).abs() < 1e-6);
    }

    // -- Unary operation tests --

    #[test]
    fn cast_i32_to_f64() {
        let col = Column::from_scalar(&Scalar::from_i32(42), 2);
        let result = col.view().cast(TypeId::FLOAT64).unwrap();
        assert_eq!(result.type_id(), TypeId::FLOAT64);
        assert_eq!(result.to_vec_f64(), vec![42.0, 42.0]);
    }

    #[test]
    fn cast_f64_to_i32() {
        let col = Column::from_scalar(&Scalar::from_f64(3.7), 2);
        let result = col.view().cast(TypeId::INT32).unwrap();
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![3, 3]);
    }

    #[test]
    fn is_null_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let result = col.view().is_null().unwrap();
        assert_eq!(result.to_vec_bool(), vec![false, false, false]);
    }

    #[test]
    fn is_valid_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let result = col.view().is_valid().unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true, true]);
    }

    #[test]
    fn negate_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = col.view().negate().unwrap();
        assert_eq!(result.to_vec_i32(), vec![-5, -5]);
    }

    #[test]
    fn abs_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(-7), 2);
        let result = col.view().abs().unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7]);
    }

    #[test]
    fn ne_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let result = c1.view().ne(&c2.view()).unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![false, false, false]);
    }

    #[test]
    fn ge_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = c1.view().ge(&c2.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn gt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let result = c1.view().gt(&c2.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn le_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let result = c1.view().le(&c2.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn is_nan_basic() {
        let ds = crate::stream::Stream::default_stream().as_raw();
        let col = Column(cudf_sys::ffi::make_column_from_host_f64(&[1.0, f64::NAN, 3.0], ds));
        let result = col.view().is_nan().unwrap();
        assert_eq!(result.to_vec_bool(), vec![false, true, false]);
    }

    #[test]
    fn scalar_column_binary_op() {
        let s = Scalar::from_i32(10);
        let col = Column::from_scalar(&Scalar::from_i32(3), 2);
        let result = s.binary_op(&col.view(), BinaryOperator::SUB, TypeId::INT32).unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7]);
    }
}
