// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Binary and unary operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

#[doc(alias = "binary_operator")]
pub use cudf_sys::ffi::BinaryOperator;
#[doc(alias = "rounding_method")]
pub use cudf_sys::ffi::RoundingMethod;
#[doc(alias = "unary_operator")]
pub use cudf_sys::ffi::UnaryOperator;

/// Builder for a column-column binary operation.
///
/// Created by [`ColumnView::binary_op`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
pub struct BinaryOpColumns<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    op: BinaryOperator,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for BinaryOpColumns<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.lhs.0,
            self.rhs.0,
            self.op,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for a column-scalar binary operation.
///
/// Created by [`ColumnView::binary_op_scalar`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
pub struct BinaryOpColumnScalar<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a Scalar,
    op: BinaryOperator,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for BinaryOpColumnScalar<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let rhs_ffi = crate::scalar::scalar_to_ffi(self.rhs);
        let col = cudf_sys::binaryop::ffi::binary_operation_column_scalar(
            self.lhs.0,
            &rhs_ffi,
            self.op,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for a scalar-column binary operation.
///
/// Created by [`scalar_binary_op`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
pub struct ScalarBinaryOp<'a> {
    lhs: &'a Scalar,
    rhs: &'a ColumnView<'a>,
    op: BinaryOperator,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for ScalarBinaryOp<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let lhs_ffi = crate::scalar::scalar_to_ffi(self.lhs);
        let col = cudf_sys::binaryop::ffi::binary_operation_scalar_column(
            &lhs_ffi,
            self.rhs.0,
            self.op,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

impl<'a> ColumnView<'a> {
    /// Applies a binary operation between two columns.
    ///
    /// Returns a [`BinaryOpColumns`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    pub fn binary_op(
        &'a self,
        rhs: &'a ColumnView<'a>,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> BinaryOpColumns<'a> {
        BinaryOpColumns {
            lhs: self,
            rhs,
            op,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Applies a binary operation between a column and a scalar.
    ///
    /// Returns a [`BinaryOpColumnScalar`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn binary_op_scalar(
        &'a self,
        rhs: &'a Scalar,
        op: BinaryOperator,
        output_type: TypeId,
    ) -> BinaryOpColumnScalar<'a> {
        BinaryOpColumnScalar {
            lhs: self,
            rhs,
            op,
            output_type,
            stream: Stream::default_stream(),
        }
    }
}

/// Applies a binary operation with a scalar on the left and a column on the right.
///
/// Returns a [`ScalarBinaryOp`] builder. Use `.stream()` to set a custom
/// CUDA stream, then `.call()` to execute.
pub fn scalar_binary_op<'a>(
    lhs: &'a Scalar,
    rhs: &'a ColumnView<'a>,
    op: BinaryOperator,
    output_type: TypeId,
) -> ScalarBinaryOp<'a> {
    ScalarBinaryOp {
        lhs,
        rhs,
        op,
        output_type,
        stream: Stream::default_stream(),
    }
}

/// Compute the output scale for a fixed-point binary operation.
pub fn binary_operation_fixed_point_scale(
    op: BinaryOperator,
    left_scale: i32,
    right_scale: i32,
) -> i32 {
    cudf_sys::binaryop::ffi::binary_operation_fixed_point_scale(op.repr, left_scale, right_scale)
}

/// Check if a binary operation is supported for the given type combination.
pub fn is_supported_binaryop(out: TypeId, lhs: TypeId, rhs: TypeId, op: BinaryOperator) -> bool {
    cudf_sys::binaryop::ffi::is_supported_binaryop(out.repr, lhs.repr, rhs.repr, op.repr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::GpuOp;

    // -- Binary operation tests --

    #[test]
    fn add_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(4), 3);
        let result = c1.view().add(&c2.view(), TypeId::INT32).call().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![5, 5, 5]);
    }

    #[test]
    fn sub_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 3);
        let result = c1.view().sub(&c2.view(), TypeId::INT32).call().unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7, 7]);
    }

    #[test]
    fn mul_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(2.0), 3);
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 3);
        let result = c1.view().mul(&c2.view(), TypeId::FLOAT64).call().unwrap();
        assert_eq!(result.to_vec_f64(), vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn div_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(10.0), 2);
        let c2 = Column::from_scalar(&Scalar::from_f64(4.0), 2);
        let result = c1.view().div(&c2.view(), TypeId::FLOAT64).call().unwrap();
        assert_eq!(result.to_vec_f64(), vec![2.5, 2.5]);
    }

    #[test]
    fn eq_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let result = c1.view().eq(&c2.view()).call().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![true, true, true]);
    }

    #[test]
    fn lt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = c1.view().lt(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn add_column_scalar() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let s = Scalar::from_i32(10);
        let result = col
            .view()
            .binary_op_scalar(&s, BinaryOperator::ADD, TypeId::INT32)
            .call()
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
            .call()
            .unwrap();
        let vals = result.to_vec_f64();
        assert!((vals[0] - 1024.0).abs() < 1e-6);
    }

    // -- Unary operation tests --

    #[test]
    fn cast_i32_to_f64() {
        let col = Column::from_scalar(&Scalar::from_i32(42), 2);
        let result = col.view().cast(TypeId::FLOAT64).call().unwrap();
        assert_eq!(result.type_id(), TypeId::FLOAT64);
        assert_eq!(result.to_vec_f64(), vec![42.0, 42.0]);
    }

    #[test]
    fn cast_f64_to_i32() {
        let col = Column::from_scalar(&Scalar::from_f64(3.7), 2);
        let result = col.view().cast(TypeId::INT32).call().unwrap();
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![3, 3]);
    }

    #[test]
    fn is_null_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let result = col.view().is_null().call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![false, false, false]);
    }

    #[test]
    fn is_valid_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3);
        let result = col.view().is_valid().call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true, true]);
    }

    #[test]
    fn negate_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = col.view().negate().call().unwrap();
        assert_eq!(result.to_vec_i32(), vec![-5, -5]);
    }

    #[test]
    fn abs_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(-7), 2);
        let result = col.view().abs().call().unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7]);
    }

    #[test]
    fn ne_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3);
        let result = c1.view().ne(&c2.view()).call().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![false, false, false]);
    }

    #[test]
    fn ge_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let result = c1.view().ge(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn gt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let result = c1.view().gt(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn le_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 2);
        let result = c1.view().le(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true]);
    }

    #[test]
    fn is_nan_basic() {
        let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]);
        let result = col.view().is_nan().call().unwrap();
        assert_eq!(result.to_vec_bool(), vec![false, true, false]);
    }

    #[test]
    fn scalar_column_binary_op() {
        let s = Scalar::from_i32(10);
        let col = Column::from_scalar(&Scalar::from_i32(3), 2);
        let result = scalar_binary_op(&s, &col.view(), BinaryOperator::SUB, TypeId::INT32)
            .call()
            .unwrap();
        assert_eq!(result.to_vec_i32(), vec![7, 7]);
    }
}
