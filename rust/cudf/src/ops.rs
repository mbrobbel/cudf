// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Binary and unary operations on GPU columns.
//!
//! This module re-exports the [`BinaryOperator`], [`UnaryOperator`], and
//! [`RoundingMethod`] enums and provides the generic binary-operation builders
//! ([`BinaryOpColumns`], [`BinaryOpColumnScalar`], [`ScalarBinaryOp`]).
//!
//! Most users will interact with the convenience methods on
//! [`ColumnView`] instead of using these builders
//! directly:
//!
//! - Arithmetic: [`add`](crate::column::ColumnView::add),
//!   [`sub`](crate::column::ColumnView::sub),
//!   [`mul`](crate::column::ColumnView::mul),
//!   [`div`](crate::column::ColumnView::div)
//! - Comparison: [`eq`](crate::column::ColumnView::eq),
//!   [`ne`](crate::column::ColumnView::ne),
//!   [`lt`](crate::column::ColumnView::lt),
//!   [`gt`](crate::column::ColumnView::gt),
//!   [`le`](crate::column::ColumnView::le),
//!   [`ge`](crate::column::ColumnView::ge)
//! - Unary: [`cast`](crate::column::ColumnView::cast),
//!   [`negate`](crate::column::ColumnView::negate),
//!   [`abs`](crate::column::ColumnView::abs),
//!   [`is_null`](crate::column::ColumnView::is_null),
//!   [`is_nan`](crate::column::ColumnView::is_nan),
//!   [`round`](crate::column::ColumnView::round),
//!   [`ceil`](crate::column::ColumnView::ceil),
//!   [`floor`](crate::column::ColumnView::floor)

use crate::column::ColumnView;
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Selects which binary operation to perform on two columns or a column and a
/// scalar.
///
/// Arithmetic variants: `ADD`, `SUB`, `MUL`, `DIV`, `TRUE_DIV`, `FLOOR_DIV`,
/// `MOD`, `PMOD`, `PYMOD`, `POW`, `INT_POW`, `LOG_BASE`, `ATAN2`.
///
/// Bitwise variants: `SHIFT_LEFT`, `SHIFT_RIGHT`, `SHIFT_RIGHT_UNSIGNED`,
/// `BITWISE_AND`, `BITWISE_OR`, `BITWISE_XOR`.
///
/// Logical variants: `LOGICAL_AND`, `LOGICAL_OR`, `NULL_LOGICAL_AND`,
/// `NULL_LOGICAL_OR`.
///
/// Comparison variants: `EQUAL`, `NOT_EQUAL`, `LESS`, `GREATER`,
/// `LESS_EQUAL`, `GREATER_EQUAL`, `NULL_EQUALS`, `NULL_NOT_EQUALS`.
///
/// Null-aware variants: `NULL_MAX`, `NULL_MIN`.
///
/// See [`ColumnView::binary_op`](crate::column::ColumnView::binary_op) for
/// usage.
#[doc(alias = "binary_operator")]
pub use cudf_sys::ffi::BinaryOperator;

/// Controls the rounding strategy used by
/// [`ColumnView::round_with_method`](crate::column::ColumnView::round_with_method).
///
/// Variants:
/// - `HALF_UP` -- round half away from zero (standard rounding).
/// - `HALF_EVEN` -- round half to the nearest even digit (banker's rounding).
#[doc(alias = "rounding_method")]
pub use cudf_sys::ffi::RoundingMethod;

/// Selects which unary operation to apply to a column via
/// [`ColumnView::unary_op`](crate::column::ColumnView::unary_op).
///
/// Trigonometric: `SIN`, `COS`, `TAN`, `ARCSIN`, `ARCCOS`, `ARCTAN`.
///
/// Hyperbolic: `SINH`, `COSH`, `TANH`, `ARCSINH`, `ARCCOSH`, `ARCTANH`.
///
/// Exponential / logarithmic: `EXP`, `LOG`, `SQRT`, `CBRT`.
///
/// Rounding: `CEIL`, `FLOOR`, `ABS`, `RINT`.
///
/// Bitwise / logical: `BIT_COUNT`, `BIT_INVERT`, `NOT`, `NEGATE`.
#[doc(alias = "unary_operator")]
pub use cudf_sys::ffi::UnaryOperator;

/// Builder for a column-column binary operation.
///
/// Created by [`ColumnView::binary_op`](crate::column::ColumnView::binary_op).
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with the result.
pub struct BinaryOpColumns<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    op: BinaryOperator,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for BinaryOpColumns<'_> {
    type Output = crate::column::UnboundColumn;

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
        Ok(crate::column::RawColumn(col))
    }
}

/// Builder for a column-scalar binary operation.
///
/// Created by [`ColumnView::binary_op_scalar`](crate::column::ColumnView::binary_op_scalar).
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with the result.
pub struct BinaryOpColumnScalar<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a Scalar,
    op: BinaryOperator,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for BinaryOpColumnScalar<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let rhs_ffi = crate::scalar::scalar_to_ffi(self.rhs)?;
        let col = cudf_sys::binaryop::ffi::binary_operation_column_scalar(
            self.lhs.0,
            &rhs_ffi,
            self.op,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(col))
    }
}

/// Builder for a scalar-column binary operation.
///
/// Created by [`scalar_binary_op`]. Call [`.call()`](crate::stream::GpuOp::call)
/// to execute, producing an owned [`Column`] with the result.
pub struct ScalarBinaryOp<'a> {
    lhs: &'a Scalar,
    rhs: &'a ColumnView<'a>,
    op: BinaryOperator,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for ScalarBinaryOp<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let lhs_ffi = crate::scalar::scalar_to_ffi(self.lhs)?;
        let col = cudf_sys::binaryop::ffi::binary_operation_scalar_column(
            &lhs_ffi,
            self.rhs.0,
            self.op,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(col))
    }
}

impl<'a> ColumnView<'a> {
    /// Applies a binary operation element-wise between two columns.
    ///
    /// Both columns must have the same length. The `op` parameter selects the
    /// operation (see [`BinaryOperator`] for the full list), and `output_type`
    /// controls the [`TypeId`] of the resulting column.
    ///
    /// Returns a [`BinaryOpColumns`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is not supported for the input and
    /// output type combination, or if the columns have different lengths.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::ops::BinaryOperator;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(2), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(10), 3).call()?;
    /// let result = a.view()
    ///     .binary_op(&b.view(), BinaryOperator::POW, TypeId::FLOAT64)
    ///     .call()?;
    /// // [2^10, 2^10, 2^10] = [1024.0, 1024.0, 1024.0]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
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

    /// Applies a binary operation element-wise between this column (left) and
    /// a scalar (right).
    ///
    /// The scalar `rhs` is broadcast to every row. The `op` parameter selects
    /// the operation (see [`BinaryOperator`]), and `output_type` controls the
    /// [`TypeId`] of the resulting column.
    ///
    /// Returns a [`BinaryOpColumnScalar`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is not supported for the given types.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::ops::BinaryOperator;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(5), 3).call()?;
    /// let s = Scalar::from_i32(10);
    /// let result = col.view()
    ///     .binary_op_scalar(&s, BinaryOperator::ADD, TypeId::INT32)
    ///     .call()?;
    /// assert_eq!(result.to_vec_i32().call()?, [15, 15, 15]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
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

/// Applies a binary operation with a scalar on the left and a column on the
/// right.
///
/// This is the reverse of
/// [`ColumnView::binary_op_scalar`](crate::column::ColumnView::binary_op_scalar)
/// -- the scalar is the left operand and each column element is the right
/// operand. This matters for non-commutative operations such as
/// [`BinaryOperator::SUB`] or [`BinaryOperator::DIV`].
///
/// The `output_type` controls the [`TypeId`] of the resulting column.
///
/// Returns a [`ScalarBinaryOp`] builder. Use `.stream()` to set a custom
/// CUDA stream, then `.call()` to execute.
///
/// # Errors
///
/// Returns an error if the operation is not supported for the given types.
///
/// # Examples
///
/// ```no_run
/// use cudf::column::Column;
/// use cudf::data_type::TypeId;
/// use cudf::ops::{scalar_binary_op, BinaryOperator};
/// use cudf::scalar::Scalar;
/// use cudf::stream::GpuOp;
///
/// let s = Scalar::from_i32(100);
/// let col = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
/// // 100 - 3 = 97 for every element
/// let result = scalar_binary_op(&s, &col.view(), BinaryOperator::SUB, TypeId::INT32)
///     .call()?;
/// assert_eq!(result.to_vec_i32().call()?, [97, 97]);
/// # Ok::<(), cudf::error::Error>(())
/// ```
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

/// Computes the output scale for a fixed-point (decimal) binary operation.
///
/// When performing arithmetic on fixed-point columns, the result scale
/// depends on the operation and the scales of the two operands. For example,
/// adding two `DECIMAL64` columns with scales -2 and -3 yields a result
/// with scale -3 (the more precise of the two).
///
/// # Parameters
///
/// - `op` -- the [`BinaryOperator`] to query (typically `ADD`, `SUB`, `MUL`,
///   or `DIV`).
/// - `left_scale` -- scale of the left operand (negative means digits after
///   the decimal point).
/// - `right_scale` -- scale of the right operand.
///
/// # Returns
///
/// The scale (as a negative `i32`) that the result column will have.
pub fn binary_operation_fixed_point_scale(
    op: BinaryOperator,
    left_scale: i32,
    right_scale: i32,
) -> i32 {
    cudf_sys::binaryop::ffi::binary_operation_fixed_point_scale(op.repr, left_scale, right_scale)
}

/// Checks whether a binary operation is supported for the given type
/// combination.
///
/// Returns `true` if `op` can be applied to columns of types `lhs` and
/// `rhs` and produce a result of type `out`. Use this to validate type
/// compatibility before calling [`ColumnView::binary_op`](crate::column::ColumnView::binary_op).
///
/// # Parameters
///
/// - `out` -- desired output [`TypeId`].
/// - `lhs` -- [`TypeId`] of the left operand.
/// - `rhs` -- [`TypeId`] of the right operand.
/// - `op` -- the [`BinaryOperator`] to check.
pub fn is_supported_binaryop(out: TypeId, lhs: TypeId, rhs: TypeId, op: BinaryOperator) -> bool {
    cudf_sys::binaryop::ffi::is_supported_binaryop(out.repr, lhs.repr, rhs.repr, op.repr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::stream::GpuOp;

    // -- Binary operation tests --

    #[test]
    fn add_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(4), 3).call().unwrap();
        let result = c1.view().add(&c2.view(), TypeId::INT32).call().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![5, 5, 5]);
    }

    #[test]
    fn sub_i32_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 3)
            .call()
            .unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 3).call().unwrap();
        let result = c1.view().sub(&c2.view(), TypeId::INT32).call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![7, 7, 7]);
    }

    #[test]
    fn mul_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(2.0), 3)
            .call()
            .unwrap();
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 3)
            .call()
            .unwrap();
        let result = c1.view().mul(&c2.view(), TypeId::FLOAT64).call().unwrap();
        assert_eq!(result.to_vec_f64().call().unwrap(), vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn div_f64_columns() {
        let c1 = Column::from_scalar(&Scalar::from_f64(10.0), 2)
            .call()
            .unwrap();
        let c2 = Column::from_scalar(&Scalar::from_f64(4.0), 2)
            .call()
            .unwrap();
        let result = c1.view().div(&c2.view(), TypeId::FLOAT64).call().unwrap();
        assert_eq!(result.to_vec_f64().call().unwrap(), vec![2.5, 2.5]);
    }

    #[test]
    fn eq_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
        let result = c1.view().eq(&c2.view()).call().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, true, true]);
    }

    #[test]
    fn lt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2).call().unwrap();
        let result = c1.view().lt(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, true]);
    }

    #[test]
    fn add_column_scalar() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let s = Scalar::from_i32(10);
        let result = col
            .view()
            .binary_op_scalar(&s, BinaryOperator::ADD, TypeId::INT32)
            .call()
            .unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![11, 11, 11]);
    }

    #[test]
    fn binary_op_generic() {
        let c1 = Column::from_scalar(&Scalar::from_i32(2), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(10), 2)
            .call()
            .unwrap();
        let result = c1
            .view()
            .binary_op(&c2.view(), BinaryOperator::POW, TypeId::FLOAT64)
            .call()
            .unwrap();
        let vals = result.to_vec_f64().call().unwrap();
        assert!((vals[0] - 1024.0).abs() < 1e-6);
    }

    // -- Unary operation tests --

    #[test]
    fn cast_i32_to_f64() {
        let col = Column::from_scalar(&Scalar::from_i32(42), 2)
            .call()
            .unwrap();
        let result = col.view().cast(TypeId::FLOAT64).call().unwrap();
        assert_eq!(result.type_id(), TypeId::FLOAT64);
        assert_eq!(result.to_vec_f64().call().unwrap(), vec![42.0, 42.0]);
    }

    #[test]
    fn cast_f64_to_i32() {
        let col = Column::from_scalar(&Scalar::from_f64(3.7), 2)
            .call()
            .unwrap();
        let result = col.view().cast(TypeId::INT32).call().unwrap();
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![3, 3]);
    }

    #[test]
    fn is_null_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let result = col.view().is_null().call().unwrap();
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![false, false, false]
        );
    }

    #[test]
    fn is_valid_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let result = col.view().is_valid().call().unwrap();
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, true, true]);
    }

    #[test]
    fn negate_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(5), 2).call().unwrap();
        let result = col.view().negate().call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![-5, -5]);
    }

    #[test]
    fn abs_i32() {
        let col = Column::from_scalar(&Scalar::from_i32(-7), 2)
            .call()
            .unwrap();
        let result = col.view().abs().call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![7, 7]);
    }

    #[test]
    fn ne_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
        let result = c1.view().ne(&c2.view()).call().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![false, false, false]
        );
    }

    #[test]
    fn ge_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(5), 2).call().unwrap();
        let result = c1.view().ge(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, true]);
    }

    #[test]
    fn gt_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(5), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let result = c1.view().gt(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, true]);
    }

    #[test]
    fn le_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let result = c1.view().le(&c2.view()).call().unwrap();
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, true]);
    }

    #[test]
    fn is_nan_basic() {
        let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0])
            .call()
            .unwrap();
        let result = col.view().is_nan().call().unwrap();
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![false, true, false]
        );
    }

    #[test]
    fn scalar_column_binary_op() {
        let s = Scalar::from_i32(10);
        let col = Column::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let result = scalar_binary_op(&s, &col.view(), BinaryOperator::SUB, TypeId::INT32)
            .call()
            .unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![7, 7]);
    }
}
