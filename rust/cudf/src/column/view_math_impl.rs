#![allow(clippy::wildcard_imports)]
use super::*;

impl ColumnView<'_> {
    /// Computes the element-wise sum of this column and `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column.
    ///
    /// Returns an [`Add`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(10), 3).call()?;
    /// let sum = a.view().add(&b.view(), TypeId::INT32).call()?;
    /// assert_eq!(sum.to_vec_i32().call()?, [11, 11, 11]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn add<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Add<'a> {
        Add {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the element-wise difference of this column minus `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column.
    ///
    /// Returns a [`Sub`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(10), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 3).call()?;
    /// let diff = a.view().sub(&b.view(), TypeId::INT32).call()?;
    /// assert_eq!(diff.to_vec_i32().call()?, [7, 7, 7]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn sub<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Sub<'a> {
        Sub {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the element-wise product of this column and `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column.
    ///
    /// Returns a [`Mul`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_f64(2.0), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_f64(3.0), 3).call()?;
    /// let prod = a.view().mul(&b.view(), TypeId::FLOAT64).call()?;
    /// assert_eq!(prod.to_vec_f64().call()?, [6.0, 6.0, 6.0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn mul<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Mul<'a> {
        Mul {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the element-wise division of this column by `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column. For integer types this
    /// performs truncating division; use `FLOAT64` output for exact results.
    ///
    /// Returns a [`Div`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_f64(10.0), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_f64(4.0), 2).call()?;
    /// let quot = a.view().div(&b.view(), TypeId::FLOAT64).call()?;
    /// assert_eq!(quot.to_vec_f64().call()?, [2.5, 2.5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn div<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Div<'a> {
        Div {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise equality comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when the corresponding elements of `self` and `rhs` are equal.
    ///
    /// Returns an [`struct@Eq`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(5), 3).call()?;
    /// let mask = a.view().eq(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn eq<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Eq<'a> {
        Eq {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise not-equal comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when the corresponding elements differ.
    ///
    /// Returns a [`Ne`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let mask = a.view().ne(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn ne<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Ne<'a> {
        Ne {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise less-than comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] < rhs[i]`.
    ///
    /// Returns an [`Lt`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let mask = a.view().lt(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn lt<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Lt<'a> {
        Lt {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise greater-than comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] > rhs[i]`.
    ///
    /// Returns a [`Gt`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let mask = a.view().gt(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn gt<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Gt<'a> {
        Gt {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise less-than-or-equal comparison, producing a `BOOL8`
    /// column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] <= rhs[i]`.
    ///
    /// Returns an [`Le`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let mask = a.view().le(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn le<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Le<'a> {
        Le {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise greater-than-or-equal comparison, producing a `BOOL8`
    /// column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] >= rhs[i]`.
    ///
    /// Returns a [`Ge`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let mask = a.view().ge(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn ge<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Ge<'a> {
        Ge {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    // -- Generic unary operation --

    /// Applies a generic unary operation to every element of this column.
    ///
    /// The `op` parameter selects the operation (see
    /// [`UnaryOperator`](crate::ops::UnaryOperator) for all variants).
    /// Convenience wrappers such as [`sin`](Self::sin), [`abs`](Self::abs),
    /// and [`ceil`](Self::ceil) delegate to this method.
    ///
    /// Returns a [`UnaryOp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn unary_op(&self, op: crate::ops::UnaryOperator) -> UnaryOp<'_> {
        UnaryOp {
            view: self,
            op,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a non-NaN element.
    ///
    /// Only applicable to floating-point columns. This is the logical
    /// inverse of [`is_nan`](Self::is_nan).
    ///
    /// Returns an [`IsNotNan`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    pub fn is_not_nan(&self) -> IsNotNan<'_> {
        IsNotNan {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Math convenience methods --

    /// Computes the sine of each element (radians), returning a new
    /// column. Applicable to floating-point types.
    pub fn sin(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SIN)
    }
    /// Computes the cosine of each element (radians).
    pub fn cos(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::COS)
    }
    /// Computes the tangent of each element (radians).
    pub fn tan(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::TAN)
    }
    /// Computes the arcsine (inverse sine) of each element.
    pub fn arcsin(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCSIN)
    }
    /// Computes the arccosine (inverse cosine) of each element.
    pub fn arccos(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCCOS)
    }
    /// Computes the arctangent (inverse tangent) of each element.
    pub fn arctan(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCTAN)
    }
    /// Computes the hyperbolic sine of each element.
    pub fn sinh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SINH)
    }
    /// Computes the hyperbolic cosine of each element.
    pub fn cosh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::COSH)
    }
    /// Computes the hyperbolic tangent of each element.
    pub fn tanh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::TANH)
    }
    /// Computes the inverse hyperbolic sine of each element.
    pub fn arcsinh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCSINH)
    }
    /// Computes the inverse hyperbolic cosine of each element.
    pub fn arccosh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCCOSH)
    }
    /// Computes the inverse hyperbolic tangent of each element.
    pub fn arctanh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCTANH)
    }
    /// Computes `e^x` for each element.
    pub fn exp(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::EXP)
    }
    /// Computes the natural logarithm (`ln`) of each element.
    pub fn log(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::LOG)
    }
    /// Computes the square root of each element.
    pub fn sqrt(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SQRT)
    }
    /// Computes the cube root of each element.
    pub fn cbrt(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::CBRT)
    }
    /// Rounds each element up to the smallest integer not less than the value.
    ///
    /// Applicable to floating-point columns. The result has the same type as
    /// the input.
    ///
    /// Returns a [`UnaryOp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn ceil(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::CEIL)
    }
    /// Rounds each element down to the largest integer not greater than the
    /// value.
    ///
    /// Applicable to floating-point columns. The result has the same type as
    /// the input.
    ///
    /// Returns a [`UnaryOp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn floor(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::FLOOR)
    }
    /// Rounds each element to the nearest integer (round half to even),
    /// returning a floating-point column.
    pub fn rint(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::RINT)
    }
    /// Bitwise inversion of each element.
    pub fn bit_invert(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::BIT_INVERT)
    }
    /// Logical NOT of each element.
    pub fn logical_not(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::NOT)
    }

    // -- Round --

    /// Rounds column values to the given number of decimal places.
    ///
    /// Uses the `HALF_UP` rounding method by default. A positive
    /// `decimal_places` rounds to that many digits after the decimal point;
    /// a negative value rounds to digits before the decimal point (e.g.,
    /// `-1` rounds to the nearest 10).
    ///
    /// For control over the rounding strategy, see
    /// [`round_with_method`](Self::round_with_method).
    ///
    /// Returns a [`Round`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.15, 2.25, 3.35]).call()?;
    /// let rounded = col.view().round(1).call()?;
    /// // [1.2, 2.3, 3.4] with HALF_UP rounding
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn round(&self, decimal_places: i32) -> Round<'_> {
        Round {
            view: self,
            decimal_places,
            stream: Stream::default_stream(),
        }
    }

    /// Rounds column values using a specific [`RoundingMethod`](crate::ops::RoundingMethod).
    ///
    /// - `decimal_places` -- number of decimal places to round to (see
    ///   [`round`](Self::round) for sign semantics).
    /// - `method` -- either
    ///   [`HALF_UP`](crate::ops::RoundingMethod::HALF_UP) or
    ///   [`HALF_EVEN`](crate::ops::RoundingMethod::HALF_EVEN) (banker's
    ///   rounding).
    ///
    /// Returns a [`RoundWithMethod`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn round_with_method(
        &self,
        decimal_places: i32,
        method: crate::ops::RoundingMethod,
    ) -> RoundWithMethod<'_> {
        RoundWithMethod {
            view: self,
            decimal_places,
            method,
            stream: Stream::default_stream(),
        }
    }
}
