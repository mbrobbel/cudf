// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Replace operations on GPU columns.
//!
//! This module provides builders and methods for replacing values in GPU
//! columns:
//!
//! - [`ColumnView::replace_nulls_with_column`](crate::column::ColumnView) --
//!   replace each null with the corresponding element from another column.
//! - [`ColumnView::replace_nulls_with_scalar`](crate::column::ColumnView) --
//!   replace all nulls with a single scalar value.
//! - [`ColumnView::replace_nans`](crate::column::ColumnView) -- replace NaN
//!   values with corresponding elements from another column (floats only).
//! - [`ColumnView::replace_nans_scalar`](crate::column::ColumnView) -- replace
//!   all NaN values with a single scalar value (floats only).
//! - [`ColumnView::clamp`](crate::column::ColumnView) -- clamp values to a
//!   `[lo, hi]` range.
//! - [`ColumnView::find_and_replace_all`](crate::column::ColumnView) -- find
//!   matching values and replace them.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Builder for replacing null values with a column.
///
/// Created by [`ColumnView::replace_nulls_with_column`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with no null values where replacements were applied.
pub struct ReplaceNullsColumn<'a> {
    view: &'a ColumnView<'a>,
    replacement: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ReplaceNullsColumn<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::replace::ffi::replace_nulls_column(
            self.view.0,
            self.replacement.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for replacing null values with a scalar.
///
/// Created by [`ColumnView::replace_nulls_with_scalar`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with all former nulls replaced by the scalar value.
pub struct ReplaceNullsScalar<'a> {
    view: &'a ColumnView<'a>,
    replacement: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for ReplaceNullsScalar<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.replacement);
        let c =
            cudf_sys::replace::ffi::replace_nulls_scalar(self.view.0, &ffi, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for replacing NaN values with values from another column.
///
/// Created by [`ColumnView::replace_nans`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with NaN values replaced element-wise.
pub struct ReplaceNansColumn<'a> {
    view: &'a ColumnView<'a>,
    replacement: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ReplaceNansColumn<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::replace::ffi::replace_nans_column(
            self.view.0,
            self.replacement.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for replacing NaN values with a single scalar.
///
/// Created by [`ColumnView::replace_nans_scalar`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with all NaN values replaced by the scalar.
pub struct ReplaceNansScalar<'a> {
    view: &'a ColumnView<'a>,
    replacement: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for ReplaceNansScalar<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.replacement);
        let c =
            cudf_sys::replace::ffi::replace_nans_scalar(self.view.0, &ffi, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for clamping column values to a `[lo, hi]` range.
///
/// Created by [`ColumnView::clamp`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with all values bounded to the range.
pub struct Clamp<'a> {
    view: &'a ColumnView<'a>,
    lo: &'a Scalar,
    hi: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for Clamp<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let lo_ffi = crate::scalar::scalar_to_ffi(self.lo);
        let hi_ffi = crate::scalar::scalar_to_ffi(self.hi);
        let c = cudf_sys::replace::ffi::clamp_column(
            self.view.0,
            &lo_ffi,
            &hi_ffi,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for finding and replacing all matching values in a column.
///
/// Created by [`ColumnView::find_and_replace_all`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute, producing an
/// owned [`Column`] with all matches replaced.
pub struct FindAndReplaceAll<'a> {
    view: &'a ColumnView<'a>,
    old: &'a ColumnView<'a>,
    new: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for FindAndReplaceAll<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::replace::ffi::find_and_replace_all(
            self.view.0,
            self.old.0,
            self.new.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

impl<'a> ColumnView<'a> {
    /// Replaces null values element-wise with values from `replacement`.
    ///
    /// Both columns must have the same length and compatible types. For each
    /// row, if this column's element is null, the corresponding element from
    /// `replacement` is used; otherwise the original value is kept.
    ///
    /// Returns a [`ReplaceNullsColumn`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or incompatible
    /// types.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let null_col = Column::from_scalar(&Scalar::null_i32(), 3).call()?;
    /// let replacement = Column::from_scalar(&Scalar::from_i32(42), 3).call()?;
    /// let result = null_col.view()
    ///     .replace_nulls_with_column(&replacement.view())
    ///     .call()?;
    /// assert_eq!(result.to_vec_i32().call()?, [42, 42, 42]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn replace_nulls_with_column(
        &'a self,
        replacement: &'a ColumnView<'a>,
    ) -> ReplaceNullsColumn<'a> {
        ReplaceNullsColumn {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces all null values in this column with the given scalar.
    ///
    /// Every null element is replaced by `replacement`; non-null elements
    /// are unchanged. The scalar must have a type compatible with the
    /// column.
    ///
    /// Returns a [`ReplaceNullsScalar`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the scalar type is incompatible with the column.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let null_col = Column::from_scalar(&Scalar::null_i32(), 3).call()?;
    /// let result = null_col.view()
    ///     .replace_nulls_with_scalar(&Scalar::from_i32(0))
    ///     .call()?;
    /// assert_eq!(result.to_vec_i32().call()?, [0, 0, 0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn replace_nulls_with_scalar(&'a self, replacement: &'a Scalar) -> ReplaceNullsScalar<'a> {
        ReplaceNullsScalar {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces NaN values element-wise with values from `replacement`.
    ///
    /// Both columns must be floating-point (`FLOAT32` or `FLOAT64`) and
    /// have the same length. For each row, if this column's element is NaN,
    /// the corresponding element from `replacement` is used.
    ///
    /// Returns a [`ReplaceNansColumn`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if either column is not a floating-point type or if
    /// they have different lengths.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]).call()?;
    /// let rep = Column::from_slice_f64(&[10.0, 20.0, 30.0]).call()?;
    /// let result = col.view().replace_nans(&rep.view()).call()?;
    /// assert_eq!(result.to_vec_f64().call()?, [1.0, 20.0, 3.0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn replace_nans(&'a self, replacement: &'a ColumnView<'a>) -> ReplaceNansColumn<'a> {
        ReplaceNansColumn {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces all NaN values in this column with the given scalar.
    ///
    /// The column must be a floating-point type. Every NaN element is
    /// replaced by `replacement`; non-NaN elements are unchanged.
    ///
    /// Returns a [`ReplaceNansScalar`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a floating-point type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]).call()?;
    /// let result = col.view()
    ///     .replace_nans_scalar(&Scalar::from_f64(0.0))
    ///     .call()?;
    /// assert_eq!(result.to_vec_f64().call()?, [1.0, 0.0, 3.0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn replace_nans_scalar(&'a self, replacement: &'a Scalar) -> ReplaceNansScalar<'a> {
        ReplaceNansScalar {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Clamps column values to the range \[`lo`, `hi`\].
    ///
    /// Values below `lo` are set to `lo`, values above `hi` are set to
    /// `hi`, and values within the range are unchanged. Null values remain
    /// null.
    ///
    /// Both `lo` and `hi` must have a type compatible with the column, and
    /// `lo` must be less than or equal to `hi`.
    ///
    /// Returns a [`Clamp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the scalar types are incompatible with the column.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 5, 10]).call()?;
    /// let lo = Scalar::from_i32(3);
    /// let hi = Scalar::from_i32(7);
    /// let result = col.view().clamp(&lo, &hi).call()?;
    /// assert_eq!(result.to_vec_i32().call()?, [3, 5, 7]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn clamp(&'a self, lo: &'a Scalar, hi: &'a Scalar) -> Clamp<'a> {
        Clamp {
            view: self,
            lo,
            hi,
            stream: Stream::default_stream(),
        }
    }

    /// Finds all occurrences of values in `old` and replaces them with the
    /// corresponding values in `new`.
    ///
    /// The `old` and `new` columns must have the same length and compatible
    /// types with this column. For each element in `self`, if it matches
    /// `old[i]`, it is replaced with `new[i]`. Elements that do not match
    /// any value in `old` are left unchanged.
    ///
    /// Returns a [`FindAndReplaceAll`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have incompatible types or if `old`
    /// and `new` have different lengths.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 2, 1]).call()?;
    /// let old = Column::from_slice_i32(&[2]).call()?;
    /// let new = Column::from_slice_i32(&[99]).call()?;
    /// let result = col.view()
    ///     .find_and_replace_all(&old.view(), &new.view())
    ///     .call()?;
    /// assert_eq!(result.to_vec_i32().call()?, [1, 99, 3, 99, 1]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn find_and_replace_all(
        &'a self,
        old: &'a ColumnView<'a>,
        new: &'a ColumnView<'a>,
    ) -> FindAndReplaceAll<'a> {
        FindAndReplaceAll {
            view: self,
            old,
            new,
            stream: Stream::default_stream(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;

    #[test]
    fn replace_nulls_with_scalar() {
        let null_s = Scalar::null_i32();
        let null_col = Column::from_scalar(&null_s, 3).call().unwrap();
        let replacement = Scalar::from_i32(0);
        let result = null_col
            .view()
            .replace_nulls_with_scalar(&replacement)
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn replace_nans_with_scalar() {
        let nan_col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0])
            .call()
            .unwrap();
        let replacement = Scalar::from_f64(0.0);
        let result = nan_col
            .view()
            .replace_nans_scalar(&replacement)
            .call()
            .unwrap();
        let data = result.to_vec_f64().call().unwrap();
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 0.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn clamp_values() {
        let col = Column::from_slice_i32(&[1, 5, 10]).call().unwrap();
        let lo = Scalar::from_i32(3);
        let hi = Scalar::from_i32(7);
        let result = col.view().clamp(&lo, &hi).call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![3, 5, 7]);
    }

    #[test]
    fn find_and_replace() {
        let col = Column::from_slice_i32(&[1, 2, 3, 2, 1]).call().unwrap();
        let old_vals = Column::from_slice_i32(&[2]).call().unwrap();
        let new_vals = Column::from_slice_i32(&[99]).call().unwrap();
        let result = col
            .view()
            .find_and_replace_all(&old_vals.view(), &new_vals.view())
            .call()
            .unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 99, 3, 99, 1]);
    }

    #[test]
    fn replace_nulls_with_column() {
        let null_s = Scalar::null_i32();
        let null_col = Column::from_scalar(&null_s, 3).call().unwrap();
        let replacement = Column::from_scalar(&Scalar::from_i32(42), 3)
            .call()
            .unwrap();
        let result = null_col
            .view()
            .replace_nulls_with_column(&replacement.view())
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![42, 42, 42]);
    }

    #[test]
    fn replace_nans_with_column() {
        let nan_col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0])
            .call()
            .unwrap();
        let replacement = Column::from_slice_f64(&[10.0, 20.0, 30.0]).call().unwrap();
        let result = nan_col
            .view()
            .replace_nans(&replacement.view())
            .call()
            .unwrap();
        let data = result.to_vec_f64().call().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 20.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }
}
