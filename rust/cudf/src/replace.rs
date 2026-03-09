// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Replace operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Builder for replacing null values with a column.
///
/// Created by [`ColumnView::replace_nulls_with_column`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
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
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
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

/// Builder for replacing NaN values with a column.
///
/// Created by [`ColumnView::replace_nans`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
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

/// Builder for replacing NaN values with a scalar.
///
/// Created by [`ColumnView::replace_nans_scalar`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
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

/// Builder for clamping column values to a range.
///
/// Created by [`ColumnView::clamp`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
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

/// Builder for finding and replacing all matching values.
///
/// Created by [`ColumnView::find_and_replace_all`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
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
    /// Replaces null values with corresponding values from a replacement column.
    ///
    /// Returns a [`ReplaceNullsColumn`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
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

    /// Replaces null values with a scalar.
    ///
    /// Returns a [`ReplaceNullsScalar`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn replace_nulls_with_scalar(&'a self, replacement: &'a Scalar) -> ReplaceNullsScalar<'a> {
        ReplaceNullsScalar {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces NaN values with corresponding values from a replacement column.
    ///
    /// Returns a [`ReplaceNansColumn`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn replace_nans(&'a self, replacement: &'a ColumnView<'a>) -> ReplaceNansColumn<'a> {
        ReplaceNansColumn {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces NaN values with a scalar.
    ///
    /// Returns a [`ReplaceNansScalar`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn replace_nans_scalar(&'a self, replacement: &'a Scalar) -> ReplaceNansScalar<'a> {
        ReplaceNansScalar {
            view: self,
            replacement,
            stream: Stream::default_stream(),
        }
    }

    /// Clamps column values to the range \[lo, hi\].
    ///
    /// Returns a [`Clamp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn clamp(&'a self, lo: &'a Scalar, hi: &'a Scalar) -> Clamp<'a> {
        Clamp {
            view: self,
            lo,
            hi,
            stream: Stream::default_stream(),
        }
    }

    /// Finds and replaces all matching values.
    ///
    /// Returns a [`FindAndReplaceAll`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
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
        let null_col = Column::from_scalar(&null_s, 3);
        let replacement = Scalar::from_i32(0);
        let result = null_col
            .view()
            .replace_nulls_with_scalar(&replacement)
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32(), vec![0, 0, 0]);
    }

    #[test]
    fn replace_nans_with_scalar() {
        let nan_col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]);
        let replacement = Scalar::from_f64(0.0);
        let result = nan_col
            .view()
            .replace_nans_scalar(&replacement)
            .call()
            .unwrap();
        let data = result.to_vec_f64();
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 0.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn clamp_values() {
        let col = Column::from_slice_i32(&[1, 5, 10]);
        let lo = Scalar::from_i32(3);
        let hi = Scalar::from_i32(7);
        let result = col.view().clamp(&lo, &hi).call().unwrap();
        assert_eq!(result.to_vec_i32(), vec![3, 5, 7]);
    }

    #[test]
    fn find_and_replace() {
        let col = Column::from_slice_i32(&[1, 2, 3, 2, 1]);
        let old_vals = Column::from_slice_i32(&[2]);
        let new_vals = Column::from_slice_i32(&[99]);
        let result = col
            .view()
            .find_and_replace_all(&old_vals.view(), &new_vals.view())
            .call()
            .unwrap();
        assert_eq!(result.to_vec_i32(), vec![1, 99, 3, 99, 1]);
    }

    #[test]
    fn replace_nulls_with_column() {
        let null_s = Scalar::null_i32();
        let null_col = Column::from_scalar(&null_s, 3);
        let replacement = Column::from_scalar(&Scalar::from_i32(42), 3);
        let result = null_col
            .view()
            .replace_nulls_with_column(&replacement.view())
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32(), vec![42, 42, 42]);
    }

    #[test]
    fn replace_nans_with_column() {
        let nan_col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]);
        let replacement = Column::from_slice_f64(&[10.0, 20.0, 30.0]);
        let result = nan_col
            .view()
            .replace_nans(&replacement.view())
            .call()
            .unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 20.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }
}
