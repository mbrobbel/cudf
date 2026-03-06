// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Replace operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Extension trait for replacing null values with a replacement.
pub trait ReplaceNullsWith<R> {
    /// Replaces null values with the replacement.
    fn replace_nulls(&self, replacement: &R) -> Result<Column>;
    /// `replace_nulls` on a custom CUDA stream.
    fn replace_nulls_on(&self, replacement: &R, stream: Stream) -> Result<Column>;
}

impl ReplaceNullsWith<ColumnView<'_>> for ColumnView<'_> {
    fn replace_nulls(&self, replacement: &ColumnView<'_>) -> Result<Column> {
        self.replace_nulls_on(replacement, Stream::default_stream())
    }
    fn replace_nulls_on(&self, replacement: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::replace_nulls_column(self.0, replacement.0, stream.as_raw())?;
        Ok(Column(c))
    }
}

impl ReplaceNullsWith<Scalar> for ColumnView<'_> {
    fn replace_nulls(&self, replacement: &Scalar) -> Result<Column> {
        self.replace_nulls_on(replacement, Stream::default_stream())
    }
    fn replace_nulls_on(&self, replacement: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(replacement);
        let c = cudf_sys::ffi::replace_nulls_scalar(self.0, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }
}

impl ColumnView<'_> {
    /// Replaces NaN values with corresponding values from a replacement column.
    pub fn replace_nans(&self, replacement: &ColumnView<'_>) -> Result<Column> {
        self.replace_nans_on(replacement, Stream::default_stream())
    }

    /// `replace_nans` on a custom CUDA stream.
    pub fn replace_nans_on(&self, replacement: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::replace_nans_column(self.0, replacement.0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Replaces NaN values with a scalar.
    pub fn replace_nans_scalar(&self, replacement: &Scalar) -> Result<Column> {
        self.replace_nans_scalar_on(replacement, Stream::default_stream())
    }

    /// `replace_nans_scalar` on a custom CUDA stream.
    pub fn replace_nans_scalar_on(&self, replacement: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(replacement);
        let c = cudf_sys::ffi::replace_nans_scalar(self.0, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Clamps column values to the range [lo, hi].
    pub fn clamp(&self, lo: &Scalar, hi: &Scalar) -> Result<Column> {
        self.clamp_on(lo, hi, Stream::default_stream())
    }

    /// clamp on a custom CUDA stream.
    pub fn clamp_on(&self, lo: &Scalar, hi: &Scalar, stream: Stream) -> Result<Column> {
        let lo_ffi = crate::scalar::scalar_to_ffi(lo);
        let hi_ffi = crate::scalar::scalar_to_ffi(hi);
        let c = cudf_sys::ffi::clamp_column(self.0, &lo_ffi, &hi_ffi, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Finds and replaces all matching values.
    pub fn find_and_replace_all(
        &self,
        old: &ColumnView<'_>,
        new: &ColumnView<'_>,
    ) -> Result<Column> {
        self.find_and_replace_all_on(old, new, Stream::default_stream())
    }

    /// `find_and_replace_all` on a custom CUDA stream.
    pub fn find_and_replace_all_on(
        &self,
        old: &ColumnView<'_>,
        new: &ColumnView<'_>,
        stream: Stream,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::find_and_replace_all(self.0, old.0, new.0, stream.as_raw())?;
        Ok(Column(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;

    fn ds() -> usize {
        Stream::default_stream().as_raw()
    }

    #[test]
    fn replace_nulls_with_scalar() {
        let null_s = Scalar::null_i32();
        let null_col = Column::from_scalar(&null_s, 3);
        let replacement = Scalar::from_i32(0);
        let result = null_col.view().replace_nulls(&replacement).unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32(), vec![0, 0, 0]);
    }

    #[test]
    fn replace_nans_with_scalar() {
        let nan_col = Column(cudf_sys::ffi::make_column_from_host_f64(
            &[1.0, f64::NAN, 3.0],
            ds(),
        ));
        let replacement = Scalar::from_f64(0.0);
        let result = nan_col.view().replace_nans_scalar(&replacement).unwrap();
        let data = result.to_vec_f64();
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 0.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn clamp_values() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 5, 10], ds()));
        let lo = Scalar::from_i32(3);
        let hi = Scalar::from_i32(7);
        let result = col.view().clamp(&lo, &hi).unwrap();
        assert_eq!(result.to_vec_i32(), vec![3, 5, 7]);
    }

    #[test]
    fn find_and_replace() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(
            &[1, 2, 3, 2, 1],
            ds(),
        ));
        let old_vals = Column(cudf_sys::ffi::make_column_from_host_i32(&[2], ds()));
        let new_vals = Column(cudf_sys::ffi::make_column_from_host_i32(&[99], ds()));
        let result = col
            .view()
            .find_and_replace_all(&old_vals.view(), &new_vals.view())
            .unwrap();
        assert_eq!(result.to_vec_i32(), vec![1, 99, 3, 99, 1]);
    }

    #[test]
    fn replace_nulls_with_column() {
        let null_s = Scalar::null_i32();
        let null_col = Column::from_scalar(&null_s, 3);
        let replacement = Column::from_scalar(&Scalar::from_i32(42), 3);
        let result = null_col.view().replace_nulls(&replacement.view()).unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32(), vec![42, 42, 42]);
    }

    #[test]
    fn replace_nans_with_column() {
        let nan_col = Column(cudf_sys::ffi::make_column_from_host_f64(
            &[1.0, f64::NAN, 3.0],
            ds(),
        ));
        let replacement = Column(cudf_sys::ffi::make_column_from_host_f64(
            &[10.0, 20.0, 30.0],
            ds(),
        ));
        let result = nan_col.view().replace_nans(&replacement.view()).unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 20.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }
}
