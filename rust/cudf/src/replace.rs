// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Replace operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;

/// Replaces null values with corresponding values from a replacement column.
pub fn replace_nulls(col: &ColumnView<'_>, replacement: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::replace_nulls_column(col.0, replacement.0)?;
    Ok(Column(c))
}

/// Replaces null values with a scalar.
pub fn replace_nulls_scalar(col: &ColumnView<'_>, replacement: &Scalar) -> Result<Column> {
    let c = cudf_sys::ffi::replace_nulls_scalar(col.0, &replacement.0)?;
    Ok(Column(c))
}

/// Replaces NaN values with corresponding values from a replacement column.
pub fn replace_nans(col: &ColumnView<'_>, replacement: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::replace_nans_column(col.0, replacement.0)?;
    Ok(Column(c))
}

/// Replaces NaN values with a scalar.
pub fn replace_nans_scalar(col: &ColumnView<'_>, replacement: &Scalar) -> Result<Column> {
    let c = cudf_sys::ffi::replace_nans_scalar(col.0, &replacement.0)?;
    Ok(Column(c))
}

/// Clamps column values to the range [lo, hi].
pub fn clamp(col: &ColumnView<'_>, lo: &Scalar, hi: &Scalar) -> Result<Column> {
    let c = cudf_sys::ffi::clamp_column(col.0, &lo.0, &hi.0)?;
    Ok(Column(c))
}

/// Finds and replaces all matching values in a column.
pub fn find_and_replace_all(
    col: &ColumnView<'_>,
    old: &ColumnView<'_>,
    new: &ColumnView<'_>,
) -> Result<Column> {
    let c = cudf_sys::ffi::find_and_replace_all(col.0, old.0, new.0)?;
    Ok(Column(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;

    #[test]
    fn replace_nulls_with_scalar() {
        // Create a column with nulls by using a null scalar
        let null_s = Scalar::null_i32();
        let null_col = Column::from_scalar(&null_s, 3);
        let replacement = Scalar::from_i32(0);
        let result = replace_nulls_scalar(&null_col.view(), &replacement).unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_nulls());
        assert_eq!(result.to_vec_i32(), vec![0, 0, 0]);
    }

    #[test]
    fn replace_nans_with_scalar() {
        // Create f64 column with NaN by using make_column_from_host_f64
        let nan_col = Column(cudf_sys::ffi::make_column_from_host_f64(&[1.0, f64::NAN, 3.0]));
        let replacement = Scalar::from_f64(0.0);
        let result = replace_nans_scalar(&nan_col.view(), &replacement).unwrap();
        let data = result.to_vec_f64();
        assert_eq!(data.len(), 3);
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 0.0).abs() < 1e-9);
        assert!((data[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn clamp_values() {
        // Create column [1, 5, 10]
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 5, 10]));
        let lo = Scalar::from_i32(3);
        let hi = Scalar::from_i32(7);
        let result = clamp(&col.view(), &lo, &hi).unwrap();
        assert_eq!(result.to_vec_i32(), vec![3, 5, 7]);
    }

    #[test]
    fn find_and_replace() {
        // Create column [1, 2, 3, 2, 1]
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 2, 1]));
        let old_vals = Column(cudf_sys::ffi::make_column_from_host_i32(&[2]));
        let new_vals = Column(cudf_sys::ffi::make_column_from_host_i32(&[99]));
        let result =
            find_and_replace_all(&col.view(), &old_vals.view(), &new_vals.view()).unwrap();
        assert_eq!(result.to_vec_i32(), vec![1, 99, 3, 99, 1]);
    }
}
