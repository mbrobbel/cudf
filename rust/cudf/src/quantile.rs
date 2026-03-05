// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Quantile operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;

pub use cudf_sys::ffi::Interpolation;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Computes quantiles of a column.
///
/// `quantiles` should contain values in the range [0, 1].
/// The default interpolation method is `LINEAR`.
pub fn quantile(col: &ColumnView<'_>, quantiles: &[f64]) -> Result<Column> {
    quantile_with_interp(col, quantiles, Interpolation::LINEAR)
}

/// Computes quantiles of a column with a specified interpolation method.
pub fn quantile_with_interp(
    col: &ColumnView<'_>,
    quantiles: &[f64],
    interp: Interpolation,
) -> Result<Column> {
    let c = cudf_sys::ffi::quantile_column(col.0, quantiles, interp.repr, ds())?;
    Ok(Column(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;

    #[test]
    fn median_of_column() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 4, 5], ds()));
        let result = quantile(&col.view(), &[0.5]).unwrap();
        assert_eq!(result.len(), 1);
        let data = result.to_vec_f64();
        assert!((data[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn quartiles() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 4, 5], ds()));
        let result = quantile(&col.view(), &[0.25, 0.5, 0.75]).unwrap();
        assert_eq!(result.len(), 3);
        let data = result.to_vec_f64();
        assert!((data[0] - 2.0).abs() < 1e-9);
        assert!((data[1] - 3.0).abs() < 1e-9);
        assert!((data[2] - 4.0).abs() < 1e-9);
    }
}
