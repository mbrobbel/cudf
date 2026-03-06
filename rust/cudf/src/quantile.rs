// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Quantile operations on GPU columns.
//!
//! Available as methods on [`ColumnView`](crate::ColumnView):
//! `col.quantile(...)`, `col.quantile_with_interp(...)`.

pub use cudf_sys::ffi::Interpolation;

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::stream::Stream;

    fn ds() -> usize {
        Stream::default_stream().as_raw()
    }

    #[test]
    fn median_of_column() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 4, 5], ds()));
        let result = col.view().quantile(&[0.5]).unwrap();
        assert_eq!(result.len(), 1);
        let data = result.to_vec_f64();
        assert!((data[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn quartiles() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 4, 5], ds()));
        let result = col.view().quantile(&[0.25, 0.5, 0.75]).unwrap();
        assert_eq!(result.len(), 3);
        let data = result.to_vec_f64();
        assert!((data[0] - 2.0).abs() < 1e-9);
        assert!((data[1] - 3.0).abs() < 1e-9);
        assert!((data[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn quantile_with_interp_lower() {
        use crate::quantile::Interpolation;
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 4], ds()));
        let result = col
            .view()
            .quantile_with_interp(&[0.5], Interpolation::LOWER)
            .unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn quantile_with_interp_higher() {
        use crate::quantile::Interpolation;
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[1, 2, 3, 4], ds()));
        let result = col
            .view()
            .quantile_with_interp(&[0.5], Interpolation::HIGHER)
            .unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 3.0).abs() < 1e-9);
    }
}
