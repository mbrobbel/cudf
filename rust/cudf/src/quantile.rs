// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Quantile operations on GPU columns.
//!
//! Available as methods on [`ColumnView`](crate::ColumnView):
//! `col.quantile(...)`, `col.quantile_with_interp(...)`.

#[doc(alias = "interpolation")]
pub use cudf_sys::ffi::Interpolation;

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::stream::GpuOp;

    #[test]
    fn median_of_column() {
        let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]);
        let result = col.view().quantile(&[0.5]).call().unwrap();
        assert_eq!(result.len(), 1);
        let data = result.to_vec_f64();
        assert!((data[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn quartiles() {
        let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]);
        let result = col.view().quantile(&[0.25, 0.5, 0.75]).call().unwrap();
        assert_eq!(result.len(), 3);
        let data = result.to_vec_f64();
        assert!((data[0] - 2.0).abs() < 1e-9);
        assert!((data[1] - 3.0).abs() < 1e-9);
        assert!((data[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn quantile_with_interp_lower() {
        use crate::quantile::Interpolation;
        let col = Column::from_slice_i32(&[1, 2, 3, 4]);
        let result = col
            .view()
            .quantile_with_interp(&[0.5], Interpolation::LOWER)
            .call()
            .unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn quantile_with_interp_higher() {
        use crate::quantile::Interpolation;
        let col = Column::from_slice_i32(&[1, 2, 3, 4]);
        let result = col
            .view()
            .quantile_with_interp(&[0.5], Interpolation::HIGHER)
            .call()
            .unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 3.0).abs() < 1e-9);
    }
}
