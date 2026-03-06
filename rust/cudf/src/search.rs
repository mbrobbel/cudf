// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Search operations on GPU columns and tables.
//!
//! - `column_view.contains_scalar(...)` and `column_view.contains_column(...)`
//! - `table.lower_bound(...)` and `table.upper_bound(...)`

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::Stream;

    fn ds() -> usize {
        Stream::default_stream().as_raw()
    }

    #[test]
    fn contains_found() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50], ds()));
        let needle = Scalar::from_i32(20);
        assert!(col.view().contains_scalar(&needle).unwrap());
    }

    #[test]
    fn contains_not_found() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50], ds()));
        let needle = Scalar::from_i32(25);
        assert!(!col.view().contains_scalar(&needle).unwrap());
    }

    #[test]
    fn contains_column_test() {
        let haystack =
            Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50], ds()));
        let needles = Column(cudf_sys::ffi::make_column_from_host_i32(&[20, 40, 60, 80], ds()));
        let result = haystack
            .view()
            .contains_column(&needles.view())
            .unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true, false, false]);
    }

    #[test]
    fn lower_bound_basic() {
        use crate::sorting::{NullOrder, Order};
        use crate::table::TableBuilder;

        // Sorted haystack: [10, 20, 30, 40, 50]
        let hay_col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50], ds()));
        let mut hb = TableBuilder::new();
        hb.push_column(hay_col);
        let haystack = hb.build().unwrap();

        // Needles: [15, 30, 55]
        let needle_col = Column(cudf_sys::ffi::make_column_from_host_i32(&[15, 30, 55], ds()));
        let mut nb = TableBuilder::new();
        nb.push_column(needle_col);
        let needles = nb.build().unwrap();

        let result = haystack
            .lower_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32(), vec![1, 2, 5]);
    }

    #[test]
    fn upper_bound_basic() {
        use crate::sorting::{NullOrder, Order};
        use crate::table::TableBuilder;

        let hay_col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50], ds()));
        let mut hb = TableBuilder::new();
        hb.push_column(hay_col);
        let haystack = hb.build().unwrap();

        let needle_col = Column(cudf_sys::ffi::make_column_from_host_i32(&[15, 30, 55], ds()));
        let mut nb = TableBuilder::new();
        nb.push_column(needle_col);
        let needles = nb.build().unwrap();

        let result = haystack
            .upper_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32(), vec![1, 3, 5]);
    }
}
