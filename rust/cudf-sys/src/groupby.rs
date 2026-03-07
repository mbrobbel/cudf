// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        type Scalar = crate::ffi::Scalar;
        type ScalarList = crate::ffi::ScalarList;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- GroupBy operations --

        /// Performs a single groupby aggregation on one value column.
        /// Returns a table with [key_columns..., aggregated_value_column].
        fn groupby_single(
            tbl: &Table,
            key_indices: &[i32],
            value_index: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Performs groupby with multiple aggregations on multiple value columns.
        /// value_indices and agg_kinds must have the same length.
        fn groupby_multi(
            tbl: &Table,
            key_indices: &[i32],
            value_indices: &[i32],
            agg_kinds: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Performs groupby scan (cumulative aggregation within groups).
        /// value_indices and agg_kinds must have the same length.
        fn groupby_scan(
            tbl: &Table,
            key_indices: &[i32],
            value_indices: &[i32],
            agg_kinds: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Shifts values within groups by specified offsets, filling with scalars.
        fn groupby_shift(
            tbl: &Table,
            key_indices: &[i32],
            value_indices: &[i32],
            offsets: &[i32],
            fill_values: Pin<&mut ScalarList>,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Replaces null values within groups using preceding/following policy.
        /// policies: 0 = PRECEDING, 1 = FOLLOWING.
        fn groupby_replace_nulls(
            tbl: &Table,
            key_indices: &[i32],
            value_indices: &[i32],
            policies: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
