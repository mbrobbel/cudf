// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Filtering --

        /// Filters a table by a boolean mask column.
        fn apply_boolean_mask(
            tbl: &Table,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Drops rows where all columns are null.
        fn drop_nulls_all(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Stream compaction --

        /// Drops rows from a table where any of the specified key columns contain NaN.
        fn drop_nans(tbl: &Table, keys: &[i32], stream: usize) -> Result<UniquePtr<Table>>;

        /// Drops rows where the number of non-null key values is below the threshold.
        fn drop_nulls_with_threshold(
            tbl: &Table,
            keys: &[i32],
            threshold: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns unique consecutive rows based on key columns.
        fn unique_table(
            tbl: &Table,
            keys: &[i32],
            keep: i32,
            null_equal: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns distinct rows based on key columns.
        fn distinct_table(
            tbl: &Table,
            keys: &[i32],
            keep: i32,
            null_equal: i32,
            nan_equal: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns indices of distinct rows.
        fn distinct_indices_column(
            tbl: &Table,
            keep: i32,
            null_equal: i32,
            nan_equal: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns distinct rows preserving input order.
        fn stable_distinct_table(
            tbl: &Table,
            keys: &[i32],
            keep: i32,
            null_equal: i32,
            nan_equal: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Distinct count --

        /// Count distinct values in a column.
        fn distinct_count_column(
            col: &column_view,
            null_policy: i32,
            nan_is_null: bool,
            stream: usize,
        ) -> i32;
        /// Count distinct rows in a table.
        fn distinct_count_table(tbl: &Table, null_equality: i32, stream: usize) -> i32;

        // -- Unique count (consecutive) --

        /// Count consecutive unique values in a column.
        fn unique_count_column(
            col: &column_view,
            null_policy: i32,
            nan_is_null: bool,
            stream: usize,
        ) -> i32;
        /// Count consecutive unique rows in a table.
        fn unique_count_table(tbl: &Table, null_equality: i32, stream: usize) -> i32;

        // -- Drop NaNs with threshold --

        /// Drop rows with NaN values, keeping rows with at least threshold non-NaN values.
        fn drop_nans_with_threshold(
            tbl: &Table,
            keys: &[i32],
            threshold: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Approximate distinct count --

        /// Approximate distinct count using HyperLogLog. Precision 4-18 (default 12).
        fn approx_distinct_count(tbl: &Table, precision: i32, stream: usize) -> usize;
    }
}
