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

        // -- Sorting --

        /// Sorts the rows of a table.
        fn sort_table(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns the row indices that would sort the table.
        fn sorted_order(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Checks whether the rows of a table are sorted.
        fn is_sorted_table(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<bool>;

        // -- Sorting (new) --

        /// Returns the row indices that would produce a stable sort of the table.
        fn stable_sorted_order(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Stable-sorts the rows of a table.
        fn stable_sort_table(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Sorts a table of values by a table of keys.
        fn sort_by_key(
            values: &Table,
            keys: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Stable-sorts a table of values by a table of keys.
        fn stable_sort_by_key(
            values: &Table,
            keys: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Computes the rank of each element in a column.
        fn rank_column(
            col: &column_view,
            method: i32,
            column_order: i32,
            null_handling: i32,
            null_precedence: i32,
            percentage: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns the top-k elements of a column.
        fn top_k(col: &column_view, k: i32, order: i32, stream: usize)
        -> Result<UniquePtr<Column>>;

        /// Returns the row indices of the top-k elements of a column.
        fn top_k_order(
            col: &column_view,
            k: i32,
            order: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns row indices that would sort each segment of a table.
        fn segmented_sorted_order(
            tbl: &Table,
            segment_offsets: &column_view,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns the top-k elements within each segment of a column.
        fn segmented_top_k(
            col: &column_view,
            segment_offsets: &column_view,
            k: i32,
            order: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns the row indices of the top-k elements within each segment.
        fn segmented_top_k_order(
            col: &column_view,
            segment_offsets: &column_view,
            k: i32,
            order: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Sorting: stable segmented --

        /// Returns row indices for a stable sort within each segment.
        fn stable_segmented_sorted_order(
            tbl: &Table,
            segment_offsets: &column_view,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Stable-sorts values by keys within each segment.
        fn stable_segmented_sort_by_key(
            values: &Table,
            keys: &Table,
            segment_offsets: &column_view,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Sorting: segmented_sort_by_key (non-stable) --

        /// Sorts values by keys within each segment.
        fn segmented_sort_by_key(
            values: &Table,
            keys: &Table,
            segment_offsets: &column_view,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
