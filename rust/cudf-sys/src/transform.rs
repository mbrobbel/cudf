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

        // -- Transform --

        /// Converts NaN values to null in a floating-point column.
        fn nans_to_nulls(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Encodes table rows as integer indices into sorted distinct rows.
        fn encode_table(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the sorted distinct key rows from encoding.
        fn encode_keys(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Transform (new) --

        /// Approximate per-row bit count.
        fn row_bit_count(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Transform: segmented_row_bit_count --

        /// Per-segment cumulative row bit count.
        fn segmented_row_bit_count(
            tbl: &Table,
            segment_length: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- One-hot encoding --

        /// One-hot encode input against categories, returning a table of BOOL8 columns.
        fn one_hot_encode(
            input: &column_view,
            categories: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
