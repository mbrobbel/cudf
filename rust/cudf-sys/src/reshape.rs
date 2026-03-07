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

        // -- Reshape --

        /// Interleaves columns of a table into a single column.
        fn interleave_columns(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// Tiles (repeats) the rows of a table.
        fn tile_table(tbl: &Table, count: i32, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Reshape (new) --

        /// Convert column elements to lists of bytes.
        fn byte_cast_column(
            col: &column_view,
            flip_endian: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Transpose --

        /// Transposes a table.
        fn transpose_table(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;
    }
}
