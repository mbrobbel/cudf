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
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Fill operations --

        /// Fills a range [begin, end) in a column with a scalar value.
        fn fill_column(
            col: &column_view,
            begin: i32,
            end: i32,
            value: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Repeats each row of a table N times.
        fn repeat_table(tbl: &Table, count: i32, stream: usize) -> Result<UniquePtr<Table>>;

        /// Generates an arithmetic sequence column.
        fn sequence_column(
            count: i32,
            init: &Scalar,
            step: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Fill (new) --

        /// Repeat table rows by per-row counts in a column.
        fn repeat_table_column(
            tbl: &Table,
            counts: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
        /// Generate calendrical month sequence.
        fn calendrical_month_sequence(
            count: i32,
            init: &Scalar,
            months: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
