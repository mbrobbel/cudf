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

        /// Builder for collecting column_views to concatenate.
        type ColumnConcatenator;

        /// Builder for collecting tables to concatenate.
        type TableConcatenator;

        // -- Concatenation --

        /// Creates a new ColumnConcatenator.
        fn new_column_concatenator() -> UniquePtr<ColumnConcatenator>;

        /// Adds a column_view to the concatenator.
        fn column_concatenator_add(cat: Pin<&mut ColumnConcatenator>, v: &column_view);

        /// Concatenates all added column_views.
        fn column_concatenator_finish(
            cat: Pin<&mut ColumnConcatenator>,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Creates a new TableConcatenator.
        fn new_table_concatenator() -> UniquePtr<TableConcatenator>;

        /// Adds a table to the concatenator.
        fn table_concatenator_add(cat: Pin<&mut TableConcatenator>, t: &Table);

        /// Concatenates all added tables.
        fn table_concatenator_finish(
            cat: Pin<&mut TableConcatenator>,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Concatenate operations --

        /// Concatenates all columns from a table into a single column.
        fn concatenate_columns(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;
        /// Concatenates two tables vertically (row-wise).
        fn concatenate_tables(lhs: &Table, rhs: &Table, stream: usize) -> Result<UniquePtr<Table>>;
    }
}
