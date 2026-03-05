// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    /// Identifies a column's logical element type.
    ///
    /// Mirrors `cudf::type_id`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(i32)]
    enum TypeId {
        EMPTY = 0,
        INT8 = 1,
        INT16 = 2,
        INT32 = 3,
        INT64 = 4,
        UINT8 = 5,
        UINT16 = 6,
        UINT32 = 7,
        UINT64 = 8,
        FLOAT32 = 9,
        FLOAT64 = 10,
        BOOL8 = 11,
        TIMESTAMP_DAYS = 12,
        TIMESTAMP_SECONDS = 13,
        TIMESTAMP_MILLISECONDS = 14,
        TIMESTAMP_MICROSECONDS = 15,
        TIMESTAMP_NANOSECONDS = 16,
        DURATION_DAYS = 17,
        DURATION_SECONDS = 18,
        DURATION_MILLISECONDS = 19,
        DURATION_MICROSECONDS = 20,
        DURATION_NANOSECONDS = 21,
        DICTIONARY32 = 22,
        STRING = 23,
        LIST = 24,
        DECIMAL32 = 25,
        DECIMAL64 = 26,
        DECIMAL128 = 27,
        STRUCT = 28,
        NUM_TYPE_IDS = 29,
    }

    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // -- DataType --

        /// Opaque wrapper around `cudf::data_type`.
        type DataType;

        /// Returns the type identifier of a `DataType`.
        fn data_type_id(dt: &DataType) -> TypeId;

        /// Returns the scale of a `DataType` (for fixed-point types).
        fn data_type_scale(dt: &DataType) -> i32;

        /// Returns the size in bytes of elements of the specified type.
        fn size_of_data_type(dt: &DataType) -> Result<usize>;

        // -- column_view --

        /// Opaque reference to `cudf::column_view`.
        #[namespace = "cudf"]
        type column_view;

        /// Returns the number of elements in a column_view.
        fn column_view_size(view: &column_view) -> i32;

        /// Returns the count of null elements in a column_view.
        fn column_view_null_count(view: &column_view) -> i32;

        /// Returns whether the column_view contains null elements.
        fn column_view_has_nulls(view: &column_view) -> bool;

        /// Returns the offset of the column_view.
        fn column_view_offset(view: &column_view) -> i32;

        /// Returns the type_id of a column_view as an i32.
        fn column_view_type_id(view: &column_view) -> i32;

        // -- Column --

        /// RAII wrapper around `cudf::column`.
        type Column;

        /// Returns the number of elements.
        fn column_size(col: &Column) -> i32;

        /// Returns the count of null elements.
        fn column_null_count(col: &Column) -> i32;

        /// Returns whether the column contains any null elements.
        fn column_has_nulls(col: &Column) -> bool;

        /// Returns whether the column has an allocated null mask.
        fn column_nullable(col: &Column) -> bool;

        /// Returns the number of child columns.
        fn column_num_children(col: &Column) -> i32;

        /// Returns the type_id as an i32.
        fn column_type_id(col: &Column) -> i32;

        /// Returns the scale (for fixed-point types).
        fn column_type_scale(col: &Column) -> i32;

        /// Returns an immutable view of the column.
        fn column_view_of(col: &Column) -> Result<&column_view>;

        // -- table_view --

        /// Opaque reference to `cudf::table_view`.
        #[namespace = "cudf"]
        type table_view;

        /// Returns the number of columns in a table_view.
        fn table_view_num_columns(view: &table_view) -> i32;

        /// Returns the number of rows in a table_view.
        fn table_view_num_rows(view: &table_view) -> i32;

        // -- Table --

        /// RAII wrapper around `cudf::table`.
        type Table;

        /// Creates an empty table.
        fn table_empty() -> UniquePtr<Table>;

        /// Returns the number of columns.
        fn table_num_columns(tbl: &Table) -> i32;

        /// Returns the number of rows.
        fn table_num_rows(tbl: &Table) -> i32;

        /// Returns the total allocation size in bytes.
        fn table_alloc_size(tbl: &Table) -> usize;

        /// Returns a column_view at the given index.
        fn table_get_column_view(tbl: &Table, index: i32) -> Result<&column_view>;

        /// Returns an immutable table_view.
        fn table_view_of(tbl: &Table) -> Result<&table_view>;
    }
}
