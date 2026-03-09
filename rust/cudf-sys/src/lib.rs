// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Low-level CXX FFI bindings for libcudf.
//!
//! This crate provides raw FFI wrappers; prefer the safe `cudf` crate.

#![deny(clippy::undocumented_unsafe_blocks)]
// CXX-generated shared enum variants and repr fields cannot carry doc comments.
#![allow(missing_docs)]
// CXX bridge functions may need many parameters to match C++ signatures.
#![allow(clippy::too_many_arguments)]

pub mod binaryop;
pub mod compaction;
pub mod concatenate;
pub mod copying;
pub mod datetime;
pub mod dictionary;
pub mod filling;
pub mod groupby;
pub mod hashing;
pub mod io;
pub mod join;
pub mod labeling;
pub mod lists;
pub mod merge;
pub mod partitioning;
pub mod quantile;
pub mod reduction;
pub mod replace;
pub mod reshape;
pub mod rolling;
pub mod search;
pub mod sorting;
pub mod strings;
pub mod transform;
pub mod unary;

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

    /// Sort order for columns.
    ///
    /// Mirrors `cudf::order`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum Order {
        ASCENDING = 0,
        DESCENDING = 1,
    }

    /// Determines where null values appear in sorted output.
    ///
    /// Mirrors `cudf::null_order`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum NullOrder {
        AFTER = 0,
        BEFORE = 1,
    }

    /// Whether to include or exclude null elements.
    ///
    /// Mirrors `cudf::null_policy`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum NullPolicy {
        EXCLUDE = 0,
        INCLUDE = 1,
    }

    /// Binary operation types.
    ///
    /// Mirrors `cudf::binary_operator`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum BinaryOperator {
        ADD = 0,
        SUB = 1,
        MUL = 2,
        DIV = 3,
        TRUE_DIV = 4,
        FLOOR_DIV = 5,
        MOD = 6,
        PMOD = 7,
        PYMOD = 8,
        POW = 9,
        INT_POW = 10,
        LOG_BASE = 11,
        ATAN2 = 12,
        SHIFT_LEFT = 13,
        SHIFT_RIGHT = 14,
        SHIFT_RIGHT_UNSIGNED = 15,
        BITWISE_AND = 16,
        BITWISE_OR = 17,
        BITWISE_XOR = 18,
        LOGICAL_AND = 19,
        LOGICAL_OR = 20,
        EQUAL = 21,
        NOT_EQUAL = 22,
        LESS = 23,
        GREATER = 24,
        LESS_EQUAL = 25,
        GREATER_EQUAL = 26,
        NULL_EQUALS = 27,
        NULL_NOT_EQUALS = 28,
        NULL_MAX = 29,
        NULL_MIN = 30,
        GENERIC_BINARY = 31,
        NULL_LOGICAL_AND = 32,
        NULL_LOGICAL_OR = 33,
        INVALID_BINARY = 34,
    }

    /// Whether null values compare equal in join operations.
    ///
    /// Mirrors `cudf::null_equality`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum NullEquality {
        EQUAL = 0,
        UNEQUAL = 1,
    }

    /// Aggregation operation kind for groupby.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum AggregationKind {
        SUM = 0,
        MIN = 1,
        MAX = 2,
        MEAN = 3,
        COUNT = 4,
        NUNIQUE = 5,
        MEDIAN = 6,
        STD = 7,
        VAR = 8,
        PRODUCT = 9,
        ANY = 10,
        ALL = 11,
        ARGMAX = 12,
        ARGMIN = 13,
        COLLECT_LIST = 14,
        COLLECT_SET = 15,
        HISTOGRAM = 16,
        SUM_OF_SQUARES = 17,
        M2 = 18,
        MERGE_LISTS = 19,
        MERGE_M2 = 20,
        MERGE_HISTOGRAM = 21,
    }

    /// Interpolation strategy for quantile computation.
    ///
    /// Mirrors `cudf::interpolation`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum Interpolation {
        LINEAR = 0,
        LOWER = 1,
        HIGHER = 2,
        MIDPOINT = 3,
        NEAREST = 4,
    }

    /// Unary operation types.
    ///
    /// Mirrors `cudf::unary_operator`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum UnaryOperator {
        SIN = 0,
        COS = 1,
        TAN = 2,
        ARCSIN = 3,
        ARCCOS = 4,
        ARCTAN = 5,
        SINH = 6,
        COSH = 7,
        TANH = 8,
        ARCSINH = 9,
        ARCCOSH = 10,
        ARCTANH = 11,
        EXP = 12,
        LOG = 13,
        SQRT = 14,
        CBRT = 15,
        CEIL = 16,
        FLOOR = 17,
        ABS = 18,
        RINT = 19,
        BIT_COUNT = 20,
        BIT_INVERT = 21,
        NOT = 22,
        NEGATE = 23,
    }

    /// Rounding method for round operations.
    ///
    /// Mirrors `cudf::rounding_method`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum RoundingMethod {
        HALF_UP = 0,
        HALF_EVEN = 1,
    }

    /// Which duplicate rows to keep.
    ///
    /// Mirrors `cudf::duplicate_keep_option`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum DuplicateKeepOption {
        KEEP_ANY = 0,
        KEEP_FIRST = 1,
        KEEP_LAST = 2,
        KEEP_NONE = 3,
    }

    /// Whether NaN values compare equal.
    ///
    /// Mirrors `cudf::nan_equality`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum NanEquality {
        ALL_EQUAL = 0,
        UNEQUAL = 1,
    }

    /// Scan direction (inclusive or exclusive prefix scan).
    ///
    /// Mirrors `cudf::scan_type`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum ScanType {
        INCLUSIVE = 0,
        EXCLUSIVE = 1,
    }

    /// Whether a bin edge is inclusive.
    ///
    /// Mirrors `cudf::inclusive`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum Inclusive {
        YES = 0,
        NO = 1,
    }

    /// Side for string padding.
    ///
    /// Mirrors `cudf::strings::side_type`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    enum SideType {
        LEFT = 0,
        RIGHT = 1,
        BOTH = 2,
    }

    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // -- Stream --

        /// Returns the default CUDA stream used by cudf as a raw handle.
        fn get_default_stream() -> usize;

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
        fn column_view_of(col: &Column) -> &column_view;

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

        // -- Scalar --

        /// RAII wrapper around `cudf::scalar`.
        type Scalar;

        /// Creates an INT32 scalar.
        fn make_int32_scalar(value: i32, valid: bool) -> UniquePtr<Scalar>;

        /// Creates an INT64 scalar.
        fn make_int64_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;

        /// Creates a FLOAT32 scalar.
        fn make_float32_scalar(value: f32, valid: bool) -> UniquePtr<Scalar>;

        /// Creates a FLOAT64 scalar.
        fn make_float64_scalar(value: f64, valid: bool) -> UniquePtr<Scalar>;

        /// Creates a BOOL8 scalar.
        fn make_bool_scalar(value: bool, valid: bool) -> UniquePtr<Scalar>;

        /// Creates a STRING scalar.
        fn make_string_scalar(value: &str) -> UniquePtr<Scalar>;
        fn make_int8_scalar(value: i8, valid: bool) -> UniquePtr<Scalar>;
        fn make_int16_scalar(value: i16, valid: bool) -> UniquePtr<Scalar>;
        fn make_uint8_scalar(value: u8, valid: bool) -> UniquePtr<Scalar>;
        fn make_uint16_scalar(value: u16, valid: bool) -> UniquePtr<Scalar>;
        fn make_uint32_scalar(value: u32, valid: bool) -> UniquePtr<Scalar>;
        fn make_uint64_scalar(value: u64, valid: bool) -> UniquePtr<Scalar>;
        fn make_timestamp_s_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_timestamp_ms_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_timestamp_us_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_timestamp_ns_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_duration_s_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_duration_ms_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_duration_us_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        fn make_duration_ns_scalar(value: i64, valid: bool) -> UniquePtr<Scalar>;
        /// Create a default-constructed scalar of the given type (invalid, zero-initialized).
        fn make_default_constructed_scalar(type_id: i32, scale: i32) -> UniquePtr<Scalar>;
        /// Create an empty scalar with the same type as the given column.
        fn make_empty_scalar_like(col: &column_view) -> UniquePtr<Scalar>;

        /// Returns whether the scalar holds a valid (non-null) value.
        fn scalar_is_valid(s: &Scalar) -> bool;

        /// Returns the type_id of the scalar as an i32.
        fn scalar_type_id(s: &Scalar) -> i32;

        /// Extracts an i32 value from a numeric scalar.
        fn scalar_to_i32(s: &Scalar) -> i32;

        /// Extracts an i64 value from a numeric scalar.
        fn scalar_to_i64(s: &Scalar) -> i64;

        /// Extracts an f32 value from a numeric scalar.
        fn scalar_to_f32(s: &Scalar) -> f32;

        /// Extracts an f64 value from a numeric scalar.
        fn scalar_to_f64(s: &Scalar) -> f64;

        /// Extracts a bool value from a numeric scalar.
        fn scalar_to_bool(s: &Scalar) -> bool;

        // -- ScalarList --

        /// Builder for collecting scalars (used by scatter_scalars, groupby_shift).
        type ScalarList;

        /// Creates a new empty ScalarList.
        fn new_scalar_list() -> UniquePtr<ScalarList>;

        /// Adds a scalar to the list.
        fn scalar_list_add(list: Pin<&mut ScalarList>, s: UniquePtr<Scalar>);

        // -- Column factories --

        /// Creates a column by repeating a scalar value `count` times.
        fn make_column_from_scalar(s: &Scalar, count: i32, stream: usize) -> UniquePtr<Column>;

        /// Creates an empty column of the given type_id.
        fn make_empty_column_by_type(type_id: i32) -> UniquePtr<Column>;

        /// Create an uninitialized fixed-width column.
        /// mask_state: 0=UNALLOCATED, 1=UNINITIALIZED, 2=ALL_VALID, 3=ALL_NULL.
        fn make_fixed_width_column(
            type_id: i32,
            scale: i32,
            num_rows: i32,
            mask_state: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Create an empty lists column with the given child element type.
        fn make_empty_lists_column(child_type_id: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Create a dictionary column filled with a single scalar value.
        fn make_dictionary_from_scalar(
            scalar: &Scalar,
            num_rows: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Create a LIST column from offsets and child (no null mask).
        fn make_lists_column(
            num_rows: i32,
            offsets: UniquePtr<Column>,
            child: UniquePtr<Column>,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Builder for constructing a STRUCT column from child columns.
        type StructColumnBuilder;
        fn new_struct_column_builder() -> UniquePtr<StructColumnBuilder>;
        fn struct_column_builder_add(
            builder: Pin<&mut StructColumnBuilder>,
            col: UniquePtr<Column>,
        );
        fn struct_column_builder_build(
            builder: Pin<&mut StructColumnBuilder>,
            num_rows: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Column data extraction (device -> host) --

        /// Copies INT16 column data to a host vector.
        fn column_to_host_i16(col: &Column, stream: usize) -> Vec<i16>;

        /// Copies INT32 column data to a host vector.
        fn column_to_host_i32(col: &Column, stream: usize) -> Vec<i32>;

        /// Copies INT64 column data to a host vector.
        fn column_to_host_i64(col: &Column, stream: usize) -> Vec<i64>;

        /// Copies FLOAT32 column data to a host vector.
        fn column_to_host_f32(col: &Column, stream: usize) -> Vec<f32>;

        /// Copies FLOAT64 column data to a host vector.
        fn column_to_host_f64(col: &Column, stream: usize) -> Vec<f64>;

        /// Copies INT8 column data to a host vector.
        fn column_to_host_i8(col: &Column, stream: usize) -> Vec<i8>;

        /// Copies UINT8 column data to a host vector.
        fn column_to_host_u8(col: &Column, stream: usize) -> Vec<u8>;

        /// Copies UINT16 column data to a host vector.
        fn column_to_host_u16(col: &Column, stream: usize) -> Vec<u16>;

        /// Copies UINT32 column data to a host vector.
        fn column_to_host_u32(col: &Column, stream: usize) -> Vec<u32>;

        /// Copies UINT64 column data to a host vector.
        fn column_to_host_u64(col: &Column, stream: usize) -> Vec<u64>;

        /// Copies BOOL8 column data to a host vector of bools.
        fn column_to_host_bool(col: &Column, stream: usize) -> Vec<bool>;

        /// Extracts per-element null mask as a host vector of bools.
        fn column_null_mask_to_host(col: &Column, stream: usize) -> Vec<bool>;

        // -- Column factories from host data --

        /// Creates an INT32 column from host data.
        fn make_column_from_host_i32(data: &[i32], stream: usize) -> UniquePtr<Column>;

        /// Creates an INT64 column from host data.
        fn make_column_from_host_i64(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a FLOAT64 column from host data.
        fn make_column_from_host_f64(data: &[f64], stream: usize) -> UniquePtr<Column>;

        /// Creates a BOOL8 column from host data.
        fn make_column_from_host_bool(data: &[bool], stream: usize) -> UniquePtr<Column>;

        /// Creates an INT8 column from host data.
        fn make_column_from_host_i8(data: &[i8], stream: usize) -> UniquePtr<Column>;

        /// Creates an INT16 column from host data.
        fn make_column_from_host_i16(data: &[i16], stream: usize) -> UniquePtr<Column>;

        /// Creates a FLOAT32 column from host data.
        fn make_column_from_host_f32(data: &[f32], stream: usize) -> UniquePtr<Column>;

        /// Creates a UINT8 column from host data.
        fn make_column_from_host_u8(data: &[u8], stream: usize) -> UniquePtr<Column>;

        /// Creates a UINT16 column from host data.
        fn make_column_from_host_u16(data: &[u16], stream: usize) -> UniquePtr<Column>;

        /// Creates a UINT32 column from host data.
        fn make_column_from_host_u32(data: &[u32], stream: usize) -> UniquePtr<Column>;

        /// Creates a UINT64 column from host data.
        fn make_column_from_host_u64(data: &[u64], stream: usize) -> UniquePtr<Column>;

        /// Creates a TIMESTAMP_SECONDS column from host epoch-second data.
        fn make_column_from_host_timestamp_s(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a TIMESTAMP_MILLISECONDS column from host data.
        fn make_column_from_host_timestamp_ms(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a TIMESTAMP_MICROSECONDS column from host data.
        fn make_column_from_host_timestamp_us(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a TIMESTAMP_NANOSECONDS column from host data.
        fn make_column_from_host_timestamp_ns(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a DURATION_SECONDS column from host data.
        fn make_column_from_host_duration_s(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a DURATION_MILLISECONDS column from host data.
        fn make_column_from_host_duration_ms(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a DURATION_MICROSECONDS column from host data.
        fn make_column_from_host_duration_us(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a DURATION_NANOSECONDS column from host data.
        fn make_column_from_host_duration_ns(data: &[i64], stream: usize) -> UniquePtr<Column>;

        // -- TableBuilder --

        /// Builder for constructing a Table from individual columns.
        type TableBuilder;

        /// Creates a new empty TableBuilder.
        fn new_table_builder() -> UniquePtr<TableBuilder>;

        /// Adds a column to the builder.
        fn table_builder_add_column(builder: Pin<&mut TableBuilder>, col: UniquePtr<Column>);

        /// Consumes the builder and returns a Table.
        fn table_builder_build(builder: Pin<&mut TableBuilder>) -> Result<UniquePtr<Table>>;

        // -- Null mask utilities --

        /// Compute bitmask allocation size in bytes.
        fn bitmask_allocation_size_bytes(number_of_bits: i32) -> usize;
        /// Compute number of bitmask words needed.
        fn num_bitmask_words(number_of_bits: i32) -> i32;
        /// Returns the null count implied by a mask_state for a given number of rows.
        fn state_null_count(mask_state: i32, num_rows: i32) -> i32;

        // -- Null mask conversions --

        /// Convert a column's null mask to a BOOL8 column (true = valid, false = null).
        fn null_mask_to_bools(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Set a column's null mask from a BOOL8 column (true = valid, false = null).
        fn set_null_mask_from_bools(
            col: Pin<&mut Column>,
            bools: &column_view,
            stream: usize,
        ) -> Result<()>;

        // -- Bitmask combining --

        /// Bitwise AND of all column null masks in a table, returned as BOOL8 column.
        fn bitmask_and_to_bools(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;
        /// Bitwise OR of all column null masks in a table, returned as BOOL8 column.
        fn bitmask_or_to_bools(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Type checking utilities --

        /// Check if two column types are equivalent (ignoring scale for fixed-point).
        fn column_types_equivalent(lhs: &column_view, rhs: &column_view) -> bool;
        /// Check if two columns have the same types (including scale, nested types).
        fn columns_have_same_types(lhs: &column_view, rhs: &column_view) -> bool;
        /// Check if two tables have columns of the same types.
        fn tables_have_same_types(lhs: &Table, rhs: &Table) -> bool;
        /// Check if a cast from one data type to another is supported.
        fn is_supported_cast(
            from_type_id: i32,
            from_scale: i32,
            to_type_id: i32,
            to_scale: i32,
        ) -> bool;

        // -- Column child access --

        /// Deep-copies a child column by index (for STRUCT, LIST, DICTIONARY columns).
        fn column_view_child_copy(
            col: &column_view,
            index: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Returns the number of child columns.
        fn column_view_num_children(col: &column_view) -> i32;

        // -- Table nested column queries --

        /// Returns true if the table contains any nested columns (LIST, STRUCT).
        fn table_has_nested_columns(tbl: &Table) -> bool;
        /// Returns true if any nested column in the table has nulls.
        fn table_has_nested_nulls(tbl: &Table) -> bool;
        /// Returns true if any nested column in the table is nullable.
        fn table_has_nested_nullable_columns(tbl: &Table) -> bool;

        // -- Column with null mask from bools --

        /// Creates a copy of the column with a null mask derived from a boolean validity column.
        fn column_with_null_mask_from_bools(
            col: &Column,
            validity: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}

// SAFETY: cudf::column wraps GPU memory which is globally accessible from any
// CPU thread. The CXX UniquePtr ensures exclusive ownership.
unsafe impl Send for ffi::Column {}
// SAFETY: &Column only allows immutable access through the FFI.
unsafe impl Sync for ffi::Column {}

// SAFETY: cudf::table wraps GPU memory which is globally accessible from any
// CPU thread. The CXX UniquePtr ensures exclusive ownership.
unsafe impl Send for ffi::Table {}
// SAFETY: &Table only allows immutable access through the FFI.
unsafe impl Sync for ffi::Table {}

impl std::fmt::Display for ffi::TypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// Converts a `TypeId` raw i32 value from C++ into the corresponding enum variant.
///
/// Returns `None` if the value is out of range.
pub fn type_id_from_i32(value: i32) -> Option<ffi::TypeId> {
    if (0..ffi::TypeId::NUM_TYPE_IDS.repr).contains(&value) {
        // SAFETY: TypeId is #[repr(i32)] and we verified the value is in range.
        Some(unsafe { std::mem::transmute::<i32, ffi::TypeId>(value) })
    } else {
        None
    }
}

/// Zero-cost reinterpretation of a `#[repr(i32)]` enum slice as `&[i32]`.
///
/// This is safe because `Order`, `NullOrder`, and `AggregationKind` are all
/// `#[repr(i32)]` CXX shared enums with the same size and alignment as `i32`.
pub fn orders_as_i32(slice: &[ffi::Order]) -> &[i32] {
    // SAFETY: Order is #[repr(i32)] with identical size and alignment.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<i32>(), slice.len()) }
}

/// Zero-cost reinterpretation of a `NullOrder` slice as `&[i32]`.
pub fn null_orders_as_i32(slice: &[ffi::NullOrder]) -> &[i32] {
    // SAFETY: NullOrder is #[repr(i32)] with identical size and alignment.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<i32>(), slice.len()) }
}

/// Zero-cost reinterpretation of an `AggregationKind` slice as `&[i32]`.
pub fn agg_kinds_as_i32(slice: &[ffi::AggregationKind]) -> &[i32] {
    // SAFETY: AggregationKind is #[repr(i32)] with identical size and alignment.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<i32>(), slice.len()) }
}
