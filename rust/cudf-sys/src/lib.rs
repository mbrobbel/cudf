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

        // -- Column factories --

        /// Creates a column by repeating a scalar value `count` times.
        fn make_column_from_scalar(s: &Scalar, count: i32, stream: usize) -> UniquePtr<Column>;

        /// Creates an empty column of the given type_id.
        fn make_empty_column_by_type(type_id: i32) -> UniquePtr<Column>;

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

        /// Copies BOOL8 column data to a host vector of bools.
        fn column_to_host_bool(col: &Column, stream: usize) -> Vec<bool>;

        /// Extracts per-element null mask as a host vector of bools.
        fn column_null_mask_to_host(col: &Column, stream: usize) -> Vec<bool>;

        // -- TableBuilder --

        /// Builder for constructing a Table from individual columns.
        type TableBuilder;

        /// Creates a new empty TableBuilder.
        fn new_table_builder() -> UniquePtr<TableBuilder>;

        /// Adds a column to the builder.
        fn table_builder_add_column(builder: Pin<&mut TableBuilder>, col: UniquePtr<Column>);

        /// Consumes the builder and returns a Table.
        fn table_builder_build(builder: Pin<&mut TableBuilder>) -> Result<UniquePtr<Table>>;

        // -- Binary operations --

        /// Binary operation between two column_views.
        fn binary_operation_columns(
            lhs: &column_view,
            rhs: &column_view,
            op: BinaryOperator,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Binary operation between a column_view and a Scalar.
        fn binary_operation_column_scalar(
            lhs: &column_view,
            rhs: &Scalar,
            op: BinaryOperator,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Binary operation between a Scalar and a column_view.
        fn binary_operation_scalar_column(
            lhs: &Scalar,
            rhs: &column_view,
            op: BinaryOperator,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Unary operations --

        /// Casts a column to a different type.
        fn unary_cast(col: &column_view, target_type_id: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates a null value.
        fn unary_is_null(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates a valid value.
        fn unary_is_valid(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates NaN.
        fn unary_is_nan(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Negates every element of the column.
        fn unary_negate(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the absolute value of every element.
        fn unary_abs(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Reduction --

        /// Computes the sum of all elements.
        fn reduce_sum(col: &column_view, output_type_id: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Computes the minimum value.
        fn reduce_min(col: &column_view, output_type_id: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Computes the maximum value.
        fn reduce_max(col: &column_view, output_type_id: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Computes the product of all elements.
        fn reduce_product(col: &column_view, output_type_id: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Returns true if any element is non-zero.
        fn reduce_any(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Returns true if all elements are non-zero.
        fn reduce_all(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        // -- Sorting --

        /// Sorts a table.
        fn sort_table(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns the sorted row indices.
        fn sorted_order(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns whether the table rows are sorted.
        fn is_sorted_table(
            tbl: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<bool>;

        // -- Filtering --

        /// Filters a table by a boolean mask column.
        fn apply_boolean_mask(tbl: &Table, mask: &column_view, stream: usize) -> Result<UniquePtr<Table>>;

        /// Drops rows where all columns are null.
        fn drop_nulls_all(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Concatenation --

        /// Accumulates column_views for concatenation.
        type ColumnConcatenator;

        /// Creates a new ColumnConcatenator.
        fn new_column_concatenator() -> UniquePtr<ColumnConcatenator>;

        /// Adds a column_view to the concatenator.
        fn column_concatenator_add(cat: Pin<&mut ColumnConcatenator>, v: &column_view);

        /// Concatenates all added column_views.
        fn column_concatenator_finish(
            cat: Pin<&mut ColumnConcatenator>,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Accumulates table views for concatenation.
        type TableConcatenator;

        /// Creates a new TableConcatenator.
        fn new_table_concatenator() -> UniquePtr<TableConcatenator>;

        /// Adds a table to the concatenator.
        fn table_concatenator_add(cat: Pin<&mut TableConcatenator>, t: &Table);

        /// Concatenates all added tables.
        fn table_concatenator_finish(
            cat: Pin<&mut TableConcatenator>,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Copying --

        /// Gathers rows from a table using index column.
        fn gather_table(tbl: &Table, indices: &column_view, stream: usize) -> Result<UniquePtr<Table>>;

        /// Creates an empty column with the same type as input.
        fn empty_like_column(col: &column_view) -> UniquePtr<Column>;

        /// Creates an empty table with the same schema as input.
        fn empty_like_table(tbl: &Table) -> UniquePtr<Table>;

        // -- Column factories from host data --

        /// Creates an INT32 column from host data.
        fn make_column_from_host_i32(data: &[i32], stream: usize) -> UniquePtr<Column>;

        /// Creates an INT64 column from host data.
        fn make_column_from_host_i64(data: &[i64], stream: usize) -> UniquePtr<Column>;

        /// Creates a FLOAT64 column from host data.
        fn make_column_from_host_f64(data: &[f64], stream: usize) -> UniquePtr<Column>;

        /// Creates a BOOL8 column from host data.
        fn make_column_from_host_bool(data: &[bool], stream: usize) -> UniquePtr<Column>;

        /// Creates a TIMESTAMP_SECONDS column from host epoch-second data.
        fn make_column_from_host_timestamp_s(data: &[i64], stream: usize) -> UniquePtr<Column>;

        // -- Replace operations --

        /// Replaces null values with corresponding values from replacement column.
        fn replace_nulls_column(
            col: &column_view,
            replacement: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces null values with a scalar.
        fn replace_nulls_scalar(
            col: &column_view,
            replacement: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces NaN values with corresponding values from replacement column.
        fn replace_nans_column(
            col: &column_view,
            replacement: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces NaN values with a scalar.
        fn replace_nans_scalar(
            col: &column_view,
            replacement: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Clamps column values to [lo, hi] range.
        fn clamp_column(
            col: &column_view,
            lo: &Scalar,
            hi: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds and replaces all matching values in a column.
        fn find_and_replace_all(
            col: &column_view,
            values_to_replace: &column_view,
            replacement_values: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

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

        // -- Search operations --

        /// Checks if a scalar value exists in a column.
        fn contains_scalar(haystack: &column_view, needle: &Scalar, stream: usize) -> Result<bool>;

        /// Checks which values from needles exist in haystack.
        fn contains_column(
            haystack: &column_view,
            needles: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds lower bound insertion points in a sorted table.
        fn lower_bound(
            haystack: &Table,
            needles: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds upper bound insertion points in a sorted table.
        fn upper_bound(
            haystack: &Table,
            needles: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Quantile operations --

        /// Computes quantiles of a column.
        fn quantile_column(
            col: &column_view,
            quantiles: &[f64],
            interp: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Join operations --

        /// Inner join: returns a table with all columns from left and right
        /// for matching rows.
        fn inner_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left join: returns all rows from left, with matching right rows
        /// (nulls for unmatched).
        fn left_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Full outer join: returns all rows from both sides.
        fn full_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left semi join: returns rows from left that have matches in right.
        fn left_semi_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left anti join: returns rows from left that have NO matches in right.
        fn left_anti_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- String operations --

        /// Converts strings to lower case.
        fn strings_to_lower(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts strings to upper case.
        fn strings_to_upper(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns BOOL8 column indicating whether each string contains the target.
        fn strings_contains(col: &column_view, target: &Scalar, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns BOOL8 column indicating whether each string starts with the target.
        fn strings_starts_with(col: &column_view, target: &Scalar, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns BOOL8 column indicating whether each string ends with the target.
        fn strings_ends_with(col: &column_view, target: &Scalar, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns INT32 column with position of first occurrence of target in each string.
        fn strings_find(col: &column_view, target: &Scalar, stream: usize) -> Result<UniquePtr<Column>>;

        /// Replaces occurrences of target with replacement in each string.
        fn strings_replace(
            col: &column_view,
            target: &Scalar,
            replacement: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Strips whitespace from both sides of each string.
        fn strings_strip(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Strips whitespace from the left side of each string.
        fn strings_lstrip(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Strips whitespace from the right side of each string.
        fn strings_rstrip(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns INT32 column with character count of each string.
        fn strings_count_characters(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns INT32 column with byte count of each string.
        fn strings_count_bytes(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts an integer column to a string column.
        fn strings_from_integers(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts a string column to an integer column.
        fn strings_to_integers(col: &column_view, output_type_id: i32, stream: usize)
            -> Result<UniquePtr<Column>>;

        /// Converts a float column to a string column.
        fn strings_from_floats(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts a string column to a float column.
        fn strings_to_floats(col: &column_view, output_type_id: i32, stream: usize)
            -> Result<UniquePtr<Column>>;

        // -- String column construction --

        /// Creates a string column from a vector of strings.
        fn make_string_column(strings: Vec<String>, stream: usize) -> UniquePtr<Column>;

        // -- String column extraction (device -> host) --

        /// Copies string column data to a host vector of strings.
        fn column_to_host_strings(col: &Column, stream: usize) -> Vec<String>;

        // -- I/O --

        /// Reads a CSV file and returns a Table.
        fn read_csv(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Reads a CSV file with options and returns a Table.
        fn read_csv_with_options(
            filepath: &str,
            delimiter: u8,
            header: bool,
            skip_rows: i32,
            num_rows: i32,
        ) -> Result<UniquePtr<Table>>;

        /// Writes a Table to a CSV file.
        fn write_csv(tbl: &Table, filepath: &str) -> Result<()>;

        /// Writes a Table to a CSV file with options.
        fn write_csv_with_options(
            tbl: &Table,
            filepath: &str,
            delimiter: u8,
            include_header: bool,
            na_rep: &str,
        ) -> Result<()>;

        /// Reads a Parquet file and returns a Table.
        fn read_parquet(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Writes a Table to a Parquet file.
        fn write_parquet(tbl: &Table, filepath: &str) -> Result<()>;

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

        // -- Datetime operations --

        /// Extracts the year component from a timestamp column (returns INT16).
        fn datetime_extract_year(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the month component from a timestamp column (returns INT16).
        fn datetime_extract_month(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the day component from a timestamp column (returns INT16).
        fn datetime_extract_day(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the weekday component from a timestamp column (returns INT16).
        fn datetime_extract_weekday(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the hour component from a timestamp column (returns INT16).
        fn datetime_extract_hour(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the minute component from a timestamp column (returns INT16).
        fn datetime_extract_minute(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the second component from a timestamp column (returns INT16).
        fn datetime_extract_second(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the day of year (1-366) for each timestamp (returns INT16).
        fn datetime_day_of_year(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns whether each timestamp's year is a leap year (returns BOOL8).
        fn datetime_is_leap_year(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the number of days in the month for each timestamp (returns INT16).
        fn datetime_days_in_month(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the last day of the month for each timestamp (returns TIMESTAMP_DAYS).
        fn datetime_last_day_of_month(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the quarter (1-4) for each timestamp (returns INT16).
        fn datetime_extract_quarter(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Hashing --

        /// Computes MurmurHash3 32-bit hash of each row.
        fn hash_murmur3(tbl: &Table, seed: u32, stream: usize) -> UniquePtr<Column>;

        /// Computes XXHash64 hash of each row.
        fn hash_xxhash64(tbl: &Table, seed: u64, stream: usize) -> UniquePtr<Column>;

        /// Computes MD5 hash of each row (returns string column).
        fn hash_md5(tbl: &Table, stream: usize) -> UniquePtr<Column>;

        /// Computes SHA-256 hash of each row (returns string column).
        fn hash_sha256(tbl: &Table, stream: usize) -> UniquePtr<Column>;

        // -- Reshape --

        /// Interleaves columns of a table into a single column.
        fn interleave_columns(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// Tiles (repeats) the rows of a table.
        fn tile_table(tbl: &Table, count: i32, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Transform --

        /// Converts NaN values to null in a floating-point column.
        fn nans_to_nulls(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Encodes table rows as integer indices into sorted distinct rows.
        fn encode_table(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the sorted distinct key rows from encoding.
        fn encode_keys(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Merge --

        /// Merges two sorted tables maintaining sort order.
        fn merge_tables(
            left: &Table,
            right: &Table,
            key_indices: &[i32],
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Partitioning --

        /// Hash-partitions a table into N partitions.
        fn hash_partition_table(
            tbl: &Table,
            columns_to_hash: &[i32],
            num_partitions: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns partition offsets for hash partitioning.
        fn hash_partition_offsets(
            tbl: &Table,
            columns_to_hash: &[i32],
            num_partitions: i32,
            stream: usize,
        ) -> Result<Vec<i32>>;

        /// Round-robin partitions a table.
        fn round_robin_partition_table(
            tbl: &Table,
            num_partitions: i32,
            start_partition: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns partition offsets for round-robin partitioning.
        fn round_robin_partition_offsets(
            tbl: &Table,
            num_partitions: i32,
            start_partition: i32,
            stream: usize,
        ) -> Result<Vec<i32>>;

        // -- Generic unary operation --

        /// Applies a unary operation to a column.
        fn unary_operation(col: &column_view, op: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates a non-NaN value.
        fn unary_is_not_nan(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Round --

        /// Rounds column values to the given number of decimal places.
        fn round_column(
            col: &column_view,
            decimal_places: i32,
            method: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

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

        // -- Copying extras --

        /// Scatters source table rows into target table at given indices.
        fn scatter_table(
            source: &Table,
            scatter_map: &column_view,
            target: &Table,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Reverses the elements of a column.
        fn reverse_column(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Reverses the rows of a table.
        fn reverse_table(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        /// Shifts column values by offset, filling with the given scalar.
        fn shift_column(
            col: &column_view,
            offset: i32,
            fill_value: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns a single element from a column as a scalar.
        fn get_element(col: &column_view, index: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Selects elements from lhs or rhs based on a boolean mask.
        fn copy_if_else_columns(
            lhs: &column_view,
            rhs: &column_view,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Selects elements from a scalar or column based on a boolean mask.
        fn copy_if_else_scalar_column(
            lhs: &Scalar,
            rhs: &column_view,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Selects elements from a column or scalar based on a boolean mask.
        fn copy_if_else_column_scalar(
            lhs: &column_view,
            rhs: &Scalar,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Extracts a slice [begin, end) of a column as a new owned column.
        fn slice_column(col: &column_view, begin: i32, end: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts a slice [begin, end) of a table as a new owned table.
        fn slice_table(tbl: &Table, begin: i32, end: i32, stream: usize) -> Result<UniquePtr<Table>>;

        /// Randomly samples rows from a table.
        fn sample_table(
            tbl: &Table,
            n: i32,
            with_replacement: bool,
            seed: i64,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Additional reductions --

        /// Computes the mean of all elements.
        fn reduce_mean(col: &column_view, output_type_id: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Computes the standard deviation.
        fn reduce_std(col: &column_view, output_type_id: i32, ddof: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Computes the variance.
        fn reduce_var(col: &column_view, output_type_id: i32, ddof: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Computes the median.
        fn reduce_median(col: &column_view, output_type_id: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Counts the number of unique elements.
        fn reduce_nunique(col: &column_view, null_policy: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Prefix scan (cumulative operation).
        fn scan_column(
            col: &column_view,
            agg_kind: i32,
            scan_type: i32,
            null_policy: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns the minimum value from a column (via minmax).
        fn minmax_min(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Returns the maximum value from a column (via minmax).
        fn minmax_max(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        // -- Transpose --

        /// Transposes a table.
        fn transpose_table(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Label bins --

        /// Assigns bin labels to column values.
        fn label_bins_column(
            col: &column_view,
            left_edges: &column_view,
            left_inclusive: i32,
            right_edges: &column_view,
            right_inclusive: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String extras --

        /// Pads strings to a minimum width.
        fn strings_pad(col: &column_view, width: i32, side: i32, fill_char: &str, stream: usize) -> Result<UniquePtr<Column>>;

        /// Zero-fills strings to a minimum width.
        fn strings_zfill(col: &column_view, width: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts a substring [start, stop) with optional step.
        fn strings_slice(col: &column_view, start: i32, stop: i32, step: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Repeats each string N times.
        fn strings_repeat(col: &column_view, repeat_times: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Splits strings by a delimiter into a table of columns.
        fn strings_split_to_table(col: &column_view, delimiter: &Scalar, maxsplit: i32, stream: usize) -> Result<UniquePtr<Table>>;

        /// Right-splits strings by a delimiter into a table of columns.
        fn strings_rsplit_to_table(col: &column_view, delimiter: &Scalar, maxsplit: i32, stream: usize) -> Result<UniquePtr<Table>>;

        /// Returns the Nth part after splitting by delimiter.
        fn strings_split_part(col: &column_view, delimiter: &Scalar, index: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Joins all strings in a column into a single string column of size 1.
        fn strings_join(col: &column_view, separator: &Scalar, narep: &Scalar, stream: usize) -> Result<UniquePtr<Column>>;

        /// Concatenates string columns element-wise with a separator.
        fn strings_concatenate_columns(tbl: &Table, separator: &Scalar, narep: &Scalar, stream: usize) -> Result<UniquePtr<Column>>;

        /// SQL LIKE pattern matching.
        fn strings_like(col: &column_view, pattern: &str, escape_char: &str, stream: usize) -> Result<UniquePtr<Column>>;

        /// Regex contains check.
        fn strings_contains_re(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Column>>;

        /// Regex match from start of string.
        fn strings_matches_re(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Column>>;

        /// Counts regex matches per string.
        fn strings_count_re(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Column>>;

        /// Replaces regex matches with a replacement string.
        fn strings_replace_re(
            col: &column_view,
            pattern: &str,
            replacement: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String extras (batch 6) --

        /// Swap case (upper→lower, lower→upper).
        fn strings_swapcase(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Strip specified characters from sides of strings.
        fn strings_strip_chars(col: &column_view, side: i32, to_strip: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Literal string replace with maxrepl.
        fn strings_replace_literal(col: &column_view, target: &str, repl: &str, maxrepl: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Find first position of target in range [start, stop).
        fn strings_find_str(col: &column_view, target: &str, start: i32, stop: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Find last position of target (reverse find).
        fn strings_rfind(col: &column_view, target: &str, start: i32, stop: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Check if string contains literal target.
        fn strings_contains_str(col: &column_view, target: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Check if string starts with literal target.
        fn strings_starts_with_str(col: &column_view, target: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Check if string ends with literal target.
        fn strings_ends_with_str(col: &column_view, target: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Reverse characters within each string.
        fn strings_reverse(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Regex extract groups into a Table (one column per group).
        fn strings_extract(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Table>>;
        /// Regex extract all matches into a lists column.
        fn strings_extract_all_record(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Find all regex matches as a lists column.
        fn strings_findall(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Find first regex match position.
        fn strings_find_re(col: &column_view, pattern: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Capitalizes the first character of each string.
        fn strings_capitalize(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Title-cases each string (first char of each word uppercase).
        fn strings_title(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Returns BOOL8 column indicating whether each string is title-cased.
        fn strings_is_title(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Wraps strings onto multiple lines shorter than `width`.
        fn strings_wrap(col: &column_view, width: i32, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String operations (new batch) --

        /// Regex split to table of columns.
        fn strings_split_re(col: &column_view, pattern: &str, maxsplit: i32, stream: usize) -> Result<UniquePtr<Table>>;
        /// Regex reverse split to table of columns.
        fn strings_rsplit_re(col: &column_view, pattern: &str, maxsplit: i32, stream: usize) -> Result<UniquePtr<Table>>;
        /// Regex split to lists column.
        fn strings_split_record_re(col: &column_view, pattern: &str, maxsplit: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Regex reverse split to lists column.
        fn strings_rsplit_record_re(col: &column_view, pattern: &str, maxsplit: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Partition around first delimiter into 3 columns.
        fn strings_partition(col: &column_view, delimiter: &str, stream: usize) -> Result<UniquePtr<Table>>;
        /// Partition around last delimiter into 3 columns.
        fn strings_rpartition(col: &column_view, delimiter: &str, stream: usize) -> Result<UniquePtr<Table>>;
        /// Regex replace with back-reference template.
        fn strings_replace_with_backrefs(col: &column_view, pattern: &str, replacement: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Repeat each string by count in another column.
        fn strings_repeat_column(col: &column_view, repeat_times: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Check if strings contain multiple targets (returns Table of BOOL8 columns).
        fn strings_contains_multiple(col: &column_view, targets: &column_view, stream: usize) -> Result<UniquePtr<Table>>;
        /// Find positions of multiple targets in each string (returns lists column).
        fn strings_find_multiple(col: &column_view, targets: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String conversions --

        /// Converts strings to timestamps using format pattern.
        fn strings_to_timestamps(col: &column_view, timestamp_type_id: i32, format: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts timestamp column to strings using format pattern.
        fn strings_from_timestamps(col: &column_view, format: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Verifies strings can be parsed as timestamps with given format.
        fn strings_is_timestamp(col: &column_view, format: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts strings to booleans using true_string match.
        fn strings_to_booleans(col: &column_view, true_string: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts boolean column to strings.
        fn strings_from_booleans(col: &column_view, true_string: &str, false_string: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts strings to durations using format pattern.
        fn strings_to_durations(col: &column_view, duration_type_id: i32, format: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts duration column to strings using format pattern.
        fn strings_from_durations(col: &column_view, format: &str, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts strings to fixed-point decimal.
        fn strings_to_fixed_point(col: &column_view, type_id: i32, scale: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts fixed-point column to strings.
        fn strings_from_fixed_point(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Verifies strings can be parsed as fixed-point.
        fn strings_is_fixed_point(col: &column_view, type_id: i32, scale: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// URL-encodes each string.
        fn strings_url_encode(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// URL-decodes each string.
        fn strings_url_decode(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts IPv4 strings to integers.
        fn strings_ipv4_to_integers(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Converts integers to IPv4 strings.
        fn strings_integers_to_ipv4(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Verifies strings are valid IPv4 format.
        fn strings_is_ipv4(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Datetime operations (new) --

        /// Ceil datetimes to frequency (DAY=0, HOUR=1, MINUTE=2, SECOND=3, MS=4, US=5, NS=6).
        fn datetime_ceil(col: &column_view, freq: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Floor datetimes to frequency.
        fn datetime_floor(col: &column_view, freq: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Round datetimes to frequency.
        fn datetime_round(col: &column_view, freq: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Add months column to timestamp column.
        fn datetime_add_months(timestamps: &column_view, months: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Hashing (new) --

        /// MurmurHash3 128-bit hash (returns table of two UINT64 columns).
        fn hash_murmurhash3_x64_128(tbl: &Table, seed: u64, stream: usize) -> Result<UniquePtr<Table>>;
        /// SHA-1 hash.
        fn hash_sha1(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;
        /// XXHash 32-bit.
        fn hash_xxhash_32(tbl: &Table, seed: u32, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Transform (new) --

        /// Approximate per-row bit count.
        fn row_bit_count(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Reshape (new) --

        /// Convert column elements to lists of bytes.
        fn byte_cast_column(col: &column_view, flip_endian: bool, stream: usize) -> Result<UniquePtr<Column>>;

        // -- List set operations --

        /// Check if two list columns have overlapping elements per row.
        fn lists_have_overlap(lhs: &column_view, rhs: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Intersect distinct elements of two list columns per row.
        fn lists_intersect_distinct(lhs: &column_view, rhs: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Union distinct elements of two list columns per row.
        fn lists_union_distinct(lhs: &column_view, rhs: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Difference distinct elements of two list columns per row.
        fn lists_difference_distinct(lhs: &column_view, rhs: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Lists operations --

        /// Count elements in each list row.
        fn lists_count_elements(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Extract element at given index from each list row.
        fn lists_extract_element(col: &column_view, index: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Sort elements within each list row.
        fn lists_sort(col: &column_view, ascending: bool, nulls_last: bool, stream: usize) -> Result<UniquePtr<Column>>;
        /// Reverse elements within each list row.
        fn lists_reverse(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Check if each list contains nulls.
        fn lists_contains_nulls(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Remove duplicates from each list.
        fn lists_distinct(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Concatenate nested list elements within each row.
        fn lists_concatenate_elements(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Generate sequences as lists from starts and sizes columns.
        fn lists_sequences(starts: &column_view, sizes: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Explode --

        /// Explode a list column in a table.
        fn explode_table(tbl: &Table, column_idx: i32, stream: usize) -> Result<UniquePtr<Table>>;
        /// Explode a list column with position column.
        fn explode_position_table(tbl: &Table, column_idx: i32, stream: usize) -> Result<UniquePtr<Table>>;
        /// Explode a list column, keeping null/empty list rows.
        fn explode_outer_table(tbl: &Table, column_idx: i32, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Rolling window --

        /// Fixed-size rolling window aggregation.
        fn rolling_window(
            col: &column_view,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Grouped fixed-size rolling window aggregation.
        fn grouped_rolling_window(
            group_keys: &Table,
            col: &column_view,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
