// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "rust/cxx.h"

namespace cudf_sys {

// CXX shared enums — defined in the generated header.
enum class TypeId : ::std::int32_t;
enum class Order : ::std::int32_t;
enum class NullOrder : ::std::int32_t;
enum class NullPolicy : ::std::int32_t;

// -- Stream --

std::size_t get_default_stream();

// -- DataType --

/// Opaque wrapper around cudf::data_type for CXX compatibility.
struct DataType {
  cudf::data_type inner;
  explicit DataType(cudf::data_type dt) : inner(dt) {}
};

TypeId data_type_id(DataType const& dt);
int32_t data_type_scale(DataType const& dt);
std::size_t size_of_data_type(DataType const& dt);

// -- Column --

/// RAII wrapper around `cudf::column`.
class Column {
 public:
  explicit Column(std::unique_ptr<cudf::column> col);

  cudf::column const& inner() const { return *column_; }
  cudf::column& mutable_inner() { return *column_; }
  cudf::column_view const& cached_view() const;

  /// Releases ownership of the inner column. The Column is left empty.
  std::unique_ptr<cudf::column> release() { return std::move(column_); }

 private:
  std::unique_ptr<cudf::column> column_;
  mutable std::once_flag view_once_;
  mutable cudf::column_view view_;
};

// Column free functions (CXX-compatible signatures)
int32_t column_size(Column const& col);
int32_t column_null_count(Column const& col);
bool column_has_nulls(Column const& col);
bool column_nullable(Column const& col);
int32_t column_num_children(Column const& col);
int32_t column_type_id(Column const& col);
int32_t column_type_scale(Column const& col);
cudf::column_view const& column_view_of(Column const& col) noexcept;

// column_view free functions
int32_t column_view_size(cudf::column_view const& view);
int32_t column_view_null_count(cudf::column_view const& view);
bool column_view_has_nulls(cudf::column_view const& view);
int32_t column_view_offset(cudf::column_view const& view);
int32_t column_view_type_id(cudf::column_view const& view);

// -- Table --

/// RAII wrapper around `cudf::table`.
class Table {
 public:
  explicit Table(std::unique_ptr<cudf::table> tbl);

  cudf::table const& inner() const { return *table_; }
  cudf::table_view const& cached_view() const;

 private:
  std::unique_ptr<cudf::table> table_;
  mutable std::once_flag view_once_;
  mutable cudf::table_view view_;
};

// Table free functions
std::unique_ptr<Table> table_empty();
int32_t table_num_columns(Table const& tbl);
int32_t table_num_rows(Table const& tbl);
std::size_t table_alloc_size(Table const& tbl);
cudf::column_view const& table_get_column_view(Table const& tbl, int32_t index);
cudf::table_view const& table_view_of(Table const& tbl);

// table_view free functions
int32_t table_view_num_columns(cudf::table_view const& view);
int32_t table_view_num_rows(cudf::table_view const& view);

// -- Scalar --

/// RAII wrapper around `cudf::scalar`.
class Scalar {
 public:
  explicit Scalar(std::unique_ptr<cudf::scalar> s);
  cudf::scalar const& inner() const { return *scalar_; }

 private:
  std::unique_ptr<cudf::scalar> scalar_;
};

// Scalar factory functions
std::unique_ptr<Scalar> make_int32_scalar(int32_t value, bool valid);
std::unique_ptr<Scalar> make_int64_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_float32_scalar(float value, bool valid);
std::unique_ptr<Scalar> make_float64_scalar(double value, bool valid);
std::unique_ptr<Scalar> make_bool_scalar(bool value, bool valid);
std::unique_ptr<Scalar> make_string_scalar(rust::Str value);
std::unique_ptr<Scalar> make_int8_scalar(int8_t value, bool valid);
std::unique_ptr<Scalar> make_int16_scalar(int16_t value, bool valid);
std::unique_ptr<Scalar> make_uint8_scalar(uint8_t value, bool valid);
std::unique_ptr<Scalar> make_uint16_scalar(uint16_t value, bool valid);
std::unique_ptr<Scalar> make_uint32_scalar(uint32_t value, bool valid);
std::unique_ptr<Scalar> make_uint64_scalar(uint64_t value, bool valid);
std::unique_ptr<Scalar> make_timestamp_s_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_timestamp_ms_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_timestamp_us_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_timestamp_ns_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_duration_s_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_duration_ms_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_duration_us_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_duration_ns_scalar(int64_t value, bool valid);
std::unique_ptr<Scalar> make_default_constructed_scalar(int32_t type_id, int32_t scale);
std::unique_ptr<Scalar> make_empty_scalar_like(cudf::column_view const& col);

bool scalar_is_valid(Scalar const& s);
int32_t scalar_type_id(Scalar const& s);

int32_t scalar_to_i32(Scalar const& s);
int64_t scalar_to_i64(Scalar const& s);
float scalar_to_f32(Scalar const& s);
double scalar_to_f64(Scalar const& s);
bool scalar_to_bool(Scalar const& s);

// -- Column factories --

std::unique_ptr<Column> make_column_from_scalar(Scalar const& s, int32_t count, std::size_t stream);
std::unique_ptr<Column> make_empty_column_by_type(int32_t type_id);

// -- Column data extraction (device -> host) --

rust::Vec<int32_t> column_to_host_i32(Column const& col, std::size_t stream);
rust::Vec<int64_t> column_to_host_i64(Column const& col, std::size_t stream);
rust::Vec<float> column_to_host_f32(Column const& col, std::size_t stream);
rust::Vec<double> column_to_host_f64(Column const& col, std::size_t stream);
rust::Vec<int16_t> column_to_host_i16(Column const& col, std::size_t stream);
rust::Vec<bool> column_to_host_bool(Column const& col, std::size_t stream);
rust::Vec<bool> column_null_mask_to_host(Column const& col, std::size_t stream);

// -- TableBuilder --

class TableBuilder {
 public:
  void add_column(std::unique_ptr<Column> wrapper);
  std::unique_ptr<Table> build();

 private:
  std::vector<std::unique_ptr<cudf::column>> cudf_columns_;
};

std::unique_ptr<TableBuilder> new_table_builder();
void table_builder_add_column(TableBuilder& builder, std::unique_ptr<Column> col);
std::unique_ptr<Table> table_builder_build(TableBuilder& builder);

// -- CXX shared enums (generated from Rust bridge) --
enum class BinaryOperator : ::std::int32_t;
enum class NullEquality : ::std::int32_t;

// -- Binary operations --

std::unique_ptr<Column> binary_operation_columns(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream);

std::unique_ptr<Column> binary_operation_column_scalar(
    cudf::column_view const& lhs,
    Scalar const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream);

std::unique_ptr<Column> binary_operation_scalar_column(
    Scalar const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream);

// -- Unary operations --

std::unique_ptr<Column> unary_cast(cudf::column_view const& col, int32_t target_type_id, std::size_t stream);
std::unique_ptr<Column> unary_is_null(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> unary_is_valid(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> unary_is_nan(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> unary_negate(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> unary_abs(cudf::column_view const& col, std::size_t stream);

// -- Reduction --

std::unique_ptr<Scalar> reduce_sum(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Scalar> reduce_min(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Scalar> reduce_max(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Scalar> reduce_product(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Scalar> reduce_any(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Scalar> reduce_all(cudf::column_view const& col, std::size_t stream);

// -- Sorting --

std::unique_ptr<Table> sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Column> sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

bool is_sorted_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

// -- Filtering --

std::unique_ptr<Table> apply_boolean_mask(
    Table const& tbl,
    cudf::column_view const& mask,
    std::size_t stream);

std::unique_ptr<Table> drop_nulls_all(Table const& tbl, std::size_t stream);

// -- Concatenation --

class ColumnConcatenator {
 public:
  void add(cudf::column_view const& v);
  std::unique_ptr<Column> finish(std::size_t stream);

 private:
  std::vector<cudf::column_view> views_;
};

class TableConcatenator {
 public:
  void add_table(Table const& t);
  std::unique_ptr<Table> finish(std::size_t stream);

 private:
  std::vector<cudf::table_view> views_;
};

std::unique_ptr<ColumnConcatenator> new_column_concatenator();
void column_concatenator_add(ColumnConcatenator& cat, cudf::column_view const& v);
std::unique_ptr<Column> column_concatenator_finish(ColumnConcatenator& cat, std::size_t stream);

std::unique_ptr<TableConcatenator> new_table_concatenator();
void table_concatenator_add(TableConcatenator& cat, Table const& t);
std::unique_ptr<Table> table_concatenator_finish(TableConcatenator& cat, std::size_t stream);

// -- Copying --

std::unique_ptr<Table> gather_table(
    Table const& tbl,
    cudf::column_view const& indices,
    std::size_t stream);

std::unique_ptr<Column> empty_like_column(cudf::column_view const& col);
std::unique_ptr<Table> empty_like_table(Table const& tbl);

// -- Column factories from host data --

std::unique_ptr<Column> make_column_from_host_i32(rust::Slice<int32_t const> data, std::size_t stream);
std::unique_ptr<Column> make_column_from_host_i64(rust::Slice<int64_t const> data, std::size_t stream);
std::unique_ptr<Column> make_column_from_host_f64(rust::Slice<double const> data, std::size_t stream);
std::unique_ptr<Column> make_column_from_host_bool(rust::Slice<bool const> data, std::size_t stream);
std::unique_ptr<Column> make_column_from_host_timestamp_s(rust::Slice<int64_t const> data, std::size_t stream);

// -- CXX shared enum (generated from Rust bridge) --
enum class Interpolation : ::std::int32_t;

// -- Replace operations --

std::unique_ptr<Column> replace_nulls_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream);

std::unique_ptr<Column> replace_nulls_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream);

std::unique_ptr<Column> replace_nans_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream);

std::unique_ptr<Column> replace_nans_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream);

std::unique_ptr<Column> clamp_column(
    cudf::column_view const& col,
    Scalar const& lo,
    Scalar const& hi,
    std::size_t stream);

std::unique_ptr<Column> find_and_replace_all(
    cudf::column_view const& col,
    cudf::column_view const& values_to_replace,
    cudf::column_view const& replacement_values,
    std::size_t stream);

// -- Fill operations --

std::unique_ptr<Column> fill_column(
    cudf::column_view const& col,
    int32_t begin,
    int32_t end,
    Scalar const& value,
    std::size_t stream);

std::unique_ptr<Table> repeat_table(
    Table const& tbl,
    int32_t count,
    std::size_t stream);

std::unique_ptr<Column> sequence_column(
    int32_t count,
    Scalar const& init,
    Scalar const& step,
    std::size_t stream);

// -- Search operations --

bool contains_scalar(
    cudf::column_view const& haystack,
    Scalar const& needle,
    std::size_t stream);

std::unique_ptr<Column> contains_column(
    cudf::column_view const& haystack,
    cudf::column_view const& needles,
    std::size_t stream);

std::unique_ptr<Column> lower_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Column> upper_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

// -- Quantile operations --

std::unique_ptr<Column> quantile_column(
    cudf::column_view const& col,
    rust::Slice<double const> quantiles,
    int32_t interp,
    std::size_t stream);

// -- Join operations --

std::unique_ptr<Table> inner_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream);

std::unique_ptr<Table> left_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream);

std::unique_ptr<Table> full_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream);

std::unique_ptr<Table> left_semi_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream);

std::unique_ptr<Table> left_anti_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream);

// -- String operations --

std::unique_ptr<Column> strings_to_lower(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_to_upper(cudf::column_view const& col, std::size_t stream);

std::unique_ptr<Column> strings_contains(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream);

std::unique_ptr<Column> strings_starts_with(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream);

std::unique_ptr<Column> strings_ends_with(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream);

std::unique_ptr<Column> strings_find(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream);

std::unique_ptr<Column> strings_replace(
    cudf::column_view const& col,
    Scalar const& target,
    Scalar const& replacement,
    std::size_t stream);

std::unique_ptr<Column> strings_strip(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_lstrip(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_rstrip(cudf::column_view const& col, std::size_t stream);

std::unique_ptr<Column> strings_count_characters(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_count_bytes(cudf::column_view const& col, std::size_t stream);

std::unique_ptr<Column> strings_from_integers(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Column> strings_from_floats(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_to_floats(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);

// -- String column construction --

std::unique_ptr<Column> make_string_column(rust::Vec<rust::String> strings, std::size_t stream);

// -- String column extraction (device -> host) --

rust::Vec<rust::String> column_to_host_strings(Column const& col, std::size_t stream);

// -- I/O --

std::unique_ptr<Table> read_csv(rust::Str filepath);
std::unique_ptr<Table> read_csv_with_options(
    rust::Str filepath,
    uint8_t delimiter,
    bool header,
    int32_t skip_rows,
    int32_t num_rows);
void write_csv(Table const& tbl, rust::Str filepath);
void write_csv_with_options(
    Table const& tbl,
    rust::Str filepath,
    uint8_t delimiter,
    bool include_header,
    rust::Str na_rep);

std::unique_ptr<Table> read_parquet(rust::Str filepath);
void write_parquet(Table const& tbl, rust::Str filepath);

// -- Datetime operations --

std::unique_ptr<Column> datetime_extract_year(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_month(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_day(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_weekday(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_hour(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_minute(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_second(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_day_of_year(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_is_leap_year(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_days_in_month(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_last_day_of_month(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_quarter(cudf::column_view const& col, std::size_t stream);

// -- Hashing --

std::unique_ptr<Column> hash_murmur3(Table const& tbl, uint32_t seed, std::size_t stream);
std::unique_ptr<Column> hash_xxhash64(Table const& tbl, uint64_t seed, std::size_t stream);
std::unique_ptr<Column> hash_md5(Table const& tbl, std::size_t stream);
std::unique_ptr<Column> hash_sha256(Table const& tbl, std::size_t stream);

// -- Reshape --

std::unique_ptr<Column> interleave_columns(Table const& tbl, std::size_t stream);
std::unique_ptr<Table> tile_table(Table const& tbl, int32_t count, std::size_t stream);

// -- Transform --

std::unique_ptr<Column> nans_to_nulls(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> encode_table(Table const& tbl, std::size_t stream);
std::unique_ptr<Table> encode_keys(Table const& tbl, std::size_t stream);

// -- Merge --

std::unique_ptr<Table> merge_tables(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

// -- Partitioning --

std::unique_ptr<Table> hash_partition_table(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions,
    std::size_t stream);

rust::Vec<int32_t> hash_partition_offsets(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions,
    std::size_t stream);

std::unique_ptr<Table> round_robin_partition_table(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition,
    std::size_t stream);

rust::Vec<int32_t> round_robin_partition_offsets(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition,
    std::size_t stream);

// -- CXX shared enums (generated from Rust bridge) --
enum class AggregationKind : ::std::int32_t;
enum class UnaryOperator : ::std::int32_t;
enum class RoundingMethod : ::std::int32_t;
enum class DuplicateKeepOption : ::std::int32_t;
enum class NanEquality : ::std::int32_t;
enum class ScanType : ::std::int32_t;
enum class Inclusive : ::std::int32_t;
enum class SideType : ::std::int32_t;

// -- GroupBy operations --

std::unique_ptr<Table> groupby_single(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    int32_t value_index,
    int32_t agg_kind,
    std::size_t stream);

std::unique_ptr<Table> groupby_multi(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds,
    std::size_t stream);

// -- Generic unary operation --

std::unique_ptr<Column> unary_operation(cudf::column_view const& col, int32_t op, std::size_t stream);
std::unique_ptr<Column> unary_is_not_nan(cudf::column_view const& col, std::size_t stream);

// -- Round --

std::unique_ptr<Column> round_column(cudf::column_view const& col, int32_t decimal_places, int32_t method, std::size_t stream);

// -- Stream compaction --

std::unique_ptr<Table> drop_nans(Table const& tbl, rust::Slice<int32_t const> keys, std::size_t stream);
std::unique_ptr<Table> drop_nulls_with_threshold(Table const& tbl, rust::Slice<int32_t const> keys, int32_t threshold, std::size_t stream);
std::unique_ptr<Table> unique_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, std::size_t stream);
std::unique_ptr<Table> distinct_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream);
std::unique_ptr<Column> distinct_indices_column(Table const& tbl, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream);
std::unique_ptr<Table> stable_distinct_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream);

// -- Copying extras --

std::unique_ptr<Table> scatter_table(Table const& source, cudf::column_view const& scatter_map, Table const& target, std::size_t stream);
std::unique_ptr<Column> reverse_column(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Table> reverse_table(Table const& tbl, std::size_t stream);
std::unique_ptr<Column> shift_column(cudf::column_view const& col, int32_t offset, Scalar const& fill_value, std::size_t stream);
std::unique_ptr<Scalar> get_element(cudf::column_view const& col, int32_t index, std::size_t stream);
std::unique_ptr<Column> copy_if_else_columns(cudf::column_view const& lhs, cudf::column_view const& rhs, cudf::column_view const& mask, std::size_t stream);
std::unique_ptr<Column> copy_if_else_scalar_column(Scalar const& lhs, cudf::column_view const& rhs, cudf::column_view const& mask, std::size_t stream);
std::unique_ptr<Column> copy_if_else_column_scalar(cudf::column_view const& lhs, Scalar const& rhs, cudf::column_view const& mask, std::size_t stream);
std::unique_ptr<Column> slice_column(cudf::column_view const& col, int32_t begin, int32_t end, std::size_t stream);
std::unique_ptr<Table> slice_table(Table const& tbl, int32_t begin, int32_t end, std::size_t stream);
std::unique_ptr<Table> sample_table(Table const& tbl, int32_t n, bool with_replacement, int64_t seed, std::size_t stream);

// -- Additional reductions --

std::unique_ptr<Scalar> reduce_mean(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Scalar> reduce_std(cudf::column_view const& col, int32_t output_type_id, int32_t ddof, std::size_t stream);
std::unique_ptr<Scalar> reduce_var(cudf::column_view const& col, int32_t output_type_id, int32_t ddof, std::size_t stream);
std::unique_ptr<Scalar> reduce_median(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Scalar> reduce_nunique(cudf::column_view const& col, int32_t null_policy, std::size_t stream);
std::unique_ptr<Column> scan_column(cudf::column_view const& col, int32_t agg_kind, int32_t scan_type, int32_t null_policy, std::size_t stream);
std::unique_ptr<Scalar> minmax_min(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Scalar> minmax_max(cudf::column_view const& col, std::size_t stream);

// -- Transpose --

std::unique_ptr<Table> transpose_table(Table const& tbl, std::size_t stream);

// -- Label bins --

std::unique_ptr<Column> label_bins_column(
    cudf::column_view const& col,
    cudf::column_view const& left_edges,
    int32_t left_inclusive,
    cudf::column_view const& right_edges,
    int32_t right_inclusive,
    std::size_t stream);

// -- String extras --

std::unique_ptr<Column> strings_pad(cudf::column_view const& col, int32_t width, int32_t side, rust::Str fill_char, std::size_t stream);
std::unique_ptr<Column> strings_zfill(cudf::column_view const& col, int32_t width, std::size_t stream);
std::unique_ptr<Column> strings_slice(cudf::column_view const& col, int32_t start, int32_t stop, int32_t step, std::size_t stream);
std::unique_ptr<Column> strings_repeat(cudf::column_view const& col, int32_t repeat_times, std::size_t stream);
std::unique_ptr<Table> strings_split_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Table> strings_rsplit_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Column> strings_split_part(cudf::column_view const& col, Scalar const& delimiter, int32_t index, std::size_t stream);
std::unique_ptr<Column> strings_join(cudf::column_view const& col, Scalar const& separator, Scalar const& narep, std::size_t stream);
std::unique_ptr<Column> strings_concatenate_columns(Table const& tbl, Scalar const& separator, Scalar const& narep, std::size_t stream);
std::unique_ptr<Column> strings_like(cudf::column_view const& col, rust::Str pattern, rust::Str escape_char, std::size_t stream);
std::unique_ptr<Column> strings_contains_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_matches_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_count_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_replace_re(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream);

// -- String extras (batch 6) --

std::unique_ptr<Column> strings_swapcase(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_strip_chars(cudf::column_view const& col, int32_t side, rust::Str to_strip, std::size_t stream);
std::unique_ptr<Column> strings_replace_literal(cudf::column_view const& col, rust::Str target, rust::Str repl, int32_t maxrepl, std::size_t stream);
std::unique_ptr<Column> strings_find_str(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream);
std::unique_ptr<Column> strings_rfind(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream);
std::unique_ptr<Column> strings_contains_str(cudf::column_view const& col, rust::Str target, std::size_t stream);
std::unique_ptr<Column> strings_starts_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream);
std::unique_ptr<Column> strings_ends_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream);
std::unique_ptr<Column> strings_reverse(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Table> strings_extract(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_extract_all_record(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_findall(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_find_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream);
std::unique_ptr<Column> strings_capitalize(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_title(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_is_title(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_wrap(cudf::column_view const& col, int32_t width, std::size_t stream);

// -- Lists operations --

std::unique_ptr<Column> lists_count_elements(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> lists_extract_element(cudf::column_view const& col, int32_t index, std::size_t stream);
std::unique_ptr<Column> lists_sort(cudf::column_view const& col, bool ascending, bool nulls_last, std::size_t stream);
std::unique_ptr<Column> lists_reverse(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> lists_contains_nulls(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> lists_distinct(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> lists_concatenate_elements(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> lists_sequences(cudf::column_view const& starts, cudf::column_view const& sizes, std::size_t stream);

// -- Explode --

std::unique_ptr<Table> explode_table(Table const& tbl, int32_t column_idx, std::size_t stream);
std::unique_ptr<Table> explode_position_table(Table const& tbl, int32_t column_idx, std::size_t stream);
std::unique_ptr<Table> explode_outer_table(Table const& tbl, int32_t column_idx, std::size_t stream);

// -- Rolling window --

std::unique_ptr<Column> rolling_window(cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream);
std::unique_ptr<Column> grouped_rolling_window(Table const& group_keys, cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream);

// -- String character types --

std::unique_ptr<Column> strings_all_characters_of_type(cudf::column_view const& col, uint32_t types, uint32_t verify_types, std::size_t stream);

// -- String operations (new batch) --

std::unique_ptr<Table> strings_split_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Table> strings_rsplit_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Column> strings_split_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Column> strings_rsplit_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Table> strings_partition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream);
std::unique_ptr<Table> strings_rpartition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream);
std::unique_ptr<Column> strings_replace_with_backrefs(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream);
std::unique_ptr<Column> strings_repeat_column(cudf::column_view const& col, cudf::column_view const& repeat_times, std::size_t stream);
std::unique_ptr<Table> strings_contains_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream);
std::unique_ptr<Column> strings_find_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream);

// -- String conversions --

std::unique_ptr<Column> strings_to_timestamps(cudf::column_view const& col, int32_t timestamp_type_id, rust::Str format, std::size_t stream);
std::unique_ptr<Column> strings_from_timestamps(cudf::column_view const& col, rust::Str format, std::size_t stream);
std::unique_ptr<Column> strings_is_timestamp(cudf::column_view const& col, rust::Str format, std::size_t stream);
std::unique_ptr<Column> strings_to_booleans(cudf::column_view const& col, rust::Str true_string, std::size_t stream);
std::unique_ptr<Column> strings_from_booleans(cudf::column_view const& col, rust::Str true_string, rust::Str false_string, std::size_t stream);
std::unique_ptr<Column> strings_to_durations(cudf::column_view const& col, int32_t duration_type_id, rust::Str format, std::size_t stream);
std::unique_ptr<Column> strings_from_durations(cudf::column_view const& col, rust::Str format, std::size_t stream);
std::unique_ptr<Column> strings_to_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream);
std::unique_ptr<Column> strings_from_fixed_point(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_is_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream);
std::unique_ptr<Column> strings_url_encode(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_url_decode(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_ipv4_to_integers(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_integers_to_ipv4(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_is_ipv4(cudf::column_view const& col, std::size_t stream);

// -- Datetime operations (new) --

std::unique_ptr<Column> datetime_ceil(cudf::column_view const& col, int32_t freq, std::size_t stream);
std::unique_ptr<Column> datetime_floor(cudf::column_view const& col, int32_t freq, std::size_t stream);
std::unique_ptr<Column> datetime_round(cudf::column_view const& col, int32_t freq, std::size_t stream);
std::unique_ptr<Column> datetime_add_months(cudf::column_view const& timestamps, cudf::column_view const& months, std::size_t stream);

// -- Hashing (new) --

std::unique_ptr<Table> hash_murmurhash3_x64_128(Table const& tbl, uint64_t seed, std::size_t stream);
std::unique_ptr<Column> hash_sha1(Table const& tbl, std::size_t stream);
std::unique_ptr<Column> hash_xxhash_32(Table const& tbl, uint32_t seed, std::size_t stream);

// -- Transform (new) --

std::unique_ptr<Column> row_bit_count(Table const& tbl, std::size_t stream);

// -- Reshape (new) --

std::unique_ptr<Column> byte_cast_column(cudf::column_view const& col, bool flip_endian, std::size_t stream);

// -- List set operations --

std::unique_ptr<Column> lists_have_overlap(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream);
std::unique_ptr<Column> lists_intersect_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream);
std::unique_ptr<Column> lists_union_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream);
std::unique_ptr<Column> lists_difference_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream);

// -- Sorting (new) --

std::unique_ptr<Column> stable_sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Table> stable_sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Table> sort_by_key(
    Table const& values,
    Table const& keys,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Table> stable_sort_by_key(
    Table const& values,
    Table const& keys,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Column> rank_column(
    cudf::column_view const& col,
    int32_t method,
    int32_t column_order,
    int32_t null_handling,
    int32_t null_precedence,
    bool percentage,
    std::size_t stream);

std::unique_ptr<Column> top_k(cudf::column_view const& col, int32_t k, int32_t order, std::size_t stream);
std::unique_ptr<Column> top_k_order(cudf::column_view const& col, int32_t k, int32_t order, std::size_t stream);

std::unique_ptr<Column> segmented_sorted_order(
    Table const& tbl,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

std::unique_ptr<Column> segmented_top_k(cudf::column_view const& col, cudf::column_view const& segment_offsets, int32_t k, int32_t order, std::size_t stream);
std::unique_ptr<Column> segmented_top_k_order(cudf::column_view const& col, cudf::column_view const& segment_offsets, int32_t k, int32_t order, std::size_t stream);

// -- Copying (new) --

std::unique_ptr<Column> copy_range(
    cudf::column_view const& source,
    cudf::column_view const& target,
    int32_t source_begin,
    int32_t source_end,
    int32_t target_begin,
    std::size_t stream);

std::unique_ptr<Column> allocate_like_column(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Table> boolean_mask_scatter_table(Table const& source, Table const& target, cudf::column_view const& mask, std::size_t stream);
bool has_nonempty_nulls(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> purge_nonempty_nulls(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> copy_if_else_scalars(Scalar const& lhs, Scalar const& rhs, cudf::column_view const& mask, std::size_t stream);

// -- Replace (new) --

std::unique_ptr<Column> replace_nulls_policy(cudf::column_view const& col, int32_t policy, std::size_t stream);
std::unique_ptr<Column> clamp_column_with_replace(cudf::column_view const& col, Scalar const& lo, Scalar const& lo_replace, Scalar const& hi, Scalar const& hi_replace, std::size_t stream);
std::unique_ptr<Column> normalize_nans_and_zeros(cudf::column_view const& col, std::size_t stream);

// -- Fill (new) --

std::unique_ptr<Table> repeat_table_column(Table const& tbl, cudf::column_view const& counts, std::size_t stream);
std::unique_ptr<Column> calendrical_month_sequence(int32_t count, Scalar const& init, int32_t months, std::size_t stream);

// -- Quantiles (new) --

std::unique_ptr<Table> quantiles_table(
    Table const& tbl,
    rust::Slice<double const> quantiles,
    int32_t interp,
    bool is_input_sorted,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream);

// -- Null mask utilities --

std::size_t bitmask_allocation_size_bytes(int32_t number_of_bits);
int32_t num_bitmask_words(int32_t number_of_bits);

// -- Dictionary operations --

std::unique_ptr<Column> dictionary_encode(cudf::column_view const& col, int32_t indices_type_id, std::size_t stream);
std::unique_ptr<Column> dictionary_decode(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> dictionary_add_keys(cudf::column_view const& col, cudf::column_view const& new_keys, std::size_t stream);
std::unique_ptr<Column> dictionary_remove_keys(cudf::column_view const& col, cudf::column_view const& keys_to_remove, std::size_t stream);
std::unique_ptr<Column> dictionary_set_keys(cudf::column_view const& col, cudf::column_view const& keys, std::size_t stream);
std::unique_ptr<Column> dictionary_remove_unused_keys(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Scalar> dictionary_get_index(cudf::column_view const& col, Scalar const& key, std::size_t stream);

// -- Lists contains / index_of / segmented_gather --

std::unique_ptr<Column> lists_contains_scalar(cudf::column_view const& col, Scalar const& search_key, std::size_t stream);
std::unique_ptr<Column> lists_contains_column(cudf::column_view const& col, cudf::column_view const& search_keys, std::size_t stream);
std::unique_ptr<Column> lists_index_of_scalar(cudf::column_view const& col, Scalar const& search_key, bool find_first, std::size_t stream);
std::unique_ptr<Column> lists_index_of_column(cudf::column_view const& col, cudf::column_view const& search_keys, bool find_first, std::size_t stream);
std::unique_ptr<Column> lists_segmented_gather(cudf::column_view const& col, cudf::column_view const& gather_map, bool nullify_oob, std::size_t stream);

// -- String validation / conversion extras --

std::unique_ptr<Column> strings_filter_characters_of_type(cudf::column_view const& col, uint32_t types_to_remove, rust::Str replacement, uint32_t types_to_keep, std::size_t stream);
std::unique_ptr<Column> strings_is_integer(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_is_integer_with_type(cudf::column_view const& col, int32_t int_type_id, std::size_t stream);
std::unique_ptr<Column> strings_is_float(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_hex_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream);
std::unique_ptr<Column> strings_is_hex(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_integers_to_hex(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> strings_format_list_column(cudf::column_view const& col, rust::Str na_rep, std::size_t stream);

// -- Strings: replace_slice / replace_multiple --

std::unique_ptr<Column> strings_replace_slice(cudf::column_view const& col, rust::Str repl, int32_t start, int32_t stop, std::size_t stream);
std::unique_ptr<Column> strings_replace_multiple(cudf::column_view const& col, cudf::column_view const& targets, cudf::column_view const& repls, std::size_t stream);

// -- Strings: split_record / rsplit_record --

std::unique_ptr<Column> strings_split_record(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream);
std::unique_ptr<Column> strings_rsplit_record(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream);

// -- Lists: apply_boolean_mask --

std::unique_ptr<Column> lists_apply_boolean_mask(cudf::column_view const& col, cudf::column_view const& boolean_mask, std::size_t stream);

// -- Strings: join_list_elements --

std::unique_ptr<Column> strings_join_list_elements(cudf::column_view const& col, rust::Str separator, rust::Str narep, std::size_t stream);

// -- Transform: segmented_row_bit_count --

std::unique_ptr<Column> segmented_row_bit_count(Table const& tbl, int32_t segment_length, std::size_t stream);

// -- Strings: code_points --

std::unique_ptr<Column> strings_code_points(cudf::column_view const& col, std::size_t stream);

// -- Strings: translate / filter_characters --

std::unique_ptr<Column> strings_translate(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    std::size_t stream);

std::unique_ptr<Column> strings_filter_characters(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    bool keep,
    rust::Str replacement,
    std::size_t stream);

// -- Strings: cast_to_integer / cast_from_integer --

std::unique_ptr<Column> strings_cast_to_integer(cudf::column_view const& col, int32_t output_type_id, bool big_endian, std::size_t stream);
std::unique_ptr<Column> strings_cast_from_integer(cudf::column_view const& col, bool big_endian, std::size_t stream);

// -- Lists: extract (column), stable sort, concatenate_rows --

std::unique_ptr<Column> lists_extract_element_column(cudf::column_view const& col, cudf::column_view const& indices, std::size_t stream);
std::unique_ptr<Column> lists_stable_sort(cudf::column_view const& col, bool ascending, bool nulls_last, std::size_t stream);
std::unique_ptr<Column> lists_concatenate_rows(Table const& tbl, std::size_t stream);

// -- Binary: fixed_point_scale / is_supported_operation --

int32_t binary_operation_fixed_point_scale(int32_t op, int32_t left_scale, int32_t right_scale);
bool is_supported_binaryop(int32_t out_type_id, int32_t lhs_type_id, int32_t rhs_type_id, int32_t op);

// -- Sorting: stable segmented --

std::unique_ptr<Column> stable_segmented_sorted_order(Table const& tbl, cudf::column_view const& segment_offsets, rust::Slice<int32_t const> column_orders, rust::Slice<int32_t const> null_orders, std::size_t stream);
std::unique_ptr<Table> stable_segmented_sort_by_key(Table const& values, Table const& keys, cudf::column_view const& segment_offsets, rust::Slice<int32_t const> column_orders, rust::Slice<int32_t const> null_orders, std::size_t stream);

// -- Explode: outer_position --

std::unique_ptr<Table> explode_outer_position_table(Table const& tbl, int32_t column_idx, std::size_t stream);

// -- Strings: slice by column, extract_single --

std::unique_ptr<Column> strings_slice_column(cudf::column_view const& col, cudf::column_view const& starts, cudf::column_view const& stops, std::size_t stream);
std::unique_ptr<Column> strings_extract_single(cudf::column_view const& col, rust::Str pattern, int32_t group_index, std::size_t stream);

// -- Copying: gather_checked, may_have_nonempty_nulls --

std::unique_ptr<Table> gather_table_checked(Table const& tbl, cudf::column_view const& gather_map, bool nullify_oob, std::size_t stream);
bool may_have_nonempty_nulls(cudf::column_view const& col);

// -- Sorting: segmented_sort_by_key (non-stable) --

std::unique_ptr<Table> segmented_sort_by_key(Table const& values, Table const& keys, cudf::column_view const& segment_offsets, rust::Slice<int32_t const> column_orders, rust::Slice<int32_t const> null_orders, std::size_t stream);

// -- Partitioning: partition by map column --

std::unique_ptr<Table> partition_by_map(Table const& tbl, cudf::column_view const& partition_map, int32_t num_partitions, std::size_t stream);
rust::Vec<int32_t> partition_by_map_offsets(Table const& tbl, cudf::column_view const& partition_map, int32_t num_partitions, std::size_t stream);

// -- Datetime: fractional seconds --

std::unique_ptr<Column> datetime_extract_millisecond(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_microsecond(cudf::column_view const& col, std::size_t stream);
std::unique_ptr<Column> datetime_extract_nanosecond(cudf::column_view const& col, std::size_t stream);

// -- Join: cross_join --

std::unique_ptr<Table> cross_join(Table const& left, Table const& right, std::size_t stream);

// -- Lists: sequences with step --

std::unique_ptr<Column> lists_sequences_with_step(cudf::column_view const& starts, cudf::column_view const& steps, cudf::column_view const& sizes, std::size_t stream);

// -- Strings: concatenate with per-row separator / join_list_elements with per-row separator --

std::unique_ptr<Column> strings_concatenate_columns_sep_col(Table const& tbl, cudf::column_view const& separators, rust::Str separator_narep, rust::Str col_narep, std::size_t stream);
std::unique_ptr<Column> strings_join_list_elements_column(cudf::column_view const& col, cudf::column_view const& separators, rust::Str separator_narep, rust::Str string_narep, std::size_t stream);

// -- Strings: zfill_by_widths --

std::unique_ptr<Column> strings_zfill_by_widths(cudf::column_view const& col, cudf::column_view const& widths, std::size_t stream);

// -- Hashing: sha224 / sha384 / sha512 --

std::unique_ptr<Column> hash_sha224(Table const& tbl, std::size_t stream);
std::unique_ptr<Column> hash_sha384(Table const& tbl, std::size_t stream);
std::unique_ptr<Column> hash_sha512(Table const& tbl, std::size_t stream);

// -- Column factories --

std::unique_ptr<Column> make_fixed_width_column(int32_t type_id, int32_t scale, int32_t num_rows, int32_t mask_state, std::size_t stream);
std::unique_ptr<Column> make_empty_lists_column(int32_t child_type_id, std::size_t stream);
std::unique_ptr<Column> make_dictionary_from_scalar(Scalar const& scalar, int32_t num_rows, std::size_t stream);

// -- JSON path extraction --

std::unique_ptr<Column> get_json_object(cudf::column_view const& col, rust::Str json_path, bool allow_single_quotes, bool strip_quotes, bool missing_fields_as_nulls, std::size_t stream);

// -- Null mask utility --

int32_t state_null_count(int32_t mask_state, int32_t num_rows);

// -- Type checking utilities --

bool column_types_equivalent(cudf::column_view const& lhs, cudf::column_view const& rhs);
bool columns_have_same_types(cudf::column_view const& lhs, cudf::column_view const& rhs);
bool tables_have_same_types(Table const& lhs, Table const& rhs);
bool is_supported_cast(int32_t from_type_id, int32_t from_scale, int32_t to_type_id, int32_t to_scale);

// -- In-place operations --

void fill_in_place(Column& col, int32_t begin, int32_t end, Scalar const& value, std::size_t stream);
void copy_range_in_place(Column& dest, cudf::column_view const& source, int32_t source_begin, int32_t source_end, int32_t dest_begin, std::size_t stream);

// -- One-hot encoding --

std::unique_ptr<Table> one_hot_encode(cudf::column_view const& input, cudf::column_view const& categories, std::size_t stream);

// -- Null mask conversions --

std::unique_ptr<Column> null_mask_to_bools(cudf::column_view const& col, std::size_t stream);
void set_null_mask_from_bools(Column& col, cudf::column_view const& bools, std::size_t stream);

// -- Bitmask combining --

std::unique_ptr<Column> bitmask_and_to_bools(Table const& tbl, std::size_t stream);
std::unique_ptr<Column> bitmask_or_to_bools(Table const& tbl, std::size_t stream);

// -- Column factories: lists and structs --

std::unique_ptr<Column> make_lists_column(int32_t num_rows, std::unique_ptr<Column> offsets, std::unique_ptr<Column> child, std::size_t stream);

class StructColumnBuilder {
 public:
  void add_child(std::unique_ptr<Column> col);
  std::unique_ptr<Column> build(int32_t num_rows, std::size_t stream);

 private:
  std::vector<std::unique_ptr<cudf::column>> children_;
};

std::unique_ptr<StructColumnBuilder> new_struct_column_builder();
void struct_column_builder_add(StructColumnBuilder& builder, std::unique_ptr<Column> col);
std::unique_ptr<Column> struct_column_builder_build(StructColumnBuilder& builder, int32_t num_rows, std::size_t stream);

// -- GroupBy scan/replace_nulls --

std::unique_ptr<Table> groupby_scan(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds,
    std::size_t stream);

std::unique_ptr<Table> groupby_replace_nulls(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> policies,
    std::size_t stream);

// -- ORC I/O --

std::unique_ptr<Table> read_orc(rust::Str filepath);
void write_orc(Table const& tbl, rust::Str filepath);

// -- JSON I/O --

std::unique_ptr<Table> read_json(rust::Str filepath, bool json_lines);
void write_json(Table const& tbl, rust::Str filepath, bool json_lines);

// -- Distinct count --

int32_t distinct_count_column(cudf::column_view const& col, int32_t null_policy, bool nan_is_null, std::size_t stream);
int32_t distinct_count_table(Table const& tbl, int32_t null_equality, std::size_t stream);

// -- Avro I/O --

std::unique_ptr<Table> read_avro(rust::Str filepath);

// -- Scatter with scalars --

class ScalarList {
 public:
  void add(std::unique_ptr<Scalar> s);
  std::vector<std::reference_wrapper<cudf::scalar const>> refs() const;

 private:
  std::vector<std::unique_ptr<Scalar>> owned_;
};

std::unique_ptr<ScalarList> new_scalar_list();
void scalar_list_add(ScalarList& list, std::unique_ptr<Scalar> s);
std::unique_ptr<Table> scatter_scalars(ScalarList& sources, cudf::column_view const& indices, Table const& target, std::size_t stream);
std::unique_ptr<Table> boolean_mask_scatter_scalars(ScalarList& sources, Table const& target, cudf::column_view const& mask, std::size_t stream);

// -- GroupBy shift --

std::unique_ptr<Table> groupby_shift(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> offsets,
    ScalarList& fill_values,
    std::size_t stream);

// -- Unique count --

int32_t unique_count_column(cudf::column_view const& col, int32_t null_policy, bool nan_is_null, std::size_t stream);
int32_t unique_count_table(Table const& tbl, int32_t null_equality, std::size_t stream);

// -- Additional hashing --

std::unique_ptr<Column> hash_murmurhash3_x86_32(Table const& tbl, uint32_t seed, std::size_t stream);

// -- Drop NaNs with threshold --

std::unique_ptr<Table> drop_nans_with_threshold(Table const& tbl, rust::Slice<int32_t const> keys, int32_t threshold, std::size_t stream);

}  // namespace cudf_sys
