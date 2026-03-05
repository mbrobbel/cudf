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
cudf::column_view const& column_view_of(Column const& col);

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

bool scalar_is_valid(Scalar const& s);
int32_t scalar_type_id(Scalar const& s);

int32_t scalar_to_i32(Scalar const& s);
int64_t scalar_to_i64(Scalar const& s);
float scalar_to_f32(Scalar const& s);
double scalar_to_f64(Scalar const& s);
bool scalar_to_bool(Scalar const& s);

// -- Column factories --

std::unique_ptr<Column> make_column_from_scalar(Scalar const& s, int32_t count);
std::unique_ptr<Column> make_empty_column_by_type(int32_t type_id);

// -- Column data extraction (device → host) --

rust::Vec<int32_t> column_to_host_i32(Column const& col);
rust::Vec<int64_t> column_to_host_i64(Column const& col);
rust::Vec<float> column_to_host_f32(Column const& col);
rust::Vec<double> column_to_host_f64(Column const& col);
rust::Vec<int16_t> column_to_host_i16(Column const& col);
rust::Vec<bool> column_to_host_bool(Column const& col);
rust::Vec<bool> column_null_mask_to_host(Column const& col);

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
    int32_t output_type_id);

std::unique_ptr<Column> binary_operation_column_scalar(
    cudf::column_view const& lhs,
    Scalar const& rhs,
    BinaryOperator op,
    int32_t output_type_id);

std::unique_ptr<Column> binary_operation_scalar_column(
    Scalar const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id);

// -- Unary operations --

std::unique_ptr<Column> unary_cast(cudf::column_view const& col, int32_t target_type_id);
std::unique_ptr<Column> unary_is_null(cudf::column_view const& col);
std::unique_ptr<Column> unary_is_valid(cudf::column_view const& col);
std::unique_ptr<Column> unary_is_nan(cudf::column_view const& col);
std::unique_ptr<Column> unary_negate(cudf::column_view const& col);
std::unique_ptr<Column> unary_abs(cudf::column_view const& col);

// -- Reduction --

std::unique_ptr<Scalar> reduce_sum(cudf::column_view const& col, int32_t output_type_id);
std::unique_ptr<Scalar> reduce_min(cudf::column_view const& col, int32_t output_type_id);
std::unique_ptr<Scalar> reduce_max(cudf::column_view const& col, int32_t output_type_id);
std::unique_ptr<Scalar> reduce_product(cudf::column_view const& col, int32_t output_type_id);
std::unique_ptr<Scalar> reduce_any(cudf::column_view const& col);
std::unique_ptr<Scalar> reduce_all(cudf::column_view const& col);

// -- Sorting --

std::unique_ptr<Table> sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders);

std::unique_ptr<Column> sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders);

bool is_sorted_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders);

// -- Filtering --

std::unique_ptr<Table> apply_boolean_mask(
    Table const& tbl,
    cudf::column_view const& mask);

std::unique_ptr<Table> drop_nulls_all(Table const& tbl);

// -- Concatenation --

class ColumnConcatenator {
 public:
  void add(cudf::column_view const& v);
  std::unique_ptr<Column> finish();

 private:
  std::vector<cudf::column_view> views_;
};

class TableConcatenator {
 public:
  void add_table(Table const& t);
  std::unique_ptr<Table> finish();

 private:
  std::vector<cudf::table_view> views_;
};

std::unique_ptr<ColumnConcatenator> new_column_concatenator();
void column_concatenator_add(ColumnConcatenator& cat, cudf::column_view const& v);
std::unique_ptr<Column> column_concatenator_finish(ColumnConcatenator& cat);

std::unique_ptr<TableConcatenator> new_table_concatenator();
void table_concatenator_add(TableConcatenator& cat, Table const& t);
std::unique_ptr<Table> table_concatenator_finish(TableConcatenator& cat);

// -- Copying --

std::unique_ptr<Table> gather_table(
    Table const& tbl,
    cudf::column_view const& indices);

std::unique_ptr<Column> empty_like_column(cudf::column_view const& col);
std::unique_ptr<Table> empty_like_table(Table const& tbl);

// -- Column factories from host data --

std::unique_ptr<Column> make_column_from_host_i32(rust::Slice<int32_t const> data);
std::unique_ptr<Column> make_column_from_host_i64(rust::Slice<int64_t const> data);
std::unique_ptr<Column> make_column_from_host_f64(rust::Slice<double const> data);
std::unique_ptr<Column> make_column_from_host_bool(rust::Slice<bool const> data);
std::unique_ptr<Column> make_column_from_host_timestamp_s(rust::Slice<int64_t const> data);

// -- CXX shared enum (generated from Rust bridge) --
enum class Interpolation : ::std::int32_t;

// -- Replace operations --

std::unique_ptr<Column> replace_nulls_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement);

std::unique_ptr<Column> replace_nulls_scalar(
    cudf::column_view const& col,
    Scalar const& replacement);

std::unique_ptr<Column> replace_nans_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement);

std::unique_ptr<Column> replace_nans_scalar(
    cudf::column_view const& col,
    Scalar const& replacement);

std::unique_ptr<Column> clamp_column(
    cudf::column_view const& col,
    Scalar const& lo,
    Scalar const& hi);

std::unique_ptr<Column> find_and_replace_all(
    cudf::column_view const& col,
    cudf::column_view const& values_to_replace,
    cudf::column_view const& replacement_values);

// -- Fill operations --

std::unique_ptr<Column> fill_column(
    cudf::column_view const& col,
    int32_t begin,
    int32_t end,
    Scalar const& value);

std::unique_ptr<Table> repeat_table(
    Table const& tbl,
    int32_t count);

std::unique_ptr<Column> sequence_column(
    int32_t count,
    Scalar const& init,
    Scalar const& step);

// -- Search operations --

bool contains_scalar(
    cudf::column_view const& haystack,
    Scalar const& needle);

std::unique_ptr<Column> contains_column(
    cudf::column_view const& haystack,
    cudf::column_view const& needles);

std::unique_ptr<Column> lower_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders);

std::unique_ptr<Column> upper_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders);

// -- Quantile operations --

std::unique_ptr<Column> quantile_column(
    cudf::column_view const& col,
    rust::Slice<double const> quantiles,
    int32_t interp);

// -- Join operations --

std::unique_ptr<Table> inner_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on);

std::unique_ptr<Table> left_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on);

std::unique_ptr<Table> full_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on);

std::unique_ptr<Table> left_semi_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on);

std::unique_ptr<Table> left_anti_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on);

// -- String operations --

std::unique_ptr<Column> strings_to_lower(cudf::column_view const& col);
std::unique_ptr<Column> strings_to_upper(cudf::column_view const& col);

std::unique_ptr<Column> strings_contains(
    cudf::column_view const& col,
    Scalar const& target);

std::unique_ptr<Column> strings_starts_with(
    cudf::column_view const& col,
    Scalar const& target);

std::unique_ptr<Column> strings_ends_with(
    cudf::column_view const& col,
    Scalar const& target);

std::unique_ptr<Column> strings_find(
    cudf::column_view const& col,
    Scalar const& target);

std::unique_ptr<Column> strings_replace(
    cudf::column_view const& col,
    Scalar const& target,
    Scalar const& replacement);

std::unique_ptr<Column> strings_strip(cudf::column_view const& col);
std::unique_ptr<Column> strings_lstrip(cudf::column_view const& col);
std::unique_ptr<Column> strings_rstrip(cudf::column_view const& col);

std::unique_ptr<Column> strings_count_characters(cudf::column_view const& col);
std::unique_ptr<Column> strings_count_bytes(cudf::column_view const& col);

std::unique_ptr<Column> strings_from_integers(cudf::column_view const& col);
std::unique_ptr<Column> strings_to_integers(cudf::column_view const& col, int32_t output_type_id);
std::unique_ptr<Column> strings_from_floats(cudf::column_view const& col);
std::unique_ptr<Column> strings_to_floats(cudf::column_view const& col, int32_t output_type_id);

// -- String column construction --

std::unique_ptr<Column> make_string_column(rust::Vec<rust::String> strings);

// -- String column extraction (device -> host) --

rust::Vec<rust::String> column_to_host_strings(Column const& col);

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

std::unique_ptr<Column> datetime_extract_year(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_month(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_day(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_weekday(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_hour(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_minute(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_second(cudf::column_view const& col);
std::unique_ptr<Column> datetime_day_of_year(cudf::column_view const& col);
std::unique_ptr<Column> datetime_is_leap_year(cudf::column_view const& col);
std::unique_ptr<Column> datetime_days_in_month(cudf::column_view const& col);
std::unique_ptr<Column> datetime_last_day_of_month(cudf::column_view const& col);
std::unique_ptr<Column> datetime_extract_quarter(cudf::column_view const& col);

// -- Hashing --

std::unique_ptr<Column> hash_murmur3(Table const& tbl, uint32_t seed);
std::unique_ptr<Column> hash_xxhash64(Table const& tbl, uint64_t seed);
std::unique_ptr<Column> hash_md5(Table const& tbl);
std::unique_ptr<Column> hash_sha256(Table const& tbl);

// -- Reshape --

std::unique_ptr<Column> interleave_columns(Table const& tbl);
std::unique_ptr<Table> tile_table(Table const& tbl, int32_t count);

// -- Transform --

std::unique_ptr<Column> nans_to_nulls(cudf::column_view const& col);
std::unique_ptr<Column> encode_table(Table const& tbl);
std::unique_ptr<Table> encode_keys(Table const& tbl);

// -- Merge --

std::unique_ptr<Table> merge_tables(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders);

// -- Partitioning --

std::unique_ptr<Table> hash_partition_table(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions);

rust::Vec<int32_t> hash_partition_offsets(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions);

std::unique_ptr<Table> round_robin_partition_table(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition);

rust::Vec<int32_t> round_robin_partition_offsets(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition);

// -- CXX shared enum (generated from Rust bridge) --
enum class AggregationKind : ::std::int32_t;

// -- GroupBy operations --

std::unique_ptr<Table> groupby_single(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    int32_t value_index,
    int32_t agg_kind);

std::unique_ptr<Table> groupby_multi(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds);

}  // namespace cudf_sys
