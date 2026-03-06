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

}  // namespace cudf_sys
