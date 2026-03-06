// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/lib.rs.h"

#include <cudf/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/replace.hpp>
#include <cudf/search.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/unary.hpp>
#include <cudf/reshape.hpp>
#include <cudf/transform.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>

#include <cudf/round.hpp>
#include <cudf/transpose.hpp>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>

#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/wrap.hpp>
#include <cudf/strings/convert/convert_lists.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/strings/convert/int_cast.hpp>

#include <cudf/lists/combine.hpp>

#include <cudf/lists/contains.hpp>
#include <cudf/lists/gather.hpp>

#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>

#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/lists/set_operations.hpp>

#include <cudf/rolling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/json/json.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cudf_sys {

// -- Stream --

std::size_t get_default_stream() {
  return reinterpret_cast<std::size_t>(cudf::get_default_stream().value());
}

// -- DataType --

TypeId data_type_id(DataType const& dt) {
  return static_cast<TypeId>(static_cast<int32_t>(dt.inner.id()));
}

int32_t data_type_scale(DataType const& dt) {
  return dt.inner.scale();
}

std::size_t size_of_data_type(DataType const& dt) {
  return cudf::size_of(dt.inner);
}

// -- Column --

Column::Column(std::unique_ptr<cudf::column> col) : column_(std::move(col)) {}

cudf::column_view const& Column::cached_view() const {
  std::call_once(view_once_, [this]() { view_ = column_->view(); });
  return view_;
}

int32_t column_size(Column const& col) { return col.inner().size(); }
int32_t column_null_count(Column const& col) { return col.inner().null_count(); }
bool column_has_nulls(Column const& col) { return col.inner().has_nulls(); }
bool column_nullable(Column const& col) { return col.inner().nullable(); }
int32_t column_num_children(Column const& col) { return col.inner().num_children(); }

int32_t column_type_id(Column const& col) {
  return static_cast<int32_t>(col.inner().type().id());
}

int32_t column_type_scale(Column const& col) {
  return col.inner().type().scale();
}

cudf::column_view const& column_view_of(Column const& col) noexcept {
  return col.cached_view();
}

// column_view free functions

int32_t column_view_size(cudf::column_view const& view) { return view.size(); }
int32_t column_view_null_count(cudf::column_view const& view) { return view.null_count(); }
bool column_view_has_nulls(cudf::column_view const& view) { return view.has_nulls(); }
int32_t column_view_offset(cudf::column_view const& view) { return view.offset(); }

int32_t column_view_type_id(cudf::column_view const& view) {
  return static_cast<int32_t>(view.type().id());
}

// -- Table --

Table::Table(std::unique_ptr<cudf::table> tbl) : table_(std::move(tbl)) {}

cudf::table_view const& Table::cached_view() const {
  std::call_once(view_once_, [this]() { view_ = table_->view(); });
  return view_;
}

std::unique_ptr<Table> table_empty() {
  auto tbl = std::make_unique<cudf::table>(
      std::vector<std::unique_ptr<cudf::column>>{});
  return std::make_unique<Table>(std::move(tbl));
}

int32_t table_num_columns(Table const& tbl) { return tbl.inner().num_columns(); }
int32_t table_num_rows(Table const& tbl) { return tbl.inner().num_rows(); }
std::size_t table_alloc_size(Table const& tbl) { return tbl.inner().alloc_size(); }

cudf::column_view const& table_get_column_view(Table const& tbl, int32_t index) {
  return tbl.cached_view().column(index);
}

cudf::table_view const& table_view_of(Table const& tbl) {
  return tbl.cached_view();
}

// table_view free functions

int32_t table_view_num_columns(cudf::table_view const& view) { return view.num_columns(); }
int32_t table_view_num_rows(cudf::table_view const& view) { return view.num_rows(); }

// -- Scalar --

Scalar::Scalar(std::unique_ptr<cudf::scalar> s) : scalar_(std::move(s)) {}

std::unique_ptr<Scalar> make_int32_scalar(int32_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<int32_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_int64_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<int64_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_float32_scalar(float value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<float>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_float64_scalar(double value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<double>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_bool_scalar(bool value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<bool>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_string_scalar(rust::Str value) {
  auto sv = std::string_view(value.data(), value.size());
  auto s = std::make_unique<cudf::string_scalar>(sv, true);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_int8_scalar(int8_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<int8_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_int16_scalar(int16_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<int16_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_uint8_scalar(uint8_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<uint8_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_uint16_scalar(uint16_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<uint16_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_uint32_scalar(uint32_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<uint32_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_uint64_scalar(uint64_t value, bool valid) {
  auto s = std::make_unique<cudf::numeric_scalar<uint64_t>>(value, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_timestamp_s_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
      cudf::timestamp_s{cudf::duration_s{value}}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_timestamp_ms_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
      cudf::timestamp_ms{cudf::duration_ms{value}}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_timestamp_us_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
      cudf::timestamp_us{cudf::duration_us{value}}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_timestamp_ns_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
      cudf::timestamp_ns{cudf::duration_ns{value}}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_duration_s_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::duration_scalar<cudf::duration_s>>(
      cudf::duration_s{value}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_duration_ms_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::duration_scalar<cudf::duration_ms>>(
      cudf::duration_ms{value}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_duration_us_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::duration_scalar<cudf::duration_us>>(
      cudf::duration_us{value}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_duration_ns_scalar(int64_t value, bool valid) {
  auto s = std::make_unique<cudf::duration_scalar<cudf::duration_ns>>(
      cudf::duration_ns{value}, valid);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_default_constructed_scalar(int32_t type_id, int32_t scale) {
  auto dt = cudf::data_type{static_cast<cudf::type_id>(type_id), scale};
  auto s = cudf::make_default_constructed_scalar(dt);
  return std::make_unique<Scalar>(std::move(s));
}

std::unique_ptr<Scalar> make_empty_scalar_like(cudf::column_view const& col) {
  auto s = cudf::make_empty_scalar_like(col);
  return std::make_unique<Scalar>(std::move(s));
}

bool scalar_is_valid(Scalar const& s) { return s.inner().is_valid(); }

int32_t scalar_type_id(Scalar const& s) {
  return static_cast<int32_t>(s.inner().type().id());
}

int32_t scalar_to_i32(Scalar const& s) {
  return static_cast<cudf::numeric_scalar<int32_t> const&>(s.inner()).value();
}

int64_t scalar_to_i64(Scalar const& s) {
  return static_cast<cudf::numeric_scalar<int64_t> const&>(s.inner()).value();
}

float scalar_to_f32(Scalar const& s) {
  return static_cast<cudf::numeric_scalar<float> const&>(s.inner()).value();
}

double scalar_to_f64(Scalar const& s) {
  return static_cast<cudf::numeric_scalar<double> const&>(s.inner()).value();
}

bool scalar_to_bool(Scalar const& s) {
  return static_cast<cudf::numeric_scalar<bool> const&>(s.inner()).value();
}

// -- Column factories --

std::unique_ptr<Column> make_column_from_scalar(Scalar const& s, int32_t count, std::size_t stream) {
  rmm::cuda_stream_view sv{reinterpret_cast<cudaStream_t>(stream)};
  auto col = cudf::make_column_from_scalar(s.inner(), count, sv);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_empty_column_by_type(int32_t type_id) {
  auto col = cudf::make_empty_column(static_cast<cudf::type_id>(type_id));
  return std::make_unique<Column>(std::move(col));
}

// -- Column data extraction (device -> host) --

rust::Vec<int16_t> column_to_host_i16(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<int16_t> result;
  result.reserve(size);
  std::vector<int16_t> host_data(size);
  cudaMemcpyAsync(host_data.data(), view.data<int16_t>(),
             size * sizeof(int16_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<int32_t> column_to_host_i32(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<int32_t> result;
  result.reserve(size);
  std::vector<int32_t> host_data(size);
  cudaMemcpyAsync(host_data.data(), view.data<int32_t>(),
             size * sizeof(int32_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<int64_t> column_to_host_i64(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<int64_t> result;
  result.reserve(size);
  std::vector<int64_t> host_data(size);
  cudaMemcpyAsync(host_data.data(), view.data<int64_t>(),
             size * sizeof(int64_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<float> column_to_host_f32(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<float> result;
  result.reserve(size);
  std::vector<float> host_data(size);
  cudaMemcpyAsync(host_data.data(), view.data<float>(),
             size * sizeof(float), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<double> column_to_host_f64(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<double> result;
  result.reserve(size);
  std::vector<double> host_data(size);
  cudaMemcpyAsync(host_data.data(), view.data<double>(),
             size * sizeof(double), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<bool> column_to_host_bool(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<bool> result;
  result.reserve(size);
  // cudf BOOL8 uses int8_t internally
  std::vector<int8_t> host_data(size);
  cudaMemcpyAsync(host_data.data(), view.data<int8_t>(),
             size * sizeof(int8_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  for (auto v : host_data) result.push_back(v != 0);
  return result;
}

rust::Vec<bool> column_null_mask_to_host(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<bool> result;
  result.reserve(size);

  if (!view.nullable()) {
    // No null mask means all values are valid
    for (int32_t i = 0; i < size; ++i) result.push_back(true);
    return result;
  }

  // Copy the bitmask to host
  auto num_bitmask_words = cudf::bitmask_allocation_size_bytes(size) / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host_mask(num_bitmask_words);
  cudaMemcpyAsync(host_mask.data(), view.null_mask(),
             num_bitmask_words * sizeof(cudf::bitmask_type), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();

  for (int32_t i = 0; i < size; ++i) {
    auto word = host_mask[i / 32];
    auto bit = (word >> (i % 32)) & 1;
    result.push_back(bit != 0);
  }
  return result;
}

// -- TableBuilder --

void TableBuilder::add_column(std::unique_ptr<Column> wrapper) {
  cudf_columns_.push_back(wrapper->release());
}

std::unique_ptr<Table> TableBuilder::build() {
  auto tbl = std::make_unique<cudf::table>(std::move(cudf_columns_));
  cudf_columns_.clear();
  return std::make_unique<Table>(std::move(tbl));
}

std::unique_ptr<TableBuilder> new_table_builder() {
  return std::make_unique<TableBuilder>();
}

void table_builder_add_column(TableBuilder& builder, std::unique_ptr<Column> col) {
  builder.add_column(std::move(col));
}

std::unique_ptr<Table> table_builder_build(TableBuilder& builder) {
  return builder.build();
}

// -- Binary operations --

std::unique_ptr<Column> binary_operation_columns(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::binary_operation(
      lhs, rhs,
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> binary_operation_column_scalar(
    cudf::column_view const& lhs,
    Scalar const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::binary_operation(
      lhs, rhs.inner(),
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> binary_operation_scalar_column(
    Scalar const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::binary_operation(
      lhs.inner(), rhs,
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      s);
  return std::make_unique<Column>(std::move(result));
}

// -- Unary operations --

std::unique_ptr<Column> unary_cast(cudf::column_view const& col, int32_t target_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::cast(col, cudf::data_type{static_cast<cudf::type_id>(target_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_null(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_null(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_valid(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_valid(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_nan(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_nan(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_negate(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::unary_operation(col, cudf::unary_operator::NEGATE, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_abs(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::unary_operation(col, cudf::unary_operator::ABS, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Reduction --

std::unique_ptr<Scalar> reduce_sum(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(
      col, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_min(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(
      col, *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_max(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(
      col, *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_product(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(
      col, *cudf::make_product_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_any(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(
      col, *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_all(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(
      col, *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8}, s);
  return std::make_unique<Scalar>(std::move(result));
}

// -- Sorting --

std::unique_ptr<Table> sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::sort(tbl.cached_view(), orders, nulls, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::sorted_order(tbl.cached_view(), orders, nulls, s);
  return std::make_unique<Column>(std::move(result));
}

bool is_sorted_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  return cudf::is_sorted(tbl.cached_view(), orders, nulls, s);
}

// -- Filtering --

std::unique_ptr<Table> apply_boolean_mask(
    Table const& tbl,
    cudf::column_view const& mask,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::apply_boolean_mask(tbl.cached_view(), mask, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> drop_nulls_all(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = tbl.cached_view();
  auto num_cols = view.num_columns();
  std::vector<cudf::size_type> keys(num_cols);
  for (int32_t i = 0; i < num_cols; ++i) keys[i] = i;
  auto result = cudf::drop_nulls(view, keys, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Concatenation --

void ColumnConcatenator::add(cudf::column_view const& v) {
  views_.push_back(v);
}

std::unique_ptr<Column> ColumnConcatenator::finish(std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::concatenate(views_, s);
  views_.clear();
  return std::make_unique<Column>(std::move(result));
}

void TableConcatenator::add_table(Table const& t) {
  views_.push_back(t.cached_view());
}

std::unique_ptr<Table> TableConcatenator::finish(std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::concatenate(views_, s);
  views_.clear();
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<ColumnConcatenator> new_column_concatenator() {
  return std::make_unique<ColumnConcatenator>();
}

void column_concatenator_add(ColumnConcatenator& cat, cudf::column_view const& v) {
  cat.add(v);
}

std::unique_ptr<Column> column_concatenator_finish(ColumnConcatenator& cat, std::size_t stream) {
  return cat.finish(stream);
}

std::unique_ptr<TableConcatenator> new_table_concatenator() {
  return std::make_unique<TableConcatenator>();
}

void table_concatenator_add(TableConcatenator& cat, Table const& t) {
  cat.add_table(t);
}

std::unique_ptr<Table> table_concatenator_finish(TableConcatenator& cat, std::size_t stream) {
  return cat.finish(stream);
}

// -- Copying --

std::unique_ptr<Table> gather_table(
    Table const& tbl,
    cudf::column_view const& indices,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::gather(tbl.cached_view(), indices,
      cudf::out_of_bounds_policy::DONT_CHECK, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> empty_like_column(cudf::column_view const& col) {
  auto result = cudf::empty_like(col);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> empty_like_table(Table const& tbl) {
  auto result = cudf::empty_like(tbl.cached_view());
  return std::make_unique<Table>(std::move(result));
}

// -- Replace operations --

std::unique_ptr<Column> replace_nulls_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nulls(col, replacement, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> replace_nulls_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nulls(col, replacement.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> replace_nans_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nans(col, replacement, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> replace_nans_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nans(col, replacement.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> clamp_column(
    cudf::column_view const& col,
    Scalar const& lo,
    Scalar const& hi,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::clamp(col, lo.inner(), hi.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> find_and_replace_all(
    cudf::column_view const& col,
    cudf::column_view const& values_to_replace,
    cudf::column_view const& replacement_values,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::find_and_replace_all(col, values_to_replace, replacement_values, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Fill operations --

std::unique_ptr<Column> fill_column(
    cudf::column_view const& col,
    int32_t begin,
    int32_t end,
    Scalar const& value,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::fill(col, begin, end, value.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> repeat_table(
    Table const& tbl,
    int32_t count,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::repeat(tbl.cached_view(), count, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> sequence_column(
    int32_t count,
    Scalar const& init,
    Scalar const& step,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::sequence(count, init.inner(), step.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Search operations --

bool contains_scalar(
    cudf::column_view const& haystack,
    Scalar const& needle,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  return cudf::contains(haystack, needle.inner(), s);
}

std::unique_ptr<Column> contains_column(
    cudf::column_view const& haystack,
    cudf::column_view const& needles,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::contains(haystack, needles, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lower_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::lower_bound(haystack.cached_view(), needles.cached_view(), orders, nulls, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> upper_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::upper_bound(haystack.cached_view(), needles.cached_view(), orders, nulls, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Quantile operations --

std::unique_ptr<Column> quantile_column(
    cudf::column_view const& col,
    rust::Slice<double const> quantiles,
    int32_t interp,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<double> q(quantiles.begin(), quantiles.end());
  auto result = cudf::quantile(col, q, static_cast<cudf::interpolation>(interp), cudf::column_view{}, true, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Column factories from host data --

std::unique_ptr<Column> make_column_from_host_i32(rust::Slice<int32_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int32_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_i64(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT64}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_f64(rust::Slice<double const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(double), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::FLOAT64}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_bool(rust::Slice<bool const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  // cudf BOOL8 uses int8_t internally
  std::vector<int8_t> int8_data(size);
  for (cudf::size_type i = 0; i < size; ++i) {
    int8_data[i] = data[i] ? 1 : 0;
  }
  rmm::device_buffer buf(int8_data.data(), size * sizeof(int8_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_timestamp_s(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_i8(rust::Slice<int8_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int8_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT8}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_i16(rust::Slice<int16_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int16_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT16}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_f32(rust::Slice<float const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(float), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::FLOAT32}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_u8(rust::Slice<uint8_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(uint8_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_u16(rust::Slice<uint16_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(uint16_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT16}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_u32(rust::Slice<uint32_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(uint32_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT32}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_u64(rust::Slice<uint64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(uint64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT64}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

// -- Timestamp/Duration column factories from host data --

std::unique_ptr<Column> make_column_from_host_timestamp_ms(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_timestamp_us(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_timestamp_ns(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_duration_s(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::DURATION_SECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_duration_ms(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::DURATION_MILLISECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_duration_us(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::DURATION_MICROSECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_column_from_host_duration_ns(rust::Slice<int64_t const> data, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(int64_t), s);
  s.synchronize();
  auto col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::DURATION_NANOSECONDS}, size, std::move(buf),
      rmm::device_buffer{}, 0);
  return std::make_unique<Column>(std::move(col));
}

// -- Column to host extras --

rust::Vec<int8_t> column_to_host_i8(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& view = col.cached_view();
  auto size = view.size();
  std::vector<int8_t> host(size);
  cudaMemcpyAsync(host.data(), view.data<int8_t>(), size * sizeof(int8_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<int8_t> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v);
  return out;
}

rust::Vec<uint8_t> column_to_host_u8(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& view = col.cached_view();
  auto size = view.size();
  std::vector<uint8_t> host(size);
  cudaMemcpyAsync(host.data(), view.data<uint8_t>(), size * sizeof(uint8_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<uint8_t> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v);
  return out;
}

rust::Vec<uint16_t> column_to_host_u16(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& view = col.cached_view();
  auto size = view.size();
  std::vector<uint16_t> host(size);
  cudaMemcpyAsync(host.data(), view.data<uint16_t>(), size * sizeof(uint16_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<uint16_t> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v);
  return out;
}

rust::Vec<uint32_t> column_to_host_u32(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& view = col.cached_view();
  auto size = view.size();
  std::vector<uint32_t> host(size);
  cudaMemcpyAsync(host.data(), view.data<uint32_t>(), size * sizeof(uint32_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<uint32_t> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v);
  return out;
}

rust::Vec<uint64_t> column_to_host_u64(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& view = col.cached_view();
  auto size = view.size();
  std::vector<uint64_t> host(size);
  cudaMemcpyAsync(host.data(), view.data<uint64_t>(), size * sizeof(uint64_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<uint64_t> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v);
  return out;
}

// -- Join operations --

// Helper: build a table_view selecting only key columns from a full table view.
static cudf::table_view select_columns(
    cudf::table_view const& view,
    rust::Slice<int32_t const> indices) {
  std::vector<cudf::column_view> key_cols;
  key_cols.reserve(indices.size());
  for (auto idx : indices) {
    key_cols.push_back(view.column(idx));
  }
  return cudf::table_view{key_cols};
}

// Helper: convert device_uvector<size_type> to a cudf column (for gather maps).
static std::unique_ptr<cudf::column> indices_to_column(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices) {
  auto size = static_cast<cudf::size_type>(indices->size());
  auto buf = indices->release();
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, size, std::move(buf),
      rmm::device_buffer{}, 0);
}

// Helper: gather + combine columns from left and right tables.
static std::unique_ptr<Table> gather_and_combine(
    cudf::table_view const& left_view,
    cudf::table_view const& right_view,
    std::unique_ptr<cudf::column> left_idx_col,
    std::unique_ptr<cudf::column> right_idx_col,
    rmm::cuda_stream_view s,
    bool nullify_left = false) {
  auto left_result = cudf::gather(
      left_view, left_idx_col->view(),
      nullify_left ? cudf::out_of_bounds_policy::NULLIFY
                   : cudf::out_of_bounds_policy::DONT_CHECK, s);
  auto right_result = cudf::gather(
      right_view, right_idx_col->view(),
      cudf::out_of_bounds_policy::NULLIFY, s);

  auto left_cols = left_result->release();
  auto right_cols = right_result->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(left_cols.size() + right_cols.size());
  for (auto& c : left_cols) all_cols.push_back(std::move(c));
  for (auto& c : right_cols) all_cols.push_back(std::move(c));

  return std::make_unique<Table>(
      std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> inner_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto left_view = left.cached_view();
  auto right_view = right.cached_view();

  auto left_keys = select_columns(left_view, left_on);
  auto right_keys = select_columns(right_view, right_on);

  auto [left_indices, right_indices] = cudf::inner_join(left_keys, right_keys,
      cudf::null_equality::EQUAL, s);

  auto left_idx_col = indices_to_column(std::move(left_indices));
  auto right_idx_col = indices_to_column(std::move(right_indices));

  return gather_and_combine(left_view, right_view,
      std::move(left_idx_col), std::move(right_idx_col), s);
}

std::unique_ptr<Table> left_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto left_view = left.cached_view();
  auto right_view = right.cached_view();

  auto left_keys = select_columns(left_view, left_on);
  auto right_keys = select_columns(right_view, right_on);

  auto [left_indices, right_indices] = cudf::left_join(left_keys, right_keys,
      cudf::null_equality::EQUAL, s);

  auto left_idx_col = indices_to_column(std::move(left_indices));
  auto right_idx_col = indices_to_column(std::move(right_indices));

  // Right indices may contain JoinNoMatch for unmatched rows -> nullify
  return gather_and_combine(left_view, right_view,
      std::move(left_idx_col), std::move(right_idx_col), s);
}

std::unique_ptr<Table> full_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto left_view = left.cached_view();
  auto right_view = right.cached_view();

  auto left_keys = select_columns(left_view, left_on);
  auto right_keys = select_columns(right_view, right_on);

  auto [left_indices, right_indices] = cudf::full_join(left_keys, right_keys,
      cudf::null_equality::EQUAL, s);

  auto left_idx_col = indices_to_column(std::move(left_indices));
  auto right_idx_col = indices_to_column(std::move(right_indices));

  // Both sides may contain JoinNoMatch -> nullify on both
  return gather_and_combine(left_view, right_view,
      std::move(left_idx_col), std::move(right_idx_col), s, true);
}

std::unique_ptr<Table> left_semi_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto left_view = left.cached_view();
  auto right_view = right.cached_view();

  auto left_keys = select_columns(left_view, left_on);
  auto right_keys = select_columns(right_view, right_on);

  // Build hash table on right keys, probe with left keys
  cudf::filtered_join joiner(right_keys, cudf::null_equality::EQUAL,
                             cudf::set_as_build_table::RIGHT, s);
  auto left_indices = joiner.semi_join(left_keys, s);

  auto left_idx_col = indices_to_column(std::move(left_indices));

  // Gather only from left table
  auto result = cudf::gather(left_view, left_idx_col->view(),
      cudf::out_of_bounds_policy::DONT_CHECK, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> left_anti_join(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> left_on,
    rust::Slice<int32_t const> right_on,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto left_view = left.cached_view();
  auto right_view = right.cached_view();

  auto left_keys = select_columns(left_view, left_on);
  auto right_keys = select_columns(right_view, right_on);

  // Build hash table on right keys, probe with left keys
  cudf::filtered_join joiner(right_keys, cudf::null_equality::EQUAL,
                             cudf::set_as_build_table::RIGHT, s);
  auto left_indices = joiner.anti_join(left_keys, s);

  auto left_idx_col = indices_to_column(std::move(left_indices));

  // Gather only from left table
  auto result = cudf::gather(left_view, left_idx_col->view(),
      cudf::out_of_bounds_policy::DONT_CHECK, s);
  return std::make_unique<Table>(std::move(result));
}

// -- String operations --

std::unique_ptr<Column> strings_to_lower(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_lower(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_upper(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_upper(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_contains(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::contains(scv, str_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_starts_with(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::starts_with(scv, str_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_ends_with(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::ends_with(scv, str_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::find(scv, str_scalar, 0, -1, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace(
    cudf::column_view const& col,
    Scalar const& target,
    Scalar const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& tgt = static_cast<cudf::string_scalar const&>(target.inner());
  auto const& repl = static_cast<cudf::string_scalar const&>(replacement.inner());
  auto result = cudf::strings::replace(scv, tgt, repl, -1, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_strip(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::strip(scv, cudf::strings::side_type::BOTH,
      cudf::string_scalar("", true), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_lstrip(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::strip(scv, cudf::strings::side_type::LEFT,
      cudf::string_scalar("", true), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rstrip(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::strip(scv, cudf::strings::side_type::RIGHT,
      cudf::string_scalar("", true), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_count_characters(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::count_characters(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_count_bytes(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::count_bytes(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_integers(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::from_integers(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_integers(scv,
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_floats(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::from_floats(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_floats(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_floats(scv,
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String column construction --

std::unique_ptr<Column> make_string_column(rust::Vec<rust::String> strings, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  // Build a string column by creating individual scalars and concatenating
  // single-element columns. This avoids needing test utilities.
  if (strings.empty()) {
    return std::make_unique<Column>(
        cudf::make_empty_column(cudf::type_id::STRING));
  }

  std::vector<cudf::column_view> views;
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(strings.size());

  for (auto const& str : strings) {
    auto sv = std::string_view(str.data(), str.size());
    auto scalar = cudf::string_scalar(sv, true, s);
    auto col = cudf::make_column_from_scalar(scalar, 1, s);
    views.push_back(col->view());
    cols.push_back(std::move(col));
  }

  auto result = cudf::concatenate(views, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String column extraction (device -> host) --

rust::Vec<rust::String> column_to_host_strings(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<rust::String> result;
  result.reserve(size);

  if (size == 0) return result;

  cudf::strings_column_view scv(view);

  // Get offsets to host
  auto offsets_view = scv.offsets();
  auto num_offsets = size + 1;
  auto offset_begin = scv.offset();
  std::vector<int32_t> host_offsets(num_offsets);
  cudaMemcpyAsync(host_offsets.data(),
             offsets_view.data<int32_t>() + offset_begin,
             num_offsets * sizeof(int32_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();

  // Get chars to host
  auto chars_begin_ptr = scv.chars_begin(s);
  auto total_bytes = host_offsets[size] - host_offsets[0];
  std::vector<char> host_chars(total_bytes);
  if (total_bytes > 0) {
    cudaMemcpyAsync(host_chars.data(),
               chars_begin_ptr + host_offsets[0],
               total_bytes, cudaMemcpyDeviceToHost, s.value());
    s.synchronize();
  }

  auto base_offset = host_offsets[0];
  for (int32_t i = 0; i < size; ++i) {
    auto start = host_offsets[i] - base_offset;
    auto end = host_offsets[i + 1] - base_offset;
    auto str = std::string(host_chars.data() + start, end - start);
    result.push_back(rust::String(str));
  }
  return result;
}

// -- I/O --

std::unique_ptr<Table> read_csv(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::csv_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_csv(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

std::unique_ptr<Table> read_csv_with_options(
    rust::Str filepath,
    uint8_t delimiter,
    bool header,
    int32_t skip_rows,
    int32_t num_rows) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::csv_reader_options::builder(cudf::io::source_info{path});
  builder.delimiter(static_cast<char>(delimiter));
  if (!header) {
    builder.header(-1);
  }
  builder.skiprows(skip_rows);
  if (num_rows >= 0) {
    builder.nrows(num_rows);
  }
  auto opts = builder.build();
  auto result = cudf::io::read_csv(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_csv(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::csv_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_csv(opts);
}

void write_csv_with_options(
    Table const& tbl,
    rust::Str filepath,
    uint8_t delimiter,
    bool include_header,
    rust::Str na_rep) {
  std::string path(filepath.data(), filepath.size());
  std::string na_str(na_rep.data(), na_rep.size());
  auto builder = cudf::io::csv_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view());
  builder.inter_column_delimiter(static_cast<char>(delimiter));
  builder.include_header(include_header);
  builder.na_rep(na_str);
  auto opts = builder.build();
  cudf::io::write_csv(opts);
}

std::unique_ptr<Table> read_parquet(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_parquet(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_parquet(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_parquet(opts);
}

// -- Datetime operations --

std::unique_ptr<Column> datetime_extract_year(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::YEAR, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_month(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MONTH, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_day(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::DAY, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_weekday(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::WEEKDAY, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_hour(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::HOUR, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_minute(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MINUTE, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_second(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::SECOND, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_day_of_year(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::day_of_year(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_is_leap_year(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::is_leap_year(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_days_in_month(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::days_in_month(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_last_day_of_month(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::last_day_of_month(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_quarter(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_quarter(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- GroupBy operations --

static std::unique_ptr<cudf::groupby_aggregation> make_groupby_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::groupby_aggregation>();
    case 5: return cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
    case 6: return cudf::make_median_aggregation<cudf::groupby_aggregation>();
    case 7: return cudf::make_std_aggregation<cudf::groupby_aggregation>();
    case 8: return cudf::make_variance_aggregation<cudf::groupby_aggregation>();
    case 9: return cudf::make_product_aggregation<cudf::groupby_aggregation>();
    case 10: return cudf::make_any_aggregation<cudf::groupby_aggregation>();
    case 11: return cudf::make_all_aggregation<cudf::groupby_aggregation>();
    case 12: return cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
    case 13: return cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
    case 14: return cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
    case 15: return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>();
    case 16: return cudf::make_histogram_aggregation<cudf::groupby_aggregation>();
    case 17: return cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
    case 18: return cudf::make_m2_aggregation<cudf::groupby_aggregation>();
    case 19: return cudf::make_merge_lists_aggregation<cudf::groupby_aggregation>();
    case 20: return cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>();
    case 21: return cudf::make_merge_histogram_aggregation<cudf::groupby_aggregation>();
    default:
      throw std::invalid_argument("Unknown aggregation kind: " + std::to_string(kind));
  }
}

static std::unique_ptr<cudf::groupby_scan_aggregation> make_groupby_scan_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::groupby_scan_aggregation>();
    case 9: return cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
    default:
      throw std::invalid_argument("Unsupported groupby scan aggregation kind: " + std::to_string(kind));
  }
}

std::unique_ptr<Table> groupby_single(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    int32_t value_index,
    int32_t agg_kind,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};

  auto view = tbl.cached_view();

  // Build keys table_view
  std::vector<cudf::size_type> key_cols(key_indices.begin(), key_indices.end());
  auto keys_view = view.select(key_cols);

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  // Build aggregation request
  std::vector<cudf::groupby::aggregation_request> requests;
  cudf::groupby::aggregation_request req;
  req.values = view.column(value_index);
  req.aggregations.push_back(make_groupby_agg(agg_kind));
  requests.push_back(std::move(req));

  auto [result_keys, result_vals] = gb.aggregate(requests, s);

  // Combine keys + result value columns into one table
  auto key_cols_owned = result_keys->release();
  auto& val_results = result_vals[0].results;

  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + val_results.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& c : val_results) all_cols.push_back(std::move(c));

  return std::make_unique<Table>(
      std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_multi(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};

  auto view = tbl.cached_view();

  // Build keys table_view
  std::vector<cudf::size_type> key_cols(key_indices.begin(), key_indices.end());
  auto keys_view = view.select(key_cols);

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  // Build aggregation requests -- one per (value_column, agg) pair
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.reserve(value_indices.size());
  for (size_t i = 0; i < value_indices.size(); ++i) {
    cudf::groupby::aggregation_request req;
    req.values = view.column(value_indices[i]);
    req.aggregations.push_back(make_groupby_agg(agg_kinds[i]));
    requests.push_back(std::move(req));
  }

  auto [result_keys, result_vals] = gb.aggregate(requests, s);

  // Combine keys + all result value columns into one table
  auto key_cols_owned = result_keys->release();

  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + result_vals.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& rv : result_vals) {
    for (auto& c : rv.results) all_cols.push_back(std::move(c));
  }

  return std::make_unique<Table>(
      std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_scan(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};

  auto view = tbl.cached_view();
  std::vector<cudf::size_type> key_cols(key_indices.begin(), key_indices.end());
  auto keys_view = view.select(key_cols);

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::groupby::scan_request> requests;
  requests.reserve(value_indices.size());
  for (size_t i = 0; i < value_indices.size(); ++i) {
    cudf::groupby::scan_request req;
    req.values = view.column(value_indices[i]);
    req.aggregations.push_back(make_groupby_scan_agg(agg_kinds[i]));
    requests.push_back(std::move(req));
  }

  auto [result_keys, result_vals] = gb.scan(requests, s);

  auto key_cols_owned = result_keys->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + result_vals.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& rv : result_vals) {
    for (auto& c : rv.results) all_cols.push_back(std::move(c));
  }

  return std::make_unique<Table>(
      std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_shift(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> offsets,
    ScalarList& fill_values,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};

  auto view = tbl.cached_view();
  std::vector<cudf::size_type> key_cols(key_indices.begin(), key_indices.end());
  auto keys_view = view.select(key_cols);

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  // Build values table
  std::vector<cudf::column_view> val_views;
  val_views.reserve(value_indices.size());
  for (auto idx : value_indices) val_views.push_back(view.column(idx));
  cudf::table_view values_view(val_views);

  std::vector<cudf::size_type> offset_vec(offsets.begin(), offsets.end());
  auto fill_refs = fill_values.refs();

  auto [result_keys, result_values] = gb.shift(values_view, offset_vec, fill_refs, s);

  auto key_cols_owned = result_keys->release();
  auto val_cols_owned = result_values->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + val_cols_owned.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& c : val_cols_owned) all_cols.push_back(std::move(c));

  return std::make_unique<Table>(
      std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_replace_nulls(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> policies,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};

  auto view = tbl.cached_view();
  std::vector<cudf::size_type> key_cols(key_indices.begin(), key_indices.end());
  auto keys_view = view.select(key_cols);

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::column_view> val_views;
  val_views.reserve(value_indices.size());
  for (auto idx : value_indices) val_views.push_back(view.column(idx));
  cudf::table_view values_view(val_views);

  std::vector<cudf::replace_policy> pol_vec;
  pol_vec.reserve(policies.size());
  for (auto p : policies) {
    pol_vec.push_back(static_cast<cudf::replace_policy>(p));
  }

  auto [result_keys, result_values] = gb.replace_nulls(values_view, pol_vec, s);

  auto key_cols_owned = result_keys->release();
  auto val_cols_owned = result_values->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + val_cols_owned.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& c : val_cols_owned) all_cols.push_back(std::move(c));

  return std::make_unique<Table>(
      std::make_unique<cudf::table>(std::move(all_cols)));
}

// -- Reshape --

std::unique_ptr<Column> interleave_columns(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::interleave_columns(tbl.cached_view(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> tile_table(Table const& tbl, int32_t count, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::tile(tbl.cached_view(), count, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Transform --

std::unique_ptr<Column> nans_to_nulls(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [null_mask, null_count] = cudf::nans_to_nulls(col, s);
  auto result = std::make_unique<cudf::column>(col, s);
  result->set_null_mask(std::move(*null_mask), null_count);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> encode_table(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [keys, indices] = cudf::encode(tbl.cached_view(), s);
  return std::make_unique<Column>(std::move(indices));
}

std::unique_ptr<Table> encode_keys(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [keys, indices] = cudf::encode(tbl.cached_view(), s);
  return std::make_unique<Table>(std::move(keys));
}

// -- Merge --

std::unique_ptr<Table> merge_tables(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::table_view> views = {left.cached_view(), right.cached_view()};

  std::vector<cudf::size_type> keys;
  keys.reserve(key_indices.size());
  for (auto k : key_indices) keys.push_back(k);

  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::merge(views, keys, orders, nulls, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Partitioning --

std::unique_ptr<Table> hash_partition_table(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> cols;
  cols.reserve(columns_to_hash.size());
  for (auto c : columns_to_hash) cols.push_back(c);

  auto [result_table, offsets] = cudf::hash_partition(
      tbl.cached_view(), cols, num_partitions,
      cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, s);
  return std::make_unique<Table>(std::move(result_table));
}

rust::Vec<int32_t> hash_partition_offsets(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> cols;
  cols.reserve(columns_to_hash.size());
  for (auto c : columns_to_hash) cols.push_back(c);

  auto [result_table, offsets] = cudf::hash_partition(
      tbl.cached_view(), cols, num_partitions,
      cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, s);
  rust::Vec<int32_t> result;
  result.reserve(offsets.size());
  for (auto o : offsets) result.push_back(o);
  return result;
}

std::unique_ptr<Table> round_robin_partition_table(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::round_robin_partition(
      tbl.cached_view(), num_partitions, start_partition, s);
  return std::make_unique<Table>(std::move(result_table));
}

rust::Vec<int32_t> round_robin_partition_offsets(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::round_robin_partition(
      tbl.cached_view(), num_partitions, start_partition, s);
  rust::Vec<int32_t> result;
  result.reserve(offsets.size());
  for (auto o : offsets) result.push_back(o);
  return result;
}

// -- Generic unary operation --

std::unique_ptr<Column> unary_operation(cudf::column_view const& col, int32_t op, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::unary_operation(col, static_cast<cudf::unary_operator>(op), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_not_nan(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_not_nan(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Round --

std::unique_ptr<Column> round_column(cudf::column_view const& col, int32_t decimal_places, int32_t method, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::round_decimal(col, decimal_places, static_cast<cudf::rounding_method>(method), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Stream compaction --

std::unique_ptr<Table> drop_nans(Table const& tbl, rust::Slice<int32_t const> keys, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> key_cols(keys.begin(), keys.end());
  auto result = cudf::drop_nans(tbl.cached_view(), key_cols, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> drop_nulls_with_threshold(Table const& tbl, rust::Slice<int32_t const> keys, int32_t threshold, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> key_cols(keys.begin(), keys.end());
  auto result = cudf::drop_nulls(tbl.cached_view(), key_cols, threshold, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> unique_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> key_cols(keys.begin(), keys.end());
  auto result = cudf::unique(tbl.cached_view(), key_cols,
      static_cast<cudf::duplicate_keep_option>(keep),
      static_cast<cudf::null_equality>(null_equal), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> distinct_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> key_cols(keys.begin(), keys.end());
  auto result = cudf::distinct(tbl.cached_view(), key_cols,
      static_cast<cudf::duplicate_keep_option>(keep),
      static_cast<cudf::null_equality>(null_equal),
      static_cast<cudf::nan_equality>(nan_equal), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> distinct_indices_column(Table const& tbl, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::distinct_indices(tbl.cached_view(),
      static_cast<cudf::duplicate_keep_option>(keep),
      static_cast<cudf::null_equality>(null_equal),
      static_cast<cudf::nan_equality>(nan_equal), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> stable_distinct_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> key_cols(keys.begin(), keys.end());
  auto result = cudf::stable_distinct(tbl.cached_view(), key_cols,
      static_cast<cudf::duplicate_keep_option>(keep),
      static_cast<cudf::null_equality>(null_equal),
      static_cast<cudf::nan_equality>(nan_equal), s);
  return std::make_unique<Table>(std::move(result));
}

// -- Copying extras --

std::unique_ptr<Table> scatter_table(Table const& source, cudf::column_view const& scatter_map, Table const& target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::scatter(source.cached_view(), scatter_map, target.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> reverse_column(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  // cudf::reverse takes a table_view, so wrap single column
  cudf::table_view tv{{col}};
  auto result = cudf::reverse(tv, s);
  auto cols = result->release();
  return std::make_unique<Column>(std::move(cols[0]));
}

std::unique_ptr<Table> reverse_table(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reverse(tbl.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> shift_column(cudf::column_view const& col, int32_t offset, Scalar const& fill_value, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::shift(col, offset, fill_value.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Scalar> get_element(cudf::column_view const& col, int32_t index, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::get_element(col, index, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_columns(cudf::column_view const& lhs, cudf::column_view const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs, rhs, mask, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_scalar_column(Scalar const& lhs, cudf::column_view const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs.inner(), rhs, mask, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_column_scalar(cudf::column_view const& lhs, Scalar const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs, rhs.inner(), mask, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> slice_column(cudf::column_view const& col, int32_t begin, int32_t end, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> indices = {begin, end};
  auto views = cudf::slice(col, indices);
  // Copy the view into an owned column
  auto result = std::make_unique<cudf::column>(views[0], s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> slice_table(Table const& tbl, int32_t begin, int32_t end, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> indices = {begin, end};
  auto views = cudf::slice(tbl.cached_view(), indices);
  // Copy the view into an owned table
  auto result = std::make_unique<cudf::table>(views[0], s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> sample_table(Table const& tbl, int32_t n, bool with_replacement, int64_t seed, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto policy = with_replacement ? cudf::sample_with_replacement::TRUE : cudf::sample_with_replacement::FALSE;
  auto result = cudf::sample(tbl.cached_view(), n, policy, seed, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Additional reductions --

std::unique_ptr<Scalar> reduce_mean(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col, *cudf::make_mean_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_std(cudf::column_view const& col, int32_t output_type_id, int32_t ddof, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col, *cudf::make_std_aggregation<cudf::reduce_aggregation>(ddof),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_var(cudf::column_view const& col, int32_t output_type_id, int32_t ddof, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col, *cudf::make_variance_aggregation<cudf::reduce_aggregation>(ddof),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_median(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col, *cudf::make_median_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_nunique(cudf::column_view const& col, int32_t null_policy, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col,
      *cudf::make_nunique_aggregation<cudf::reduce_aggregation>(static_cast<cudf::null_policy>(null_policy)),
      cudf::data_type{cudf::type_id::INT64}, s);
  return std::make_unique<Scalar>(std::move(result));
}

static std::unique_ptr<cudf::scan_aggregation> make_scan_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::scan_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::scan_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::scan_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::scan_aggregation>();
    case 9: return cudf::make_product_aggregation<cudf::scan_aggregation>();
    default:
      throw std::invalid_argument("Unsupported scan aggregation kind: " + std::to_string(kind));
  }
}

static std::unique_ptr<cudf::reduce_aggregation> make_reduce_agg(int32_t kind, int32_t ddof) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::reduce_aggregation>();
    case 5: return cudf::make_nunique_aggregation<cudf::reduce_aggregation>();
    case 6: return cudf::make_median_aggregation<cudf::reduce_aggregation>();
    case 7: return cudf::make_std_aggregation<cudf::reduce_aggregation>(ddof);
    case 8: return cudf::make_variance_aggregation<cudf::reduce_aggregation>(ddof);
    case 9: return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    case 10: return cudf::make_any_aggregation<cudf::reduce_aggregation>();
    case 11: return cudf::make_all_aggregation<cudf::reduce_aggregation>();
    case 17: return cudf::make_sum_of_squares_aggregation<cudf::reduce_aggregation>();
    default:
      throw std::invalid_argument("Unsupported reduce aggregation kind: " + std::to_string(kind));
  }
}

std::unique_ptr<Scalar> reduce_generic(cudf::column_view const& col, int32_t agg_kind, int32_t ddof, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col, *make_reduce_agg(agg_kind, ddof),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_with_init(cudf::column_view const& col, int32_t agg_kind, int32_t ddof, int32_t output_type_id, Scalar const& init, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reduce(col, *make_reduce_agg(agg_kind, ddof),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      std::optional<std::reference_wrapper<cudf::scalar const>>{init.inner()}, s);
  return std::make_unique<Scalar>(std::move(result));
}

static std::unique_ptr<cudf::segmented_reduce_aggregation> make_segmented_reduce_agg(int32_t kind, int32_t ddof) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::segmented_reduce_aggregation>();
    case 5: return cudf::make_nunique_aggregation<cudf::segmented_reduce_aggregation>();
    case 6: return cudf::make_median_aggregation<cudf::segmented_reduce_aggregation>();
    case 7: return cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>(ddof);
    case 8: return cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>(ddof);
    case 9: return cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();
    case 10: return cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
    case 11: return cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>();
    case 17: return cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
    default: throw std::invalid_argument("unsupported segmented_reduce aggregation kind");
  }
}

std::unique_ptr<Column> segmented_reduce(cudf::column_view const& col, cudf::column_view const& offsets, int32_t agg_kind, int32_t ddof, int32_t output_type_id, int32_t null_handling, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const* offsets_data = offsets.data<cudf::size_type>();
  auto offsets_span = cudf::device_span<cudf::size_type const>(offsets_data, offsets.size());
  auto result = cudf::segmented_reduce(col, offsets_span, *make_segmented_reduce_agg(agg_kind, ddof),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      static_cast<cudf::null_policy>(null_handling), s);
  return std::make_unique<Column>(std::move(result));
}

static std::unique_ptr<cudf::rolling_aggregation> make_rolling_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::rolling_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::rolling_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::rolling_aggregation>();
    case 12: return cudf::make_argmax_aggregation<cudf::rolling_aggregation>();
    case 13: return cudf::make_argmin_aggregation<cudf::rolling_aggregation>();
    case 14: return cudf::make_collect_list_aggregation<cudf::rolling_aggregation>();
    case 15: return cudf::make_collect_set_aggregation<cudf::rolling_aggregation>();
    default:
      throw std::invalid_argument("Unsupported rolling aggregation kind: " + std::to_string(kind));
  }
}

std::unique_ptr<Column> rolling_window_with_defaults(
    cudf::column_view const& col,
    cudf::column_view const& default_outputs,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::rolling_window(col, default_outputs, preceding, following, min_periods,
      *make_rolling_agg(agg_kind), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> scan_column(cudf::column_view const& col, int32_t agg_kind, int32_t scan_type, int32_t null_policy, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::scan(col, *make_scan_agg(agg_kind),
      static_cast<cudf::scan_type>(scan_type),
      static_cast<cudf::null_policy>(null_policy), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Scalar> minmax_min(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [min_val, max_val] = cudf::minmax(col, s);
  return std::make_unique<Scalar>(std::move(min_val));
}

std::unique_ptr<Scalar> minmax_max(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [min_val, max_val] = cudf::minmax(col, s);
  return std::make_unique<Scalar>(std::move(max_val));
}

// -- Transpose --

std::unique_ptr<Table> transpose_table(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_col, result_view] = cudf::transpose(tbl.cached_view(), s);
  // result_col owns the data, result_view is a table_view into it
  // We need to build a table from the transposed view
  auto result = std::make_unique<cudf::table>(result_view, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Label bins --

std::unique_ptr<Column> label_bins_column(
    cudf::column_view const& col,
    cudf::column_view const& left_edges,
    int32_t left_inclusive,
    cudf::column_view const& right_edges,
    int32_t right_inclusive,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::label_bins(col, left_edges,
      static_cast<cudf::inclusive>(left_inclusive),
      right_edges,
      static_cast<cudf::inclusive>(right_inclusive), s);
  return std::make_unique<Column>(std::move(result));
}

// -- String extras --

std::unique_ptr<Column> strings_pad(cudf::column_view const& col, int32_t width, int32_t side, rust::Str fill_char, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::string fc(fill_char.data(), fill_char.size());
  auto result = cudf::strings::pad(scv, width,
      static_cast<cudf::strings::side_type>(side), fc, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_zfill(cudf::column_view const& col, int32_t width, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::zfill(scv, width, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_slice(cudf::column_view const& col, int32_t start, int32_t stop, int32_t step, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto start_scalar = cudf::numeric_scalar<cudf::size_type>(start, true, s);
  auto stop_scalar = cudf::numeric_scalar<cudf::size_type>(stop, true, s);
  auto step_scalar = cudf::numeric_scalar<cudf::size_type>(step, true, s);
  auto result = cudf::strings::slice_strings(scv, start_scalar, stop_scalar, step_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_repeat(cudf::column_view const& col, int32_t repeat_times, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::repeat_strings(scv, repeat_times, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_split_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::split(scv, str_scalar, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> strings_rsplit_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::rsplit(scv, str_scalar, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_split_part(cudf::column_view const& col, Scalar const& delimiter, int32_t index, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::split_part(scv, str_scalar, index, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_join(cudf::column_view const& col, Scalar const& separator, Scalar const& narep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& sep = static_cast<cudf::string_scalar const&>(separator.inner());
  auto const& na = static_cast<cudf::string_scalar const&>(narep.inner());
  auto result = cudf::strings::join_strings(scv, sep, na, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_concatenate_columns(Table const& tbl, Scalar const& separator, Scalar const& narep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& sep = static_cast<cudf::string_scalar const&>(separator.inner());
  auto const& na = static_cast<cudf::string_scalar const&>(narep.inner());
  auto result = cudf::strings::concatenate(tbl.cached_view(), sep, na,
      cudf::strings::separator_on_nulls::YES, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_like(cudf::column_view const& col, rust::Str pattern, rust::Str escape_char, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string_view(pattern.data(), pattern.size());
  auto esc = std::string_view(escape_char.data(), escape_char.size());
  auto result = cudf::strings::like(scv, pat, esc, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_contains_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::contains_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_matches_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::matches_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_count_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::count_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_re(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto repl = std::string(replacement.data(), replacement.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::replace_re(scv, *prog, cudf::string_scalar(repl, true, s), std::nullopt, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String extras (batch 6) --

std::unique_ptr<Column> strings_swapcase(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::swapcase(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_strip_chars(cudf::column_view const& col, int32_t side, rust::Str to_strip, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto chars = std::string(to_strip.data(), to_strip.size());
  auto result = cudf::strings::strip(scv, static_cast<cudf::strings::side_type>(side), cudf::string_scalar(chars, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_literal(cudf::column_view const& col, rust::Str target, rust::Str repl, int32_t maxrepl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto rep = std::string(repl.data(), repl.size());
  auto result = cudf::strings::replace(scv, cudf::string_scalar(tgt, true, s), cudf::string_scalar(rep, true, s), maxrepl, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find_str(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::find(scv, cudf::string_scalar(tgt, true, s), start, stop, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rfind(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::rfind(scv, cudf::string_scalar(tgt, true, s), start, stop, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_contains_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::contains(scv, cudf::string_scalar(tgt, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_starts_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::starts_with(scv, cudf::string_scalar(tgt, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_ends_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::ends_with(scv, cudf::string_scalar(tgt, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_reverse(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::reverse(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_extract(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::extract(scv, *prog, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_extract_all_record(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::extract_all_record(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_findall(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::findall(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::find_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_capitalize(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::capitalize(scv, cudf::string_scalar("", true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_title(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::title(scv, cudf::strings::string_character_types::ALPHA, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_title(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_title(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_wrap(cudf::column_view const& col, int32_t width, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::wrap(scv, width, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Lists operations --

std::unique_ptr<Column> lists_count_elements(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::count_elements(lcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_extract_element(cudf::column_view const& col, int32_t index, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::extract_list_element(lcv, index, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_sort(cudf::column_view const& col, bool ascending, bool nulls_last, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto order = ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING;
  auto null_order = nulls_last ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  auto result = cudf::lists::sort_lists(lcv, order, null_order, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_reverse(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::reverse(lcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_contains_nulls(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::contains_nulls(lcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_distinct(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::distinct(lcv, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, cudf::duplicate_keep_option::KEEP_ANY, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_concatenate_elements(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::lists::concatenate_list_elements(col, cudf::lists::concatenate_null_policy::IGNORE, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_sequences(cudf::column_view const& starts, cudf::column_view const& sizes, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::lists::sequences(starts, sizes, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Explode --

std::unique_ptr<Table> explode_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::explode(tbl.cached_view(), column_idx, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> explode_position_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::explode_position(tbl.cached_view(), column_idx, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> explode_outer_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::explode_outer(tbl.cached_view(), column_idx, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Rolling window --

std::unique_ptr<Column> rolling_window(cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto agg = make_rolling_agg(agg_kind);
  auto result = cudf::rolling_window(col, preceding, following, min_periods, *agg, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> grouped_rolling_window(Table const& group_keys, cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto agg = make_rolling_agg(agg_kind);
  auto result = cudf::grouped_rolling_window(group_keys.cached_view(), col, preceding, following, min_periods, *agg, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String character types --

std::unique_ptr<Column> strings_all_characters_of_type(cudf::column_view const& col, uint32_t types, uint32_t verify_types, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::all_characters_of_type(scv, static_cast<cudf::strings::string_character_types>(types), static_cast<cudf::strings::string_character_types>(verify_types), s);
  return std::make_unique<Column>(std::move(result));
}

// -- String operations (new batch) --

std::unique_ptr<Table> strings_split_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::split_re(scv, *prog, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> strings_rsplit_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::rsplit_re(scv, *prog, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_split_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::split_record_re(scv, *prog, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rsplit_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::rsplit_record_re(scv, *prog, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_partition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto delim = std::string(delimiter.data(), delimiter.size());
  cudf::string_scalar delim_scalar(delim, true, s);
  auto result = cudf::strings::partition(scv, delim_scalar, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> strings_rpartition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto delim = std::string(delimiter.data(), delimiter.size());
  cudf::string_scalar delim_scalar(delim, true, s);
  auto result = cudf::strings::rpartition(scv, delim_scalar, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_replace_with_backrefs(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto repl = std::string(replacement.data(), replacement.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::replace_with_backrefs(scv, *prog, repl, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_repeat_column(cudf::column_view const& col, cudf::column_view const& repeat_times, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::repeat_strings(scv, repeat_times, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_contains_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view tgts(targets);
  auto result = cudf::strings::contains_multiple(scv, tgts, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_find_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view tgts(targets);
  auto result = cudf::strings::find_multiple(scv, tgts, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions --

std::unique_ptr<Column> strings_to_timestamps(cudf::column_view const& col, int32_t timestamp_type_id, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::to_timestamps(scv, cudf::data_type{static_cast<cudf::type_id>(timestamp_type_id)}, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_timestamps(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::from_timestamps(col, fmt, cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_timestamp(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::is_timestamp(scv, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_booleans(cudf::column_view const& col, rust::Str true_string, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto ts = std::string(true_string.data(), true_string.size());
  cudf::string_scalar true_scalar(ts, true, s);
  auto result = cudf::strings::to_booleans(scv, true_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_booleans(cudf::column_view const& col, rust::Str true_string, rust::Str false_string, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto ts = std::string(true_string.data(), true_string.size());
  auto fs = std::string(false_string.data(), false_string.size());
  cudf::string_scalar true_scalar(ts, true, s);
  cudf::string_scalar false_scalar(fs, true, s);
  auto result = cudf::strings::from_booleans(col, true_scalar, false_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_durations(cudf::column_view const& col, int32_t duration_type_id, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::to_durations(scv, cudf::data_type{static_cast<cudf::type_id>(duration_type_id)}, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_durations(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::from_durations(col, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_fixed_point(scv, cudf::data_type{static_cast<cudf::type_id>(type_id), scale}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_fixed_point(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::from_fixed_point(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_fixed_point(scv, cudf::data_type{static_cast<cudf::type_id>(type_id), scale}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_url_encode(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::url_encode(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_url_decode(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::url_decode(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_ipv4_to_integers(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::ipv4_to_integers(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_integers_to_ipv4(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::integers_to_ipv4(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_ipv4(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_ipv4(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Datetime operations (new) --

std::unique_ptr<Column> datetime_ceil(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::ceil_datetimes(col, static_cast<cudf::datetime::rounding_frequency>(freq), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_floor(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::floor_datetimes(col, static_cast<cudf::datetime::rounding_frequency>(freq), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_round(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::round_datetimes(col, static_cast<cudf::datetime::rounding_frequency>(freq), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_add_months(cudf::column_view const& timestamps, cudf::column_view const& months, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::add_calendrical_months(timestamps, months, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Hashing --

// -- Transform (new) --

std::unique_ptr<Column> row_bit_count(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::row_bit_count(tbl.cached_view(), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Reshape (new) --

std::unique_ptr<Column> byte_cast_column(cudf::column_view const& col, bool flip_endian, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto endian = flip_endian ? cudf::flip_endianness::YES : cudf::flip_endianness::NO;
  auto result = cudf::byte_cast(col, endian, s);
  return std::make_unique<Column>(std::move(result));
}

// -- List set operations --

std::unique_ptr<Column> lists_have_overlap(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view llhs(lhs);
  cudf::lists_column_view lrhs(rhs);
  auto result = cudf::lists::have_overlap(llhs, lrhs, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_intersect_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view llhs(lhs);
  cudf::lists_column_view lrhs(rhs);
  auto result = cudf::lists::intersect_distinct(llhs, lrhs, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_union_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view llhs(lhs);
  cudf::lists_column_view lrhs(rhs);
  auto result = cudf::lists::union_distinct(llhs, lrhs, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_difference_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view llhs(lhs);
  cudf::lists_column_view lrhs(rhs);
  auto result = cudf::lists::difference_distinct(llhs, lrhs, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Sorting (new) --

// Helper: convert i32 slice to enum vectors
static std::vector<cudf::order> to_orders(rust::Slice<int32_t const> s) {
  std::vector<cudf::order> v;
  v.reserve(s.size());
  for (auto o : s) v.push_back(static_cast<cudf::order>(o));
  return v;
}
static std::vector<cudf::null_order> to_null_orders(rust::Slice<int32_t const> s) {
  std::vector<cudf::null_order> v;
  v.reserve(s.size());
  for (auto n : s) v.push_back(static_cast<cudf::null_order>(n));
  return v;
}

std::unique_ptr<Column> stable_sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::stable_sorted_order(tbl.cached_view(), to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> stable_sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::stable_sort(tbl.cached_view(), to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> sort_by_key(
    Table const& values,
    Table const& keys,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::sort_by_key(values.cached_view(), keys.cached_view(), to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> stable_sort_by_key(
    Table const& values,
    Table const& keys,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::stable_sort_by_key(values.cached_view(), keys.cached_view(), to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> rank_column(
    cudf::column_view const& col,
    int32_t method,
    int32_t column_order,
    int32_t null_handling,
    int32_t null_precedence,
    bool percentage,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::rank(
      col,
      static_cast<cudf::rank_method>(method),
      static_cast<cudf::order>(column_order),
      static_cast<cudf::null_policy>(null_handling),
      static_cast<cudf::null_order>(null_precedence),
      percentage,
      s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> top_k(cudf::column_view const& col, int32_t k, int32_t order, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::top_k(col, k, static_cast<cudf::order>(order), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> top_k_order(cudf::column_view const& col, int32_t k, int32_t order, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::top_k_order(col, k, static_cast<cudf::order>(order), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> segmented_sorted_order(
    Table const& tbl,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::segmented_sorted_order(tbl.cached_view(), segment_offsets, to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> segmented_top_k(cudf::column_view const& col, cudf::column_view const& segment_offsets, int32_t k, int32_t order, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::segmented_top_k(col, segment_offsets, k, static_cast<cudf::order>(order), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> segmented_top_k_order(cudf::column_view const& col, cudf::column_view const& segment_offsets, int32_t k, int32_t order, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::segmented_top_k_order(col, segment_offsets, k, static_cast<cudf::order>(order), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Copying (new) --

std::unique_ptr<Column> copy_range(
    cudf::column_view const& source,
    cudf::column_view const& target,
    int32_t source_begin,
    int32_t source_end,
    int32_t target_begin,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_range(source, target, source_begin, source_end, target_begin, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> allocate_like_column(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::allocate_like(col, cudf::mask_allocation_policy::RETAIN, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> boolean_mask_scatter_table(Table const& source, Table const& target, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::boolean_mask_scatter(source.cached_view(), target.cached_view(), mask, s);
  return std::make_unique<Table>(std::move(result));
}

bool has_nonempty_nulls(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  return cudf::has_nonempty_nulls(col, s);
}

std::unique_ptr<Column> purge_nonempty_nulls(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::purge_nonempty_nulls(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_scalars(Scalar const& lhs, Scalar const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs.inner(), rhs.inner(), mask, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Replace (new) --

std::unique_ptr<Column> replace_nulls_policy(cudf::column_view const& col, int32_t policy, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nulls(col, static_cast<cudf::replace_policy>(policy), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> clamp_column_with_replace(cudf::column_view const& col, Scalar const& lo, Scalar const& lo_replace, Scalar const& hi, Scalar const& hi_replace, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::clamp(col, lo.inner(), lo_replace.inner(), hi.inner(), hi_replace.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> normalize_nans_and_zeros(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::normalize_nans_and_zeros(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Fill (new) --

std::unique_ptr<Table> repeat_table_column(Table const& tbl, cudf::column_view const& counts, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::repeat(tbl.cached_view(), counts, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> calendrical_month_sequence(int32_t count, Scalar const& init, int32_t months, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::calendrical_month_sequence(count, init.inner(), months, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Quantiles (new) --

std::unique_ptr<Table> quantiles_table(
    Table const& tbl,
    rust::Slice<double const> quantiles,
    int32_t interp,
    bool is_input_sorted,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<double> q(quantiles.begin(), quantiles.end());
  auto sorted = is_input_sorted ? cudf::sorted::YES : cudf::sorted::NO;
  auto result = cudf::quantiles(
      tbl.cached_view(),
      q,
      static_cast<cudf::interpolation>(interp),
      sorted,
      to_orders(column_orders),
      to_null_orders(null_orders),
      s);
  return std::make_unique<Table>(std::move(result));
}

// -- Null mask utilities --

std::size_t bitmask_allocation_size_bytes(int32_t number_of_bits) {
  return cudf::bitmask_allocation_size_bytes(number_of_bits);
}

int32_t num_bitmask_words(int32_t number_of_bits) {
  return cudf::num_bitmask_words(number_of_bits);
}

// -- Dictionary operations --

std::unique_ptr<Column> dictionary_encode(cudf::column_view const& col, int32_t indices_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::dictionary::encode(col, cudf::data_type{static_cast<cudf::type_id>(indices_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_decode(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::decode(dcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_add_keys(cudf::column_view const& col, cudf::column_view const& new_keys, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::add_keys(dcv, new_keys, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_remove_keys(cudf::column_view const& col, cudf::column_view const& keys_to_remove, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::remove_keys(dcv, keys_to_remove, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_set_keys(cudf::column_view const& col, cudf::column_view const& keys, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::set_keys(dcv, keys, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_remove_unused_keys(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::remove_unused_keys(dcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Scalar> dictionary_get_index(cudf::column_view const& col, Scalar const& key, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::get_index(dcv, key.inner(), s);
  return std::make_unique<Scalar>(std::move(result));
}

// -- Lists contains / index_of / segmented_gather --

std::unique_ptr<Column> lists_contains_scalar(cudf::column_view const& col, Scalar const& search_key, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::contains(lcv, search_key.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_contains_column(cudf::column_view const& col, cudf::column_view const& search_keys, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::contains(lcv, search_keys, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_index_of_scalar(cudf::column_view const& col, Scalar const& search_key, bool find_first, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto opt = find_first ? cudf::lists::duplicate_find_option::FIND_FIRST
                        : cudf::lists::duplicate_find_option::FIND_LAST;
  auto result = cudf::lists::index_of(lcv, search_key.inner(), opt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_index_of_column(cudf::column_view const& col, cudf::column_view const& search_keys, bool find_first, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto opt = find_first ? cudf::lists::duplicate_find_option::FIND_FIRST
                        : cudf::lists::duplicate_find_option::FIND_LAST;
  auto result = cudf::lists::index_of(lcv, search_keys, opt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_segmented_gather(cudf::column_view const& col, cudf::column_view const& gather_map, bool nullify_oob, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  cudf::lists_column_view gcv(gather_map);
  auto policy = nullify_oob ? cudf::out_of_bounds_policy::NULLIFY
                            : cudf::out_of_bounds_policy::DONT_CHECK;
  auto result = cudf::lists::segmented_gather(lcv, gcv, policy, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String validation / conversion extras --

std::unique_ptr<Column> strings_filter_characters_of_type(cudf::column_view const& col, uint32_t types_to_remove, rust::Str replacement, uint32_t types_to_keep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto repl = std::string(replacement.data(), replacement.size());
  cudf::string_scalar repl_scalar(repl, true, s);
  auto result = cudf::strings::filter_characters_of_type(
      scv,
      static_cast<cudf::strings::string_character_types>(types_to_remove),
      repl_scalar,
      static_cast<cudf::strings::string_character_types>(types_to_keep),
      s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_integer(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_integer(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_integer_with_type(cudf::column_view const& col, int32_t int_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_integer(scv, cudf::data_type{static_cast<cudf::type_id>(int_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_float(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_float(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_hex_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::hex_to_integers(scv, cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_hex(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_hex(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_integers_to_hex(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::integers_to_hex(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_format_list_column(cudf::column_view const& col, rust::Str na_rep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto na = std::string(na_rep.data(), na_rep.size());
  cudf::string_scalar na_scalar(na, true, s);
  auto result = cudf::strings::format_list_column(lcv, na_scalar, cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: replace_slice / replace_multiple --

std::unique_ptr<Column> strings_replace_slice(
    cudf::column_view const& col,
    rust::Str repl,
    int32_t start,
    int32_t stop,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto repl_str = std::string(repl.data(), repl.size());
  cudf::string_scalar repl_scalar(repl_str, true, s);
  auto result = cudf::strings::replace_slice(scv, repl_scalar, start, stop, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_multiple(
    cudf::column_view const& col,
    cudf::column_view const& targets,
    cudf::column_view const& repls,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view tcv(targets);
  cudf::strings_column_view rcv(repls);
  auto result = cudf::strings::replace_multiple(scv, tcv, rcv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: split_record / rsplit_record --

std::unique_ptr<Column> strings_split_record(
    cudf::column_view const& col,
    Scalar const& delimiter,
    int32_t maxsplit,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::split_record(scv, str_scalar, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rsplit_record(
    cudf::column_view const& col,
    Scalar const& delimiter,
    int32_t maxsplit,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::rsplit_record(scv, str_scalar, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Lists: apply_boolean_mask --

std::unique_ptr<Column> lists_apply_boolean_mask(
    cudf::column_view const& col,
    cudf::column_view const& boolean_mask,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  cudf::lists_column_view mcv(boolean_mask);
  auto result = cudf::lists::apply_boolean_mask(lcv, mcv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: join_list_elements --

std::unique_ptr<Column> strings_join_list_elements(
    cudf::column_view const& col,
    rust::Str separator,
    rust::Str narep,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto sep_str = std::string(separator.data(), separator.size());
  auto na_str = std::string(narep.data(), narep.size());
  cudf::string_scalar sep_scalar(sep_str, true, s);
  cudf::string_scalar na_scalar(na_str, !na_str.empty(), s);
  auto result = cudf::strings::join_list_elements(lcv, sep_scalar, na_scalar,
      cudf::strings::separator_on_nulls::YES,
      cudf::strings::output_if_empty_list::EMPTY_STRING, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Transform: segmented_row_bit_count --

std::unique_ptr<Column> segmented_row_bit_count(Table const& tbl, int32_t segment_length, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::segmented_row_bit_count(tbl.cached_view(), segment_length, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: code_points --

std::unique_ptr<Column> strings_code_points(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::code_points(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: translate / filter_characters --

std::unique_ptr<Column> strings_translate(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> table;
  table.reserve(from_chars.size());
  for (size_t i = 0; i < from_chars.size(); ++i) {
    table.emplace_back(static_cast<cudf::char_utf8>(from_chars[i]),
                       static_cast<cudf::char_utf8>(to_chars[i]));
  }
  auto result = cudf::strings::translate(scv, table, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_filter_characters(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    bool keep,
    rust::Str replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> ranges;
  ranges.reserve(from_chars.size());
  for (size_t i = 0; i < from_chars.size(); ++i) {
    ranges.emplace_back(static_cast<cudf::char_utf8>(from_chars[i]),
                        static_cast<cudf::char_utf8>(to_chars[i]));
  }
  auto repl = std::string(replacement.data(), replacement.size());
  cudf::string_scalar repl_scalar(repl, true, s);
  auto ft = keep ? cudf::strings::filter_type::KEEP
                 : cudf::strings::filter_type::REMOVE;
  auto result = cudf::strings::filter_characters(scv, ranges, ft, repl_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: cast_to_integer / cast_from_integer --

std::unique_ptr<Column> strings_cast_to_integer(
    cudf::column_view const& col,
    int32_t output_type_id,
    bool big_endian,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto swap = big_endian ? cudf::strings::endian::BIG
                         : cudf::strings::endian::LITTLE;
  auto result = cudf::strings::cast_to_integer(
      scv, cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, swap, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_cast_from_integer(
    cudf::column_view const& col,
    bool big_endian,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto swap = big_endian ? cudf::strings::endian::BIG
                         : cudf::strings::endian::LITTLE;
  auto result = cudf::strings::cast_from_integer(col, swap, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Lists: extract_element (column), stable_sort, concatenate_rows --

std::unique_ptr<Column> lists_extract_element_column(
    cudf::column_view const& col,
    cudf::column_view const& indices,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto result = cudf::lists::extract_list_element(lcv, indices, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_stable_sort(
    cudf::column_view const& col,
    bool ascending,
    bool nulls_last,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto order = ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING;
  auto null_order = nulls_last ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  auto result = cudf::lists::stable_sort_lists(lcv, order, null_order, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lists_concatenate_rows(
    Table const& tbl,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::lists::concatenate_rows(
      tbl.cached_view(), cudf::lists::concatenate_null_policy::IGNORE, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Binary: fixed_point_scale / is_supported_operation --

int32_t binary_operation_fixed_point_scale(int32_t op, int32_t left_scale, int32_t right_scale) {
  return cudf::binary_operation_fixed_point_scale(
      static_cast<cudf::binary_operator>(op), left_scale, right_scale);
}

bool is_supported_binaryop(int32_t out_type_id, int32_t lhs_type_id, int32_t rhs_type_id, int32_t op) {
  return cudf::binops::is_supported_operation(
      cudf::data_type{static_cast<cudf::type_id>(out_type_id)},
      cudf::data_type{static_cast<cudf::type_id>(lhs_type_id)},
      cudf::data_type{static_cast<cudf::type_id>(rhs_type_id)},
      static_cast<cudf::binary_operator>(op));
}

// -- Sorting: stable segmented --

std::unique_ptr<Column> stable_segmented_sorted_order(
    Table const& tbl,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::stable_segmented_sorted_order(
      tbl.cached_view(), segment_offsets,
      to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> stable_segmented_sort_by_key(
    Table const& values,
    Table const& keys,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::stable_segmented_sort_by_key(
      values.cached_view(), keys.cached_view(), segment_offsets,
      to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Table>(std::move(result));
}

// -- Explode: outer_position --

std::unique_ptr<Table> explode_outer_position_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::explode_outer_position(tbl.cached_view(), column_idx, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Strings: slice by column offsets --

std::unique_ptr<Column> strings_slice_column(
    cudf::column_view const& col,
    cudf::column_view const& starts,
    cudf::column_view const& stops,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::slice_strings(scv, starts, stops, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: extract_single --

std::unique_ptr<Column> strings_extract_single(
    cudf::column_view const& col,
    rust::Str pattern,
    int32_t group_index,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::extract_single(scv, *prog, group_index, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Copying: gather_checked, may_have_nonempty_nulls --

std::unique_ptr<Table> gather_table_checked(
    Table const& tbl,
    cudf::column_view const& gather_map,
    bool nullify_oob,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto policy = nullify_oob ? cudf::out_of_bounds_policy::NULLIFY
                            : cudf::out_of_bounds_policy::DONT_CHECK;
  auto result = cudf::gather(tbl.cached_view(), gather_map, policy, s);
  return std::make_unique<Table>(std::move(result));
}

bool may_have_nonempty_nulls(cudf::column_view const& col) {
  return cudf::may_have_nonempty_nulls(col);
}

// -- Sorting: segmented_sort_by_key (non-stable) --

std::unique_ptr<Table> segmented_sort_by_key(
    Table const& values,
    Table const& keys,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::segmented_sort_by_key(
      values.cached_view(), keys.cached_view(), segment_offsets,
      to_orders(column_orders), to_null_orders(null_orders), s);
  return std::make_unique<Table>(std::move(result));
}

// -- Partitioning: partition by map column --

std::unique_ptr<Table> partition_by_map(
    Table const& tbl,
    cudf::column_view const& partition_map,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::partition(tbl.cached_view(), partition_map, num_partitions, s);
  return std::make_unique<Table>(std::move(result_table));
}

rust::Vec<int32_t> partition_by_map_offsets(
    Table const& tbl,
    cudf::column_view const& partition_map,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::partition(tbl.cached_view(), partition_map, num_partitions, s);
  rust::Vec<int32_t> result;
  result.reserve(offsets.size());
  for (auto o : offsets) result.push_back(o);
  return result;
}

// -- Datetime: fractional seconds --

std::unique_ptr<Column> datetime_extract_millisecond(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MILLISECOND, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_microsecond(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MICROSECOND, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_nanosecond(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::NANOSECOND, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Join: cross_join --

std::unique_ptr<Table> cross_join(Table const& left, Table const& right, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::cross_join(left.cached_view(), right.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

// -- Lists: sequences with step --

std::unique_ptr<Column> lists_sequences_with_step(
    cudf::column_view const& starts,
    cudf::column_view const& steps,
    cudf::column_view const& sizes,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::lists::sequences(starts, steps, sizes, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: concatenate with per-row separator column --

std::unique_ptr<Column> strings_concatenate_columns_sep_col(
    Table const& tbl,
    cudf::column_view const& separators,
    rust::Str separator_narep,
    rust::Str col_narep,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view sep_col(separators);
  auto sep_na_str = std::string(separator_narep.data(), separator_narep.size());
  auto col_na_str = std::string(col_narep.data(), col_narep.size());
  cudf::string_scalar sep_na(sep_na_str, !sep_na_str.empty(), s);
  cudf::string_scalar col_na(col_na_str, !col_na_str.empty(), s);
  auto result = cudf::strings::concatenate(tbl.cached_view(), sep_col, sep_na, col_na,
      cudf::strings::separator_on_nulls::YES, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: join_list_elements with per-row separator column --

std::unique_ptr<Column> strings_join_list_elements_column(
    cudf::column_view const& col,
    cudf::column_view const& separators,
    rust::Str separator_narep,
    rust::Str string_narep,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  cudf::strings_column_view sep_col(separators);
  auto sep_na_str = std::string(separator_narep.data(), separator_narep.size());
  auto str_na_str = std::string(string_narep.data(), string_narep.size());
  cudf::string_scalar sep_na(sep_na_str, !sep_na_str.empty(), s);
  cudf::string_scalar str_na(str_na_str, !str_na_str.empty(), s);
  auto result = cudf::strings::join_list_elements(lcv, sep_col, sep_na, str_na,
      cudf::strings::separator_on_nulls::YES,
      cudf::strings::output_if_empty_list::EMPTY_STRING, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Strings: zfill_by_widths --

std::unique_ptr<Column> strings_zfill_by_widths(
    cudf::column_view const& col,
    cudf::column_view const& widths,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::zfill_by_widths(scv, widths, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Column factories --

std::unique_ptr<Column> make_fixed_width_column(
    int32_t type_id,
    int32_t scale,
    int32_t num_rows,
    int32_t mask_state,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto dt = cudf::data_type{static_cast<cudf::type_id>(type_id), scale};
  auto result = cudf::make_fixed_width_column(dt, num_rows,
      static_cast<cudf::mask_state>(mask_state), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> make_empty_lists_column(
    int32_t child_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto dt = cudf::data_type{static_cast<cudf::type_id>(child_type_id)};
  auto result = cudf::make_empty_lists_column(dt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> make_dictionary_from_scalar(
    Scalar const& scalar,
    int32_t num_rows,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::make_dictionary_from_scalar(scalar.inner(), num_rows, s);
  return std::make_unique<Column>(std::move(result));
}

// -- JSON path extraction --

std::unique_ptr<Column> get_json_object(
    cudf::column_view const& col,
    rust::Str json_path,
    bool allow_single_quotes,
    bool strip_quotes,
    bool missing_fields_as_nulls,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto path_str = std::string(json_path.data(), json_path.size());
  cudf::string_scalar path_scalar(path_str, true, s);
  cudf::get_json_object_options opts;
  opts.set_allow_single_quotes(allow_single_quotes);
  opts.set_strip_quotes_from_single_strings(strip_quotes);
  opts.set_missing_fields_as_nulls(missing_fields_as_nulls);
  auto result = cudf::get_json_object(scv, path_scalar, opts, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Null mask utility --

int32_t state_null_count(int32_t mask_state, int32_t num_rows) {
  return cudf::state_null_count(static_cast<cudf::mask_state>(mask_state), num_rows);
}

// -- Type checking utilities --

bool column_types_equivalent(cudf::column_view const& lhs, cudf::column_view const& rhs) {
  return cudf::column_types_equivalent(lhs, rhs);
}

bool columns_have_same_types(cudf::column_view const& lhs, cudf::column_view const& rhs) {
  return cudf::have_same_types(lhs, rhs);
}

bool tables_have_same_types(Table const& lhs, Table const& rhs) {
  return cudf::have_same_types(lhs.cached_view(), rhs.cached_view());
}

bool is_supported_cast(int32_t from_type_id, int32_t from_scale, int32_t to_type_id, int32_t to_scale) {
  auto from = cudf::data_type{static_cast<cudf::type_id>(from_type_id), from_scale};
  auto to = cudf::data_type{static_cast<cudf::type_id>(to_type_id), to_scale};
  return cudf::is_supported_cast(from, to);
}

// -- In-place operations --

void fill_in_place(Column& col, int32_t begin, int32_t end, Scalar const& value, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto mcv = col.mutable_inner().mutable_view();
  cudf::fill_in_place(mcv, begin, end, value.inner(), s);
}

void copy_range_in_place(Column& dest, cudf::column_view const& source, int32_t source_begin, int32_t source_end, int32_t dest_begin, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto mcv = dest.mutable_inner().mutable_view();
  cudf::copy_range_in_place(source, mcv, source_begin, source_end, dest_begin, s);
}

// -- One-hot encoding --

std::unique_ptr<Table> one_hot_encode(cudf::column_view const& input, cudf::column_view const& categories, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_col, result_view] = cudf::one_hot_encode(input, categories, s);
  // Materialize table from the view while owning column is still alive
  auto result = std::make_unique<cudf::table>(result_view, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Null mask conversions --

std::unique_ptr<Column> null_mask_to_bools(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto mask = col.null_mask();
  if (!mask) {
    // No null mask means all valid — return all-true column
    cudf::numeric_scalar<bool> val(true, true, s);
    auto result = cudf::make_column_from_scalar(val, col.size(), s);
    return std::make_unique<Column>(std::move(result));
  }
  auto begin = col.offset();
  auto end = col.offset() + col.size();
  auto result = cudf::mask_to_bools(mask, begin, end, s);
  return std::make_unique<Column>(std::move(result));
}

void set_null_mask_from_bools(Column& col, cudf::column_view const& bools, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [mask_buf, null_count] = cudf::bools_to_mask(bools, s);
  col.mutable_inner().set_null_mask(std::move(*mask_buf), null_count);
}

// -- Bitmask combining --

std::unique_ptr<Column> bitmask_and_to_bools(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [mask, null_count] = cudf::bitmask_and(tbl.cached_view(), s);
  auto num_rows = tbl.cached_view().num_rows();
  auto bitmask_ptr = static_cast<cudf::bitmask_type const*>(mask.data());
  auto result = cudf::mask_to_bools(bitmask_ptr, 0, num_rows, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> bitmask_or_to_bools(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [mask, null_count] = cudf::bitmask_or(tbl.cached_view(), s);
  auto num_rows = tbl.cached_view().num_rows();
  auto bitmask_ptr = static_cast<cudf::bitmask_type const*>(mask.data());
  auto result = cudf::mask_to_bools(bitmask_ptr, 0, num_rows, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Column factories: lists and structs --

std::unique_ptr<Column> make_lists_column(int32_t num_rows, std::unique_ptr<Column> offsets, std::unique_ptr<Column> child, std::size_t stream) {
  auto result = cudf::make_lists_column(
      num_rows, offsets->release(), child->release(), 0, rmm::device_buffer{});
  return std::make_unique<Column>(std::move(result));
}

// StructColumnBuilder

void StructColumnBuilder::add_child(std::unique_ptr<Column> col) {
  children_.push_back(col->release());
}

std::unique_ptr<Column> StructColumnBuilder::build(int32_t num_rows, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::make_structs_column(
      num_rows, std::move(children_), 0, rmm::device_buffer{}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<StructColumnBuilder> new_struct_column_builder() {
  return std::make_unique<StructColumnBuilder>();
}

void struct_column_builder_add(StructColumnBuilder& builder, std::unique_ptr<Column> col) {
  builder.add_child(std::move(col));
}

std::unique_ptr<Column> struct_column_builder_build(StructColumnBuilder& builder, int32_t num_rows, std::size_t stream) {
  return builder.build(num_rows, stream);
}

// -- ORC I/O --

std::unique_ptr<Table> read_orc(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_orc(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_orc(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::orc_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_orc(opts);
}

// -- JSON I/O --

std::unique_ptr<Table> read_json(rust::Str filepath, bool json_lines) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::json_reader_options::builder(cudf::io::source_info{path});
  builder.lines(json_lines);
  auto opts = builder.build();
  auto result = cudf::io::read_json(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_json(Table const& tbl, rust::Str filepath, bool json_lines) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::json_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view());
  builder.lines(json_lines);
  auto opts = builder.build();
  cudf::io::write_json(opts);
}

// -- Distinct count --

int32_t distinct_count_column(cudf::column_view const& col, int32_t null_policy, bool nan_is_null, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto np = static_cast<cudf::null_policy>(null_policy);
  auto nan_p = nan_is_null ? cudf::nan_policy::NAN_IS_NULL : cudf::nan_policy::NAN_IS_VALID;
  return cudf::distinct_count(col, np, nan_p, s);
}

int32_t distinct_count_table(Table const& tbl, int32_t null_equality, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto ne = static_cast<cudf::null_equality>(null_equality);
  return cudf::distinct_count(tbl.cached_view(), ne, s);
}

// -- Avro I/O --

std::unique_ptr<Table> read_avro(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::avro_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_avro(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

// -- Scatter with scalars --

void ScalarList::add(std::unique_ptr<Scalar> s) {
  owned_.push_back(std::move(s));
}

std::vector<std::reference_wrapper<cudf::scalar const>> ScalarList::refs() const {
  std::vector<std::reference_wrapper<cudf::scalar const>> result;
  result.reserve(owned_.size());
  for (auto const& s : owned_) {
    result.push_back(std::cref(s->inner()));
  }
  return result;
}

std::unique_ptr<ScalarList> new_scalar_list() {
  return std::make_unique<ScalarList>();
}

void scalar_list_add(ScalarList& list, std::unique_ptr<Scalar> s) {
  list.add(std::move(s));
}

std::unique_ptr<Table> scatter_scalars(ScalarList& sources, cudf::column_view const& indices, Table const& target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto refs = sources.refs();
  auto result = cudf::scatter(refs, indices, target.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> boolean_mask_scatter_scalars(ScalarList& sources, Table const& target, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto refs = sources.refs();
  auto result = cudf::boolean_mask_scatter(refs, target.cached_view(), mask, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Unique count --

int32_t unique_count_column(cudf::column_view const& col, int32_t null_policy, bool nan_is_null, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto np = static_cast<cudf::null_policy>(null_policy);
  auto nan_p = nan_is_null ? cudf::nan_policy::NAN_IS_NULL : cudf::nan_policy::NAN_IS_VALID;
  return cudf::unique_count(col, np, nan_p, s);
}

int32_t unique_count_table(Table const& tbl, int32_t null_equality, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto ne = static_cast<cudf::null_equality>(null_equality);
  return cudf::unique_count(tbl.cached_view(), ne, s);
}

// -- Drop NaNs with threshold --

std::unique_ptr<Table> drop_nans_with_threshold(Table const& tbl, rust::Slice<int32_t const> keys, int32_t threshold, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> key_cols(keys.begin(), keys.end());
  auto result = cudf::drop_nans(tbl.cached_view(), key_cols, threshold, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Approximate distinct count --

std::size_t approx_distinct_count(Table const& tbl, int32_t precision, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::approx_distinct_count adc(tbl.cached_view(), precision,
      cudf::null_policy::EXCLUDE, cudf::nan_policy::NAN_IS_NULL, s);
  return adc.estimate(s);
}

// -- Column child access --

std::unique_ptr<Column> column_view_child_copy(cudf::column_view const& col, int32_t index, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto child_view = col.child(index);
  auto result = std::make_unique<cudf::column>(child_view, s);
  return std::make_unique<Column>(std::move(result));
}

int32_t column_view_num_children(cudf::column_view const& col) {
  return col.num_children();
}

// -- Concatenate operations --

std::unique_ptr<Column> concatenate_columns(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = tbl.cached_view();
  std::vector<cudf::column_view> columns;
  columns.reserve(view.num_columns());
  for (int i = 0; i < view.num_columns(); ++i) {
    columns.push_back(view.column(i));
  }
  auto result = cudf::concatenate(columns, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> concatenate_tables(Table const& lhs, Table const& rhs, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::table_view> views{lhs.cached_view(), rhs.cached_view()};
  auto result = cudf::concatenate(views, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Repeat string scalar --

std::unique_ptr<Scalar> repeat_string_scalar(Scalar const& input, int32_t repeat_times, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(input.inner());
  auto result = cudf::strings::repeat_string(str_scalar, repeat_times, s);
  return std::make_unique<Scalar>(std::move(result));
}

// -- Strings: join_strings, find_instance, like_column --

std::unique_ptr<Column> strings_join_strings(cudf::column_view const& col, rust::Str separator, rust::Str narep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::string sep_str(separator.data(), separator.size());
  std::string narep_str(narep.data(), narep.size());
  cudf::string_scalar sep(sep_str);
  cudf::string_scalar na(narep_str);
  auto result = cudf::strings::join_strings(scv, sep, na, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find_instance(cudf::column_view const& col, rust::Str target, int32_t instance, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::string target_str(target.data(), target.size());
  cudf::string_scalar tgt(target_str);
  auto result = cudf::strings::find_instance(scv, tgt, instance, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_like_column(cudf::column_view const& col, cudf::column_view const& patterns, rust::Str escape_char, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view patterns_scv(patterns);
  std::string esc_str(escape_char.data(), escape_char.size());
  cudf::string_scalar esc(esc_str);
  auto result = cudf::strings::like(scv, patterns_scv, esc, s);
  return std::make_unique<Column>(std::move(result));
}

// -- DLPack interop --

std::unique_ptr<Table> from_dlpack(std::size_t managed_tensor_ptr, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto* tensor = reinterpret_cast<DLManagedTensor const*>(managed_tensor_ptr);
  auto result = cudf::from_dlpack(tensor, s);
  return std::make_unique<Table>(std::move(result));
}

std::size_t to_dlpack(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto* tensor = cudf::to_dlpack(tbl.cached_view(), s);
  return reinterpret_cast<std::size_t>(tensor);
}

// -- Grouped rolling window with defaults --

std::unique_ptr<Column> grouped_rolling_window_with_defaults(Table const& group_keys, cudf::column_view const& col, cudf::column_view const& default_outputs, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto agg = make_rolling_agg(agg_kind);
  auto result = cudf::grouped_rolling_window(group_keys.cached_view(), col, default_outputs, preceding, following, min_periods, *agg, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Percentile approx --

std::unique_ptr<Column> percentile_approx(cudf::column_view const& tdigest_col, cudf::column_view const& percentiles, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::tdigest::tdigest_column_view tdv(tdigest_col);
  auto result = cudf::percentile_approx(tdv, percentiles, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Datetime: add months with scalar --

std::unique_ptr<Column> datetime_add_months_scalar(cudf::column_view const& timestamps, Scalar const& months, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::add_calendrical_months(timestamps, months.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Table nested column queries --

bool table_has_nested_columns(Table const& tbl) {
  return cudf::has_nested_columns(tbl.cached_view());
}

bool table_has_nested_nulls(Table const& tbl) {
  return cudf::has_nested_nulls(tbl.cached_view());
}

bool table_has_nested_nullable_columns(Table const& tbl) {
  return cudf::has_nested_nullable_columns(tbl.cached_view());
}

// -- Column with null mask from bools --

std::unique_ptr<Column> column_with_null_mask_from_bools(Column const& col, cudf::column_view const& validity, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  // Convert bool column to bitmask
  auto [null_mask, null_count] = cudf::bools_to_mask(validity, s);
  // Deep-copy the source column
  auto result = std::make_unique<cudf::column>(col.cached_view(), s);
  result->set_null_mask(std::move(*null_mask), null_count);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
