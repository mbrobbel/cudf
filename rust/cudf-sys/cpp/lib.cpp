// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cudf_sys {

// Helper: construct a data_type, only passing scale for decimal types.
static cudf::data_type make_data_type(cudf::type_id tid, int32_t scale) {
  if (tid == cudf::type_id::DECIMAL32 ||
      tid == cudf::type_id::DECIMAL64 ||
      tid == cudf::type_id::DECIMAL128) {
    return cudf::data_type{tid, scale};
  }
  return cudf::data_type{tid};
}

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
  return TBL(std::make_unique<cudf::table>(
      std::vector<std::unique_ptr<cudf::column>>{}));
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
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<int32_t>>(value, valid));
}

std::unique_ptr<Scalar> make_int64_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<int64_t>>(value, valid));
}

std::unique_ptr<Scalar> make_float32_scalar(float value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<float>>(value, valid));
}

std::unique_ptr<Scalar> make_float64_scalar(double value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<double>>(value, valid));
}

std::unique_ptr<Scalar> make_bool_scalar(bool value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<bool>>(value, valid));
}

std::unique_ptr<Scalar> make_string_scalar(rust::Str value) {
  auto sv = std::string_view(value.data(), value.size());
  return std::make_unique<Scalar>(std::make_unique<cudf::string_scalar>(sv, true));
}

std::unique_ptr<Scalar> make_int8_scalar(int8_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<int8_t>>(value, valid));
}

std::unique_ptr<Scalar> make_int16_scalar(int16_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<int16_t>>(value, valid));
}

std::unique_ptr<Scalar> make_uint8_scalar(uint8_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<uint8_t>>(value, valid));
}

std::unique_ptr<Scalar> make_uint16_scalar(uint16_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<uint16_t>>(value, valid));
}

std::unique_ptr<Scalar> make_uint32_scalar(uint32_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<uint32_t>>(value, valid));
}

std::unique_ptr<Scalar> make_uint64_scalar(uint64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::numeric_scalar<uint64_t>>(value, valid));
}

std::unique_ptr<Scalar> make_timestamp_s_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
      cudf::timestamp_s{cudf::duration_s{value}}, valid));
}

std::unique_ptr<Scalar> make_timestamp_ms_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
      cudf::timestamp_ms{cudf::duration_ms{value}}, valid));
}

std::unique_ptr<Scalar> make_timestamp_us_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
      cudf::timestamp_us{cudf::duration_us{value}}, valid));
}

std::unique_ptr<Scalar> make_timestamp_ns_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
      cudf::timestamp_ns{cudf::duration_ns{value}}, valid));
}

std::unique_ptr<Scalar> make_duration_s_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::duration_scalar<cudf::duration_s>>(
      cudf::duration_s{value}, valid));
}

std::unique_ptr<Scalar> make_duration_ms_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::duration_scalar<cudf::duration_ms>>(
      cudf::duration_ms{value}, valid));
}

std::unique_ptr<Scalar> make_duration_us_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::duration_scalar<cudf::duration_us>>(
      cudf::duration_us{value}, valid));
}

std::unique_ptr<Scalar> make_duration_ns_scalar(int64_t value, bool valid) {
  return std::make_unique<Scalar>(std::make_unique<cudf::duration_scalar<cudf::duration_ns>>(
      cudf::duration_ns{value}, valid));
}

std::unique_ptr<Scalar> make_default_constructed_scalar(int32_t type_id, int32_t scale) {
  auto dt = make_data_type(static_cast<cudf::type_id>(type_id), scale);
  return std::make_unique<Scalar>(cudf::make_default_constructed_scalar(dt));
}

std::unique_ptr<Scalar> make_empty_scalar_like(cudf::column_view const& col) {
  return std::make_unique<Scalar>(cudf::make_empty_scalar_like(col));
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

// -- ScalarList --

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

// -- Column factories --

std::unique_ptr<Column> make_column_from_scalar(Scalar const& s, int32_t count, std::size_t stream) {
  return COL(cudf::make_column_from_scalar(s.inner(), count, S(stream)));
}

std::unique_ptr<Column> make_empty_column_by_type(int32_t type_id) {
  return COL(cudf::make_empty_column(static_cast<cudf::type_id>(type_id)));
}

std::unique_ptr<Column> make_fixed_width_column(
    int32_t type_id,
    int32_t scale,
    int32_t num_rows,
    int32_t mask_state,
    std::size_t stream) {
  auto dt = make_data_type(static_cast<cudf::type_id>(type_id), scale);
  return COL(cudf::make_fixed_width_column(dt, num_rows, ENUM<cudf::mask_state>(mask_state), S(stream)));
}

std::unique_ptr<Column> make_empty_lists_column(
    int32_t child_type_id,
    std::size_t stream) {
  (void)stream;
  return COL(cudf::make_empty_lists_column(DT(child_type_id)));
}

std::unique_ptr<Column> make_dictionary_from_scalar(
    Scalar const& scalar,
    int32_t num_rows,
    std::size_t stream) {
  return COL(cudf::make_dictionary_from_scalar(scalar.inner(), num_rows, S(stream)));
}

std::unique_ptr<Column> make_lists_column(int32_t num_rows, std::unique_ptr<Column> offsets, std::unique_ptr<Column> child, std::size_t stream) {
  return COL(cudf::make_lists_column(
      num_rows, offsets->release(), child->release(), 0, rmm::device_buffer{}));
}

// StructColumnBuilder

void StructColumnBuilder::add_child(std::unique_ptr<Column> col) {
  children_.push_back(col->release());
}

std::unique_ptr<Column> StructColumnBuilder::build(int32_t num_rows, std::size_t stream) {
  return COL(cudf::make_structs_column(
      num_rows, std::move(children_), 0, rmm::device_buffer{}, S(stream)));
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

// -- Column data extraction (device -> host) --

template <typename T>
static rust::Vec<T> column_to_host(Column const& col, std::size_t stream) {
  auto s = S(stream);
  auto view = col.cached_view();
  auto size = view.size();
  std::vector<T> host(size);
  cudaMemcpyAsync(host.data(), view.data<T>(), size * sizeof(T), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<T> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v);
  return out;
}

rust::Vec<int8_t> column_to_host_i8(Column const& col, std::size_t stream) {
  return column_to_host<int8_t>(col, stream);
}
rust::Vec<int16_t> column_to_host_i16(Column const& col, std::size_t stream) {
  return column_to_host<int16_t>(col, stream);
}
rust::Vec<int32_t> column_to_host_i32(Column const& col, std::size_t stream) {
  return column_to_host<int32_t>(col, stream);
}
rust::Vec<int64_t> column_to_host_i64(Column const& col, std::size_t stream) {
  return column_to_host<int64_t>(col, stream);
}
rust::Vec<float> column_to_host_f32(Column const& col, std::size_t stream) {
  return column_to_host<float>(col, stream);
}
rust::Vec<double> column_to_host_f64(Column const& col, std::size_t stream) {
  return column_to_host<double>(col, stream);
}
rust::Vec<uint8_t> column_to_host_u8(Column const& col, std::size_t stream) {
  return column_to_host<uint8_t>(col, stream);
}
rust::Vec<uint16_t> column_to_host_u16(Column const& col, std::size_t stream) {
  return column_to_host<uint16_t>(col, stream);
}
rust::Vec<uint32_t> column_to_host_u32(Column const& col, std::size_t stream) {
  return column_to_host<uint32_t>(col, stream);
}
rust::Vec<uint64_t> column_to_host_u64(Column const& col, std::size_t stream) {
  return column_to_host<uint64_t>(col, stream);
}

rust::Vec<bool> column_to_host_bool(Column const& col, std::size_t stream) {
  auto s = S(stream);
  auto view = col.cached_view();
  auto size = view.size();
  // cudf BOOL8 uses int8_t internally
  std::vector<int8_t> host(size);
  cudaMemcpyAsync(host.data(), view.data<int8_t>(), size * sizeof(int8_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();
  rust::Vec<bool> out;
  out.reserve(size);
  for (auto v : host) out.push_back(v != 0);
  return out;
}

rust::Vec<bool> column_null_mask_to_host(Column const& col, std::size_t stream) {
  auto s = S(stream);
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<bool> result;
  result.reserve(size);

  if (!view.nullable()) {
    for (int32_t i = 0; i < size; ++i) result.push_back(true);
    return result;
  }

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

// -- Column factories from host data --

template <typename T>
static std::unique_ptr<Column> make_column_from_host(cudf::type_id tid, rust::Slice<T const> data, std::size_t stream) {
  auto s = S(stream);
  auto size = static_cast<cudf::size_type>(data.size());
  rmm::device_buffer buf(data.data(), size * sizeof(T), s);
  s.synchronize();
  return COL(std::make_unique<cudf::column>(
      cudf::data_type{tid}, size, std::move(buf), rmm::device_buffer{}, 0));
}

std::unique_ptr<Column> make_column_from_host_i8(rust::Slice<int8_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::INT8, data, stream);
}
std::unique_ptr<Column> make_column_from_host_i16(rust::Slice<int16_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::INT16, data, stream);
}
std::unique_ptr<Column> make_column_from_host_i32(rust::Slice<int32_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::INT32, data, stream);
}
std::unique_ptr<Column> make_column_from_host_i64(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::INT64, data, stream);
}
std::unique_ptr<Column> make_column_from_host_f32(rust::Slice<float const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::FLOAT32, data, stream);
}
std::unique_ptr<Column> make_column_from_host_f64(rust::Slice<double const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::FLOAT64, data, stream);
}
std::unique_ptr<Column> make_column_from_host_u8(rust::Slice<uint8_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::UINT8, data, stream);
}
std::unique_ptr<Column> make_column_from_host_u16(rust::Slice<uint16_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::UINT16, data, stream);
}
std::unique_ptr<Column> make_column_from_host_u32(rust::Slice<uint32_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::UINT32, data, stream);
}
std::unique_ptr<Column> make_column_from_host_u64(rust::Slice<uint64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::UINT64, data, stream);
}
std::unique_ptr<Column> make_column_from_host_timestamp_s(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::TIMESTAMP_SECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_timestamp_ms(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::TIMESTAMP_MILLISECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_timestamp_us(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::TIMESTAMP_MICROSECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_timestamp_ns(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::TIMESTAMP_NANOSECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_duration_s(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::DURATION_SECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_duration_ms(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::DURATION_MILLISECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_duration_us(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::DURATION_MICROSECONDS, data, stream);
}
std::unique_ptr<Column> make_column_from_host_duration_ns(rust::Slice<int64_t const> data, std::size_t stream) {
  return make_column_from_host(cudf::type_id::DURATION_NANOSECONDS, data, stream);
}

std::unique_ptr<Column> make_column_from_host_bool(rust::Slice<bool const> data, std::size_t stream) {
  auto s = S(stream);
  auto size = static_cast<cudf::size_type>(data.size());
  std::vector<int8_t> int_data(size);
  for (cudf::size_type i = 0; i < size; ++i) int_data[i] = data[i] ? 1 : 0;
  rmm::device_buffer buf(int_data.data(), size * sizeof(int8_t), s);
  s.synchronize();
  return COL(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8}, size, std::move(buf), rmm::device_buffer{}, 0));
}

// -- TableBuilder --

void TableBuilder::add_column(std::unique_ptr<Column> wrapper) {
  cudf_columns_.push_back(wrapper->release());
}

std::unique_ptr<Table> TableBuilder::build() {
  auto tbl = std::make_unique<cudf::table>(std::move(cudf_columns_));
  cudf_columns_.clear();
  return TBL(std::move(tbl));
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

// -- Null mask utilities --

std::size_t bitmask_allocation_size_bytes(int32_t number_of_bits) {
  return cudf::bitmask_allocation_size_bytes(number_of_bits);
}

int32_t num_bitmask_words(int32_t number_of_bits) {
  return cudf::num_bitmask_words(number_of_bits);
}

int32_t state_null_count(int32_t mask_state, int32_t num_rows) {
  return cudf::state_null_count(ENUM<cudf::mask_state>(mask_state), num_rows);
}

// -- Null mask conversions --

std::unique_ptr<Column> null_mask_to_bools(cudf::column_view const& col, std::size_t stream) {
  auto s = S(stream);
  auto mask = col.null_mask();
  if (!mask) {
    cudf::numeric_scalar<bool> val(true, true, s);
    return COL(cudf::make_column_from_scalar(val, col.size(), s));
  }
  return COL(cudf::mask_to_bools(mask, col.offset(), col.offset() + col.size(), s));
}

void set_null_mask_from_bools(Column& col, cudf::column_view const& bools, std::size_t stream) {
  auto [mask_buf, null_count] = cudf::bools_to_mask(bools, S(stream));
  col.mutable_inner().set_null_mask(std::move(*mask_buf), null_count);
}

// -- Bitmask combining --

std::unique_ptr<Column> bitmask_and_to_bools(Table const& tbl, std::size_t stream) {
  auto s = S(stream);
  auto [mask, null_count] = cudf::bitmask_and(tbl.cached_view(), s);
  auto bitmask_ptr = static_cast<cudf::bitmask_type const*>(mask.data());
  return COL(cudf::mask_to_bools(bitmask_ptr, 0, tbl.cached_view().num_rows(), s));
}

std::unique_ptr<Column> bitmask_or_to_bools(Table const& tbl, std::size_t stream) {
  auto s = S(stream);
  auto [mask, null_count] = cudf::bitmask_or(tbl.cached_view(), s);
  auto bitmask_ptr = static_cast<cudf::bitmask_type const*>(mask.data());
  return COL(cudf::mask_to_bools(bitmask_ptr, 0, tbl.cached_view().num_rows(), s));
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
  auto from = make_data_type(static_cast<cudf::type_id>(from_type_id), from_scale);
  auto to = make_data_type(static_cast<cudf::type_id>(to_type_id), to_scale);
  return cudf::is_supported_cast(from, to);
}

// -- Column child access --

std::unique_ptr<Column> column_view_child_copy(cudf::column_view const& col, int32_t index, std::size_t stream) {
  return COL(std::make_unique<cudf::column>(col.child(index), S(stream)));
}

int32_t column_view_num_children(cudf::column_view const& col) {
  return col.num_children();
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
  auto s = S(stream);
  auto [null_mask, null_count] = cudf::bools_to_mask(validity, s);
  auto result = std::make_unique<cudf::column>(col.cached_view(), s);
  result->set_null_mask(std::move(*null_mask), null_count);
  return COL(std::move(result));
}

}  // namespace cudf_sys
