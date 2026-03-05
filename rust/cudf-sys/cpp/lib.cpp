// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/lib.rs.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cudf_sys {

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

cudf::column_view const& column_view_of(Column const& col) {
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

std::unique_ptr<Column> make_column_from_scalar(Scalar const& s, int32_t count) {
  auto col = cudf::make_column_from_scalar(s.inner(), count);
  return std::make_unique<Column>(std::move(col));
}

std::unique_ptr<Column> make_empty_column_by_type(int32_t type_id) {
  auto col = cudf::make_empty_column(static_cast<cudf::type_id>(type_id));
  return std::make_unique<Column>(std::move(col));
}

// -- Column data extraction (device → host) --

rust::Vec<int32_t> column_to_host_i32(Column const& col) {
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<int32_t> result;
  result.reserve(size);
  std::vector<int32_t> host_data(size);
  cudaMemcpy(host_data.data(), view.data<int32_t>(),
             size * sizeof(int32_t), cudaMemcpyDeviceToHost);
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<int64_t> column_to_host_i64(Column const& col) {
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<int64_t> result;
  result.reserve(size);
  std::vector<int64_t> host_data(size);
  cudaMemcpy(host_data.data(), view.data<int64_t>(),
             size * sizeof(int64_t), cudaMemcpyDeviceToHost);
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<float> column_to_host_f32(Column const& col) {
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<float> result;
  result.reserve(size);
  std::vector<float> host_data(size);
  cudaMemcpy(host_data.data(), view.data<float>(),
             size * sizeof(float), cudaMemcpyDeviceToHost);
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<double> column_to_host_f64(Column const& col) {
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<double> result;
  result.reserve(size);
  std::vector<double> host_data(size);
  cudaMemcpy(host_data.data(), view.data<double>(),
             size * sizeof(double), cudaMemcpyDeviceToHost);
  for (auto v : host_data) result.push_back(v);
  return result;
}

rust::Vec<bool> column_to_host_bool(Column const& col) {
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<bool> result;
  result.reserve(size);
  // cudf BOOL8 uses int8_t internally
  std::vector<int8_t> host_data(size);
  cudaMemcpy(host_data.data(), view.data<int8_t>(),
             size * sizeof(int8_t), cudaMemcpyDeviceToHost);
  for (auto v : host_data) result.push_back(v != 0);
  return result;
}

rust::Vec<bool> column_null_mask_to_host(Column const& col) {
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
  cudaMemcpy(host_mask.data(), view.null_mask(),
             num_bitmask_words * sizeof(cudf::bitmask_type), cudaMemcpyDeviceToHost);

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
    int32_t output_type_id) {
  auto result = cudf::binary_operation(
      lhs, rhs,
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> binary_operation_column_scalar(
    cudf::column_view const& lhs,
    Scalar const& rhs,
    BinaryOperator op,
    int32_t output_type_id) {
  auto result = cudf::binary_operation(
      lhs, rhs.inner(),
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> binary_operation_scalar_column(
    Scalar const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id) {
  auto result = cudf::binary_operation(
      lhs.inner(), rhs,
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Column>(std::move(result));
}

// -- Unary operations --

std::unique_ptr<Column> unary_cast(cudf::column_view const& col, int32_t target_type_id) {
  auto result = cudf::cast(col, cudf::data_type{static_cast<cudf::type_id>(target_type_id)});
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_null(cudf::column_view const& col) {
  auto result = cudf::is_null(col);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_valid(cudf::column_view const& col) {
  auto result = cudf::is_valid(col);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_nan(cudf::column_view const& col) {
  auto result = cudf::is_nan(col);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_negate(cudf::column_view const& col) {
  auto result = cudf::unary_operation(col, cudf::unary_operator::NEGATE);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_abs(cudf::column_view const& col) {
  auto result = cudf::unary_operation(col, cudf::unary_operator::ABS);
  return std::make_unique<Column>(std::move(result));
}

// -- Reduction --

std::unique_ptr<Scalar> reduce_sum(cudf::column_view const& col, int32_t output_type_id) {
  auto result = cudf::reduce(
      col, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_min(cudf::column_view const& col, int32_t output_type_id) {
  auto result = cudf::reduce(
      col, *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_max(cudf::column_view const& col, int32_t output_type_id) {
  auto result = cudf::reduce(
      col, *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_product(cudf::column_view const& col, int32_t output_type_id) {
  auto result = cudf::reduce(
      col, *cudf::make_product_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)});
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_any(cudf::column_view const& col) {
  auto result = cudf::reduce(
      col, *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8});
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_all(cudf::column_view const& col) {
  auto result = cudf::reduce(
      col, *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8});
  return std::make_unique<Scalar>(std::move(result));
}

// -- Sorting --

std::unique_ptr<Table> sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders) {
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::sort(tbl.cached_view(), orders, nulls);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders) {
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::sorted_order(tbl.cached_view(), orders, nulls);
  return std::make_unique<Column>(std::move(result));
}

bool is_sorted_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders) {
  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  return cudf::is_sorted(tbl.cached_view(), orders, nulls);
}

// -- Filtering --

std::unique_ptr<Table> apply_boolean_mask(
    Table const& tbl,
    cudf::column_view const& mask) {
  auto result = cudf::apply_boolean_mask(tbl.cached_view(), mask);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> drop_nulls_all(Table const& tbl) {
  auto view = tbl.cached_view();
  auto num_cols = view.num_columns();
  std::vector<cudf::size_type> keys(num_cols);
  for (int32_t i = 0; i < num_cols; ++i) keys[i] = i;
  auto result = cudf::drop_nulls(view, keys);
  return std::make_unique<Table>(std::move(result));
}

// -- Concatenation --

void ColumnConcatenator::add(cudf::column_view const& v) {
  views_.push_back(v);
}

std::unique_ptr<Column> ColumnConcatenator::finish() {
  auto result = cudf::concatenate(views_);
  views_.clear();
  return std::make_unique<Column>(std::move(result));
}

void TableConcatenator::add_table(Table const& t) {
  views_.push_back(t.cached_view());
}

std::unique_ptr<Table> TableConcatenator::finish() {
  auto result = cudf::concatenate(views_);
  views_.clear();
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<ColumnConcatenator> new_column_concatenator() {
  return std::make_unique<ColumnConcatenator>();
}

void column_concatenator_add(ColumnConcatenator& cat, cudf::column_view const& v) {
  cat.add(v);
}

std::unique_ptr<Column> column_concatenator_finish(ColumnConcatenator& cat) {
  return cat.finish();
}

std::unique_ptr<TableConcatenator> new_table_concatenator() {
  return std::make_unique<TableConcatenator>();
}

void table_concatenator_add(TableConcatenator& cat, Table const& t) {
  cat.add_table(t);
}

std::unique_ptr<Table> table_concatenator_finish(TableConcatenator& cat) {
  return cat.finish();
}

// -- Copying --

std::unique_ptr<Table> gather_table(
    Table const& tbl,
    cudf::column_view const& indices) {
  auto result = cudf::gather(tbl.cached_view(), indices);
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

}  // namespace cudf_sys
