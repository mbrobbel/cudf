// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/lib.rs.h"

#include <cudf/types.hpp>

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

}  // namespace cudf_sys
