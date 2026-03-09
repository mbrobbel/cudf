// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/concatenate.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/concatenate.hpp>

namespace cudf_sys {

void ColumnConcatenator::add(cudf::column_view const& v) {
  views_.push_back(v);
}

std::unique_ptr<Column> ColumnConcatenator::finish(std::size_t stream) {
  auto result = COL(cudf::concatenate(views_, S(stream)));
  views_.clear();
  return result;
}

void TableConcatenator::add_table(Table const& t) {
  views_.push_back(t.cached_view());
}

std::unique_ptr<Table> TableConcatenator::finish(std::size_t stream) {
  auto result = TBL(cudf::concatenate(views_, S(stream)));
  views_.clear();
  return result;
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

std::unique_ptr<Column> concatenate_columns(Table const& tbl, std::size_t stream) {
  auto view = tbl.cached_view();
  std::vector<cudf::column_view> columns;
  columns.reserve(view.num_columns());
  for (int i = 0; i < view.num_columns(); ++i) {
    columns.push_back(view.column(i));
  }
  return COL(cudf::concatenate(columns, S(stream)));
}

std::unique_ptr<Table> concatenate_tables(Table const& lhs, Table const& rhs, std::size_t stream) {
  std::vector<cudf::table_view> views{lhs.cached_view(), rhs.cached_view()};
  return TBL(cudf::concatenate(views, S(stream)));
}

}  // namespace cudf_sys
