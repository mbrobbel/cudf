// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace cudf_sys {

// TypeId is a CXX shared enum — defined in the generated header.
enum class TypeId : ::std::int32_t;

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

}  // namespace cudf_sys
