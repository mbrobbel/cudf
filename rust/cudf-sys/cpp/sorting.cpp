// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/sorting.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/sorting.hpp>

namespace cudf_sys {

std::unique_ptr<Table> sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return TBL(cudf::sort(tbl.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Column> sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return COL(cudf::sorted_order(tbl.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

bool is_sorted_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return cudf::is_sorted(tbl.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream));
}

std::unique_ptr<Column> stable_sorted_order(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return COL(cudf::stable_sorted_order(tbl.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Table> stable_sort_table(
    Table const& tbl,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return TBL(cudf::stable_sort(tbl.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Table> sort_by_key(
    Table const& values,
    Table const& keys,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return TBL(cudf::sort_by_key(values.cached_view(), keys.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Table> stable_sort_by_key(
    Table const& values,
    Table const& keys,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return TBL(cudf::stable_sort_by_key(values.cached_view(), keys.cached_view(), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Column> rank_column(
    cudf::column_view const& col,
    int32_t method,
    int32_t column_order,
    int32_t null_handling,
    int32_t null_precedence,
    bool percentage,
    std::size_t stream) {
  return COL(cudf::rank(col,
      ENUM<cudf::rank_method>(method),
      ENUM<cudf::order>(column_order),
      ENUM<cudf::null_policy>(null_handling),
      ENUM<cudf::null_order>(null_precedence),
      percentage, S(stream)));
}

std::unique_ptr<Column> top_k(cudf::column_view const& col, int32_t k, int32_t order, std::size_t stream) {
  return COL(cudf::top_k(col, k, ENUM<cudf::order>(order), S(stream)));
}

std::unique_ptr<Column> top_k_order(cudf::column_view const& col, int32_t k, int32_t order, std::size_t stream) {
  return COL(cudf::top_k_order(col, k, ENUM<cudf::order>(order), S(stream)));
}

std::unique_ptr<Column> segmented_sorted_order(
    Table const& tbl,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return COL(cudf::segmented_sorted_order(tbl.cached_view(), segment_offsets, ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Column> segmented_top_k(cudf::column_view const& col, cudf::column_view const& segment_offsets, int32_t k, int32_t order, std::size_t stream) {
  return COL(cudf::segmented_top_k(col, segment_offsets, k, ENUM<cudf::order>(order), S(stream)));
}

std::unique_ptr<Column> segmented_top_k_order(cudf::column_view const& col, cudf::column_view const& segment_offsets, int32_t k, int32_t order, std::size_t stream) {
  return COL(cudf::segmented_top_k_order(col, segment_offsets, k, ENUM<cudf::order>(order), S(stream)));
}

std::unique_ptr<Column> stable_segmented_sorted_order(
    Table const& tbl,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return COL(cudf::stable_segmented_sorted_order(
      tbl.cached_view(), segment_offsets,
      ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Table> stable_segmented_sort_by_key(
    Table const& values,
    Table const& keys,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return TBL(cudf::stable_segmented_sort_by_key(
      values.cached_view(), keys.cached_view(), segment_offsets,
      ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Table> segmented_sort_by_key(
    Table const& values,
    Table const& keys,
    cudf::column_view const& segment_offsets,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return TBL(cudf::segmented_sort_by_key(
      values.cached_view(), keys.cached_view(), segment_offsets,
      ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

}  // namespace cudf_sys
