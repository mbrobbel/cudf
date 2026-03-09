// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/lists.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

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
#include <cudf/lists/gather.hpp>

namespace cudf_sys {

std::unique_ptr<Column> lists_count_elements(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::lists::count_elements(cudf::lists_column_view(col), S(stream)));
}

std::unique_ptr<Column> lists_extract_element(cudf::column_view const& col, int32_t index, std::size_t stream) {
  return COL(cudf::lists::extract_list_element(cudf::lists_column_view(col), index, S(stream)));
}

std::unique_ptr<Column> lists_sort(cudf::column_view const& col, bool ascending, bool nulls_last, std::size_t stream) {
  auto order = ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING;
  auto null_order = nulls_last ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  return COL(cudf::lists::sort_lists(cudf::lists_column_view(col), order, null_order, S(stream)));
}

std::unique_ptr<Column> lists_reverse(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::lists::reverse(cudf::lists_column_view(col), S(stream)));
}

std::unique_ptr<Column> lists_contains_nulls(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::lists::contains_nulls(cudf::lists_column_view(col), S(stream)));
}

std::unique_ptr<Column> lists_distinct(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::lists::distinct(cudf::lists_column_view(col), cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, cudf::duplicate_keep_option::KEEP_ANY, S(stream)));
}

std::unique_ptr<Column> lists_concatenate_elements(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::lists::concatenate_list_elements(col, cudf::lists::concatenate_null_policy::IGNORE, S(stream)));
}

std::unique_ptr<Column> lists_sequences(cudf::column_view const& starts, cudf::column_view const& sizes, std::size_t stream) {
  return COL(cudf::lists::sequences(starts, sizes, S(stream)));
}

// -- List set operations --

std::unique_ptr<Column> lists_have_overlap(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  return COL(cudf::lists::have_overlap(cudf::lists_column_view(lhs), cudf::lists_column_view(rhs), cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, S(stream)));
}

std::unique_ptr<Column> lists_intersect_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  return COL(cudf::lists::intersect_distinct(cudf::lists_column_view(lhs), cudf::lists_column_view(rhs), cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, S(stream)));
}

std::unique_ptr<Column> lists_union_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  return COL(cudf::lists::union_distinct(cudf::lists_column_view(lhs), cudf::lists_column_view(rhs), cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, S(stream)));
}

std::unique_ptr<Column> lists_difference_distinct(cudf::column_view const& lhs, cudf::column_view const& rhs, std::size_t stream) {
  return COL(cudf::lists::difference_distinct(cudf::lists_column_view(lhs), cudf::lists_column_view(rhs), cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL, S(stream)));
}

// -- Lists contains / index_of / segmented_gather --

std::unique_ptr<Column> lists_contains_scalar(cudf::column_view const& col, Scalar const& search_key, std::size_t stream) {
  return COL(cudf::lists::contains(cudf::lists_column_view(col), search_key.inner(), S(stream)));
}

std::unique_ptr<Column> lists_contains_column(cudf::column_view const& col, cudf::column_view const& search_keys, std::size_t stream) {
  return COL(cudf::lists::contains(cudf::lists_column_view(col), search_keys, S(stream)));
}

std::unique_ptr<Column> lists_index_of_scalar(cudf::column_view const& col, Scalar const& search_key, bool find_first, std::size_t stream) {
  auto opt = find_first ? cudf::lists::duplicate_find_option::FIND_FIRST
                        : cudf::lists::duplicate_find_option::FIND_LAST;
  return COL(cudf::lists::index_of(cudf::lists_column_view(col), search_key.inner(), opt, S(stream)));
}

std::unique_ptr<Column> lists_index_of_column(cudf::column_view const& col, cudf::column_view const& search_keys, bool find_first, std::size_t stream) {
  auto opt = find_first ? cudf::lists::duplicate_find_option::FIND_FIRST
                        : cudf::lists::duplicate_find_option::FIND_LAST;
  return COL(cudf::lists::index_of(cudf::lists_column_view(col), search_keys, opt, S(stream)));
}

std::unique_ptr<Column> lists_segmented_gather(cudf::column_view const& col, cudf::column_view const& gather_map, bool nullify_oob, std::size_t stream) {
  auto policy = nullify_oob ? cudf::out_of_bounds_policy::NULLIFY
                            : cudf::out_of_bounds_policy::DONT_CHECK;
  return COL(cudf::lists::segmented_gather(cudf::lists_column_view(col), cudf::lists_column_view(gather_map), policy, S(stream)));
}

// -- Lists: apply_boolean_mask --

std::unique_ptr<Column> lists_apply_boolean_mask(
    cudf::column_view const& col,
    cudf::column_view const& boolean_mask,
    std::size_t stream) {
  return COL(cudf::lists::apply_boolean_mask(cudf::lists_column_view(col), cudf::lists_column_view(boolean_mask), S(stream)));
}

// -- Lists: extract_element (column), stable_sort, concatenate_rows --

std::unique_ptr<Column> lists_extract_element_column(
    cudf::column_view const& col,
    cudf::column_view const& indices,
    std::size_t stream) {
  return COL(cudf::lists::extract_list_element(cudf::lists_column_view(col), indices, S(stream)));
}

std::unique_ptr<Column> lists_stable_sort(
    cudf::column_view const& col,
    bool ascending,
    bool nulls_last,
    std::size_t stream) {
  auto order = ascending ? cudf::order::ASCENDING : cudf::order::DESCENDING;
  auto null_order = nulls_last ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  return COL(cudf::lists::stable_sort_lists(cudf::lists_column_view(col), order, null_order, S(stream)));
}

std::unique_ptr<Column> lists_concatenate_rows(
    Table const& tbl,
    std::size_t stream) {
  return COL(cudf::lists::concatenate_rows(
      tbl.cached_view(), cudf::lists::concatenate_null_policy::IGNORE, S(stream)));
}

// -- Lists: sequences with step --

std::unique_ptr<Column> lists_sequences_with_step(
    cudf::column_view const& starts,
    cudf::column_view const& steps,
    cudf::column_view const& sizes,
    std::size_t stream) {
  return COL(cudf::lists::sequences(starts, steps, sizes, S(stream)));
}

// -- Explode --

std::unique_ptr<Table> explode_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  return TBL(cudf::explode(tbl.cached_view(), column_idx, S(stream)));
}

std::unique_ptr<Table> explode_position_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  return TBL(cudf::explode_position(tbl.cached_view(), column_idx, S(stream)));
}

std::unique_ptr<Table> explode_outer_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  return TBL(cudf::explode_outer(tbl.cached_view(), column_idx, S(stream)));
}

std::unique_ptr<Table> explode_outer_position_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  return TBL(cudf::explode_outer_position(tbl.cached_view(), column_idx, S(stream)));
}

}  // namespace cudf_sys
