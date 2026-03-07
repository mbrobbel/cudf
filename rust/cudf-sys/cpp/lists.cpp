// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/lists.rs.h"
#include "cudf-sys/src/lib.rs.h"

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

// -- Explode: outer_position --

std::unique_ptr<Table> explode_outer_position_table(Table const& tbl, int32_t column_idx, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::explode_outer_position(tbl.cached_view(), column_idx, s);
  return std::make_unique<Table>(std::move(result));
}

}  // namespace cudf_sys
