// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/join.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/copying.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/filtered_join.hpp>

namespace cudf_sys {

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

std::unique_ptr<Table> cross_join(Table const& left, Table const& right, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::cross_join(left.cached_view(), right.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

}  // namespace cudf_sys
