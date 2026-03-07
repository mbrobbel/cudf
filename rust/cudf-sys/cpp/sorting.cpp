// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/sorting.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/sorting.hpp>

namespace cudf_sys {

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

// -- Sorting (new) --

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

}  // namespace cudf_sys
