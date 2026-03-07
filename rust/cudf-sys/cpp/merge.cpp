// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/merge.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/merge.hpp>

namespace cudf_sys {

// -- Merge --

std::unique_ptr<Table> merge_tables(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::table_view> views = {left.cached_view(), right.cached_view()};

  std::vector<cudf::size_type> keys;
  keys.reserve(key_indices.size());
  for (auto k : key_indices) keys.push_back(k);

  std::vector<cudf::order> orders;
  orders.reserve(column_orders.size());
  for (auto o : column_orders) orders.push_back(static_cast<cudf::order>(o));

  std::vector<cudf::null_order> nulls;
  nulls.reserve(null_orders.size());
  for (auto n : null_orders) nulls.push_back(static_cast<cudf::null_order>(n));

  auto result = cudf::merge(views, keys, orders, nulls, s);
  return std::make_unique<Table>(std::move(result));
}

}  // namespace cudf_sys
