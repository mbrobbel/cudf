// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/search.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/search.hpp>

namespace cudf_sys {

// -- Search operations --

bool contains_scalar(
    cudf::column_view const& haystack,
    Scalar const& needle,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  return cudf::contains(haystack, needle.inner(), s);
}

std::unique_ptr<Column> contains_column(
    cudf::column_view const& haystack,
    cudf::column_view const& needles,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::contains(haystack, needles, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> lower_bound(
    Table const& haystack,
    Table const& needles,
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

  auto result = cudf::lower_bound(haystack.cached_view(), needles.cached_view(), orders, nulls, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> upper_bound(
    Table const& haystack,
    Table const& needles,
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

  auto result = cudf::upper_bound(haystack.cached_view(), needles.cached_view(), orders, nulls, s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
