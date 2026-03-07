// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/quantile.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/quantiles.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>

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

// -- Quantile operations --

std::unique_ptr<Column> quantile_column(
    cudf::column_view const& col,
    rust::Slice<double const> quantiles,
    int32_t interp,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<double> q(quantiles.begin(), quantiles.end());
  auto result = cudf::quantile(col, q, static_cast<cudf::interpolation>(interp), cudf::column_view{}, true, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Quantiles (new) --

std::unique_ptr<Table> quantiles_table(
    Table const& tbl,
    rust::Slice<double const> quantiles,
    int32_t interp,
    bool is_input_sorted,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<double> q(quantiles.begin(), quantiles.end());
  auto sorted = is_input_sorted ? cudf::sorted::YES : cudf::sorted::NO;
  auto result = cudf::quantiles(
      tbl.cached_view(),
      q,
      static_cast<cudf::interpolation>(interp),
      sorted,
      to_orders(column_orders),
      to_null_orders(null_orders),
      s);
  return std::make_unique<Table>(std::move(result));
}

// -- Percentile approx --

std::unique_ptr<Column> percentile_approx(cudf::column_view const& tdigest_col, cudf::column_view const& percentiles, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::tdigest::tdigest_column_view tdv(tdigest_col);
  auto result = cudf::percentile_approx(tdv, percentiles, s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
