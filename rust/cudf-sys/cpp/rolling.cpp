// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/rolling.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/rolling.hpp>
#include <cudf/aggregation.hpp>

namespace cudf_sys {

static std::unique_ptr<cudf::rolling_aggregation> make_rolling_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::rolling_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::rolling_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::rolling_aggregation>();
    case 12: return cudf::make_argmax_aggregation<cudf::rolling_aggregation>();
    case 13: return cudf::make_argmin_aggregation<cudf::rolling_aggregation>();
    case 14: return cudf::make_collect_list_aggregation<cudf::rolling_aggregation>();
    case 15: return cudf::make_collect_set_aggregation<cudf::rolling_aggregation>();
    default:
      throw std::invalid_argument("Unsupported rolling aggregation kind: " + std::to_string(kind));
  }
}

// -- Rolling window --

std::unique_ptr<Column> rolling_window(cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto agg = make_rolling_agg(agg_kind);
  auto result = cudf::rolling_window(col, preceding, following, min_periods, *agg, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> grouped_rolling_window(Table const& group_keys, cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto agg = make_rolling_agg(agg_kind);
  auto result = cudf::grouped_rolling_window(group_keys.cached_view(), col, preceding, following, min_periods, *agg, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Rolling window with defaults --

std::unique_ptr<Column> rolling_window_with_defaults(
    cudf::column_view const& col,
    cudf::column_view const& default_outputs,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::rolling_window(col, default_outputs, preceding, following, min_periods,
      *make_rolling_agg(agg_kind), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Grouped rolling window with defaults --

std::unique_ptr<Column> grouped_rolling_window_with_defaults(Table const& group_keys, cudf::column_view const& col, cudf::column_view const& default_outputs, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto agg = make_rolling_agg(agg_kind);
  auto result = cudf::grouped_rolling_window(group_keys.cached_view(), col, default_outputs, preceding, following, min_periods, *agg, s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
