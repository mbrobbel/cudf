// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/rolling.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
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

std::unique_ptr<Column> rolling_window(cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  return COL(cudf::rolling_window(col, preceding, following, min_periods, *make_rolling_agg(agg_kind), S(stream)));
}

std::unique_ptr<Column> grouped_rolling_window(Table const& group_keys, cudf::column_view const& col, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  return COL(cudf::grouped_rolling_window(group_keys.cached_view(), col, preceding, following, min_periods, *make_rolling_agg(agg_kind), S(stream)));
}

std::unique_ptr<Column> rolling_window_with_defaults(
    cudf::column_view const& col,
    cudf::column_view const& default_outputs,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind,
    std::size_t stream) {
  return COL(cudf::rolling_window(col, default_outputs, preceding, following, min_periods,
      *make_rolling_agg(agg_kind), S(stream)));
}

std::unique_ptr<Column> grouped_rolling_window_with_defaults(Table const& group_keys, cudf::column_view const& col, cudf::column_view const& default_outputs, int32_t preceding, int32_t following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  return COL(cudf::grouped_rolling_window(group_keys.cached_view(), col, default_outputs, preceding, following, min_periods, *make_rolling_agg(agg_kind), S(stream)));
}

std::unique_ptr<RangeWindowBounds> range_window_bounds_get(Scalar const& boundary, std::size_t stream) {
  return std::make_unique<RangeWindowBounds>(cudf::range_window_bounds::get(boundary.inner(), S(stream)));
}

std::unique_ptr<RangeWindowBounds> range_window_bounds_current_row(int32_t type_id, std::size_t stream) {
  return std::make_unique<RangeWindowBounds>(cudf::range_window_bounds::current_row(DT(type_id), S(stream)));
}

std::unique_ptr<RangeWindowBounds> range_window_bounds_unbounded(int32_t type_id, std::size_t stream) {
  return std::make_unique<RangeWindowBounds>(cudf::range_window_bounds::unbounded(DT(type_id), S(stream)));
}

std::unique_ptr<Column> grouped_range_rolling_window(Table const& group_keys, cudf::column_view const& orderby, int32_t order, cudf::column_view const& input, RangeWindowBounds const& preceding, RangeWindowBounds const& following, int32_t min_periods, int32_t agg_kind, std::size_t stream) {
  return COL(cudf::grouped_range_rolling_window(group_keys.cached_view(), orderby,
      ENUM<cudf::order>(order), input,
      preceding.inner(), following.inner(),
      min_periods, *make_rolling_agg(agg_kind), S(stream)));
}

}  // namespace cudf_sys
