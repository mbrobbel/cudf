// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/quantile.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/quantiles.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>

namespace cudf_sys {

std::unique_ptr<Column> quantile_column(
    cudf::column_view const& col,
    rust::Slice<double const> quantiles,
    int32_t interp,
    std::size_t stream) {
  std::vector<double> q(quantiles.begin(), quantiles.end());
  return COL(cudf::quantile(col, q, ENUM<cudf::interpolation>(interp), cudf::column_view{}, true, S(stream)));
}

std::unique_ptr<Table> quantiles_table(
    Table const& tbl,
    rust::Slice<double const> quantiles,
    int32_t interp,
    bool is_input_sorted,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  std::vector<double> q(quantiles.begin(), quantiles.end());
  auto sorted = is_input_sorted ? cudf::sorted::YES : cudf::sorted::NO;
  return TBL(cudf::quantiles(tbl.cached_view(), q,
      ENUM<cudf::interpolation>(interp), sorted,
      ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Column> percentile_approx(cudf::column_view const& tdigest_col, cudf::column_view const& percentiles, std::size_t stream) {
  return COL(cudf::percentile_approx(cudf::tdigest::tdigest_column_view(tdigest_col), percentiles, S(stream)));
}

}  // namespace cudf_sys
