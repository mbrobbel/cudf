// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/search.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/search.hpp>

namespace cudf_sys {

bool contains_scalar(
    cudf::column_view const& haystack,
    Scalar const& needle,
    std::size_t stream) {
  return cudf::contains(haystack, needle.inner(), S(stream));
}

std::unique_ptr<Column> contains_column(
    cudf::column_view const& haystack,
    cudf::column_view const& needles,
    std::size_t stream) {
  return COL(cudf::contains(haystack, needles, S(stream)));
}

std::unique_ptr<Column> lower_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return COL(cudf::lower_bound(haystack.cached_view(), needles.cached_view(),
      ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

std::unique_ptr<Column> upper_bound(
    Table const& haystack,
    Table const& needles,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  return COL(cudf::upper_bound(haystack.cached_view(), needles.cached_view(),
      ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

}  // namespace cudf_sys
