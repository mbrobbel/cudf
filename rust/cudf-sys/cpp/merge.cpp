// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/merge.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/merge.hpp>

namespace cudf_sys {

std::unique_ptr<Table> merge_tables(
    Table const& left, Table const& right,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> column_orders,
    rust::Slice<int32_t const> null_orders,
    std::size_t stream) {
  std::vector<cudf::table_view> views = {left.cached_view(), right.cached_view()};
  return TBL(cudf::merge(views, VEC(key_indices), ORDERS(column_orders), NULL_ORDERS(null_orders), S(stream)));
}

}  // namespace cudf_sys
