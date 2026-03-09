// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/labeling.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/labeling/label_bins.hpp>

namespace cudf_sys {

std::unique_ptr<Column> label_bins_column(
    cudf::column_view const& col,
    cudf::column_view const& left_edges,
    int32_t left_inclusive,
    cudf::column_view const& right_edges,
    int32_t right_inclusive,
    std::size_t stream) {
  return COL(cudf::label_bins(col, left_edges,
      ENUM<cudf::inclusive>(left_inclusive),
      right_edges,
      ENUM<cudf::inclusive>(right_inclusive), S(stream)));
}

}  // namespace cudf_sys
