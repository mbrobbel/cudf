// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/labeling.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/labeling/label_bins.hpp>

namespace cudf_sys {

// -- Label bins --

std::unique_ptr<Column> label_bins_column(
    cudf::column_view const& col,
    cudf::column_view const& left_edges,
    int32_t left_inclusive,
    cudf::column_view const& right_edges,
    int32_t right_inclusive,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::label_bins(col, left_edges,
      static_cast<cudf::inclusive>(left_inclusive),
      right_edges,
      static_cast<cudf::inclusive>(right_inclusive), s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
