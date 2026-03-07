// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/reshape.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/reshape.hpp>
#include <cudf/transpose.hpp>

namespace cudf_sys {

// -- Reshape --

std::unique_ptr<Column> interleave_columns(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::interleave_columns(tbl.cached_view(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> tile_table(Table const& tbl, int32_t count, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::tile(tbl.cached_view(), count, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Reshape (new) --

std::unique_ptr<Column> byte_cast_column(cudf::column_view const& col, bool flip_endian, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto endian = flip_endian ? cudf::flip_endianness::YES : cudf::flip_endianness::NO;
  auto result = cudf::byte_cast(col, endian, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Transpose --

std::unique_ptr<Table> transpose_table(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_col, result_view] = cudf::transpose(tbl.cached_view(), s);
  // result_col owns the data, result_view is a table_view into it
  // We need to build a table from the transposed view
  auto result = std::make_unique<cudf::table>(result_view, s);
  return std::make_unique<Table>(std::move(result));
}

}  // namespace cudf_sys
