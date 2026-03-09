// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/reshape.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/reshape.hpp>
#include <cudf/transpose.hpp>

namespace cudf_sys {

std::unique_ptr<Column> interleave_columns(Table const& tbl, std::size_t stream) {
  return COL(cudf::interleave_columns(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Table> tile_table(Table const& tbl, int32_t count, std::size_t stream) {
  return TBL(cudf::tile(tbl.cached_view(), count, S(stream)));
}

std::unique_ptr<Column> byte_cast_column(cudf::column_view const& col, bool flip_endian, std::size_t stream) {
  auto endian = flip_endian ? cudf::flip_endianness::YES : cudf::flip_endianness::NO;
  return COL(cudf::byte_cast(col, endian, S(stream)));
}

std::unique_ptr<Table> transpose_table(Table const& tbl, std::size_t stream) {
  auto s = S(stream);
  auto [result_col, result_view] = cudf::transpose(tbl.cached_view(), s);
  return TBL(std::make_unique<cudf::table>(result_view, s));
}

}  // namespace cudf_sys
