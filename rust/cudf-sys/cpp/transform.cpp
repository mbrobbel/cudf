// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/transform.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/transform.hpp>

namespace cudf_sys {

std::unique_ptr<Column> nans_to_nulls(cudf::column_view const& col, std::size_t stream) {
  auto s = S(stream);
  auto [null_mask, null_count] = cudf::nans_to_nulls(col, s);
  auto result = std::make_unique<cudf::column>(col, s);
  result->set_null_mask(std::move(*null_mask), null_count);
  return COL(std::move(result));
}

std::unique_ptr<Column> encode_table(Table const& tbl, std::size_t stream) {
  auto [keys, indices] = cudf::encode(tbl.cached_view(), S(stream));
  return COL(std::move(indices));
}

std::unique_ptr<Table> encode_keys(Table const& tbl, std::size_t stream) {
  auto [keys, indices] = cudf::encode(tbl.cached_view(), S(stream));
  return TBL(std::move(keys));
}

std::unique_ptr<Column> row_bit_count(Table const& tbl, std::size_t stream) {
  return COL(cudf::row_bit_count(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Column> segmented_row_bit_count(Table const& tbl, int32_t segment_length, std::size_t stream) {
  return COL(cudf::segmented_row_bit_count(tbl.cached_view(), segment_length, S(stream)));
}

std::unique_ptr<Table> one_hot_encode(cudf::column_view const& input, cudf::column_view const& categories, std::size_t stream) {
  auto s = S(stream);
  auto [result_col, result_view] = cudf::one_hot_encode(input, categories, s);
  return TBL(std::make_unique<cudf::table>(result_view, s));
}

}  // namespace cudf_sys
