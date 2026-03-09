// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/filling.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/filling.hpp>

namespace cudf_sys {

std::unique_ptr<Column> fill_column(
    cudf::column_view const& col,
    int32_t begin,
    int32_t end,
    Scalar const& value,
    std::size_t stream) {
  return COL(cudf::fill(col, begin, end, value.inner(), S(stream)));
}

std::unique_ptr<Table> repeat_table(
    Table const& tbl,
    int32_t count,
    std::size_t stream) {
  return TBL(cudf::repeat(tbl.cached_view(), count, S(stream)));
}

std::unique_ptr<Column> sequence_column(
    int32_t count,
    Scalar const& init,
    Scalar const& step,
    std::size_t stream) {
  return COL(cudf::sequence(count, init.inner(), step.inner(), S(stream)));
}

std::unique_ptr<Table> repeat_table_column(Table const& tbl, cudf::column_view const& counts, std::size_t stream) {
  return TBL(cudf::repeat(tbl.cached_view(), counts, S(stream)));
}

std::unique_ptr<Column> calendrical_month_sequence(int32_t count, Scalar const& init, int32_t months, std::size_t stream) {
  return COL(cudf::calendrical_month_sequence(count, init.inner(), months, S(stream)));
}

}  // namespace cudf_sys
