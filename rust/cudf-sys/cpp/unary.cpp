// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/unary.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/unary.hpp>
#include <cudf/round.hpp>

namespace cudf_sys {

std::unique_ptr<Column> unary_cast(cudf::column_view const& col, int32_t target_type_id, std::size_t stream) {
  return COL(cudf::cast(col, DT(target_type_id), S(stream)));
}

std::unique_ptr<Column> unary_is_null(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::is_null(col, S(stream)));
}

std::unique_ptr<Column> unary_is_valid(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::is_valid(col, S(stream)));
}

std::unique_ptr<Column> unary_is_nan(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::is_nan(col, S(stream)));
}

std::unique_ptr<Column> unary_negate(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::unary_operation(col, cudf::unary_operator::NEGATE, S(stream)));
}

std::unique_ptr<Column> unary_abs(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::unary_operation(col, cudf::unary_operator::ABS, S(stream)));
}

std::unique_ptr<Column> unary_operation(cudf::column_view const& col, int32_t op, std::size_t stream) {
  return COL(cudf::unary_operation(col, ENUM<cudf::unary_operator>(op), S(stream)));
}

std::unique_ptr<Column> unary_is_not_nan(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::is_not_nan(col, S(stream)));
}

std::unique_ptr<Column> round_column(cudf::column_view const& col, int32_t decimal_places, int32_t method, std::size_t stream) {
  return COL(cudf::round_decimal(col, decimal_places, ENUM<cudf::rounding_method>(method), S(stream)));
}

}  // namespace cudf_sys
