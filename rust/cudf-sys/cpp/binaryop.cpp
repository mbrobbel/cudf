// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/binaryop.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/binaryop.hpp>

namespace cudf_sys {

std::unique_ptr<Column> binary_operation_columns(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  return COL(cudf::binary_operation(lhs, rhs,
      ENUM<cudf::binary_operator>(static_cast<int32_t>(op)), DT(output_type_id), S(stream)));
}

std::unique_ptr<Column> binary_operation_column_scalar(
    cudf::column_view const& lhs,
    Scalar const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  return COL(cudf::binary_operation(lhs, rhs.inner(),
      ENUM<cudf::binary_operator>(static_cast<int32_t>(op)), DT(output_type_id), S(stream)));
}

std::unique_ptr<Column> binary_operation_scalar_column(
    Scalar const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  return COL(cudf::binary_operation(lhs.inner(), rhs,
      ENUM<cudf::binary_operator>(static_cast<int32_t>(op)), DT(output_type_id), S(stream)));
}

int32_t binary_operation_fixed_point_scale(int32_t op, int32_t left_scale, int32_t right_scale) {
  return cudf::binary_operation_fixed_point_scale(
      ENUM<cudf::binary_operator>(op), left_scale, right_scale);
}

bool is_supported_binaryop(int32_t out_type_id, int32_t lhs_type_id, int32_t rhs_type_id, int32_t op) {
  return cudf::binops::is_supported_operation(
      DT(out_type_id), DT(lhs_type_id), DT(rhs_type_id), ENUM<cudf::binary_operator>(op));
}

}  // namespace cudf_sys
