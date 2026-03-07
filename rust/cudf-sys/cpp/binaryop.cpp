// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/binaryop.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/binaryop.hpp>

namespace cudf_sys {

// -- Binary operations --

std::unique_ptr<Column> binary_operation_columns(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::binary_operation(
      lhs, rhs,
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> binary_operation_column_scalar(
    cudf::column_view const& lhs,
    Scalar const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::binary_operation(
      lhs, rhs.inner(),
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> binary_operation_scalar_column(
    Scalar const& lhs,
    cudf::column_view const& rhs,
    BinaryOperator op,
    int32_t output_type_id,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::binary_operation(
      lhs.inner(), rhs,
      static_cast<cudf::binary_operator>(static_cast<int32_t>(op)),
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)},
      s);
  return std::make_unique<Column>(std::move(result));
}

// -- Binary: fixed_point_scale / is_supported_operation --

int32_t binary_operation_fixed_point_scale(int32_t op, int32_t left_scale, int32_t right_scale) {
  return cudf::binary_operation_fixed_point_scale(
      static_cast<cudf::binary_operator>(op), left_scale, right_scale);
}

bool is_supported_binaryop(int32_t out_type_id, int32_t lhs_type_id, int32_t rhs_type_id, int32_t op) {
  return cudf::binops::is_supported_operation(
      cudf::data_type{static_cast<cudf::type_id>(out_type_id)},
      cudf::data_type{static_cast<cudf::type_id>(lhs_type_id)},
      cudf::data_type{static_cast<cudf::type_id>(rhs_type_id)},
      static_cast<cudf::binary_operator>(op));
}

}  // namespace cudf_sys
