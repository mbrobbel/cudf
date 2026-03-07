// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/unary.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/unary.hpp>
#include <cudf/round.hpp>

namespace cudf_sys {

// -- Unary operations --

std::unique_ptr<Column> unary_cast(cudf::column_view const& col, int32_t target_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::cast(col, cudf::data_type{static_cast<cudf::type_id>(target_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_null(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_null(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_valid(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_valid(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_nan(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_nan(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_negate(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::unary_operation(col, cudf::unary_operator::NEGATE, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_abs(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::unary_operation(col, cudf::unary_operator::ABS, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Generic unary operation --

std::unique_ptr<Column> unary_operation(cudf::column_view const& col, int32_t op, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::unary_operation(col, static_cast<cudf::unary_operator>(op), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> unary_is_not_nan(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::is_not_nan(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Round --

std::unique_ptr<Column> round_column(cudf::column_view const& col, int32_t decimal_places, int32_t method, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::round_decimal(col, decimal_places, static_cast<cudf::rounding_method>(method), s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
