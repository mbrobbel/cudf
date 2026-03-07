// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/replace.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/replace.hpp>

namespace cudf_sys {

// -- Replace operations --

std::unique_ptr<Column> replace_nulls_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nulls(col, replacement, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> replace_nulls_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nulls(col, replacement.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> replace_nans_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nans(col, replacement, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> replace_nans_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nans(col, replacement.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> clamp_column(
    cudf::column_view const& col,
    Scalar const& lo,
    Scalar const& hi,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::clamp(col, lo.inner(), hi.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> find_and_replace_all(
    cudf::column_view const& col,
    cudf::column_view const& values_to_replace,
    cudf::column_view const& replacement_values,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::find_and_replace_all(col, values_to_replace, replacement_values, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Replace (new) --

std::unique_ptr<Column> replace_nulls_policy(cudf::column_view const& col, int32_t policy, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::replace_nulls(col, static_cast<cudf::replace_policy>(policy), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> clamp_column_with_replace(cudf::column_view const& col, Scalar const& lo, Scalar const& lo_replace, Scalar const& hi, Scalar const& hi_replace, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::clamp(col, lo.inner(), lo_replace.inner(), hi.inner(), hi_replace.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> normalize_nans_and_zeros(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::normalize_nans_and_zeros(col, s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
