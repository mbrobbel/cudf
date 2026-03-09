// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/replace.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/replace.hpp>

namespace cudf_sys {

std::unique_ptr<Column> replace_nulls_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream) {
  return COL(cudf::replace_nulls(col, replacement, S(stream)));
}

std::unique_ptr<Column> replace_nulls_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream) {
  return COL(cudf::replace_nulls(col, replacement.inner(), S(stream)));
}

std::unique_ptr<Column> replace_nans_column(
    cudf::column_view const& col,
    cudf::column_view const& replacement,
    std::size_t stream) {
  return COL(cudf::replace_nans(col, replacement, S(stream)));
}

std::unique_ptr<Column> replace_nans_scalar(
    cudf::column_view const& col,
    Scalar const& replacement,
    std::size_t stream) {
  return COL(cudf::replace_nans(col, replacement.inner(), S(stream)));
}

std::unique_ptr<Column> clamp_column(
    cudf::column_view const& col,
    Scalar const& lo,
    Scalar const& hi,
    std::size_t stream) {
  return COL(cudf::clamp(col, lo.inner(), hi.inner(), S(stream)));
}

std::unique_ptr<Column> find_and_replace_all(
    cudf::column_view const& col,
    cudf::column_view const& values_to_replace,
    cudf::column_view const& replacement_values,
    std::size_t stream) {
  return COL(cudf::find_and_replace_all(col, values_to_replace, replacement_values, S(stream)));
}

std::unique_ptr<Column> replace_nulls_policy(cudf::column_view const& col, int32_t policy, std::size_t stream) {
  return COL(cudf::replace_nulls(col, ENUM<cudf::replace_policy>(policy), S(stream)));
}

std::unique_ptr<Column> clamp_column_with_replace(cudf::column_view const& col, Scalar const& lo, Scalar const& lo_replace, Scalar const& hi, Scalar const& hi_replace, std::size_t stream) {
  return COL(cudf::clamp(col, lo.inner(), lo_replace.inner(), hi.inner(), hi_replace.inner(), S(stream)));
}

std::unique_ptr<Column> normalize_nans_and_zeros(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::normalize_nans_and_zeros(col, S(stream)));
}

}  // namespace cudf_sys
