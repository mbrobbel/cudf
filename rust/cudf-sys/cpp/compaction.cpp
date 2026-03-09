// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/compaction.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/reduction/unique_count.hpp>

namespace cudf_sys {

std::unique_ptr<Table> apply_boolean_mask(
    Table const& tbl,
    cudf::column_view const& mask,
    std::size_t stream) {
  return TBL(cudf::apply_boolean_mask(tbl.cached_view(), mask, S(stream)));
}

std::unique_ptr<Table> drop_nulls_all(Table const& tbl, std::size_t stream) {
  auto view = tbl.cached_view();
  auto num_cols = view.num_columns();
  std::vector<cudf::size_type> keys(num_cols);
  for (int32_t i = 0; i < num_cols; ++i) keys[i] = i;
  return TBL(cudf::drop_nulls(view, keys, S(stream)));
}

std::unique_ptr<Table> drop_nans(Table const& tbl, rust::Slice<int32_t const> keys, std::size_t stream) {
  return TBL(cudf::drop_nans(tbl.cached_view(), VEC(keys), S(stream)));
}

std::unique_ptr<Table> drop_nulls_with_threshold(Table const& tbl, rust::Slice<int32_t const> keys, int32_t threshold, std::size_t stream) {
  return TBL(cudf::drop_nulls(tbl.cached_view(), VEC(keys), threshold, S(stream)));
}

std::unique_ptr<Table> unique_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, std::size_t stream) {
  return TBL(cudf::unique(tbl.cached_view(), VEC(keys),
      ENUM<cudf::duplicate_keep_option>(keep),
      ENUM<cudf::null_equality>(null_equal), S(stream)));
}

std::unique_ptr<Table> distinct_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream) {
  return TBL(cudf::distinct(tbl.cached_view(), VEC(keys),
      ENUM<cudf::duplicate_keep_option>(keep),
      ENUM<cudf::null_equality>(null_equal),
      ENUM<cudf::nan_equality>(nan_equal), S(stream)));
}

std::unique_ptr<Column> distinct_indices_column(Table const& tbl, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream) {
  return COL(cudf::distinct_indices(tbl.cached_view(),
      ENUM<cudf::duplicate_keep_option>(keep),
      ENUM<cudf::null_equality>(null_equal),
      ENUM<cudf::nan_equality>(nan_equal), S(stream)));
}

std::unique_ptr<Table> stable_distinct_table(Table const& tbl, rust::Slice<int32_t const> keys, int32_t keep, int32_t null_equal, int32_t nan_equal, std::size_t stream) {
  return TBL(cudf::stable_distinct(tbl.cached_view(), VEC(keys),
      ENUM<cudf::duplicate_keep_option>(keep),
      ENUM<cudf::null_equality>(null_equal),
      ENUM<cudf::nan_equality>(nan_equal), S(stream)));
}

int32_t distinct_count_column(cudf::column_view const& col, int32_t null_policy, bool nan_is_null, std::size_t stream) {
  auto nan_p = nan_is_null ? cudf::nan_policy::NAN_IS_NULL : cudf::nan_policy::NAN_IS_VALID;
  return cudf::distinct_count(col, ENUM<cudf::null_policy>(null_policy), nan_p, S(stream));
}

int32_t distinct_count_table(Table const& tbl, int32_t null_equality, std::size_t stream) {
  return cudf::distinct_count(tbl.cached_view(), ENUM<cudf::null_equality>(null_equality), S(stream));
}

int32_t unique_count_column(cudf::column_view const& col, int32_t null_policy, bool nan_is_null, std::size_t stream) {
  auto nan_p = nan_is_null ? cudf::nan_policy::NAN_IS_NULL : cudf::nan_policy::NAN_IS_VALID;
  return cudf::unique_count(col, ENUM<cudf::null_policy>(null_policy), nan_p, S(stream));
}

int32_t unique_count_table(Table const& tbl, int32_t null_equality, std::size_t stream) {
  return cudf::unique_count(tbl.cached_view(), ENUM<cudf::null_equality>(null_equality), S(stream));
}

std::unique_ptr<Table> drop_nans_with_threshold(Table const& tbl, rust::Slice<int32_t const> keys, int32_t threshold, std::size_t stream) {
  return TBL(cudf::drop_nans(tbl.cached_view(), VEC(keys), threshold, S(stream)));
}

std::size_t approx_distinct_count(Table const& tbl, int32_t precision, std::size_t stream) {
  auto s = S(stream);
  cudf::approx_distinct_count adc(tbl.cached_view(), precision,
      cudf::null_policy::EXCLUDE, cudf::nan_policy::NAN_IS_NULL, s);
  return adc.estimate(s);
}

}  // namespace cudf_sys
