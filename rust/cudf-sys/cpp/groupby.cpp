// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/groupby.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>

namespace cudf_sys {

static std::unique_ptr<cudf::groupby_aggregation> make_groupby_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::groupby_aggregation>();
    case 5: return cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
    case 6: return cudf::make_median_aggregation<cudf::groupby_aggregation>();
    case 7: return cudf::make_std_aggregation<cudf::groupby_aggregation>();
    case 8: return cudf::make_variance_aggregation<cudf::groupby_aggregation>();
    case 9: return cudf::make_product_aggregation<cudf::groupby_aggregation>();
    // Note: ANY (10) and ALL (11) are not instantiated for groupby_aggregation in libcudf.
    case 12: return cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
    case 13: return cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
    case 14: return cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
    case 15: return cudf::make_collect_set_aggregation<cudf::groupby_aggregation>();
    case 16: return cudf::make_histogram_aggregation<cudf::groupby_aggregation>();
    case 17: return cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
    case 18: return cudf::make_m2_aggregation<cudf::groupby_aggregation>();
    case 19: return cudf::make_merge_lists_aggregation<cudf::groupby_aggregation>();
    case 20: return cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>();
    case 21: return cudf::make_merge_histogram_aggregation<cudf::groupby_aggregation>();
    default:
      throw std::invalid_argument("Unknown aggregation kind: " + std::to_string(kind));
  }
}

static std::unique_ptr<cudf::groupby_scan_aggregation> make_groupby_scan_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::groupby_scan_aggregation>();
    case 9: return cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
    default:
      throw std::invalid_argument("Unsupported groupby scan aggregation kind: " + std::to_string(kind));
  }
}

std::unique_ptr<Table> groupby_single(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    int32_t value_index,
    int32_t agg_kind,
    std::size_t stream) {
  auto view = tbl.cached_view();
  auto keys_view = view.select(VEC(key_indices));

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::groupby::aggregation_request> requests;
  cudf::groupby::aggregation_request req;
  req.values = view.column(value_index);
  req.aggregations.push_back(make_groupby_agg(agg_kind));
  requests.push_back(std::move(req));

  auto [result_keys, result_vals] = gb.aggregate(requests, S(stream));

  auto key_cols_owned = result_keys->release();
  auto& val_results = result_vals[0].results;

  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + val_results.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& c : val_results) all_cols.push_back(std::move(c));

  return TBL(std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_multi(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds,
    std::size_t stream) {
  auto view = tbl.cached_view();
  auto keys_view = view.select(VEC(key_indices));

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.reserve(value_indices.size());
  for (size_t i = 0; i < value_indices.size(); ++i) {
    cudf::groupby::aggregation_request req;
    req.values = view.column(value_indices[i]);
    req.aggregations.push_back(make_groupby_agg(agg_kinds[i]));
    requests.push_back(std::move(req));
  }

  auto [result_keys, result_vals] = gb.aggregate(requests, S(stream));

  auto key_cols_owned = result_keys->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + result_vals.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& rv : result_vals) {
    for (auto& c : rv.results) all_cols.push_back(std::move(c));
  }

  return TBL(std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_scan(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> agg_kinds,
    std::size_t stream) {
  auto view = tbl.cached_view();
  auto keys_view = view.select(VEC(key_indices));

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::groupby::scan_request> requests;
  requests.reserve(value_indices.size());
  for (size_t i = 0; i < value_indices.size(); ++i) {
    cudf::groupby::scan_request req;
    req.values = view.column(value_indices[i]);
    req.aggregations.push_back(make_groupby_scan_agg(agg_kinds[i]));
    requests.push_back(std::move(req));
  }

  auto [result_keys, result_vals] = gb.scan(requests, S(stream));

  auto key_cols_owned = result_keys->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + result_vals.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& rv : result_vals) {
    for (auto& c : rv.results) all_cols.push_back(std::move(c));
  }

  return TBL(std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_shift(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> offsets,
    ScalarList& fill_values,
    std::size_t stream) {
  auto view = tbl.cached_view();
  auto keys_view = view.select(VEC(key_indices));

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::column_view> val_views;
  val_views.reserve(value_indices.size());
  for (auto idx : value_indices) val_views.push_back(view.column(idx));
  cudf::table_view values_view(val_views);

  auto [result_keys, result_values] = gb.shift(values_view, VEC(offsets), fill_values.refs(), S(stream));

  auto key_cols_owned = result_keys->release();
  auto val_cols_owned = result_values->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + val_cols_owned.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& c : val_cols_owned) all_cols.push_back(std::move(c));

  return TBL(std::make_unique<cudf::table>(std::move(all_cols)));
}

std::unique_ptr<Table> groupby_replace_nulls(
    Table const& tbl,
    rust::Slice<int32_t const> key_indices,
    rust::Slice<int32_t const> value_indices,
    rust::Slice<int32_t const> policies,
    std::size_t stream) {
  auto view = tbl.cached_view();
  auto keys_view = view.select(VEC(key_indices));

  cudf::groupby::groupby gb(keys_view, cudf::null_policy::EXCLUDE,
      cudf::sorted::NO, {}, {});

  std::vector<cudf::column_view> val_views;
  val_views.reserve(value_indices.size());
  for (auto idx : value_indices) val_views.push_back(view.column(idx));
  cudf::table_view values_view(val_views);

  std::vector<cudf::replace_policy> pol_vec;
  pol_vec.reserve(policies.size());
  for (auto p : policies) {
    pol_vec.push_back(ENUM<cudf::replace_policy>(p));
  }

  auto [result_keys, result_values] = gb.replace_nulls(values_view, pol_vec, S(stream));

  auto key_cols_owned = result_keys->release();
  auto val_cols_owned = result_values->release();
  std::vector<std::unique_ptr<cudf::column>> all_cols;
  all_cols.reserve(key_cols_owned.size() + val_cols_owned.size());
  for (auto& c : key_cols_owned) all_cols.push_back(std::move(c));
  for (auto& c : val_cols_owned) all_cols.push_back(std::move(c));

  return TBL(std::make_unique<cudf::table>(std::move(all_cols)));
}

}  // namespace cudf_sys
