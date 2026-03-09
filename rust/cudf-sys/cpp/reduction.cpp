// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/reduction.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>

namespace cudf_sys {

std::unique_ptr<Scalar> reduce_sum(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_min(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_min_aggregation<cudf::reduce_aggregation>(), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_max(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_max_aggregation<cudf::reduce_aggregation>(), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_product(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_product_aggregation<cudf::reduce_aggregation>(), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_any(cudf::column_view const& col, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8}, S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_all(cudf::column_view const& col, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8}, S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_mean(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_mean_aggregation<cudf::reduce_aggregation>(), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_std(cudf::column_view const& col, int32_t output_type_id, int32_t ddof, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_std_aggregation<cudf::reduce_aggregation>(ddof), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_var(cudf::column_view const& col, int32_t output_type_id, int32_t ddof, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_variance_aggregation<cudf::reduce_aggregation>(ddof), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_median(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *cudf::make_median_aggregation<cudf::reduce_aggregation>(), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_nunique(cudf::column_view const& col, int32_t null_policy, std::size_t stream) {
  auto result = cudf::reduce(col,
      *cudf::make_nunique_aggregation<cudf::reduce_aggregation>(ENUM<cudf::null_policy>(null_policy)),
      cudf::data_type{cudf::type_id::INT64}, S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

static std::unique_ptr<cudf::scan_aggregation> make_scan_agg(int32_t kind) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::scan_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::scan_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::scan_aggregation>();
    case 9: return cudf::make_product_aggregation<cudf::scan_aggregation>();
    default:
      throw std::invalid_argument("Unsupported scan aggregation kind: " + std::to_string(kind));
  }
}

std::unique_ptr<Column> scan_column(cudf::column_view const& col, int32_t agg_kind, int32_t scan_type, int32_t null_policy, std::size_t stream) {
  return COL(cudf::scan(col, *make_scan_agg(agg_kind),
      ENUM<cudf::scan_type>(scan_type),
      ENUM<cudf::null_policy>(null_policy), S(stream)));
}

std::unique_ptr<Scalar> minmax_min(cudf::column_view const& col, std::size_t stream) {
  auto [min_val, max_val] = cudf::minmax(col, S(stream));
  return std::make_unique<Scalar>(std::move(min_val));
}

std::unique_ptr<Scalar> minmax_max(cudf::column_view const& col, std::size_t stream) {
  auto [min_val, max_val] = cudf::minmax(col, S(stream));
  return std::make_unique<Scalar>(std::move(max_val));
}

static std::unique_ptr<cudf::reduce_aggregation> make_reduce_agg(int32_t kind, int32_t ddof) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::reduce_aggregation>();
    case 5: return cudf::make_nunique_aggregation<cudf::reduce_aggregation>();
    case 6: return cudf::make_median_aggregation<cudf::reduce_aggregation>();
    case 7: return cudf::make_std_aggregation<cudf::reduce_aggregation>(ddof);
    case 8: return cudf::make_variance_aggregation<cudf::reduce_aggregation>(ddof);
    case 9: return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    case 10: return cudf::make_any_aggregation<cudf::reduce_aggregation>();
    case 11: return cudf::make_all_aggregation<cudf::reduce_aggregation>();
    case 17: return cudf::make_sum_of_squares_aggregation<cudf::reduce_aggregation>();
    default:
      throw std::invalid_argument("Unsupported reduce aggregation kind: " + std::to_string(kind));
  }
}

std::unique_ptr<Scalar> reduce_generic(cudf::column_view const& col, int32_t agg_kind, int32_t ddof, int32_t output_type_id, std::size_t stream) {
  auto result = cudf::reduce(col, *make_reduce_agg(agg_kind, ddof), DT(output_type_id), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

std::unique_ptr<Scalar> reduce_with_init(cudf::column_view const& col, int32_t agg_kind, int32_t ddof, int32_t output_type_id, Scalar const& init, std::size_t stream) {
  auto result = cudf::reduce(col, *make_reduce_agg(agg_kind, ddof), DT(output_type_id),
      std::optional<std::reference_wrapper<cudf::scalar const>>{init.inner()}, S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

static std::unique_ptr<cudf::segmented_reduce_aggregation> make_segmented_reduce_agg(int32_t kind, int32_t ddof) {
  switch (kind) {
    case 0: return cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
    case 1: return cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
    case 2: return cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
    case 3: return cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
    case 4: return cudf::make_count_aggregation<cudf::segmented_reduce_aggregation>();
    case 5: return cudf::make_nunique_aggregation<cudf::segmented_reduce_aggregation>();
    case 6: return cudf::make_median_aggregation<cudf::segmented_reduce_aggregation>();
    case 7: return cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>(ddof);
    case 8: return cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>(ddof);
    case 9: return cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();
    case 10: return cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
    case 11: return cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>();
    case 17: return cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
    default: throw std::invalid_argument("unsupported segmented_reduce aggregation kind");
  }
}

std::unique_ptr<Column> segmented_reduce(cudf::column_view const& col, cudf::column_view const& offsets, int32_t agg_kind, int32_t ddof, int32_t output_type_id, int32_t null_handling, std::size_t stream) {
  auto const* offsets_data = offsets.data<cudf::size_type>();
  auto offsets_span = cudf::device_span<cudf::size_type const>(offsets_data, offsets.size());
  return COL(cudf::segmented_reduce(col, offsets_span, *make_segmented_reduce_agg(agg_kind, ddof),
      DT(output_type_id), ENUM<cudf::null_policy>(null_handling), S(stream)));
}

}  // namespace cudf_sys
