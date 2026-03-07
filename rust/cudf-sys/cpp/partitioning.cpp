// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/partitioning.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/partitioning.hpp>

namespace cudf_sys {

// -- Partitioning --

std::unique_ptr<Table> hash_partition_table(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> cols;
  cols.reserve(columns_to_hash.size());
  for (auto c : columns_to_hash) cols.push_back(c);

  auto [result_table, offsets] = cudf::hash_partition(
      tbl.cached_view(), cols, num_partitions,
      cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, s);
  return std::make_unique<Table>(std::move(result_table));
}

rust::Vec<int32_t> hash_partition_offsets(
    Table const& tbl,
    rust::Slice<int32_t const> columns_to_hash,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> cols;
  cols.reserve(columns_to_hash.size());
  for (auto c : columns_to_hash) cols.push_back(c);

  auto [result_table, offsets] = cudf::hash_partition(
      tbl.cached_view(), cols, num_partitions,
      cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, s);
  rust::Vec<int32_t> result;
  result.reserve(offsets.size());
  for (auto o : offsets) result.push_back(o);
  return result;
}

std::unique_ptr<Table> round_robin_partition_table(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::round_robin_partition(
      tbl.cached_view(), num_partitions, start_partition, s);
  return std::make_unique<Table>(std::move(result_table));
}

rust::Vec<int32_t> round_robin_partition_offsets(
    Table const& tbl,
    int32_t num_partitions,
    int32_t start_partition,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::round_robin_partition(
      tbl.cached_view(), num_partitions, start_partition, s);
  rust::Vec<int32_t> result;
  result.reserve(offsets.size());
  for (auto o : offsets) result.push_back(o);
  return result;
}

// -- Partitioning: partition by map column --

std::unique_ptr<Table> partition_by_map(
    Table const& tbl,
    cudf::column_view const& partition_map,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::partition(tbl.cached_view(), partition_map, num_partitions, s);
  return std::make_unique<Table>(std::move(result_table));
}

rust::Vec<int32_t> partition_by_map_offsets(
    Table const& tbl,
    cudf::column_view const& partition_map,
    int32_t num_partitions,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto [result_table, offsets] = cudf::partition(tbl.cached_view(), partition_map, num_partitions, s);
  rust::Vec<int32_t> result;
  result.reserve(offsets.size());
  for (auto o : offsets) result.push_back(o);
  return result;
}

}  // namespace cudf_sys
