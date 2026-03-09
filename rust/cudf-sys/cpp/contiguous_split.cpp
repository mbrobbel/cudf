// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/contiguous_split.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/contiguous_split.hpp>

namespace cudf_sys {

// -- PackedColumns --

rust::Vec<uint8_t> PackedColumns::metadata_to_host() const {
  rust::Vec<uint8_t> out;
  if (packed_.metadata) {
    auto const& meta = *packed_.metadata;
    out.reserve(meta.size());
    for (auto b : meta) {
      out.push_back(b);
    }
  }
  return out;
}

std::size_t PackedColumns::metadata_size() const {
  return packed_.metadata ? packed_.metadata->size() : 0;
}

std::size_t PackedColumns::gpu_data_size() const {
  return packed_.gpu_data ? packed_.gpu_data->size() : 0;
}

// -- PackedTableVec --

std::size_t PackedTableVec::size() const {
  return vec_.size();
}

std::unique_ptr<Table> PackedTableVec::unpack_at(std::size_t index) const {
  if (index >= vec_.size()) {
    throw std::out_of_range("PackedTableVec index out of range");
  }
  auto view = vec_[index].table;
  return std::make_unique<Table>(std::make_unique<cudf::table>(view));
}

// -- Free functions --

std::unique_ptr<PackedColumns> pack_table(Table const& tbl, std::size_t stream) {
  auto packed = cudf::pack(tbl.cached_view(), S(stream));
  return std::make_unique<PackedColumns>(std::move(packed));
}

std::size_t packed_size_of(Table const& tbl, std::size_t stream) {
  return cudf::packed_size(tbl.cached_view(), S(stream));
}

std::unique_ptr<Table> unpack_packed(PackedColumns const& packed) {
  auto view = cudf::unpack(packed.inner());
  return std::make_unique<Table>(std::make_unique<cudf::table>(view));
}

std::unique_ptr<PackedTableVec> contiguous_split_table(Table const& tbl, rust::Slice<int32_t const> splits, std::size_t stream) {
  std::vector<cudf::size_type> split_vec(splits.begin(), splits.end());
  auto result = cudf::contiguous_split(tbl.cached_view(), split_vec, S(stream));
  return std::make_unique<PackedTableVec>(std::move(result));
}

}  // namespace cudf_sys
