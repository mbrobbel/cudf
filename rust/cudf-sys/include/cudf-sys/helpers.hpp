// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

// Common inline helpers that eliminate boilerplate from CXX bridge wrappers.
//
// Every wrapper function does the same 3 things:
//   1. Convert args  (size_t→stream, int32_t→enum, Slice→vector)
//   2. Call libcudf
//   3. Wrap result   (unique_ptr<Column/Table>)
//
// These helpers collapse each step to a single short call.

#pragma once

#include "cudf-sys/lib.hpp"

#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rust/cxx.h"

namespace cudf_sys {

// ── Arg conversion ──────────────────────────────────────────────

/// size_t → rmm::cuda_stream_view
inline rmm::cuda_stream_view S(std::size_t raw) {
  return rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(raw)};
}

/// int32_t → cudf::data_type (non-decimal)
inline cudf::data_type DT(int32_t id) {
  return cudf::data_type{static_cast<cudf::type_id>(id)};
}

/// int32_t slice → std::vector<cudf::size_type>
inline std::vector<cudf::size_type> VEC(rust::Slice<int32_t const> s) {
  return {s.begin(), s.end()};
}

/// int32_t slice → std::vector<cudf::order>
inline std::vector<cudf::order> ORDERS(rust::Slice<int32_t const> s) {
  std::vector<cudf::order> v;
  v.reserve(s.size());
  for (auto o : s) v.push_back(static_cast<cudf::order>(o));
  return v;
}

/// int32_t slice → std::vector<cudf::null_order>
inline std::vector<cudf::null_order> NULL_ORDERS(rust::Slice<int32_t const> s) {
  std::vector<cudf::null_order> v;
  v.reserve(s.size());
  for (auto n : s) v.push_back(static_cast<cudf::null_order>(n));
  return v;
}

/// Generic int32_t → enum cast
template <typename E>
inline E ENUM(int32_t v) { return static_cast<E>(v); }

// ── Result wrapping ─────────────────────────────────────────────

/// unique_ptr<cudf::column> → unique_ptr<Column>
inline std::unique_ptr<Column> COL(std::unique_ptr<cudf::column> c) {
  return std::make_unique<Column>(std::move(c));
}

/// unique_ptr<cudf::table> → unique_ptr<Table>
inline std::unique_ptr<Table> TBL(std::unique_ptr<cudf::table> t) {
  return std::make_unique<Table>(std::move(t));
}

}  // namespace cudf_sys
