// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/dictionary.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>

namespace cudf_sys {

// -- Dictionary operations --

std::unique_ptr<Column> dictionary_encode(cudf::column_view const& col, int32_t indices_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::dictionary::encode(col, cudf::data_type{static_cast<cudf::type_id>(indices_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_decode(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::decode(dcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_add_keys(cudf::column_view const& col, cudf::column_view const& new_keys, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::add_keys(dcv, new_keys, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_remove_keys(cudf::column_view const& col, cudf::column_view const& keys_to_remove, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::remove_keys(dcv, keys_to_remove, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_set_keys(cudf::column_view const& col, cudf::column_view const& keys, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::set_keys(dcv, keys, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> dictionary_remove_unused_keys(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::remove_unused_keys(dcv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Scalar> dictionary_get_index(cudf::column_view const& col, Scalar const& key, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::dictionary_column_view dcv(col);
  auto result = cudf::dictionary::get_index(dcv, key.inner(), s);
  return std::make_unique<Scalar>(std::move(result));
}

}  // namespace cudf_sys
