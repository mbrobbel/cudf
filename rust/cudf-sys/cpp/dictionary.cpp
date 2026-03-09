// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/dictionary.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>

namespace cudf_sys {

std::unique_ptr<Column> dictionary_encode(cudf::column_view const& col, int32_t indices_type_id, std::size_t stream) {
  return COL(cudf::dictionary::encode(col, DT(indices_type_id), S(stream)));
}

std::unique_ptr<Column> dictionary_decode(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::dictionary::decode(cudf::dictionary_column_view(col), S(stream)));
}

std::unique_ptr<Column> dictionary_add_keys(cudf::column_view const& col, cudf::column_view const& new_keys, std::size_t stream) {
  return COL(cudf::dictionary::add_keys(cudf::dictionary_column_view(col), new_keys, S(stream)));
}

std::unique_ptr<Column> dictionary_remove_keys(cudf::column_view const& col, cudf::column_view const& keys_to_remove, std::size_t stream) {
  return COL(cudf::dictionary::remove_keys(cudf::dictionary_column_view(col), keys_to_remove, S(stream)));
}

std::unique_ptr<Column> dictionary_set_keys(cudf::column_view const& col, cudf::column_view const& keys, std::size_t stream) {
  return COL(cudf::dictionary::set_keys(cudf::dictionary_column_view(col), keys, S(stream)));
}

std::unique_ptr<Column> dictionary_remove_unused_keys(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::dictionary::remove_unused_keys(cudf::dictionary_column_view(col), S(stream)));
}

std::unique_ptr<Scalar> dictionary_get_index(cudf::column_view const& col, Scalar const& key, std::size_t stream) {
  auto result = cudf::dictionary::get_index(cudf::dictionary_column_view(col), key.inner(), S(stream));
  return std::make_unique<Scalar>(std::move(result));
}

}  // namespace cudf_sys
