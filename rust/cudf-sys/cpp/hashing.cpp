// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/hashing.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/hashing.hpp>

namespace cudf_sys {

std::unique_ptr<Column> hash_murmur3(Table const& tbl, uint32_t seed, std::size_t stream) {
  return COL(cudf::hashing::murmurhash3_x86_32(tbl.cached_view(), seed, S(stream)));
}

std::unique_ptr<Column> hash_xxhash64(Table const& tbl, uint64_t seed, std::size_t stream) {
  return COL(cudf::hashing::xxhash_64(tbl.cached_view(), seed, S(stream)));
}

std::unique_ptr<Column> hash_md5(Table const& tbl, std::size_t stream) {
  return COL(cudf::hashing::md5(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Column> hash_sha256(Table const& tbl, std::size_t stream) {
  return COL(cudf::hashing::sha256(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Table> hash_murmurhash3_x64_128(Table const& tbl, uint64_t seed, std::size_t stream) {
  return TBL(cudf::hashing::murmurhash3_x64_128(tbl.cached_view(), seed, S(stream)));
}

std::unique_ptr<Column> hash_sha1(Table const& tbl, std::size_t stream) {
  return COL(cudf::hashing::sha1(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Column> hash_xxhash_32(Table const& tbl, uint32_t seed, std::size_t stream) {
  return COL(cudf::hashing::xxhash_32(tbl.cached_view(), seed, S(stream)));
}

std::unique_ptr<Column> hash_sha224(Table const& tbl, std::size_t stream) {
  return COL(cudf::hashing::sha224(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Column> hash_sha384(Table const& tbl, std::size_t stream) {
  return COL(cudf::hashing::sha384(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Column> hash_sha512(Table const& tbl, std::size_t stream) {
  return COL(cudf::hashing::sha512(tbl.cached_view(), S(stream)));
}

std::unique_ptr<Column> hash_murmurhash3_x86_32(Table const& tbl, uint32_t seed, std::size_t stream) {
  return COL(cudf::hashing::murmurhash3_x86_32(tbl.cached_view(), seed, S(stream)));
}

}  // namespace cudf_sys
