// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;

        // -- Hashing --

        /// Computes `MurmurHash3` 32-bit hash of each row.
        fn hash_murmur3(tbl: &Table, seed: u32, stream: usize) -> UniquePtr<Column>;

        /// Computes `XXHash64` hash of each row.
        fn hash_xxhash64(tbl: &Table, seed: u64, stream: usize) -> UniquePtr<Column>;

        /// Computes MD5 hash of each row (returns string column).
        fn hash_md5(tbl: &Table, stream: usize) -> UniquePtr<Column>;

        /// Computes SHA-256 hash of each row (returns string column).
        fn hash_sha256(tbl: &Table, stream: usize) -> UniquePtr<Column>;

        /// `MurmurHash3` 128-bit hash (returns table of two UINT64 columns).
        fn hash_murmurhash3_x64_128(
            tbl: &Table,
            seed: u64,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// SHA-1 hash of each row (returns string column).
        fn hash_sha1(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// `XXHash` 32-bit hash of each row.
        fn hash_xxhash_32(tbl: &Table, seed: u32, stream: usize) -> Result<UniquePtr<Column>>;

        /// SHA-224 hash of each row (returns string column).
        fn hash_sha224(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// SHA-384 hash of each row (returns string column).
        fn hash_sha384(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// SHA-512 hash of each row (returns string column).
        fn hash_sha512(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        /// `MurmurHash3` x86 32-bit hash of each row (returns UINT32 column).
        fn hash_murmurhash3_x86_32(
            tbl: &Table,
            seed: u32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
