// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Hashing operations on GPU tables.

use crate::column::Column;
use crate::table::Table;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Computes MurmurHash3 32-bit hash of each row in the table.
///
/// Returns a UINT32 column containing one hash value per row.
pub fn murmur3(table: &Table, seed: u32) -> Column {
    Column(cudf_sys::ffi::hash_murmur3(&table.0, seed, ds()))
}

/// Computes XXHash64 hash of each row in the table.
///
/// Returns a UINT64 column containing one hash value per row.
pub fn xxhash64(table: &Table, seed: u64) -> Column {
    Column(cudf_sys::ffi::hash_xxhash64(&table.0, seed, ds()))
}

/// Computes MD5 hash of each row in the table.
///
/// Returns a STRING column containing hex-encoded hash values.
pub fn md5(table: &Table) -> Column {
    Column(cudf_sys::ffi::hash_md5(&table.0, ds()))
}

/// Computes SHA-256 hash of each row in the table.
///
/// Returns a STRING column containing hex-encoded hash values.
pub fn sha256(table: &Table) -> Column {
    Column(cudf_sys::ffi::hash_sha256(&table.0, ds()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    fn make_test_table() -> Table {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let c2 = Col::from_slice_i32(&[4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        builder.build().unwrap()
    }

    #[test]
    fn murmur3_basic() {
        let table = make_test_table();
        let hashes = murmur3(&table, 0);
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes.type_id(), TypeId::UINT32);
        assert!(!hashes.has_nulls());
    }

    #[test]
    fn xxhash64_basic() {
        let table = make_test_table();
        let hashes = xxhash64(&table, 0);
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes.type_id(), TypeId::UINT64);
        assert!(!hashes.has_nulls());
    }

    #[test]
    fn murmur3_deterministic() {
        let table = make_test_table();
        let h1 = murmur3(&table, 42);
        let h2 = murmur3(&table, 42);
        assert_eq!(h1.to_vec_i32(), h2.to_vec_i32());
    }

    #[test]
    fn md5_basic() {
        let c = Col::from_scalar(&Scalar::from_i32(42), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c);
        let table = builder.build().unwrap();
        let hashes = md5(&table);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.type_id(), TypeId::STRING);
        let strings = hashes.to_vec_string();
        // MD5 produces 32 hex characters
        assert_eq!(strings[0].len(), 32);
        // Same input produces same hash
        assert_eq!(strings[0], strings[1]);
    }

    #[test]
    fn sha256_basic() {
        let c = Col::from_scalar(&Scalar::from_i32(1), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c);
        let table = builder.build().unwrap();
        let hashes = sha256(&table);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.type_id(), TypeId::STRING);
        let strings = hashes.to_vec_string();
        // SHA-256 produces 64 hex characters
        assert_eq!(strings[0].len(), 64);
        assert_eq!(strings[0], strings[1]);
    }
}
