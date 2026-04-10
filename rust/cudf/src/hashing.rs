// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Hashing operations on GPU tables.
//!
//! All hashing methods produce one output value per input row, computed across
//! all columns of the table. Available as methods on
//! [`Table`](crate::table::Table):
//!
//! **Non-cryptographic hashes** (fast, for partitioning and hash joins):
//! * [`Table::murmur3`](crate::table::Table::murmur3) -- `MurmurHash3` 32-bit
//!   hash, returns a `UINT32` column.
//! * [`Table::murmurhash3_x64_128`](crate::table::Table::murmurhash3_x64_128) --
//!   `MurmurHash3` 128-bit, returns a `Table` of two `UINT64` columns.
//! * [`Table::xxhash64`](crate::table::Table::xxhash64) -- `XXHash64`, returns
//!   a `UINT64` column.
//! * [`Table::xxhash_32`](crate::table::Table::xxhash_32) -- `XXHash` 32-bit,
//!   returns a `UINT32` column.
//!
//! **Cryptographic hashes** (returns hex-encoded `STRING` columns):
//! * [`Table::md5`](crate::table::Table::md5) -- 32-char hex.
//! * [`Table::sha1`](crate::table::Table::sha1) -- 40-char hex.
//! * [`Table::sha224`](crate::table::Table::sha224) -- 56-char hex.
//! * [`Table::sha256`](crate::table::Table::sha256) -- 64-char hex.
//! * [`Table::sha384`](crate::table::Table::sha384) -- 96-char hex.
//! * [`Table::sha512`](crate::table::Table::sha512) -- 128-char hex.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::{TableBuilder, UnboundTable};

    fn make_test_table() -> UnboundTable {
        let c1 = Col::from_slice_i32(&[1, 2, 3]).call().unwrap();
        let c2 = Col::from_slice_i32(&[4, 5, 6]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        builder.build().unwrap()
    }

    #[test]
    fn murmur3_basic() {
        let table = make_test_table();
        let hashes = table.murmur3(0).call().unwrap();
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes.type_id(), TypeId::UINT32);
        assert!(!hashes.has_nulls());
    }

    #[test]
    fn xxhash64_basic() {
        let table = make_test_table();
        let hashes = table.xxhash64(0).call().unwrap();
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes.type_id(), TypeId::UINT64);
        assert!(!hashes.has_nulls());
    }

    #[test]
    fn murmur3_deterministic() {
        let table = make_test_table();
        let h1 = table.murmur3(42).call().unwrap();
        let h2 = table.murmur3(42).call().unwrap();
        assert_eq!(
            h1.to_vec_i32().call().unwrap(),
            h2.to_vec_i32().call().unwrap()
        );
    }

    #[test]
    fn md5_basic() {
        let c = Col::from_scalar(&Scalar::from_i32(42), 2).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c);
        let table = builder.build().unwrap();
        let hashes = table.md5().call().unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.type_id(), TypeId::STRING);
        let strings = hashes.to_vec_string().call().unwrap();
        assert_eq!(strings[0].len(), 32);
        assert_eq!(strings[0], strings[1]);
    }

    #[test]
    fn sha256_basic() {
        let c = Col::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c);
        let table = builder.build().unwrap();
        let hashes = table.sha256().call().unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.type_id(), TypeId::STRING);
        let strings = hashes.to_vec_string().call().unwrap();
        assert_eq!(strings[0].len(), 64);
        assert_eq!(strings[0], strings[1]);
    }
}
