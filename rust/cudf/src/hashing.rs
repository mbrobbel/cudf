// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Hashing operations on GPU tables.
//!
//! Available as methods on [`Table`](crate::table::Table):
//! `table.murmur3(...)`, `table.xxhash64(...)`, `table.md5()`, `table.sha256()`.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::{Table, TableBuilder};

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
        assert_eq!(h1.to_vec_i32(), h2.to_vec_i32());
    }

    #[test]
    fn md5_basic() {
        let c = Col::from_scalar(&Scalar::from_i32(42), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c);
        let table = builder.build().unwrap();
        let hashes = table.md5().call().unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.type_id(), TypeId::STRING);
        let strings = hashes.to_vec_string();
        assert_eq!(strings[0].len(), 32);
        assert_eq!(strings[0], strings[1]);
    }

    #[test]
    fn sha256_basic() {
        let c = Col::from_scalar(&Scalar::from_i32(1), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c);
        let table = builder.build().unwrap();
        let hashes = table.sha256().call().unwrap();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes.type_id(), TypeId::STRING);
        let strings = hashes.to_vec_string();
        assert_eq!(strings[0].len(), 64);
        assert_eq!(strings[0], strings[1]);
    }
}
