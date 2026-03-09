// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Table partitioning operations.
//!
//! Partitioning rearranges table rows so that rows belonging to the same
//! partition are contiguous. Available as methods on
//! [`Table`](crate::table::Table):
//!
//! * [`Table::hash_partition`](crate::table::Table::hash_partition) -- partition
//!   rows by hashing specified columns.
//! * [`Table::hash_partition_offsets`](crate::table::Table::hash_partition_offsets) --
//!   returns only the partition boundary offsets.
//! * [`Table::round_robin`](crate::table::Table::round_robin) -- distribute rows
//!   round-robin across partitions.
//! * [`Table::round_robin_offsets`](crate::table::Table::round_robin_offsets) --
//!   returns only the partition boundary offsets.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn hash_partition_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let result = table.hash_partition(&[0i32], 2).call().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }

    #[test]
    fn hash_partition_offsets_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let offsets = table.hash_partition_offsets(&[0i32], 2).call().unwrap();
        assert_eq!(offsets.len(), 3); // num_partitions + 1
    }

    #[test]
    fn round_robin_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let result = table.round_robin(3, 0).call().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }

    #[test]
    fn round_robin_offsets_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let offsets = table.round_robin_offsets(3, 0).call().unwrap();
        assert_eq!(offsets.len(), 4); // num_partitions + 1
    }
}
