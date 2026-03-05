// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Table partitioning operations.

use crate::error::Result;
use crate::table::Table;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Hash-partitions a table into `num_partitions` partitions.
///
/// Rows are rearranged so that rows in the same partition are contiguous.
/// The columns specified by `columns` are used for computing the hash.
pub fn hash_partition(table: &Table, columns: &[i32], num_partitions: usize) -> Result<Table> {
    let tbl =
        cudf_sys::ffi::hash_partition_table(&table.0, columns, num_partitions as i32, ds())?;
    Ok(Table(tbl))
}

/// Returns partition offsets for hash partitioning.
///
/// The returned vector has `num_partitions` elements indicating
/// the start row of each partition.
pub fn hash_partition_offsets(
    table: &Table,
    columns: &[i32],
    num_partitions: usize,
) -> Result<Vec<usize>> {
    let offsets =
        cudf_sys::ffi::hash_partition_offsets(&table.0, columns, num_partitions as i32, ds())?;
    Ok(offsets.into_iter().map(|o| o as usize).collect())
}

/// Round-robin partitions a table into `num_partitions` partitions.
///
/// Rows are assigned to partitions in a round-robin fashion starting
/// from `start_partition`.
pub fn round_robin(
    table: &Table,
    num_partitions: usize,
    start_partition: usize,
) -> Result<Table> {
    let tbl = cudf_sys::ffi::round_robin_partition_table(
        &table.0,
        num_partitions as i32,
        start_partition as i32,
        ds(),
    )?;
    Ok(Table(tbl))
}

/// Returns partition offsets for round-robin partitioning.
pub fn round_robin_offsets(
    table: &Table,
    num_partitions: usize,
    start_partition: usize,
) -> Result<Vec<usize>> {
    let offsets = cudf_sys::ffi::round_robin_partition_offsets(
        &table.0,
        num_partitions as i32,
        start_partition as i32,
        ds(),
    )?;
    Ok(offsets.into_iter().map(|o| o as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::table::TableBuilder;

    #[test]
    fn hash_partition_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let result = hash_partition(&table, &[0i32], 2).unwrap();
        // Row count is preserved
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }

    #[test]
    fn hash_partition_offsets_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let offsets = hash_partition_offsets(&table, &[0i32], 2).unwrap();
        // cudf returns num_partitions offsets (start of each partition)
        assert_eq!(offsets.len(), 2);
    }

    #[test]
    fn round_robin_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let result = round_robin(&table, 3, 0).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }

    #[test]
    fn round_robin_offsets_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3, 4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let offsets = round_robin_offsets(&table, 3, 0).unwrap();
        // cudf returns num_partitions offsets (start of each partition)
        assert_eq!(offsets.len(), 3);
    }
}
