// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Table = crate::ffi::Table;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Partitioning --

        /// Hash-partitions a table into N partitions.
        fn hash_partition_table(
            tbl: &Table,
            columns_to_hash: &[i32],
            num_partitions: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns partition offsets for hash partitioning.
        fn hash_partition_offsets(
            tbl: &Table,
            columns_to_hash: &[i32],
            num_partitions: i32,
            stream: usize,
        ) -> Result<Vec<i32>>;

        /// Round-robin partitions a table.
        fn round_robin_partition_table(
            tbl: &Table,
            num_partitions: i32,
            start_partition: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns partition offsets for round-robin partitioning.
        fn round_robin_partition_offsets(
            tbl: &Table,
            num_partitions: i32,
            start_partition: i32,
            stream: usize,
        ) -> Result<Vec<i32>>;

        // -- Partitioning: partition by map column --

        /// Partition table rows by a map column.
        fn partition_by_map(
            tbl: &Table,
            partition_map: &column_view,
            num_partitions: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
        /// Get partition offsets for partition by map.
        fn partition_by_map_offsets(
            tbl: &Table,
            partition_map: &column_view,
            num_partitions: i32,
            stream: usize,
        ) -> Result<Vec<i32>>;
    }
}
