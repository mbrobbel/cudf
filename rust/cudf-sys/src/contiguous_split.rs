// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Table = crate::ffi::Table;

        /// Opaque packed columns (host metadata + device data).
        type PackedColumns;

        /// Opaque vector of packed tables from `contiguous_split`.
        type PackedTableVec;

        // -- Pack / Unpack --

        /// Packs a table into a contiguous device buffer with host metadata.
        fn pack_table(tbl: &Table, stream: usize) -> Result<UniquePtr<PackedColumns>>;

        /// Returns the size in bytes that packing would require.
        fn packed_size_of(tbl: &Table, stream: usize) -> Result<usize>;

        /// Unpacks packed columns into an owned table.
        fn unpack_packed(packed: &PackedColumns) -> Result<UniquePtr<Table>>;

        /// Returns the host metadata as bytes.
        fn metadata_to_host(self: &PackedColumns) -> Vec<u8>;

        /// Returns the size of the host metadata in bytes.
        fn metadata_size(self: &PackedColumns) -> usize;

        /// Returns the size of the GPU data in bytes.
        fn gpu_data_size(self: &PackedColumns) -> usize;

        // -- Contiguous split --

        /// Splits a table into contiguous partitions.
        fn contiguous_split_table(
            tbl: &Table,
            splits: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<PackedTableVec>>;

        /// Returns the number of partitions.
        fn size(self: &PackedTableVec) -> usize;

        /// Unpacks partition at `index` into an owned table.
        fn unpack_at(self: &PackedTableVec, index: usize) -> Result<UniquePtr<Table>>;
    }
}
