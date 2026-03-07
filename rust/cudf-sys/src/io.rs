// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Table = crate::ffi::Table;

        // -- CSV I/O --

        /// Reads a CSV file and returns a Table.
        fn read_csv(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Reads a CSV file with options and returns a Table.
        fn read_csv_with_options(
            filepath: &str,
            delimiter: u8,
            header: bool,
            skip_rows: i32,
            num_rows: i32,
        ) -> Result<UniquePtr<Table>>;

        /// Writes a Table to a CSV file.
        fn write_csv(tbl: &Table, filepath: &str) -> Result<()>;

        /// Writes a Table to a CSV file with options.
        fn write_csv_with_options(
            tbl: &Table,
            filepath: &str,
            delimiter: u8,
            include_header: bool,
            na_rep: &str,
        ) -> Result<()>;

        // -- Parquet I/O --

        /// Reads a Parquet file and returns a Table.
        fn read_parquet(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Writes a Table to a Parquet file.
        fn write_parquet(tbl: &Table, filepath: &str) -> Result<()>;

        // -- ORC I/O --

        /// Read an ORC file into a table.
        fn read_orc(filepath: &str) -> Result<UniquePtr<Table>>;
        /// Write a table to an ORC file.
        fn write_orc(tbl: &Table, filepath: &str) -> Result<()>;

        // -- JSON I/O --

        /// Read a JSON file into a table. Set json_lines=true for JSON Lines format.
        fn read_json(filepath: &str, json_lines: bool) -> Result<UniquePtr<Table>>;
        /// Write a table to a JSON file. Set json_lines=true for JSON Lines format.
        fn write_json(tbl: &Table, filepath: &str, json_lines: bool) -> Result<()>;

        // -- Avro I/O --

        /// Read an Avro file into a table.
        fn read_avro(filepath: &str) -> Result<UniquePtr<Table>>;

        // -- DLPack interop --

        /// Converts a DLPack DLManagedTensor pointer into a cudf Table.
        /// The pointer must point to a valid DLManagedTensor.
        fn from_dlpack(managed_tensor_ptr: usize, stream: usize) -> Result<UniquePtr<Table>>;
        /// Converts a cudf Table into a DLPack DLManagedTensor pointer.
        /// Returns the pointer as usize; caller must free via the tensor's deleter.
        fn to_dlpack(tbl: &Table, stream: usize) -> usize;
    }
}
