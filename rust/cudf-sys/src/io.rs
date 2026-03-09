// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Table = crate::ffi::Table;

        /// Opaque table-with-metadata returned by `read_*_meta` functions.
        type TableWithMetadata;

        // -- TableWithMetadata accessors --

        /// Takes the table out of a TableWithMetadata (consuming the table).
        fn table_with_metadata_take_table(
            twm: Pin<&mut TableWithMetadata>,
        ) -> Result<UniquePtr<Table>>;

        /// Returns all column names.
        fn table_with_metadata_column_names(twm: &TableWithMetadata) -> Vec<String>;

        /// Returns the number of top-level columns in the metadata.
        fn table_with_metadata_num_columns(twm: &TableWithMetadata) -> i32;

        /// Returns the column name at `index`.
        fn table_with_metadata_column_name(twm: &TableWithMetadata, index: i32) -> Result<String>;

        // -- CSV I/O --

        /// Reads a CSV file and returns a Table.
        fn read_csv(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Reads a CSV file and returns a Table with column name metadata.
        fn read_csv_meta(filepath: &str) -> Result<UniquePtr<TableWithMetadata>>;

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

        /// Reads a Parquet file and returns a Table with column name metadata.
        fn read_parquet_meta(filepath: &str) -> Result<UniquePtr<TableWithMetadata>>;

        /// Reads a Parquet file with column selection and row range options.
        fn read_parquet_with_columns(
            filepath: &str,
            columns: &[&str],
            skip_rows: i64,
            num_rows: i64,
        ) -> Result<UniquePtr<TableWithMetadata>>;

        /// Writes a Table to a Parquet file.
        fn write_parquet(tbl: &Table, filepath: &str) -> Result<()>;

        /// Writes a Table to a Parquet file with column names.
        fn write_parquet_with_names(
            tbl: &Table,
            filepath: &str,
            column_names: &[&str],
        ) -> Result<()>;

        // -- ORC I/O --

        /// Read an ORC file into a table.
        fn read_orc(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Read an ORC file into a table with column name metadata.
        fn read_orc_meta(filepath: &str) -> Result<UniquePtr<TableWithMetadata>>;

        /// Write a table to an ORC file.
        fn write_orc(tbl: &Table, filepath: &str) -> Result<()>;

        // -- JSON I/O --

        /// Read a JSON file into a table. Set json_lines=true for JSON Lines format.
        fn read_json(filepath: &str, json_lines: bool) -> Result<UniquePtr<Table>>;

        /// Read a JSON file with metadata. Set json_lines=true for JSON Lines format.
        fn read_json_meta(
            filepath: &str,
            json_lines: bool,
        ) -> Result<UniquePtr<TableWithMetadata>>;

        /// Write a table to a JSON file. Set json_lines=true for JSON Lines format.
        fn write_json(tbl: &Table, filepath: &str, json_lines: bool) -> Result<()>;

        // -- Avro I/O --

        /// Read an Avro file into a table.
        fn read_avro(filepath: &str) -> Result<UniquePtr<Table>>;

        /// Read an Avro file with metadata.
        fn read_avro_meta(filepath: &str) -> Result<UniquePtr<TableWithMetadata>>;

        // -- DLPack interop --

        /// Converts a DLPack DLManagedTensor pointer into a cudf Table.
        /// The pointer must point to a valid DLManagedTensor.
        fn from_dlpack(managed_tensor_ptr: usize, stream: usize) -> Result<UniquePtr<Table>>;
        /// Converts a cudf Table into a DLPack DLManagedTensor pointer.
        /// Returns the pointer as usize; caller must free via the tensor's deleter.
        fn to_dlpack(tbl: &Table, stream: usize) -> usize;
    }
}
