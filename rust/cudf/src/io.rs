// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CSV, Parquet, and ORC I/O operations.

use std::path::Path;

use crate::Table;

/// CSV read and write operations.
pub mod csv {
    use super::*;

    /// Reads a CSV file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        let tbl = cudf_sys::ffi::read_csv(path_str)?;
        Ok(Table(tbl))
    }

    /// Options for reading a CSV file.
    pub struct ReadOptions {
        /// Field delimiter character (default: `b','`).
        pub delimiter: u8,
        /// Whether the first row is a header (default: `true`).
        pub header: bool,
        /// Number of rows to skip from the start (default: `0`).
        pub skip_rows: i32,
        /// Maximum number of rows to read, or `-1` for all (default: `-1`).
        pub num_rows: i32,
    }

    impl Default for ReadOptions {
        fn default() -> Self {
            Self {
                delimiter: b',',
                header: true,
                skip_rows: 0,
                num_rows: -1,
            }
        }
    }

    /// Reads a CSV file into a [`Table`] with the given options.
    pub fn read_with_options<P: AsRef<Path>>(
        path: P,
        opts: &ReadOptions,
    ) -> crate::Result<Table> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        let tbl = cudf_sys::ffi::read_csv_with_options(
            path_str,
            opts.delimiter,
            opts.header,
            opts.skip_rows,
            opts.num_rows,
        )?;
        Ok(Table(tbl))
    }

    /// Writes a [`Table`] to a CSV file.
    pub fn write<P: AsRef<Path>>(table: &Table, path: P) -> crate::Result<()> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        cudf_sys::ffi::write_csv(&table.0, path_str)?;
        Ok(())
    }

    /// Options for writing a CSV file.
    pub struct WriteOptions<'a> {
        /// Field delimiter character (default: `b','`).
        pub delimiter: u8,
        /// Whether to include a header row (default: `true`).
        pub include_header: bool,
        /// String representation for null values (default: `""`).
        pub na_rep: &'a str,
    }

    impl Default for WriteOptions<'_> {
        fn default() -> Self {
            Self {
                delimiter: b',',
                include_header: true,
                na_rep: "",
            }
        }
    }

    /// Writes a [`Table`] to a CSV file with the given options.
    pub fn write_with_options<P: AsRef<Path>>(
        table: &Table,
        path: P,
        opts: &WriteOptions<'_>,
    ) -> crate::Result<()> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        cudf_sys::ffi::write_csv_with_options(
            &table.0,
            path_str,
            opts.delimiter,
            opts.include_header,
            opts.na_rep,
        )?;
        Ok(())
    }
}

/// Parquet read and write operations.
pub mod parquet {
    use super::*;

    /// Reads a Parquet file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        let tbl = cudf_sys::ffi::read_parquet(path_str)?;
        Ok(Table(tbl))
    }

    /// Writes a [`Table`] to a Parquet file.
    pub fn write<P: AsRef<Path>>(table: &Table, path: P) -> crate::Result<()> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        cudf_sys::ffi::write_parquet(&table.0, path_str)?;
        Ok(())
    }
}

/// ORC read and write operations.
pub mod orc {
    use super::*;

    /// Reads an ORC file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        let tbl = cudf_sys::ffi::read_orc(path_str)?;
        Ok(Table(tbl))
    }

    /// Writes a [`Table`] to an ORC file.
    pub fn write<P: AsRef<Path>>(table: &Table, path: P) -> crate::Result<()> {
        let path_str = path.as_ref().to_str().ok_or(crate::error::Error::InvalidPath)?;
        cudf_sys::ffi::write_orc(&table.0, path_str)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, Scalar, TableBuilder};

    fn make_test_table() -> Table {
        let c1 = Column::from_scalar(&Scalar::from_i32(42), 3);
        let c2 = Column::from_scalar(&Scalar::from_f64(3.14), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        builder.build().unwrap()
    }

    #[test]
    fn csv_roundtrip() {
        std::fs::create_dir_all("/tmp/claude").ok();
        let table = make_test_table();
        let path = "/tmp/claude/test_io_roundtrip.csv";
        csv::write(&table, path).unwrap();
        let loaded = csv::read(path).unwrap();
        assert_eq!(loaded.columns_len(), 2);
        assert_eq!(loaded.len(), 3);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn csv_write_with_options() {
        std::fs::create_dir_all("/tmp/claude").ok();
        let table = make_test_table();
        let path = "/tmp/claude/test_io_csv_opts.csv";
        let opts = csv::WriteOptions {
            delimiter: b'\t',
            include_header: false,
            na_rep: "N/A",
        };
        csv::write_with_options(&table, path, &opts).unwrap();
        let read_opts = csv::ReadOptions {
            delimiter: b'\t',
            header: false,
            ..Default::default()
        };
        let loaded = csv::read_with_options(path, &read_opts).unwrap();
        assert_eq!(loaded.columns_len(), 2);
        assert_eq!(loaded.len(), 3);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn parquet_roundtrip() {
        std::fs::create_dir_all("/tmp/claude").ok();
        let table = make_test_table();
        let path = "/tmp/claude/test_io_roundtrip.parquet";
        parquet::write(&table, path).unwrap();
        let loaded = parquet::read(path).unwrap();
        assert_eq!(loaded.columns_len(), 2);
        assert_eq!(loaded.len(), 3);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn csv_read_nonexistent_file() {
        let result = csv::read("/tmp/claude/nonexistent_io.csv");
        assert!(result.is_err());
    }

    #[test]
    fn parquet_read_nonexistent_file() {
        let result = parquet::read("/tmp/claude/nonexistent_io.parquet");
        assert!(result.is_err());
    }
}
