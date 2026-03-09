// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CSV, Parquet, ORC, JSON, and Avro I/O operations.

use std::path::Path;

use crate::table::Table;

/// Column name metadata returned alongside a table from `read_with_metadata`.
#[derive(Debug, Clone)]
pub struct TableMetadata {
    /// Top-level column names.
    pub column_names: Vec<String>,
}

/// A table together with column name metadata.
pub struct TableWithMetadata {
    /// The table data.
    pub table: Table,
    /// Column name metadata.
    pub metadata: TableMetadata,
}

fn path_str(path: &Path) -> crate::Result<&str> {
    path.to_str().ok_or(crate::error::Error::InvalidPath)
}

/// CSV read and write operations.
pub mod csv {
    use super::{Path, Table, TableMetadata, TableWithMetadata, path_str};

    /// Reads a CSV file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let s = path_str(path.as_ref())?;
        let tbl = cudf_sys::io::ffi::read_csv(s)?;
        Ok(Table(tbl))
    }

    /// Reads a CSV file into a [`TableWithMetadata`] (table + column names).
    pub fn read_with_metadata<P: AsRef<Path>>(path: P) -> crate::Result<TableWithMetadata> {
        let s = path_str(path.as_ref())?;
        let mut twm = cudf_sys::io::ffi::read_csv_meta(s)?;
        let names = cudf_sys::io::ffi::table_with_metadata_column_names(&twm);
        let tbl = cudf_sys::io::ffi::table_with_metadata_take_table(twm.pin_mut())?;
        Ok(TableWithMetadata {
            table: Table(tbl),
            metadata: TableMetadata {
                column_names: names,
            },
        })
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
    pub fn read_with_options<P: AsRef<Path>>(path: P, opts: &ReadOptions) -> crate::Result<Table> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or(crate::error::Error::InvalidPath)?;
        let tbl = cudf_sys::io::ffi::read_csv_with_options(
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
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or(crate::error::Error::InvalidPath)?;
        cudf_sys::io::ffi::write_csv(&table.0, path_str)?;
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
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or(crate::error::Error::InvalidPath)?;
        cudf_sys::io::ffi::write_csv_with_options(
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
    use super::{Path, Table, TableMetadata, TableWithMetadata, path_str};

    /// Reads a Parquet file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let s = path_str(path.as_ref())?;
        let tbl = cudf_sys::io::ffi::read_parquet(s)?;
        Ok(Table(tbl))
    }

    /// Reads a Parquet file into a [`TableWithMetadata`] (table + column names).
    pub fn read_with_metadata<P: AsRef<Path>>(path: P) -> crate::Result<TableWithMetadata> {
        let s = path_str(path.as_ref())?;
        let mut twm = cudf_sys::io::ffi::read_parquet_meta(s)?;
        let names = cudf_sys::io::ffi::table_with_metadata_column_names(&twm);
        let tbl = cudf_sys::io::ffi::table_with_metadata_take_table(twm.pin_mut())?;
        Ok(TableWithMetadata {
            table: Table(tbl),
            metadata: TableMetadata {
                column_names: names,
            },
        })
    }

    /// Reads specific columns from a Parquet file.
    ///
    /// `columns` selects columns by name (empty = all). `skip_rows` skips
    /// leading rows. `num_rows` limits the result (-1 = all).
    pub fn read_columns<P: AsRef<Path>>(
        path: P,
        columns: &[&str],
        skip_rows: i64,
        num_rows: i64,
    ) -> crate::Result<TableWithMetadata> {
        let s = path_str(path.as_ref())?;
        let mut twm =
            cudf_sys::io::ffi::read_parquet_with_columns(s, columns, skip_rows, num_rows)?;
        let names = cudf_sys::io::ffi::table_with_metadata_column_names(&twm);
        let tbl = cudf_sys::io::ffi::table_with_metadata_take_table(twm.pin_mut())?;
        Ok(TableWithMetadata {
            table: Table(tbl),
            metadata: TableMetadata {
                column_names: names,
            },
        })
    }

    /// Writes a [`Table`] to a Parquet file.
    pub fn write<P: AsRef<Path>>(table: &Table, path: P) -> crate::Result<()> {
        let s = path_str(path.as_ref())?;
        cudf_sys::io::ffi::write_parquet(&table.0, s)?;
        Ok(())
    }

    /// Writes a [`Table`] to a Parquet file with column names.
    pub fn write_with_names<P: AsRef<Path>>(
        table: &Table,
        path: P,
        column_names: &[&str],
    ) -> crate::Result<()> {
        let s = path_str(path.as_ref())?;
        cudf_sys::io::ffi::write_parquet_with_names(&table.0, s, column_names)?;
        Ok(())
    }
}

/// JSON read and write operations.
pub mod json {
    use super::{Path, Table, TableMetadata, TableWithMetadata, path_str};

    /// Reads a JSON file into a [`Table`].
    ///
    /// Set `json_lines` to `true` for JSON Lines (newline-delimited) format.
    pub fn read<P: AsRef<Path>>(path: P, json_lines: bool) -> crate::Result<Table> {
        let s = path_str(path.as_ref())?;
        let tbl = cudf_sys::io::ffi::read_json(s, json_lines)?;
        Ok(Table(tbl))
    }

    /// Reads a JSON file into a [`TableWithMetadata`] (table + column names).
    pub fn read_with_metadata<P: AsRef<Path>>(
        path: P,
        json_lines: bool,
    ) -> crate::Result<TableWithMetadata> {
        let s = path_str(path.as_ref())?;
        let mut twm = cudf_sys::io::ffi::read_json_meta(s, json_lines)?;
        let names = cudf_sys::io::ffi::table_with_metadata_column_names(&twm);
        let tbl = cudf_sys::io::ffi::table_with_metadata_take_table(twm.pin_mut())?;
        Ok(TableWithMetadata {
            table: Table(tbl),
            metadata: TableMetadata {
                column_names: names,
            },
        })
    }

    /// Writes a [`Table`] to a JSON file.
    ///
    /// Set `json_lines` to `true` for JSON Lines (newline-delimited) format.
    pub fn write<P: AsRef<Path>>(table: &Table, path: P, json_lines: bool) -> crate::Result<()> {
        let s = path_str(path.as_ref())?;
        cudf_sys::io::ffi::write_json(&table.0, s, json_lines)?;
        Ok(())
    }
}

/// Avro read operations.
pub mod avro {
    use super::{Path, Table, TableMetadata, TableWithMetadata, path_str};

    /// Reads an Avro file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let s = path_str(path.as_ref())?;
        let tbl = cudf_sys::io::ffi::read_avro(s)?;
        Ok(Table(tbl))
    }

    /// Reads an Avro file into a [`TableWithMetadata`] (table + column names).
    pub fn read_with_metadata<P: AsRef<Path>>(path: P) -> crate::Result<TableWithMetadata> {
        let s = path_str(path.as_ref())?;
        let mut twm = cudf_sys::io::ffi::read_avro_meta(s)?;
        let names = cudf_sys::io::ffi::table_with_metadata_column_names(&twm);
        let tbl = cudf_sys::io::ffi::table_with_metadata_take_table(twm.pin_mut())?;
        Ok(TableWithMetadata {
            table: Table(tbl),
            metadata: TableMetadata {
                column_names: names,
            },
        })
    }
}

/// ORC read and write operations.
pub mod orc {
    use super::{Path, Table, TableMetadata, TableWithMetadata, path_str};

    /// Reads an ORC file into a [`Table`].
    pub fn read<P: AsRef<Path>>(path: P) -> crate::Result<Table> {
        let s = path_str(path.as_ref())?;
        let tbl = cudf_sys::io::ffi::read_orc(s)?;
        Ok(Table(tbl))
    }

    /// Reads an ORC file into a [`TableWithMetadata`] (table + column names).
    pub fn read_with_metadata<P: AsRef<Path>>(path: P) -> crate::Result<TableWithMetadata> {
        let s = path_str(path.as_ref())?;
        let mut twm = cudf_sys::io::ffi::read_orc_meta(s)?;
        let names = cudf_sys::io::ffi::table_with_metadata_column_names(&twm);
        let tbl = cudf_sys::io::ffi::table_with_metadata_take_table(twm.pin_mut())?;
        Ok(TableWithMetadata {
            table: Table(tbl),
            metadata: TableMetadata {
                column_names: names,
            },
        })
    }

    /// Writes a [`Table`] to an ORC file.
    pub fn write<P: AsRef<Path>>(table: &Table, path: P) -> crate::Result<()> {
        let s = path_str(path.as_ref())?;
        cudf_sys::io::ffi::write_orc(&table.0, s)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    fn make_test_table() -> Table {
        let c1 = Column::from_scalar(&Scalar::from_i32(42), 3)
            .call()
            .unwrap();
        let c2 = Column::from_scalar(&Scalar::from_f64(3.14), 3)
            .call()
            .unwrap();
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
