// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reshape operations on tables and columns.

use crate::column::Column;
use crate::error::Result;
use crate::table::Table;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Interleaves columns of a table into a single column.
///
/// All columns must have the same type. The result interleaves
/// elements row-by-row: `[A1, B1, A2, B2, ...]`.
pub fn interleave_columns(table: &Table) -> Result<Column> {
    let col = cudf_sys::ffi::interleave_columns(&table.0, ds())?;
    Ok(Column(col))
}

/// Tiles (repeats) the rows of a table `count` times.
///
/// The output has `table.len() * count` rows.
pub fn tile(table: &Table, count: usize) -> Result<Table> {
    let tbl = cudf_sys::ffi::tile_table(&table.0, count as i32, ds())?;
    Ok(Table(tbl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::table::TableBuilder;

    #[test]
    fn interleave_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let c2 = Col::from_slice_i32(&[4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();

        let result = interleave_columns(&table).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.to_vec_i32(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn tile_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let result = tile(&table, 2).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }
}
