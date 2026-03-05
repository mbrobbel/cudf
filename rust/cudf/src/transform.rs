// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Transform operations on columns.

use crate::column::Column;
use crate::error::Result;
use crate::table::Table;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Converts NaN values to null in a floating-point column.
///
/// Returns a new column where NaN values are replaced with null entries.
pub fn nans_to_nulls(col: &Column) -> Result<Column> {
    let view = cudf_sys::ffi::column_view_of(&col.0);
    let result = cudf_sys::ffi::nans_to_nulls(view, ds())?;
    Ok(Column(result))
}

/// Encodes table rows as integer indices into sorted distinct rows.
///
/// Returns a column of INT32 indices such that `keys[result[i]] == input[i]`.
pub fn encode(table: &Table) -> Result<Column> {
    let col = cudf_sys::ffi::encode_table(&table.0, ds())?;
    Ok(Column(col))
}

/// Returns the sorted distinct key rows from encoding.
pub fn encode_keys(table: &Table) -> Result<Table> {
    let tbl = cudf_sys::ffi::encode_keys(&table.0, ds())?;
    Ok(Table(tbl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::table::TableBuilder;

    #[test]
    fn nans_to_nulls_basic() {
        let data = [1.0_f64, f64::NAN, 3.0, f64::NAN, 5.0];
        let col = Col::from_slice_f64(&data);
        let result = nans_to_nulls(&col).unwrap();
        assert_eq!(result.len(), 5);
        assert!(result.has_nulls());
        assert_eq!(result.null_count(), 2);
    }

    #[test]
    fn encode_basic() {
        let c1 = Col::from_slice_i32(&[3, 1, 2, 1, 3]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build();

        let indices = encode(&table).unwrap();
        assert_eq!(indices.len(), 5);
        let vals = indices.to_vec_i32();
        // 1->0, 2->1, 3->2 in sorted order
        assert_eq!(vals, vec![2, 0, 1, 0, 2]);
    }
}
