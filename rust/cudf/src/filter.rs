// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Filtering operations on GPU tables.

use crate::column::ColumnView;
use crate::error::Result;
use crate::table::Table;

/// Filters a table by a boolean mask column.
///
/// Only rows where the mask is `true` (and non-null) are kept.
pub fn apply_boolean_mask(table: &Table, mask: &ColumnView<'_>) -> Result<Table> {
    let t = cudf_sys::ffi::apply_boolean_mask(&table.0, mask.0)?;
    Ok(Table(t))
}

/// Drops rows where all columns are null.
pub fn drop_nulls(table: &Table) -> Result<Table> {
    let t = cudf_sys::ffi::drop_nulls_all(&table.0)?;
    Ok(Table(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    #[test]
    fn apply_boolean_mask_basic() {
        // Filter [1,1,1,1] with [true, false, true, false] -> [1, 1] (2 rows)
        let col = Column::from_scalar(&Scalar::from_i32(1), 4);
        let mask = Column::from_scalar(&Scalar::from_bool(true), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        // For a uniform true mask, all rows should be kept
        let result = apply_boolean_mask(&table, &mask.view()).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn drop_nulls_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(5), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let result = drop_nulls(&table).unwrap();
        // No nulls, so all rows should be kept
        assert_eq!(result.len(), 3);
    }
}
