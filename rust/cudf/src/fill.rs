// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Fill, repeat, and sequence operations on GPU columns and tables.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::table::Table;

/// Fills the range [begin, end) in a column with a scalar value (out-of-place).
pub fn fill(col: &ColumnView<'_>, begin: usize, end: usize, value: &Scalar) -> Result<Column> {
    let c = cudf_sys::ffi::fill_column(col.0, begin as i32, end as i32, &value.0)?;
    Ok(Column(c))
}

/// Repeats each row of a table `count` times.
pub fn repeat(table: &Table, count: usize) -> Result<Table> {
    let t = cudf_sys::ffi::repeat_table(&table.0, count as i32)?;
    Ok(Table(t))
}

/// Generates an arithmetic sequence: [init, init+step, init+2*step, ...].
pub fn sequence(count: usize, init: &Scalar, step: &Scalar) -> Result<Column> {
    let c = cudf_sys::ffi::sequence_column(count as i32, &init.0, &step.0)?;
    Ok(Column(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    #[test]
    fn fill_range() {
        // Create column [1, 1, 1, 1, 1], fill [1,3) with 99
        let col = Column::from_scalar(&Scalar::from_i32(1), 5);
        let value = Scalar::from_i32(99);
        let result = fill(&col.view(), 1, 3, &value).unwrap();
        assert_eq!(result.to_vec_i32(), vec![1, 99, 99, 1, 1]);
    }

    #[test]
    fn repeat_table_test() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30]));
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let result = repeat(&table, 2).unwrap();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn sequence_i32() {
        let init = Scalar::from_i32(0);
        let step = Scalar::from_i32(2);
        let result = sequence(5, &init, &step).unwrap();
        assert_eq!(result.to_vec_i32(), vec![0, 2, 4, 6, 8]);
    }
}
