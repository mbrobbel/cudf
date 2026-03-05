// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Concatenation operations for GPU columns and tables.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::table::Table;

/// Concatenates multiple columns vertically into a single column.
///
/// All columns must have the same type.
pub fn concatenate_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    let mut cat = cudf_sys::ffi::new_column_concatenator();
    for col in columns {
        cudf_sys::ffi::column_concatenator_add(cat.pin_mut(), col.0);
    }
    let c = cudf_sys::ffi::column_concatenator_finish(cat.pin_mut())?;
    Ok(Column(c))
}

/// Concatenates multiple tables vertically into a single table.
///
/// All tables must have the same number of columns and matching types.
pub fn concatenate_tables(tables: &[&Table]) -> Result<Table> {
    let mut cat = cudf_sys::ffi::new_table_concatenator();
    for t in tables {
        cudf_sys::ffi::table_concatenator_add(cat.pin_mut(), &t.0);
    }
    let t = cudf_sys::ffi::table_concatenator_finish(cat.pin_mut())?;
    Ok(Table(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    #[test]
    fn concat_two_columns() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2);
        let c2 = Col::from_scalar(&Scalar::from_i32(3), 2);
        let result = concatenate_columns(&[&c1.view(), &c2.view()]).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn concat_two_tables() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2);
        let c2 = Col::from_scalar(&Scalar::from_i32(3), 2);
        let mut b1 = TableBuilder::new();
        b1.push_column(c1);
        let t1 = b1.build();
        let mut b2 = TableBuilder::new();
        b2.push_column(c2);
        let t2 = b2.build();
        let result = concatenate_tables(&[&t1, &t2]).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.columns_len(), 1);
    }

    #[test]
    fn concat_empty_column() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 3);
        let c2 = Col::empty(TypeId::INT32);
        let result = concatenate_columns(&[&c1.view(), &c2.view()]).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32(), vec![1, 1, 1]);
    }
}
