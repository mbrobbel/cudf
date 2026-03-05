// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Sorting operations on GPU tables.

use crate::column::Column;
use crate::error::Result;
use crate::table::Table;

pub use cudf_sys::ffi::{NullOrder, Order};

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Sorts a table by all columns using the given orders.
///
/// `column_orders` and `null_orders` should have one entry per column.
/// If empty, all columns are sorted ascending with nulls before.
pub fn sort(
    table: &Table,
    column_orders: &[Order],
    null_orders: &[NullOrder],
) -> Result<Table> {
    let orders_i32: Vec<i32> = column_orders.iter().map(|o| o.repr).collect();
    let nulls_i32: Vec<i32> = null_orders.iter().map(|n| n.repr).collect();
    let t = cudf_sys::ffi::sort_table(&table.0, &orders_i32, &nulls_i32, ds())?;
    Ok(Table(t))
}

/// Sorts a table by all columns in ascending order with nulls before.
pub fn sort_ascending(table: &Table) -> Result<Table> {
    sort(table, &[], &[])
}

/// Returns the sorted row indices of a table.
pub fn sorted_order(
    table: &Table,
    column_orders: &[Order],
    null_orders: &[NullOrder],
) -> Result<Column> {
    let orders_i32: Vec<i32> = column_orders.iter().map(|o| o.repr).collect();
    let nulls_i32: Vec<i32> = null_orders.iter().map(|n| n.repr).collect();
    let col = cudf_sys::ffi::sorted_order(&table.0, &orders_i32, &nulls_i32, ds())?;
    Ok(Column(col))
}

/// Returns whether the table rows are sorted according to the given orders.
pub fn is_sorted(
    table: &Table,
    column_orders: &[Order],
    null_orders: &[NullOrder],
) -> Result<bool> {
    let orders_i32: Vec<i32> = column_orders.iter().map(|o| o.repr).collect();
    let nulls_i32: Vec<i32> = null_orders.iter().map(|n| n.repr).collect();
    cudf_sys::ffi::is_sorted_table(&table.0, &orders_i32, &nulls_i32, ds()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    #[test]
    fn sort_single_column_ascending() {
        // Column of all-same values; sorted should be identical
        let col = Col::from_scalar(&Scalar::from_i32(5), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let sorted = sort_ascending(&table).unwrap();
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted.columns_len(), 1);
    }

    #[test]
    fn sort_descending() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let sorted = sort(&table, &[Order::DESCENDING], &[NullOrder::AFTER]).unwrap();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn sorted_order_basic() {
        let col = Col::from_scalar(&Scalar::from_i32(1), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let indices = sorted_order(&table, &[], &[]).unwrap();
        assert_eq!(indices.len(), 3);
        assert_eq!(indices.type_id(), TypeId::INT32);
    }

    #[test]
    fn is_sorted_true_for_constant() {
        let col = Col::from_scalar(&Scalar::from_i32(7), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let result = is_sorted(&table, &[Order::ASCENDING], &[NullOrder::BEFORE]).unwrap();
        assert!(result);
    }

    #[test]
    fn is_sorted_descending_constant() {
        let col = Col::from_scalar(&Scalar::from_i32(7), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build();
        let result = is_sorted(&table, &[Order::DESCENDING], &[NullOrder::AFTER]).unwrap();
        assert!(result);
    }
}
