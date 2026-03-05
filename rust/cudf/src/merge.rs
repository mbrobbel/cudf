// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Merge operations on sorted tables.

use crate::error::Result;
use crate::sorting::{NullOrder, Order};
use crate::table::Table;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Merges two sorted tables maintaining sort order.
///
/// Both input tables must be sorted by the key columns according to
/// the specified `orders` and `null_orders`. The result is a single
/// sorted table containing rows from both inputs.
pub fn merge(
    left: &Table,
    right: &Table,
    key_columns: &[i32],
    orders: &[Order],
    null_orders: &[NullOrder],
) -> Result<Table> {
    let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
    let nulls_i32 = unsafe { crate::enum_slice_as_i32(null_orders) };
    let tbl = cudf_sys::ffi::merge_tables(&left.0, &right.0, key_columns, orders_i32, nulls_i32, ds())?;
    Ok(Table(tbl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::table::TableBuilder;

    #[test]
    fn merge_sorted_tables() {
        // left: [1, 3, 5], right: [2, 4, 6]
        let c1 = Col::from_slice_i32(&[1, 3, 5]);
        let mut b1 = TableBuilder::new();
        b1.push_column(c1);
        let left = b1.build().unwrap();

        let c2 = Col::from_slice_i32(&[2, 4, 6]);
        let mut b2 = TableBuilder::new();
        b2.push_column(c2);
        let right = b2.build().unwrap();

        let result = merge(
            &left,
            &right,
            &[0i32],
            &[Order::ASCENDING],
            &[NullOrder::BEFORE],
        )
        .unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }
}
