// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Merge operations on sorted tables.
//!
//! Merges two tables that are each independently sorted by the same key
//! columns, producing a single sorted output table. This is the merge step of
//! a merge-sort and runs in O(n + m) time.
//!
//! Available as a method on [`Table`](crate::table::Table):
//!
//! * [`Table::merge`](crate::table::Table::merge) -- merge two pre-sorted
//!   tables while maintaining sort order.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::sorting::{NullOrder, Order};
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn merge_sorted_tables() {
        let c1 = Col::from_slice_i32(&[1, 3, 5]).call().unwrap();
        let mut b1 = TableBuilder::new();
        b1.push_column(c1);
        let left = b1.build().unwrap();

        let c2 = Col::from_slice_i32(&[2, 4, 6]).call().unwrap();
        let mut b2 = TableBuilder::new();
        b2.push_column(c2);
        let right = b2.build().unwrap();

        let result = left
            .merge(&right, &[0i32], &[Order::ASCENDING], &[NullOrder::BEFORE])
            .call()
            .unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }
}
