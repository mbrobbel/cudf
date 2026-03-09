// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Merge operations on sorted tables.
//!
//! Available as a method on [`Table`](crate::table::Table): `table.merge(...)`.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::sorting::{NullOrder, Order};
    use crate::table::TableBuilder;

    #[test]
    fn merge_sorted_tables() {
        let c1 = Col::from_slice_i32(&[1, 3, 5]);
        let mut b1 = TableBuilder::new();
        b1.push_column(c1);
        let left = b1.build().unwrap();

        let c2 = Col::from_slice_i32(&[2, 4, 6]);
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
