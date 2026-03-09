// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Search operations on GPU columns and tables.
//!
//! All methods are defined as inherent methods on their respective types
//! and follow the builder pattern:
//!
//! - [`ColumnView::contains_scalar`](crate::column::ColumnView::contains_scalar)
//!   -- returns `bool` indicating whether a scalar exists in the column.
//! - [`ColumnView::contains_column`](crate::column::ColumnView::contains_column)
//!   -- returns a `BOOL8` column with per-element membership results.
//! - [`Table::lower_bound`](crate::table::Table::lower_bound) -- for a sorted
//!   table, finds the first insertion position for each needle row.
//! - [`Table::upper_bound`](crate::table::Table::upper_bound) -- for a sorted
//!   table, finds the position after the last matching element for each needle
//!   row.

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;

    #[test]
    fn contains_found() {
        let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
            .call()
            .unwrap();
        let needle = Scalar::from_i32(20);
        assert!(col.view().contains_scalar(&needle).call().unwrap());
    }

    #[test]
    fn contains_not_found() {
        let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
            .call()
            .unwrap();
        let needle = Scalar::from_i32(25);
        assert!(!col.view().contains_scalar(&needle).call().unwrap());
    }

    #[test]
    fn contains_column_test() {
        let haystack = Column::from_slice_i32(&[10, 20, 30, 40, 50])
            .call()
            .unwrap();
        let needles = Column::from_slice_i32(&[20, 40, 60, 80]).call().unwrap();
        let result = haystack
            .view()
            .contains_column(&needles.view())
            .call()
            .unwrap();
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn lower_bound_basic() {
        use crate::sorting::{NullOrder, Order};
        use crate::table::TableBuilder;

        // Sorted haystack: [10, 20, 30, 40, 50]
        let hay_col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
            .call()
            .unwrap();
        let mut hb = TableBuilder::new();
        hb.push_column(hay_col);
        let haystack = hb.build().unwrap();

        // Needles: [15, 30, 55]
        let needle_col = Column::from_slice_i32(&[15, 30, 55]).call().unwrap();
        let mut nb = TableBuilder::new();
        nb.push_column(needle_col);
        let needles = nb.build().unwrap();

        let result = haystack
            .lower_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 2, 5]);
    }

    #[test]
    fn upper_bound_basic() {
        use crate::sorting::{NullOrder, Order};
        use crate::table::TableBuilder;

        let hay_col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
            .call()
            .unwrap();
        let mut hb = TableBuilder::new();
        hb.push_column(hay_col);
        let haystack = hb.build().unwrap();

        let needle_col = Column::from_slice_i32(&[15, 30, 55]).call().unwrap();
        let mut nb = TableBuilder::new();
        nb.push_column(needle_col);
        let needles = nb.build().unwrap();

        let result = haystack
            .upper_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 3, 5]);
    }
}
