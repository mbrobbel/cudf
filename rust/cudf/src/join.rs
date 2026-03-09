// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Join operations on GPU tables.
//!
//! Joins are available as methods on [`Table`](crate::table::Table):
//! `table.inner_join(...)`, `table.left_join(...)`, etc.

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::stream::GpuOp;
    use crate::table::{Table, TableBuilder};

    /// Helper to build a two-column table from i32 slices.
    fn make_two_col_table(keys: &[i32], vals: &[i32]) -> Table {
        let key_col = Column::from_slice_i32(keys).call().unwrap();
        let val_col = Column::from_slice_i32(vals).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(key_col);
        builder.push_column(val_col);
        builder.build().unwrap()
    }

    #[test]
    fn inner_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn left_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left.left_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn full_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left.full_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn left_semi_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left
            .left_semi_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn left_anti_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left
            .left_anti_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn inner_join_empty_result() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[4, 5, 6], &[400, 500, 600]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn inner_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn inner_join_duplicate_keys() {
        let left = make_two_col_table(&[1, 1, 2], &[10, 11, 20]);
        let right = make_two_col_table(&[1, 1, 3], &[100, 101, 300]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn left_join_no_matches() {
        let left = make_two_col_table(&[1, 2], &[10, 20]);
        let right = make_two_col_table(&[3, 4], &[300, 400]);
        let result = left.left_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn full_join_no_overlap() {
        let left = make_two_col_table(&[1, 2], &[10, 20]);
        let right = make_two_col_table(&[3, 4], &[300, 400]);
        let result = left.full_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn semi_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);
        let result = left
            .left_semi_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn anti_join_no_matches() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[4, 5, 6], &[400, 500, 600]);
        let result = left
            .left_anti_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn anti_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);
        let result = left
            .left_anti_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn inner_join_with_f64_values() {
        let key_left = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
        let val_left = Column::from_slice_f64(&[1.1, 2.2, 3.3]).call().unwrap();
        let key_right = Column::from_slice_i32(&[2, 3, 4]).call().unwrap();
        let val_right = Column::from_slice_f64(&[20.0, 30.0, 40.0]).call().unwrap();

        let mut lb = TableBuilder::new();
        lb.push_column(key_left);
        lb.push_column(val_left);
        let left = lb.build().unwrap();

        let mut rb = TableBuilder::new();
        rb.push_column(key_right);
        rb.push_column(val_right);
        let right = rb.build().unwrap();

        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn join_single_row_tables() {
        let left = make_two_col_table(&[1], &[10]);
        let right = make_two_col_table(&[1], &[100]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.columns_len(), 4);
    }
}
