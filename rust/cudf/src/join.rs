// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use crate::Table;

/// Performs an inner join between two tables.
///
/// Returns a table containing all columns from both `left` and `right` for rows
/// where the key columns match. `left_on` and `right_on` are column indices
/// specifying the join keys.
pub fn inner_join(left: &Table, right: &Table, left_on: &[usize], right_on: &[usize]) -> Table {
    let left_keys: Vec<i32> = left_on.iter().map(|&i| i as i32).collect();
    let right_keys: Vec<i32> = right_on.iter().map(|&i| i as i32).collect();
    Table(
        cudf_sys::ffi::inner_join(&left.0, &right.0, &left_keys, &right_keys)
            .expect("inner_join failed"),
    )
}

/// Performs a left join between two tables.
///
/// Returns a table containing all columns from both `left` and `right`.
/// All rows from the left table are preserved; unmatched right columns are null.
pub fn left_join(left: &Table, right: &Table, left_on: &[usize], right_on: &[usize]) -> Table {
    let left_keys: Vec<i32> = left_on.iter().map(|&i| i as i32).collect();
    let right_keys: Vec<i32> = right_on.iter().map(|&i| i as i32).collect();
    Table(
        cudf_sys::ffi::left_join(&left.0, &right.0, &left_keys, &right_keys)
            .expect("left_join failed"),
    )
}

/// Performs a full outer join between two tables.
///
/// Returns a table containing all columns from both `left` and `right`.
/// All rows from both tables are preserved; unmatched columns are null.
pub fn full_join(left: &Table, right: &Table, left_on: &[usize], right_on: &[usize]) -> Table {
    let left_keys: Vec<i32> = left_on.iter().map(|&i| i as i32).collect();
    let right_keys: Vec<i32> = right_on.iter().map(|&i| i as i32).collect();
    Table(
        cudf_sys::ffi::full_join(&left.0, &right.0, &left_keys, &right_keys)
            .expect("full_join failed"),
    )
}

/// Performs a left semi join between two tables.
///
/// Returns a table containing only the columns from `left` for rows that have
/// at least one match in `right`.
pub fn left_semi_join(
    left: &Table,
    right: &Table,
    left_on: &[usize],
    right_on: &[usize],
) -> Table {
    let left_keys: Vec<i32> = left_on.iter().map(|&i| i as i32).collect();
    let right_keys: Vec<i32> = right_on.iter().map(|&i| i as i32).collect();
    Table(
        cudf_sys::ffi::left_semi_join(&left.0, &right.0, &left_keys, &right_keys)
            .expect("left_semi_join failed"),
    )
}

/// Performs a left anti join between two tables.
///
/// Returns a table containing only the columns from `left` for rows that have
/// no match in `right`.
pub fn left_anti_join(
    left: &Table,
    right: &Table,
    left_on: &[usize],
    right_on: &[usize],
) -> Table {
    let left_keys: Vec<i32> = left_on.iter().map(|&i| i as i32).collect();
    let right_keys: Vec<i32> = right_on.iter().map(|&i| i as i32).collect();
    Table(
        cudf_sys::ffi::left_anti_join(&left.0, &right.0, &left_keys, &right_keys)
            .expect("left_anti_join failed"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::table::TableBuilder;

    /// Helper to build a two-column table from i32 slices.
    fn make_two_col_table(keys: &[i32], vals: &[i32]) -> Table {
        let key_col = Column::from_slice_i32(keys);
        let val_col = Column::from_slice_i32(vals);
        let mut builder = TableBuilder::new();
        builder.push_column(key_col);
        builder.push_column(val_col);
        builder.build()
    }

    #[test]
    fn inner_join_basic() {
        // left: key=[1,2,3], val=[10,20,30]
        // right: key=[2,3,4], val=[200,300,400]
        // inner join on key: matches on 2,3
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);

        let result = inner_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4); // 2 from left + 2 from right
        assert_eq!(result.len(), 2); // matches on key=2 and key=3
    }

    #[test]
    fn left_join_basic() {
        // All left rows preserved, unmatched right is null
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);

        let result = left_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 3); // all left rows
    }

    #[test]
    fn full_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);

        let result = full_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4); // 1,2,3 from left + 4 from right
    }

    #[test]
    fn left_semi_join_basic() {
        // Returns left rows that have matches in right
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);

        let result = left_semi_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 2); // only left columns
        assert_eq!(result.len(), 2); // keys 2 and 3 match
    }

    #[test]
    fn left_anti_join_basic() {
        // Returns left rows that have NO matches in right
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);

        let result = left_anti_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 2); // only left columns
        assert_eq!(result.len(), 1); // only key=1 has no match
    }

    #[test]
    fn inner_join_empty_result() {
        // No matching keys -> empty result
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[4, 5, 6], &[400, 500, 600]);

        let result = inner_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn inner_join_all_match() {
        // All keys match
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);

        let result = inner_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn inner_join_duplicate_keys() {
        // Duplicate keys produce cartesian product for each key
        let left = make_two_col_table(&[1, 1, 2], &[10, 11, 20]);
        let right = make_two_col_table(&[1, 1, 3], &[100, 101, 300]);

        let result = inner_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        // key=1 matches: 2*2=4, key=2: 0, key=3: 0 => total 4
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn left_join_no_matches() {
        // Left join with no matches: all right columns should be null
        let left = make_two_col_table(&[1, 2], &[10, 20]);
        let right = make_two_col_table(&[3, 4], &[300, 400]);

        let result = left_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2); // both left rows preserved
    }

    #[test]
    fn full_join_no_overlap() {
        let left = make_two_col_table(&[1, 2], &[10, 20]);
        let right = make_two_col_table(&[3, 4], &[300, 400]);

        let result = full_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4); // 2 from each side
    }

    #[test]
    fn semi_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);

        let result = left_semi_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn anti_join_no_matches() {
        // When no right keys match, anti join returns all left rows
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[4, 5, 6], &[400, 500, 600]);

        let result = left_anti_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 3); // all left rows
    }

    #[test]
    fn anti_join_all_match() {
        // When all right keys match, anti join returns empty
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);

        let result = left_anti_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn inner_join_with_f64_values() {
        // Test with float columns as non-key data
        let key_left = Column::from_slice_i32(&[1, 2, 3]);
        let val_left = Column::from_slice_f64(&[1.1, 2.2, 3.3]);
        let key_right = Column::from_slice_i32(&[2, 3, 4]);
        let val_right = Column::from_slice_f64(&[20.0, 30.0, 40.0]);

        let mut lb = TableBuilder::new();
        lb.push_column(key_left);
        lb.push_column(val_left);
        let left = lb.build();

        let mut rb = TableBuilder::new();
        rb.push_column(key_right);
        rb.push_column(val_right);
        let right = rb.build();

        let result = inner_join(&left, &right, &[0], &[0]);
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn join_single_row_tables() {
        let left = make_two_col_table(&[1], &[10]);
        let right = make_two_col_table(&[1], &[100]);

        let result = inner_join(&left, &right, &[0], &[0]);
        assert_eq!(result.len(), 1);
        assert_eq!(result.columns_len(), 4);
    }
}
