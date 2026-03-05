// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GroupBy aggregation operations.

use crate::table::Table;

/// Aggregation operation kind for groupby.
pub use cudf_sys::ffi::AggregationKind;

/// Performs a single groupby aggregation on one value column.
///
/// Groups `table` by the columns at `key_columns` indices, then applies
/// `aggregation` to the column at `value_column` index.
///
/// Returns a table with `[key_columns..., aggregated_value_column]`.
pub fn groupby(
    table: &Table,
    key_columns: &[usize],
    value_column: usize,
    aggregation: AggregationKind,
) -> Table {
    let key_indices: Vec<i32> = key_columns.iter().map(|&i| i as i32).collect();
    let result = cudf_sys::ffi::groupby_single(
        &table.0,
        &key_indices,
        value_column as i32,
        aggregation.repr,
    )
    .expect("groupby_single failed");
    Table(result)
}

/// Performs groupby with multiple aggregations on multiple value columns.
///
/// Groups `table` by the columns at `key_columns` indices, then applies
/// each `aggregations[i]` to the column at `value_columns[i]`.
///
/// `value_columns` and `aggregations` must have the same length.
///
/// Returns a table with `[key_columns..., agg_result_0, agg_result_1, ...]`.
pub fn groupby_multi(
    table: &Table,
    key_columns: &[usize],
    value_columns: &[usize],
    aggregations: &[AggregationKind],
) -> Table {
    assert_eq!(
        value_columns.len(),
        aggregations.len(),
        "value_columns and aggregations must have the same length"
    );
    let key_indices: Vec<i32> = key_columns.iter().map(|&i| i as i32).collect();
    let val_indices: Vec<i32> = value_columns.iter().map(|&i| i as i32).collect();
    let agg_kinds: Vec<i32> = aggregations.iter().map(|a| a.repr).collect();
    let result = cudf_sys::ffi::groupby_multi(
        &table.0,
        &key_indices,
        &val_indices,
        &agg_kinds,
    )
    .expect("groupby_multi failed");
    Table(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::table::TableBuilder;

    /// Helper to build a two-column table: keys (i32) and values (i32).
    fn make_kv_table(keys: &[i32], values: &[i32]) -> Table {
        let key_col = Column::from_slice_i32(keys);
        let val_col = Column::from_slice_i32(values);
        let mut builder = TableBuilder::new();
        builder.push_column(key_col);
        builder.push_column(val_col);
        builder.build()
    }

    /// Extract column data as i32 from a table column using unary_cast identity trick.
    fn col_to_host_i32(tbl: &Table, col_idx: i32) -> Vec<i32> {
        let view = cudf_sys::ffi::table_get_column_view(&tbl.0, col_idx).unwrap();
        let col = cudf_sys::ffi::unary_cast(view, cudf_sys::ffi::column_view_type_id(view))
            .unwrap();
        cudf_sys::ffi::column_to_host_i32(&col)
    }

    /// Extract column data as i64 from a table column.
    fn col_to_host_i64(tbl: &Table, col_idx: i32) -> Vec<i64> {
        let view = cudf_sys::ffi::table_get_column_view(&tbl.0, col_idx).unwrap();
        let col = cudf_sys::ffi::unary_cast(view, cudf_sys::ffi::column_view_type_id(view))
            .unwrap();
        cudf_sys::ffi::column_to_host_i64(&col)
    }

    /// Extract column data as f64 from a table column.
    fn col_to_host_f64(tbl: &Table, col_idx: i32) -> Vec<f64> {
        let view = cudf_sys::ffi::table_get_column_view(&tbl.0, col_idx).unwrap();
        let col = cudf_sys::ffi::unary_cast(view, cudf_sys::ffi::column_view_type_id(view))
            .unwrap();
        cudf_sys::ffi::column_to_host_f64(&col)
    }

    /// Sort a groupby result by first column and extract (keys_i32, vals_i32).
    fn sorted_kv_i32(result: &Table) -> (Vec<i32>, Vec<i32>) {
        let sorted = crate::sorting::sort(
            result,
            &[crate::sorting::Order::Ascending],
            &[crate::sorting::NullOrder::After],
        );
        (col_to_host_i32(&sorted, 0), col_to_host_i32(&sorted, 1))
    }

    /// Sort a groupby result by first column and extract (keys_i32, vals_i64).
    fn sorted_kv_i64(result: &Table) -> (Vec<i32>, Vec<i64>) {
        let sorted = crate::sorting::sort(
            result,
            &[crate::sorting::Order::Ascending],
            &[crate::sorting::NullOrder::After],
        );
        (col_to_host_i32(&sorted, 0), col_to_host_i64(&sorted, 1))
    }

    /// Sort a groupby result by first column and extract (keys_i32, vals_f64).
    fn sorted_kv_f64(result: &Table) -> (Vec<i32>, Vec<f64>) {
        let sorted = crate::sorting::sort(
            result,
            &[crate::sorting::Order::Ascending],
            &[crate::sorting::NullOrder::After],
        );
        (col_to_host_i32(&sorted, 0), col_to_host_f64(&sorted, 1))
    }

    #[test]
    fn groupby_sum() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[1, 2, 3, 4]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::SUM);
        assert_eq!(result.columns_len(), 2);
        let (keys, vals) = sorted_kv_i64(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![3, 7]);
    }

    #[test]
    fn groupby_min() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[5, 2, 8, 3]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::MIN);
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn groupby_max() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[5, 2, 8, 3]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::MAX);
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![5, 8]);
    }

    #[test]
    fn groupby_mean() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[10, 20, 30, 50]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::MEAN);
        let (keys, vals) = sorted_kv_f64(&result);
        assert_eq!(keys, vec![1, 2]);
        assert!((vals[0] - 15.0).abs() < 1e-9);
        assert!((vals[1] - 40.0).abs() < 1e-9);
    }

    #[test]
    fn groupby_count() {
        let tbl = make_kv_table(&[1, 1, 1, 2, 2], &[10, 20, 30, 40, 50]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::COUNT);
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![3, 2]);
    }

    #[test]
    fn groupby_nunique() {
        let tbl = make_kv_table(&[1, 1, 1, 2, 2], &[10, 10, 20, 30, 30]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::NUNIQUE);
        let (keys, vals) = sorted_kv_i32(&result);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(vals, vec![2, 1]);
    }

    #[test]
    fn groupby_multi_agg() {
        let tbl = make_kv_table(&[1, 1, 2, 2], &[10, 20, 30, 40]);
        let result = groupby_multi(
            &tbl,
            &[0],
            &[1, 1],
            &[AggregationKind::SUM, AggregationKind::MIN],
        );
        assert_eq!(result.columns_len(), 3);

        let sorted = crate::sorting::sort(
            &result,
            &[crate::sorting::Order::Ascending],
            &[crate::sorting::NullOrder::After],
        );

        let keys = col_to_host_i32(&sorted, 0);
        let sums = col_to_host_i64(&sorted, 1);
        let mins = col_to_host_i32(&sorted, 2);

        assert_eq!(keys, vec![1, 2]);
        assert_eq!(sums, vec![30, 70]);
        assert_eq!(mins, vec![10, 30]);
    }

    #[test]
    fn groupby_single_group() {
        let tbl = make_kv_table(&[1, 1, 1, 1], &[10, 20, 30, 40]);
        let result = groupby(&tbl, &[0], 1, AggregationKind::SUM);
        assert_eq!(result.len(), 1);
        let (keys, vals) = sorted_kv_i64(&result);
        assert_eq!(keys, vec![1]);
        assert_eq!(vals, vec![100]);
    }
}
