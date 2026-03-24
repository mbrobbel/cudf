// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the cudf safe API.
//!
//! These tests exercise multi-step workflows that combine operations
//! across modules, validating end-to-end correctness of the GPU pipeline.

#![allow(clippy::unwrap_used)]
#![allow(clippy::approx_constant)]

use cudf::column::Column;
use cudf::data_type::TypeId;
use cudf::datetime::{DatetimeExt, RoundingFrequency};
use cudf::groupby::AggregationKind;
use cudf::scalar::Scalar;
use cudf::sorting::{NullOrder, Order, RankMethod};
use cudf::stream::GpuOp;
use cudf::strings::{SideType, StringExt};
use cudf::table::{Table, TableBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_table(columns: Vec<Column>) -> Table {
    let mut b = TableBuilder::new();
    for c in columns {
        b.push_column(c);
    }
    b.build().unwrap()
}

/// Extract a table column to host as i32 (via cast to get an owning Column).
fn table_col_i32(tbl: &Table, idx: usize) -> Vec<i32> {
    tbl.column(idx)
        .unwrap()
        .cast(TypeId::INT32)
        .call()
        .unwrap()
        .to_vec_i32()
        .call()
        .unwrap()
}

/// Extract a table column to host as i64.
fn table_col_i64(tbl: &Table, idx: usize) -> Vec<i64> {
    tbl.column(idx)
        .unwrap()
        .cast(TypeId::INT64)
        .call()
        .unwrap()
        .to_vec_i64()
        .call()
        .unwrap()
}

/// Extract a table column to host as f64.
fn table_col_f64(tbl: &Table, idx: usize) -> Vec<f64> {
    tbl.column(idx)
        .unwrap()
        .cast(TypeId::FLOAT64)
        .call()
        .unwrap()
        .to_vec_f64()
        .call()
        .unwrap()
}

/// Sort a table and return it (for deterministic result comparison).
fn sorted(tbl: &Table) -> Table {
    tbl.sort(&[], &[]).call().unwrap()
}

// ===========================================================================
// Test: Column arithmetic pipeline
// ===========================================================================

#[test]
fn column_arithmetic_pipeline() {
    let a = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let b = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call().unwrap();

    // Element-wise add
    let sum_col = a.view().add(&b.view(), TypeId::INT32).call().unwrap();
    assert_eq!(sum_col.to_vec_i32().call().unwrap(), [11, 22, 33, 44, 55]);

    // Reduce to scalar
    let total = sum_col.view().sum(TypeId::INT64).call().unwrap();
    assert_eq!(total.as_i64(), Some(165));

    // Min / Max
    let min = sum_col.view().min(TypeId::INT32).call().unwrap();
    let max = sum_col.view().max(TypeId::INT32).call().unwrap();
    assert_eq!(min.as_i32(), Some(11));
    assert_eq!(max.as_i32(), Some(55));
}

// ===========================================================================
// Test: Cast → math → reduce
// ===========================================================================

#[test]
fn cast_and_floating_point_math() {
    let ints = Column::from_slice_i32(&[1, 4, 9, 16, 25]).call().unwrap();

    // Cast to f64, compute sqrt
    let floats = ints.view().cast(TypeId::FLOAT64).call().unwrap();
    let roots = floats.view().sqrt().call().unwrap();
    let result = roots.to_vec_f64().call().unwrap();
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0]);

    // Sum of square roots
    let total = roots.view().sum(TypeId::FLOAT64).call().unwrap();
    assert_eq!(total.as_f64(), Some(15.0));
}

// ===========================================================================
// Test: Table sort → gather → verify order
// ===========================================================================

#[test]
fn sort_and_verify_order() {
    let ids = Column::from_slice_i32(&[3, 1, 4, 1, 5]).call().unwrap();
    let vals = Column::from_slice_f64(&[30.0, 10.0, 40.0, 11.0, 50.0])
        .call()
        .unwrap();
    let tbl = build_table(vec![ids, vals]);

    let asc = tbl
        .sort(
            &[Order::ASCENDING, Order::ASCENDING],
            &[NullOrder::BEFORE, NullOrder::BEFORE],
        )
        .call()
        .unwrap();
    assert_eq!(table_col_i32(&asc, 0), [1, 1, 3, 4, 5]);
    assert_eq!(table_col_f64(&asc, 1), [10.0, 11.0, 30.0, 40.0, 50.0]);

    let desc = tbl
        .sort(
            &[Order::DESCENDING, Order::DESCENDING],
            &[NullOrder::AFTER, NullOrder::AFTER],
        )
        .call()
        .unwrap();
    assert_eq!(table_col_i32(&desc, 0), [5, 4, 3, 1, 1]);
}

// ===========================================================================
// Test: Filter rows with boolean mask
// ===========================================================================

#[test]
fn filter_rows_by_comparison() {
    let values = Column::from_slice_i32(&[10, 25, 5, 30, 15]).call().unwrap();
    let tbl = build_table(vec![values]);

    // Build mask: values > 12
    let threshold = Column::from_scalar(&Scalar::from_i32(12), 5)
        .call()
        .unwrap();
    let mask = tbl.column(0).unwrap().gt(&threshold.view()).call().unwrap();

    let filtered = tbl.filter(&mask.view()).call().unwrap();
    assert_eq!(filtered.len(), 3);

    let result = sorted(&filtered);
    assert_eq!(table_col_i32(&result, 0), [15, 25, 30]);
}

// ===========================================================================
// Test: GroupBy sum / count / mean
// ===========================================================================

#[test]
fn groupby_aggregate_sum_and_count() {
    // keys: [a, b, a, b, a], values: [10, 20, 30, 40, 50]
    let keys = Column::from_slice_i32(&[1, 2, 1, 2, 1]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    // SUM by key
    let sum_result = tbl.groupby(&[0], 1, AggregationKind::SUM).call().unwrap();
    let sum_sorted = sorted(&sum_result);
    assert_eq!(table_col_i32(&sum_sorted, 0), [1, 2]); // keys
    assert_eq!(table_col_i64(&sum_sorted, 1), [90, 60]); // sums

    // COUNT by key
    let count_result = tbl.groupby(&[0], 1, AggregationKind::COUNT).call().unwrap();
    let count_sorted = sorted(&count_result);
    assert_eq!(table_col_i32(&count_sorted, 0), [1, 2]);
    assert_eq!(table_col_i32(&count_sorted, 1), [3, 2]);
}

// ===========================================================================
// Test: GroupBy with multiple aggregations
// ===========================================================================

#[test]
fn groupby_multi_aggregation() {
    let keys = Column::from_slice_i32(&[1, 2, 1, 2, 1]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    let result = tbl
        .groupby_multi(&[0], &[1, 1], &[AggregationKind::MIN, AggregationKind::MAX])
        .call()
        .unwrap();
    let result = sorted(&result);

    // keys, min, max
    assert_eq!(result.columns_len(), 3);
    assert_eq!(table_col_i32(&result, 0), [1, 2]); // keys
    assert_eq!(table_col_i32(&result, 1), [10, 20]); // min
    assert_eq!(table_col_i32(&result, 2), [50, 40]); // max
}

// ===========================================================================
// Test: Inner join
// ===========================================================================

#[test]
fn inner_join_two_tables() {
    let left_keys = Column::from_slice_i32(&[1, 2, 3, 4]).call().unwrap();
    let left_vals = Column::from_slice_f64(&[10.0, 20.0, 30.0, 40.0])
        .call()
        .unwrap();
    let left = build_table(vec![left_keys, left_vals]);

    let right_keys = Column::from_slice_i32(&[2, 4, 5]).call().unwrap();
    let right_vals = Column::from_slice_f64(&[200.0, 400.0, 500.0])
        .call()
        .unwrap();
    let right = build_table(vec![right_keys, right_vals]);

    let joined = left.inner_join(&right, &[0], &[0]).call().unwrap();

    assert_eq!(joined.len(), 2);
    let joined = sorted(&joined);
    // left_key, left_val, right_key, right_val
    assert_eq!(table_col_i32(&joined, 0), [2, 4]);
    assert_eq!(table_col_f64(&joined, 1), [20.0, 40.0]);
    assert_eq!(table_col_f64(&joined, 3), [200.0, 400.0]);
}

// ===========================================================================
// Test: Left join with unmatched rows
// ===========================================================================

#[test]
fn left_join_preserves_all_left_rows() {
    let left_keys = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let left_vals = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
    let left = build_table(vec![left_keys, left_vals]);

    let right_keys = Column::from_slice_i32(&[2]).call().unwrap();
    let right_vals = Column::from_slice_i32(&[200]).call().unwrap();
    let right = build_table(vec![right_keys, right_vals]);

    let joined = left.left_join(&right, &[0], &[0]).call().unwrap();

    // All 3 left rows preserved
    assert_eq!(joined.len(), 3);
    let joined = sorted(&joined);
    assert_eq!(table_col_i32(&joined, 0), [1, 2, 3]);
    assert_eq!(table_col_i32(&joined, 1), [10, 20, 30]);
}

// ===========================================================================
// Test: String processing pipeline
// ===========================================================================

#[test]
fn string_case_and_contains() {
    let col = Column::from_strings(&["Hello", "WORLD", "foo Bar"])
        .call()
        .unwrap();

    let lower = col.view().to_lower().call().unwrap();
    assert_eq!(
        lower.to_vec_string().call().unwrap(),
        ["hello", "world", "foo bar"]
    );

    let upper = col.view().to_upper().call().unwrap();
    assert_eq!(
        upper.to_vec_string().call().unwrap(),
        ["HELLO", "WORLD", "FOO BAR"]
    );

    // Check which strings contain "o"
    let target = Scalar::from_string("o");
    let mask = col.view().str_contains(&target).call().unwrap();
    assert_eq!(mask.to_vec_bool().call().unwrap(), [true, false, true]);
}

#[test]
fn string_split_and_length() {
    let col = Column::from_strings(&["a,b,c", "x,y", "hello"])
        .call()
        .unwrap();

    // Count characters
    let lengths = col.view().count_characters().call().unwrap();
    assert_eq!(lengths.to_vec_i32().call().unwrap(), [5, 3, 5]);

    // Split by comma
    let delimiter = Scalar::from_string(",");
    let parts = col.view().str_split(&delimiter, -1).call().unwrap();
    assert_eq!(parts.columns_len(), 3); // max 3 parts
}

#[test]
fn string_regex_operations() {
    let col = Column::from_strings(&["abc123", "def", "456ghi", "789"])
        .call()
        .unwrap();

    // Check regex contains digits
    let has_digits = col.view().str_contains_re("\\d+").call().unwrap();
    assert_eq!(
        has_digits.to_vec_bool().call().unwrap(),
        [true, false, true, true]
    );

    // Count digit sequences
    let counts = col.view().str_count_re("\\d+").call().unwrap();
    assert_eq!(counts.to_vec_i32().call().unwrap(), [1, 0, 1, 1]);

    // Replace digits with "#"
    let replaced = col.view().str_replace_re("\\d+", "#").call().unwrap();
    assert_eq!(
        replaced.to_vec_string().call().unwrap(),
        ["abc#", "def", "#ghi", "#"]
    );
}

// ===========================================================================
// Test: Null handling workflow
// ===========================================================================

#[test]
fn create_column_with_nulls_and_filter() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    // Create validity mask: positions 1 and 3 are null
    let validity = Column::from_slice_bool(&[true, false, true, false, true])
        .call()
        .unwrap();
    let with_nulls = col
        .with_null_mask_from_bools(&validity.view())
        .call()
        .unwrap();

    assert_eq!(with_nulls.len(), 5);
    assert_eq!(with_nulls.null_count(), 2);
    assert!(with_nulls.has_nulls());

    // Verify the null mask round-trips
    assert_eq!(
        with_nulls.null_mask_to_host().call().unwrap(),
        [true, false, true, false, true]
    );
}

#[test]
fn drop_null_rows_from_table() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40]).call().unwrap();
    let validity = Column::from_slice_bool(&[true, false, true, false])
        .call()
        .unwrap();
    let with_nulls = col
        .with_null_mask_from_bools(&validity.view())
        .call()
        .unwrap();
    let tbl = build_table(vec![with_nulls]);

    let dropped = tbl.drop_nulls().call().unwrap();
    assert_eq!(dropped.len(), 2);
    assert_eq!(table_col_i32(&dropped, 0), [10, 30]);
}

// ===========================================================================
// Test: Concatenate tables
// ===========================================================================

#[test]
fn concatenate_two_tables() {
    let t1 = build_table(vec![Column::from_slice_i32(&[1, 2, 3]).call().unwrap()]);
    let t2 = build_table(vec![Column::from_slice_i32(&[4, 5]).call().unwrap()]);

    let combined = t1.concatenate(&t2).call().unwrap();
    assert_eq!(combined.len(), 5);
    assert_eq!(table_col_i32(&combined, 0), [1, 2, 3, 4, 5]);
}

// ===========================================================================
// Test: Partitioning round-trip
// ===========================================================================

#[test]
fn hash_partition_and_recombine() {
    let ids = Column::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8])
        .call()
        .unwrap();
    let tbl = build_table(vec![ids]);

    // Partition into 3 buckets
    let offsets = tbl.hash_partition_offsets(&[0], 3).call().unwrap();
    assert_eq!(offsets.len(), 4); // num_partitions + 1

    // All rows accounted for
    let last = offsets[offsets.len() - 1];
    assert_eq!(last, 8);
}

// ===========================================================================
// Test: Unique / distinct deduplication
// ===========================================================================

#[test]
fn distinct_removes_duplicates() {
    let keys = Column::from_slice_i32(&[1, 2, 2, 3, 3, 3]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 20, 21, 30, 31, 32])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    let result = tbl.distinct(&[0]).call().unwrap();
    assert_eq!(result.len(), 3); // 3 distinct keys

    let result = sorted(&result);
    assert_eq!(table_col_i32(&result, 0), [1, 2, 3]);
}

// ===========================================================================
// Test: End-to-end ETL pipeline
// ===========================================================================

#[test]
fn etl_pipeline_filter_sort_aggregate() {
    // Simulate: sales data with region (1=east, 2=west) and amount
    let regions = Column::from_slice_i32(&[1, 2, 1, 2, 1, 2, 1, 2])
        .call()
        .unwrap();
    let amounts = Column::from_slice_i32(&[100, 200, 150, 50, 300, 75, 25, 400])
        .call()
        .unwrap();
    let tbl = build_table(vec![regions, amounts]);

    // Step 1: Filter to amounts >= 100
    let threshold = Column::from_scalar(&Scalar::from_i32(100), tbl.len())
        .call()
        .unwrap();
    let mask = tbl.column(1).unwrap().ge(&threshold.view()).call().unwrap();
    let filtered = tbl.filter(&mask.view()).call().unwrap();
    assert_eq!(filtered.len(), 5); // 100, 200, 150, 300, 400

    // Step 2: GroupBy region → SUM
    let agg = filtered
        .groupby(&[0], 1, AggregationKind::SUM)
        .call()
        .unwrap();
    let agg = sorted(&agg);
    assert_eq!(table_col_i32(&agg, 0), [1, 2]); // regions
    assert_eq!(table_col_i64(&agg, 1), [550, 600]); // sums

    // Step 3: Sort by total descending (col1 desc, col0 desc)
    let final_result = agg
        .sort(
            &[Order::DESCENDING, Order::DESCENDING],
            &[NullOrder::AFTER, NullOrder::AFTER],
        )
        .call()
        .unwrap();
    // Region 2 (sum=600) before region 1 (sum=550) — sorted by key desc
    // since keys are the primary sort column
    assert_eq!(table_col_i32(&final_result, 0), [2, 1]);
    assert_eq!(table_col_i64(&final_result, 1), [600, 550]);
}

// ===========================================================================
// Test: Slice and reverse
// ===========================================================================

#[test]
fn slice_and_reverse_table() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let tbl = build_table(vec![col]);

    // Slice rows [1, 4)
    let sliced = tbl.slice(1, 4).call().unwrap();
    assert_eq!(sliced.len(), 3);
    assert_eq!(table_col_i32(&sliced, 0), [20, 30, 40]);

    // Reverse
    let rev = sliced.reverse().call().unwrap();
    assert_eq!(table_col_i32(&rev, 0), [40, 30, 20]);
}

// ===========================================================================
// Test: Column fill sequence
// ===========================================================================

#[test]
fn fill_sequence_column() {
    let init = Scalar::from_i32(0);
    let step = Scalar::from_i32(10);
    let seq = cudf::fill::sequence(5, &init, &step).call().unwrap();
    assert_eq!(seq.to_vec_i32().call().unwrap(), [0, 10, 20, 30, 40]);
}

// ===========================================================================
// Test: Hashing produces consistent results
// ===========================================================================

#[test]
fn murmur3_hash_deterministic() {
    let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call().unwrap();
    let tbl = build_table(vec![col]);

    // Same input should produce same hash
    let hash1 = tbl.murmur3(42).call().unwrap();
    let hash2 = tbl.murmur3(42).call().unwrap();
    assert_eq!(
        hash1.to_vec_u32().call().unwrap(),
        hash2.to_vec_u32().call().unwrap()
    );

    // Different seed should produce different hash
    let hash3 = tbl.murmur3(99).call().unwrap();
    assert_ne!(
        hash1.to_vec_u32().call().unwrap(),
        hash3.to_vec_u32().call().unwrap()
    );
}

// ===========================================================================
// Test: Repeat and tile
// ===========================================================================

#[test]
fn repeat_and_tile_table() {
    let col = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let tbl = build_table(vec![col]);

    // repeat duplicates each row N times (row-wise)
    let repeated = tbl.repeat(3).call().unwrap();
    assert_eq!(repeated.len(), 9);
    assert_eq!(table_col_i32(&repeated, 0), [1, 1, 1, 2, 2, 2, 3, 3, 3]);

    // tile repeats the whole table N times
    let tiled = tbl.tile(2).call().unwrap();
    assert_eq!(tiled.len(), 6);
    assert_eq!(table_col_i32(&tiled, 0), [1, 2, 3, 1, 2, 3]);
}

// ===========================================================================
// Test: CSV I/O round-trip
// ===========================================================================

#[test]
fn csv_write_and_read_back() {
    let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
    let tbl = build_table(vec![col]);

    let path = std::env::temp_dir().join("cudf_integration_test.csv");

    cudf::io::csv::write(&tbl, &path).unwrap();
    let read_back = cudf::io::csv::read(&path).unwrap();

    assert_eq!(read_back.len(), 3);
    assert_eq!(read_back.columns_len(), 1);
    assert_eq!(table_col_i64(&read_back, 0), [10, 20, 30]);

    // Clean up
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Test: Scatter rows
// ===========================================================================

#[test]
fn scatter_values_into_table() {
    let target = build_table(vec![
        Column::from_slice_i32(&[0, 0, 0, 0, 0]).call().unwrap(),
    ]);
    let source = build_table(vec![Column::from_slice_i32(&[99, 88]).call().unwrap()]);
    let indices = Column::from_slice_i32(&[1, 3]).call().unwrap();

    let result = target.scatter(&source, &indices.view()).call().unwrap();
    assert_eq!(table_col_i32(&result, 0), [0, 99, 0, 88, 0]);
}

// ===========================================================================
// Test: Interleave columns
// ===========================================================================

#[test]
fn interleave_table_columns() {
    let a = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let b = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
    let tbl = build_table(vec![a, b]);

    let interleaved = tbl.interleave_columns().call().unwrap();
    assert_eq!(
        interleaved.to_vec_i32().call().unwrap(),
        [1, 10, 2, 20, 3, 30]
    );
}

// ===========================================================================
// Test: Search (lower_bound / upper_bound)
// ===========================================================================

#[test]
fn lower_and_upper_bound_search() {
    // Sorted haystack
    let haystack = build_table(vec![
        Column::from_slice_i32(&[10, 20, 20, 30, 40])
            .call()
            .unwrap(),
    ]);
    // Needles to search for
    let needles = build_table(vec![Column::from_slice_i32(&[20, 35]).call().unwrap()]);

    let lower = haystack
        .lower_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
        .call()
        .unwrap();
    assert_eq!(lower.to_vec_i32().call().unwrap(), [1, 4]); // first 20 at index 1, 35 would go at 4

    let upper = haystack
        .upper_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
        .call()
        .unwrap();
    assert_eq!(upper.to_vec_i32().call().unwrap(), [3, 4]); // past last 20 at index 3
}

// ===========================================================================
// Test: Boolean comparison chaining
// ===========================================================================

#[test]
fn chained_boolean_comparisons() {
    let col = Column::from_slice_i32(&[5, 15, 25, 35, 45]).call().unwrap();

    // 10 <= col <= 30
    let lo = Column::from_scalar(&Scalar::from_i32(10), 5)
        .call()
        .unwrap();
    let hi = Column::from_scalar(&Scalar::from_i32(30), 5)
        .call()
        .unwrap();

    let ge_lo = col.view().ge(&lo.view()).call().unwrap();
    let le_hi = col.view().le(&hi.view()).call().unwrap();

    // AND the two masks using binary op
    use cudf::ops::BinaryOperator;
    let in_range = ge_lo
        .view()
        .binary_op(&le_hi.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();
    assert_eq!(
        in_range.to_vec_bool().call().unwrap(),
        [false, true, true, false, false]
    );
}

// ===========================================================================
// Test: String-to-integer conversion pipeline
// ===========================================================================

#[test]
fn string_to_integer_conversion() {
    let strings = Column::from_strings(&["100", "200", "300"]).call().unwrap();

    // Convert to integers
    let ints = strings
        .view()
        .str_to_integers(TypeId::INT32)
        .call()
        .unwrap();
    assert_eq!(ints.to_vec_i32().call().unwrap(), [100, 200, 300]);

    // Convert back to strings
    let back = ints.view().str_from_integers().call().unwrap();
    assert_eq!(back.to_vec_string().call().unwrap(), ["100", "200", "300"]);
}

// ===========================================================================
// Test: Replace literal strings
// ===========================================================================

#[test]
fn string_replace_literal() {
    let col = Column::from_strings(&["hello world", "world peace", "no match"])
        .call()
        .unwrap();

    let replaced = col
        .view()
        .str_replace_literal("world", "earth", -1)
        .call()
        .unwrap();
    assert_eq!(
        replaced.to_vec_string().call().unwrap(),
        ["hello earth", "earth peace", "no match"]
    );
}

// ===========================================================================
// Test: Cross join produces cartesian product
// ===========================================================================

#[test]
fn cross_join_cartesian_product() {
    let left = build_table(vec![Column::from_slice_i32(&[1, 2]).call().unwrap()]);
    let right = build_table(vec![Column::from_slice_i32(&[10, 20, 30]).call().unwrap()]);

    let result = left.cross_join(&right).call().unwrap();
    assert_eq!(result.len(), 6); // 2 * 3
    assert_eq!(result.columns_len(), 2);
}

// ===========================================================================
// Test: Merge two sorted tables
// ===========================================================================

#[test]
fn merge_two_sorted_tables() {
    let t1 = build_table(vec![Column::from_slice_i32(&[1, 3, 5]).call().unwrap()]);
    let t2 = build_table(vec![Column::from_slice_i32(&[2, 4, 6]).call().unwrap()]);

    let merged = t1
        .merge(&t2, &[0], &[Order::ASCENDING], &[NullOrder::BEFORE])
        .call()
        .unwrap();
    assert_eq!(merged.len(), 6);
    assert_eq!(table_col_i32(&merged, 0), [1, 2, 3, 4, 5, 6]);
}

// ===========================================================================
// Test: Column clone (deep copy)
// ===========================================================================

#[test]
fn from_scalar_creates_uniform_column() {
    let col = Column::from_scalar(&Scalar::from_i32(42), 5)
        .call()
        .unwrap();
    assert_eq!(col.len(), 5);
    assert_eq!(col.to_vec_i32().call().unwrap(), [42, 42, 42, 42, 42]);
    assert!(!col.has_nulls());
}

// ===========================================================================
// Test: Gather rows by index
// ===========================================================================

#[test]
fn gather_specific_rows() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let tbl = build_table(vec![col]);

    let indices = Column::from_slice_i32(&[4, 2, 0]).call().unwrap();
    let gathered = tbl.gather(&indices.view()).call().unwrap();

    assert_eq!(gathered.len(), 3);
    assert_eq!(table_col_i32(&gathered, 0), [50, 30, 10]);
}

// ===========================================================================
// Test: Subtraction, multiplication, division
// ===========================================================================

#[test]
fn column_sub_mul_div() {
    let a = Column::from_slice_i32(&[100, 200, 300]).call().unwrap();
    let b = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();

    let diff = a.view().sub(&b.view(), TypeId::INT32).call().unwrap();
    assert_eq!(diff.to_vec_i32().call().unwrap(), [90, 180, 270]);

    let prod = a.view().mul(&b.view(), TypeId::INT32).call().unwrap();
    assert_eq!(prod.to_vec_i32().call().unwrap(), [1000, 4000, 9000]);

    let quot = a.view().div(&b.view(), TypeId::INT32).call().unwrap();
    assert_eq!(quot.to_vec_i32().call().unwrap(), [10, 10, 10]);
}

// ===========================================================================
// Test: Unary ops — negate, abs, round
// ===========================================================================

#[test]
fn unary_negate_abs_round() {
    let col = Column::from_slice_i32(&[-5, 0, 7]).call().unwrap();

    let neg = col.view().negate().call().unwrap();
    assert_eq!(neg.to_vec_i32().call().unwrap(), [5, 0, -7]);

    let a = col.view().abs().call().unwrap();
    assert_eq!(a.to_vec_i32().call().unwrap(), [5, 0, 7]);

    let floats = Column::from_slice_f64(&[1.456, 2.789, 3.123])
        .call()
        .unwrap();
    let rounded = floats.view().round(1).call().unwrap();
    let result = rounded.to_vec_f64().call().unwrap();
    assert!((result[0] - 1.5).abs() < 1e-9);
    assert!((result[1] - 2.8).abs() < 1e-9);
    assert!((result[2] - 3.1).abs() < 1e-9);
}

// ===========================================================================
// Test: Comparison ops — eq, ne, lt, le
// ===========================================================================

#[test]
fn comparison_operators() {
    let a = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call().unwrap();
    let b = Column::from_slice_i32(&[3, 3, 3, 3, 3]).call().unwrap();

    assert_eq!(
        a.view()
            .eq(&b.view())
            .call()
            .unwrap()
            .to_vec_bool()
            .call()
            .unwrap(),
        [false, false, true, false, false]
    );
    assert_eq!(
        a.view()
            .ne(&b.view())
            .call()
            .unwrap()
            .to_vec_bool()
            .call()
            .unwrap(),
        [true, true, false, true, true]
    );
    assert_eq!(
        a.view()
            .lt(&b.view())
            .call()
            .unwrap()
            .to_vec_bool()
            .call()
            .unwrap(),
        [true, true, false, false, false]
    );
    assert_eq!(
        a.view()
            .le(&b.view())
            .call()
            .unwrap()
            .to_vec_bool()
            .call()
            .unwrap(),
        [true, true, true, false, false]
    );
}

// ===========================================================================
// Test: Reductions — product, any, all, mean, median, std_dev, variance
// ===========================================================================

#[test]
fn reduction_product() {
    let col = Column::from_slice_i32(&[2, 3, 4]).call().unwrap();
    let prod = col.view().product(TypeId::INT64).call().unwrap();
    assert_eq!(prod.as_i64(), Some(24));
}

#[test]
fn reduction_any_all() {
    let all_true = Column::from_slice_bool(&[true, true, true]).call().unwrap();
    assert_eq!(all_true.view().any().call().unwrap().as_bool(), Some(true));
    assert_eq!(all_true.view().all().call().unwrap().as_bool(), Some(true));

    let mixed = Column::from_slice_bool(&[true, false, true])
        .call()
        .unwrap();
    assert_eq!(mixed.view().any().call().unwrap().as_bool(), Some(true));
    assert_eq!(mixed.view().all().call().unwrap().as_bool(), Some(false));

    let all_false = Column::from_slice_bool(&[false, false, false])
        .call()
        .unwrap();
    assert_eq!(
        all_false.view().any().call().unwrap().as_bool(),
        Some(false)
    );
}

#[test]
fn reduction_mean_median_std_var() {
    let col = Column::from_slice_f64(&[10.0, 20.0, 30.0, 40.0, 50.0])
        .call()
        .unwrap();

    let mean = col.view().mean(TypeId::FLOAT64).call().unwrap();
    assert_eq!(mean.as_f64(), Some(30.0));

    let median = col.view().median(TypeId::FLOAT64).call().unwrap();
    assert_eq!(median.as_f64(), Some(30.0));

    // variance (ddof=1) of [10,20,30,40,50] = 250
    let var = col.view().variance(TypeId::FLOAT64, 1).call().unwrap();
    assert!((var.as_f64().unwrap() - 250.0).abs() < 1e-9);

    // std_dev = sqrt(250) ≈ 15.8114
    let std = col.view().std_dev(TypeId::FLOAT64, 1).call().unwrap();
    assert!((std.as_f64().unwrap() - 15.811_388_300_841_896).abs() < 1e-6);
}

#[test]
fn reduction_nunique() {
    let col = Column::from_slice_i32(&[1, 2, 2, 3, 3, 3]).call().unwrap();
    let n = col.view().nunique().call().unwrap();
    assert_eq!(n.as_i64(), Some(3)); // nunique returns INT64
}

// ===========================================================================
// Test: Quantile
// ===========================================================================

#[test]
fn quantile_column() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let q = col.view().quantile(&[0.0, 0.5, 1.0]).call().unwrap();
    let data = q.to_vec_f64().call().unwrap();
    assert!((data[0] - 10.0).abs() < 1e-9);
    assert!((data[1] - 30.0).abs() < 1e-9);
    assert!((data[2] - 50.0).abs() < 1e-9);
}

// ===========================================================================
// Test: Cumulative scan
// ===========================================================================

#[test]
fn scan_cumulative_sum() {
    let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call().unwrap();
    let cumsum = col.view().scan(AggregationKind::SUM, true).call().unwrap();
    assert_eq!(cumsum.to_vec_i32().call().unwrap(), [1, 3, 6, 10, 15]);
}

// ===========================================================================
// Test: Shift
// ===========================================================================

#[test]
fn shift_column_with_fill() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let fill = Scalar::from_i32(0);

    let shifted_right = col.view().shift(2, &fill).call().unwrap();
    assert_eq!(
        shifted_right.to_vec_i32().call().unwrap(),
        [0, 0, 10, 20, 30]
    );

    let shifted_left = col.view().shift(-1, &fill).call().unwrap();
    assert_eq!(
        shifted_left.to_vec_i32().call().unwrap(),
        [20, 30, 40, 50, 0]
    );
}

// ===========================================================================
// Test: Contains scalar / column
// ===========================================================================

#[test]
fn contains_scalar_check() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let yes = Scalar::from_i32(30);
    let no = Scalar::from_i32(99);
    assert!(col.view().contains_scalar(&yes).call().unwrap());
    assert!(!col.view().contains_scalar(&no).call().unwrap());
}

#[test]
fn contains_column_check() {
    let haystack = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let needles = Column::from_slice_i32(&[20, 99, 40]).call().unwrap();
    let result = haystack
        .view()
        .contains_column(&needles.view())
        .call()
        .unwrap();
    assert_eq!(result.to_vec_bool().call().unwrap(), [true, false, true]);
}

// ===========================================================================
// Test: Replace nulls
// ===========================================================================

#[test]
fn replace_nulls_with_scalar_and_column() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let validity = Column::from_slice_bool(&[true, false, true, false, true])
        .call()
        .unwrap();
    let with_nulls = col
        .with_null_mask_from_bools(&validity.view())
        .call()
        .unwrap();

    // Replace with scalar
    let fill = Scalar::from_i32(-1);
    let filled = with_nulls
        .view()
        .replace_nulls_with_scalar(&fill)
        .call()
        .unwrap();
    assert_eq!(filled.to_vec_i32().call().unwrap(), [10, -1, 30, -1, 50]);
    assert!(!filled.has_nulls());

    // Replace with column
    let replacement = Column::from_slice_i32(&[0, 0, 0, 0, 0]).call().unwrap();
    let filled2 = with_nulls
        .view()
        .replace_nulls_with_column(&replacement.view())
        .call()
        .unwrap();
    assert_eq!(filled2.to_vec_i32().call().unwrap(), [10, 0, 30, 0, 50]);
}

// ===========================================================================
// Test: Replace NaNs
// ===========================================================================

#[test]
fn replace_nans_with_scalar() {
    let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0])
        .call()
        .unwrap();
    let fill = Scalar::from_f64(0.0);
    let result = col.view().replace_nans_scalar(&fill).call().unwrap();
    assert_eq!(
        result.to_vec_f64().call().unwrap(),
        [1.0, 0.0, 3.0, 0.0, 5.0]
    );
}

// ===========================================================================
// Test: Clamp
// ===========================================================================

#[test]
fn clamp_column_values() {
    let col = Column::from_slice_i32(&[1, 5, 10, 15, 20]).call().unwrap();
    let lo = Scalar::from_i32(5);
    let hi = Scalar::from_i32(15);
    let clamped = col.view().clamp(&lo, &hi).call().unwrap();
    assert_eq!(clamped.to_vec_i32().call().unwrap(), [5, 5, 10, 15, 15]);
}

// ===========================================================================
// Test: Find and replace all
// ===========================================================================

#[test]
fn find_and_replace_all_values() {
    let col = Column::from_slice_i32(&[1, 2, 3, 2, 1]).call().unwrap();
    let old_vals = Column::from_slice_i32(&[1, 2]).call().unwrap();
    let new_vals = Column::from_slice_i32(&[10, 20]).call().unwrap();
    let result = col
        .view()
        .find_and_replace_all(&old_vals.view(), &new_vals.view())
        .call()
        .unwrap();
    assert_eq!(result.to_vec_i32().call().unwrap(), [10, 20, 3, 20, 10]);
}

// ===========================================================================
// Test: is_null / is_valid
// ===========================================================================

#[test]
fn is_null_and_is_valid_masks() {
    let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
    let validity = Column::from_slice_bool(&[true, false, true])
        .call()
        .unwrap();
    let with_nulls = col
        .with_null_mask_from_bools(&validity.view())
        .call()
        .unwrap();

    let null_mask = with_nulls.view().is_null().call().unwrap();
    assert_eq!(
        null_mask.to_vec_bool().call().unwrap(),
        [false, true, false]
    );

    let valid_mask = with_nulls.view().is_valid().call().unwrap();
    assert_eq!(
        valid_mask.to_vec_bool().call().unwrap(),
        [true, false, true]
    );
}

// ===========================================================================
// Test: is_nan / normalize_nans_and_zeros
// ===========================================================================

#[test]
fn is_nan_detection() {
    let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN])
        .call()
        .unwrap();
    let mask = col.view().is_nan().call().unwrap();
    assert_eq!(
        mask.to_vec_bool().call().unwrap(),
        [false, true, false, true]
    );
}

#[test]
fn normalize_nans_and_zeros_col() {
    // Negative zero becomes positive zero, NaN stays NaN
    let col = Column::from_slice_f64(&[-0.0, 0.0, f64::NAN, 1.0])
        .call()
        .unwrap();
    let normed = col.view().normalize_nans_and_zeros().call().unwrap();
    let data = normed.to_vec_f64().call().unwrap();
    // Both zeros should now be +0.0 (bit-identical)
    assert!(data[0].to_bits() == 0u64); // +0.0
    assert!(data[1].to_bits() == 0u64); // +0.0
    assert!(data[2].is_nan());
    assert!((data[3] - 1.0).abs() < 1e-9);
}

// ===========================================================================
// Test: copy_if_else
// ===========================================================================

#[test]
fn copy_if_else_by_mask() {
    let lhs = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call().unwrap();
    let rhs = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let mask = Column::from_slice_bool(&[true, false, true, false, true])
        .call()
        .unwrap();

    let result = lhs
        .view()
        .copy_if_else(&rhs.view(), &mask.view())
        .call()
        .unwrap();
    assert_eq!(result.to_vec_i32().call().unwrap(), [1, 20, 3, 40, 5]);
}

// ===========================================================================
// Test: get_element
// ===========================================================================

#[test]
fn get_element_from_column() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let elem = col.view().get_element(2).call().unwrap();
    assert_eq!(elem.as_i32(), Some(30));

    let elem_last = col.view().get_element(4).call().unwrap();
    assert_eq!(elem_last.as_i32(), Some(50));
}

// ===========================================================================
// Test: Rank
// ===========================================================================

#[test]
fn rank_column() {
    use cudf::compaction::NullPolicy;
    let col = Column::from_slice_i32(&[30, 10, 20, 10, 30])
        .call()
        .unwrap();
    let ranked = col
        .view()
        .rank(
            RankMethod::Dense,
            Order::ASCENDING,
            NullPolicy::EXCLUDE,
            NullOrder::BEFORE,
            false,
        )
        .call()
        .unwrap();
    // Dense ranking: 10→1, 20→2, 30→3
    let data = ranked.to_vec_i32().call().unwrap();
    assert_eq!(data, [3, 1, 2, 1, 3]);
}

// ===========================================================================
// Test: Top-k
// ===========================================================================

#[test]
fn top_k_largest() {
    let col = Column::from_slice_i32(&[10, 50, 30, 40, 20])
        .call()
        .unwrap();
    let top3 = col.view().top_k(3, Order::DESCENDING).call().unwrap();
    // Top 3 in descending order
    assert_eq!(top3.len(), 3);
    assert_eq!(top3.to_vec_i32().call().unwrap(), [50, 40, 30]);
}

// ===========================================================================
// Test: Column distinct_count
// ===========================================================================

#[test]
fn column_distinct_count() {
    let col = Column::from_slice_i32(&[1, 2, 2, 3, 3, 3]).call().unwrap();
    let count = col.view().distinct_count(false, false).call().unwrap();
    assert_eq!(count, 3);
}

// ===========================================================================
// Test: Full join
// ===========================================================================

#[test]
fn full_join_all_rows_present() {
    let left = build_table(vec![
        Column::from_slice_i32(&[1, 2]).call().unwrap(),
        Column::from_slice_i32(&[10, 20]).call().unwrap(),
    ]);
    let right = build_table(vec![
        Column::from_slice_i32(&[2, 3]).call().unwrap(),
        Column::from_slice_i32(&[200, 300]).call().unwrap(),
    ]);

    let result = left.full_join(&right, &[0], &[0]).call().unwrap();
    // Full join: 3 rows (key=1 from left only, key=2 from both, key=3 from right only)
    assert_eq!(result.len(), 3);
}

// ===========================================================================
// Test: Semi and anti joins
// ===========================================================================

#[test]
fn semi_join_filters_to_matching() {
    let left = build_table(vec![
        Column::from_slice_i32(&[1, 2, 3, 4]).call().unwrap(),
        Column::from_slice_i32(&[10, 20, 30, 40]).call().unwrap(),
    ]);
    let right = build_table(vec![Column::from_slice_i32(&[2, 4]).call().unwrap()]);

    let semi = left.left_semi_join(&right, &[0], &[0]).call().unwrap();
    let semi = sorted(&semi);
    assert_eq!(semi.len(), 2);
    assert_eq!(table_col_i32(&semi, 0), [2, 4]);
    assert_eq!(table_col_i32(&semi, 1), [20, 40]);
}

#[test]
fn anti_join_filters_to_non_matching() {
    let left = build_table(vec![
        Column::from_slice_i32(&[1, 2, 3, 4]).call().unwrap(),
        Column::from_slice_i32(&[10, 20, 30, 40]).call().unwrap(),
    ]);
    let right = build_table(vec![Column::from_slice_i32(&[2, 4]).call().unwrap()]);

    let anti = left.left_anti_join(&right, &[0], &[0]).call().unwrap();
    let anti = sorted(&anti);
    assert_eq!(anti.len(), 2);
    assert_eq!(table_col_i32(&anti, 0), [1, 3]);
    assert_eq!(table_col_i32(&anti, 1), [10, 30]);
}

// ===========================================================================
// Test: Unique vs distinct
// ===========================================================================

#[test]
fn unique_keeps_first_and_last() {
    let keys = Column::from_slice_i32(&[1, 1, 2, 2, 3]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 11, 20, 21, 30])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    let result = tbl.unique(&[0]).call().unwrap();
    let result = sorted(&result);
    // Unique keeps first and last occurrences of each key
    assert!(result.len() >= 3); // at least 3 distinct keys
    assert_eq!(table_col_i32(&result, 0)[0], 1);
}

// ===========================================================================
// Test: Stable distinct preserves order
// ===========================================================================

#[test]
fn stable_distinct_preserves_insertion_order() {
    let keys = Column::from_slice_i32(&[3, 1, 2, 1, 3]).call().unwrap();
    let tbl = build_table(vec![keys]);

    let result = tbl.stable_distinct(&[0]).call().unwrap();
    assert_eq!(result.len(), 3);
    // Stable distinct preserves order of first occurrence: 3, 1, 2
    assert_eq!(table_col_i32(&result, 0), [3, 1, 2]);
}

// ===========================================================================
// Test: Encode table (dictionary encoding)
// ===========================================================================

#[test]
fn encode_table_returns_indices() {
    let col = Column::from_slice_i32(&[10, 20, 10, 30, 20])
        .call()
        .unwrap();
    let tbl = build_table(vec![col]);

    let indices = tbl.encode().call().unwrap();
    assert_eq!(indices.len(), 5);
    // Same input rows should produce same indices
    let idx = indices.to_vec_i32().call().unwrap();
    assert_eq!(idx[0], idx[2]); // rows 0 and 2 are both [10]
    assert_eq!(idx[1], idx[4]); // rows 1 and 4 are both [20]
}

// ===========================================================================
// Test: Transpose table
// ===========================================================================

#[test]
fn transpose_table() {
    let c0 = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let c1 = Column::from_slice_i32(&[4, 5, 6]).call().unwrap();
    let tbl = build_table(vec![c0, c1]);

    let transposed = tbl.transpose().call().unwrap();
    // Original: 3 rows × 2 cols → Transposed: 2 rows × 3 cols
    assert_eq!(transposed.len(), 2);
    assert_eq!(transposed.columns_len(), 3);
    assert_eq!(table_col_i32(&transposed, 0), [1, 4]);
    assert_eq!(table_col_i32(&transposed, 1), [2, 5]);
    assert_eq!(table_col_i32(&transposed, 2), [3, 6]);
}

// ===========================================================================
// Test: is_sorted
// ===========================================================================

#[test]
fn is_sorted_check() {
    let asc = build_table(vec![
        Column::from_slice_i32(&[1, 2, 3, 4, 5]).call().unwrap(),
    ]);
    assert!(
        asc.is_sorted(&[Order::ASCENDING], &[NullOrder::BEFORE])
            .call()
            .unwrap()
    );

    let desc = build_table(vec![
        Column::from_slice_i32(&[5, 4, 3, 2, 1]).call().unwrap(),
    ]);
    assert!(
        !desc
            .is_sorted(&[Order::ASCENDING], &[NullOrder::BEFORE])
            .call()
            .unwrap()
    );
    assert!(
        desc.is_sorted(&[Order::DESCENDING], &[NullOrder::AFTER])
            .call()
            .unwrap()
    );
}

// ===========================================================================
// Test: Stable sort preserves insertion order of equal elements
// ===========================================================================

#[test]
fn stable_sort_preserves_equal_order() {
    let keys = Column::from_slice_i32(&[2, 1, 2, 1, 2]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    let result = tbl
        .stable_sort(
            &[Order::ASCENDING, Order::ASCENDING],
            &[NullOrder::BEFORE, NullOrder::BEFORE],
        )
        .call()
        .unwrap();
    // key=1 rows keep their relative order: (1,20), (1,40)
    // key=2 rows keep their relative order: (2,10), (2,30), (2,50)
    assert_eq!(table_col_i32(&result, 0), [1, 1, 2, 2, 2]);
    assert_eq!(table_col_i32(&result, 1), [20, 40, 10, 30, 50]);
}

// ===========================================================================
// Test: Boolean mask scatter
// ===========================================================================

#[test]
fn boolean_mask_scatter_rows() {
    let target = build_table(vec![
        Column::from_slice_i32(&[0, 0, 0, 0, 0]).call().unwrap(),
    ]);
    let source = build_table(vec![Column::from_slice_i32(&[99, 88]).call().unwrap()]);
    let mask = Column::from_slice_bool(&[false, true, false, true, false])
        .call()
        .unwrap();

    let result = target
        .boolean_mask_scatter(&source, &mask.view())
        .call()
        .unwrap();
    assert_eq!(table_col_i32(&result, 0), [0, 99, 0, 88, 0]);
}

// ===========================================================================
// Test: Repeat by column (variable repetition counts)
// ===========================================================================

#[test]
fn repeat_by_column_variable_counts() {
    let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
    let tbl = build_table(vec![col]);
    let counts = Column::from_slice_i32(&[1, 3, 2]).call().unwrap();

    let result = tbl.repeat_by_column(&counts.view()).call().unwrap();
    assert_eq!(result.len(), 6); // 1+3+2
    assert_eq!(table_col_i32(&result, 0), [10, 20, 20, 20, 30, 30]);
}

// ===========================================================================
// Test: Sample rows
// ===========================================================================

#[test]
fn sample_rows_from_table() {
    let col = Column::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .call()
        .unwrap();
    let tbl = build_table(vec![col]);

    let sampled = tbl.sample(3, false, 42).call().unwrap();
    assert_eq!(sampled.len(), 3);
    assert_eq!(sampled.columns_len(), 1);
}

// ===========================================================================
// Test: Drop NaNs
// ===========================================================================

#[test]
fn drop_nans_from_table() {
    let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0])
        .call()
        .unwrap();
    let tbl = build_table(vec![col]);

    let cleaned = tbl.drop_nans(&[0]).call().unwrap();
    assert_eq!(cleaned.len(), 3);
    assert_eq!(table_col_f64(&cleaned, 0), [1.0, 3.0, 5.0]);
}

// ===========================================================================
// Test: xxhash64 produces different results than murmur3
// ===========================================================================

#[test]
fn xxhash64_hashing() {
    let col = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let tbl = build_table(vec![col]);

    let h1 = tbl.xxhash64(42).call().unwrap();
    let h2 = tbl.xxhash64(42).call().unwrap();
    assert_eq!(
        h1.to_vec_i64().call().unwrap(),
        h2.to_vec_i64().call().unwrap()
    ); // deterministic

    let h3 = tbl.xxhash64(99).call().unwrap();
    assert_ne!(
        h1.to_vec_i64().call().unwrap(),
        h3.to_vec_i64().call().unwrap()
    ); // different seed
}

// ===========================================================================
// Test: MD5 and SHA256 hashing
// ===========================================================================

#[test]
fn md5_and_sha256_hashing() {
    let col = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let tbl = build_table(vec![col]);

    let md5 = tbl.md5().call().unwrap();
    assert_eq!(md5.len(), 3);
    // MD5 outputs strings
    let hashes = md5.to_vec_string().call().unwrap();
    assert_eq!(hashes.len(), 3);
    // Each hash is a 32-char hex string
    assert_eq!(hashes[0].len(), 32);
    assert_ne!(hashes[0], hashes[1]);

    let sha = tbl.sha256().call().unwrap();
    assert_eq!(sha.len(), 3);
    let sha_hashes = sha.to_vec_string().call().unwrap();
    // SHA-256 is 64 hex chars
    assert_eq!(sha_hashes[0].len(), 64);
}

// ===========================================================================
// Test: Round-robin partitioning
// ===========================================================================

#[test]
fn round_robin_partition() {
    let col = Column::from_slice_i32(&[1, 2, 3, 4, 5, 6]).call().unwrap();
    let tbl = build_table(vec![col]);

    let offsets = tbl.round_robin_offsets(3, 0).call().unwrap();
    assert_eq!(offsets.len(), 4); // num_partitions + 1
    assert_eq!(offsets[offsets.len() - 1], 6); // all rows accounted for
}

// ===========================================================================
// Test: Row bit count
// ===========================================================================

#[test]
fn row_bit_count_table() {
    let col = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let tbl = build_table(vec![col]);

    let bits = tbl.row_bit_count().call().unwrap();
    assert_eq!(bits.len(), 3);
    // Each i32 row is 32 bits + 1 validity bit = 33 bits
    let counts = bits.to_vec_i32().call().unwrap();
    assert!(counts.iter().all(|&c| c > 0));
}

// ===========================================================================
// Test: Table distinct_count and unique_count
// ===========================================================================

#[test]
fn table_distinct_and_unique_count() {
    let col = Column::from_slice_i32(&[1, 2, 2, 3, 3, 3]).call().unwrap();
    let tbl = build_table(vec![col]);

    let dc = tbl.distinct_count(true).call().unwrap();
    assert_eq!(dc, 3); // 3 distinct values

    let uc = tbl.unique_count(true).call().unwrap();
    // unique_count counts consecutive unique groups
    assert_eq!(uc, 3); // [1], [2,2], [3,3,3] → 3 groups
}

// ===========================================================================
// Test: Concatenate columns function
// ===========================================================================

#[test]
fn concatenate_columns_function() {
    let a = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let b = Column::from_slice_i32(&[4, 5]).call().unwrap();
    let c = Column::from_slice_i32(&[6]).call().unwrap();

    let result = cudf::concatenate::concatenate_columns(&[&a.view(), &b.view(), &c.view()])
        .call()
        .unwrap();
    assert_eq!(result.len(), 6);
    assert_eq!(result.to_vec_i32().call().unwrap(), [1, 2, 3, 4, 5, 6]);
}

// ===========================================================================
// Test: Concatenate tables function
// ===========================================================================

#[test]
fn concatenate_tables_function() {
    let t1 = build_table(vec![Column::from_slice_i32(&[1, 2]).call().unwrap()]);
    let t2 = build_table(vec![Column::from_slice_i32(&[3, 4]).call().unwrap()]);
    let t3 = build_table(vec![Column::from_slice_i32(&[5]).call().unwrap()]);

    let result = cudf::concatenate::concatenate_tables(&[&t1, &t2, &t3])
        .call()
        .unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(table_col_i32(&result, 0), [1, 2, 3, 4, 5]);
}

// ===========================================================================
// Test: copy_if_else_scalars
// ===========================================================================

#[test]
fn copy_if_else_with_scalars() {
    let mask = Column::from_slice_bool(&[true, false, true, false])
        .call()
        .unwrap();
    let lhs = Scalar::from_i32(1);
    let rhs = Scalar::from_i32(0);

    let result = cudf::copy_if_else_scalars(&lhs, &rhs, &mask.view())
        .call()
        .unwrap();
    assert_eq!(result.to_vec_i32().call().unwrap(), [1, 0, 1, 0]);
}

// ===========================================================================
// Test: One-hot encode
// ===========================================================================

#[test]
fn one_hot_encode_categories() {
    let input = Column::from_slice_i32(&[0, 1, 2, 1, 0]).call().unwrap();
    let categories = Column::from_slice_i32(&[0, 1, 2]).call().unwrap();

    let result = cudf::reshape::one_hot_encode(&input.view(), &categories.view())
        .call()
        .unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(result.columns_len(), 3); // one column per category
}

// ===========================================================================
// Test: Label bins
// ===========================================================================

#[test]
fn label_bins_column() {
    use cudf::labeling::Inclusive;
    let col = Column::from_slice_i32(&[5, 15, 25]).call().unwrap();
    let left_edges = Column::from_slice_i32(&[0, 10, 20]).call().unwrap();
    let right_edges = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();

    let labels = col
        .view()
        .label_bins(
            &left_edges.view(),
            Inclusive::YES,
            &right_edges.view(),
            Inclusive::YES,
        )
        .call()
        .unwrap();
    assert_eq!(labels.len(), 3);
    assert_eq!(labels.to_vec_i32().call().unwrap(), [0, 1, 2]);
}

// ===========================================================================
// Test: String capitalize, title, swapcase
// ===========================================================================

#[test]
fn string_capitalize_title_swapcase() {
    let col = Column::from_strings(&["hello world", "fOO bAR"])
        .call()
        .unwrap();

    let capitalized = col.view().capitalize().call().unwrap();
    assert_eq!(
        capitalized.to_vec_string().call().unwrap(),
        ["Hello world", "Foo bar"]
    );

    let titled = col.view().title().call().unwrap();
    assert_eq!(
        titled.to_vec_string().call().unwrap(),
        ["Hello World", "Foo Bar"]
    );

    let swapped = col.view().swapcase().call().unwrap();
    assert_eq!(
        swapped.to_vec_string().call().unwrap(),
        ["HELLO WORLD", "Foo Bar"]
    );
}

// ===========================================================================
// Test: String strip / lstrip / rstrip
// ===========================================================================

#[test]
fn string_strip_whitespace() {
    let col = Column::from_strings(&["  hello  ", "  world", "foo  "])
        .call()
        .unwrap();

    let stripped = col.view().str_strip().call().unwrap();
    assert_eq!(
        stripped.to_vec_string().call().unwrap(),
        ["hello", "world", "foo"]
    );

    let lstripped = col.view().str_lstrip().call().unwrap();
    assert_eq!(
        lstripped.to_vec_string().call().unwrap(),
        ["hello  ", "world", "foo  "]
    );

    let rstripped = col.view().str_rstrip().call().unwrap();
    assert_eq!(
        rstripped.to_vec_string().call().unwrap(),
        ["  hello", "  world", "foo"]
    );
}

// ===========================================================================
// Test: String strip specific chars
// ===========================================================================

#[test]
fn string_strip_chars() {
    let col = Column::from_strings(&["##hello##", "##world##"])
        .call()
        .unwrap();
    let stripped = col
        .view()
        .str_strip_chars(SideType::BOTH, "#")
        .call()
        .unwrap();
    assert_eq!(stripped.to_vec_string().call().unwrap(), ["hello", "world"]);
}

// ===========================================================================
// Test: String pad / zfill
// ===========================================================================

#[test]
fn string_pad_and_zfill() {
    let col = Column::from_strings(&["1", "42", "100"]).call().unwrap();

    let zfilled = col.view().str_zfill(5).call().unwrap();
    assert_eq!(
        zfilled.to_vec_string().call().unwrap(),
        ["00001", "00042", "00100"]
    );

    let padded = col.view().str_pad(5, SideType::LEFT, " ").call().unwrap();
    assert_eq!(
        padded.to_vec_string().call().unwrap(),
        ["    1", "   42", "  100"]
    );
}

// ===========================================================================
// Test: String starts_with / ends_with
// ===========================================================================

#[test]
fn string_starts_with_ends_with() {
    let col = Column::from_strings(&["hello world", "help me", "world hello"])
        .call()
        .unwrap();

    let starts = col.view().str_starts_with_str("hel").call().unwrap();
    assert_eq!(starts.to_vec_bool().call().unwrap(), [true, true, false]);

    let ends = col.view().str_ends_with_str("llo").call().unwrap();
    assert_eq!(ends.to_vec_bool().call().unwrap(), [false, false, true]);
}

// ===========================================================================
// Test: String SQL LIKE pattern
// ===========================================================================

#[test]
fn string_like_pattern() {
    let col = Column::from_strings(&["apple", "application", "banana", "appetite"])
        .call()
        .unwrap();
    // SQL LIKE: "app%" matches anything starting with "app"
    let result = col.view().str_like("app%", "").call().unwrap();
    assert_eq!(
        result.to_vec_bool().call().unwrap(),
        [true, true, false, true]
    );
}

// ===========================================================================
// Test: String slice
// ===========================================================================

#[test]
fn string_slice_chars() {
    let col = Column::from_strings(&["hello", "world", "test"])
        .call()
        .unwrap();
    let sliced = col.view().str_slice(0, 3, 1).call().unwrap();
    assert_eq!(
        sliced.to_vec_string().call().unwrap(),
        ["hel", "wor", "tes"]
    );
}

// ===========================================================================
// Test: String repeat
// ===========================================================================

#[test]
fn string_repeat_times() {
    let col = Column::from_strings(&["ab", "cd"]).call().unwrap();
    let repeated = col.view().str_repeat(3).call().unwrap();
    assert_eq!(
        repeated.to_vec_string().call().unwrap(),
        ["ababab", "cdcdcd"]
    );
}

// ===========================================================================
// Test: String reverse
// ===========================================================================

#[test]
fn string_reverse_chars() {
    let col = Column::from_strings(&["hello", "world"]).call().unwrap();
    let reversed = col.view().str_reverse().call().unwrap();
    assert_eq!(reversed.to_vec_string().call().unwrap(), ["olleh", "dlrow"]);
}

// ===========================================================================
// Test: String to/from floats
// ===========================================================================

#[test]
fn string_to_and_from_floats() {
    let strings = Column::from_strings(&["1.5", "2.75", "3.0"])
        .call()
        .unwrap();
    let floats = strings
        .view()
        .str_to_floats(TypeId::FLOAT64)
        .call()
        .unwrap();
    assert_eq!(floats.to_vec_f64().call().unwrap(), [1.5, 2.75, 3.0]);

    let back = floats.view().str_from_floats().call().unwrap();
    let strs = back.to_vec_string().call().unwrap();
    // May have trailing zeros, just verify round-trip values
    assert!(strs[0].parse::<f64>().unwrap() - 1.5 < 1e-9);
    assert!(strs[1].parse::<f64>().unwrap() - 2.75 < 1e-9);
}

// ===========================================================================
// Test: Datetime extraction from timestamps
// ===========================================================================

#[test]
fn datetime_extract_components() {
    // 2024-03-15 10:30:45 UTC as seconds since epoch
    // 2024-01-01 00:00:00 = 1704067200
    // March 15 = +74 days = +6393600 seconds
    // 10:30:45 = +37845 seconds
    // Total = 1704067200 + 6393600 + 37845 = 1710498645
    let timestamps = Column::from_timestamps_s(&[
        1_710_498_645, // 2024-03-15 10:30:45
        1_719_792_000, // 2024-07-01 00:00:00
    ])
    .call()
    .unwrap();

    let years = timestamps.view().extract_year().call().unwrap();
    assert_eq!(years.to_vec_i16().call().unwrap(), [2024, 2024]);

    let months = timestamps.view().extract_month().call().unwrap();
    assert_eq!(months.to_vec_i16().call().unwrap(), [3, 7]);

    let days = timestamps.view().extract_day().call().unwrap();
    assert_eq!(days.to_vec_i16().call().unwrap(), [15, 1]);

    let hours = timestamps.view().extract_hour().call().unwrap();
    assert_eq!(hours.to_vec_i16().call().unwrap(), [10, 0]);

    let minutes = timestamps.view().extract_minute().call().unwrap();
    assert_eq!(minutes.to_vec_i16().call().unwrap(), [30, 0]);

    let seconds = timestamps.view().extract_second().call().unwrap();
    assert_eq!(seconds.to_vec_i16().call().unwrap(), [45, 0]);
}

#[test]
fn datetime_day_of_year_and_leap_year() {
    let timestamps = Column::from_timestamps_s(&[
        1_710_498_645, // 2024-03-15 (leap year, day 75)
        1_672_531_200, // 2023-01-01 (not leap year, day 1)
    ])
    .call()
    .unwrap();

    let doy = timestamps.view().day_of_year().call().unwrap();
    assert_eq!(doy.to_vec_i16().call().unwrap(), [75, 1]);

    let leap = timestamps.view().is_leap_year().call().unwrap();
    assert_eq!(leap.to_vec_bool().call().unwrap(), [true, false]);
}

#[test]
fn datetime_quarter() {
    let timestamps = Column::from_timestamps_s(&[
        1_704_067_200, // 2024-01-01 → Q1
        1_712_016_000, // 2024-04-02 → Q2
        1_719_792_000, // 2024-07-01 → Q3
        1_727_740_800, // 2024-10-01 → Q4
    ])
    .call()
    .unwrap();

    let quarters = timestamps.view().extract_quarter().call().unwrap();
    assert_eq!(quarters.to_vec_i16().call().unwrap(), [1, 2, 3, 4]);
}

#[test]
fn datetime_floor_to_day() {
    // 2024-03-15 10:30:45 UTC
    let timestamps = Column::from_timestamps_s(&[1_710_498_645]).call().unwrap();
    let floored = timestamps
        .view()
        .dt_floor(RoundingFrequency::Day)
        .call()
        .unwrap();
    // Floor to day: 2024-03-15 00:00:00 = 1710460800
    assert_eq!(floored.to_vec_i64().call().unwrap(), [1_710_460_800]);
}

// ===========================================================================
// Test: GroupBy scan (cumulative within groups)
// ===========================================================================

#[test]
fn groupby_scan_cumulative_sum() {
    let keys = Column::from_slice_i32(&[1, 1, 1, 2, 2]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 20, 30, 100, 200])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    let result = tbl
        .groupby_scan(&[0], &[1], &[AggregationKind::SUM])
        .call()
        .unwrap();
    // Result: keys + cumulative sum within each group
    assert_eq!(result.len(), 5);
    let cumsum = table_col_i64(&result, 1);
    assert_eq!(cumsum, [10, 30, 60, 100, 300]);
}

// ===========================================================================
// Test: GroupBy shift
// ===========================================================================

#[test]
fn groupby_shift_values() {
    let keys = Column::from_slice_i32(&[1, 1, 1, 2, 2]).call().unwrap();
    let vals = Column::from_slice_i32(&[10, 20, 30, 100, 200])
        .call()
        .unwrap();
    let tbl = build_table(vec![keys, vals]);

    let fill = Scalar::from_i32(0);
    let result = tbl.groupby_shift(&[0], &[1], &[1], &[fill]).call().unwrap();
    // Shift by 1 within each group: first element gets fill value
    assert_eq!(result.len(), 5);
    let shifted = table_col_i32(&result, 1);
    assert_eq!(shifted[0], 0); // group1 first → fill
    assert_eq!(shifted[1], 10); // group1 second → previous
}

// ===========================================================================
// Test: Empty table operations
// ===========================================================================

#[test]
fn empty_table_operations() {
    let empty = build_table(vec![Column::from_slice_i32(&[]).call().unwrap()]);
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.columns_len(), 1); // has 1 column, 0 rows

    let sorted = empty
        .sort(&[Order::ASCENDING], &[NullOrder::BEFORE])
        .call()
        .unwrap();
    assert_eq!(sorted.len(), 0);

    let reversed = empty.reverse().call().unwrap();
    assert_eq!(reversed.len(), 0);
}

// ===========================================================================
// Test: RMM device queries
// ===========================================================================

#[test]
fn rmm_device_queries() {
    let num = cudf::rmm::device::num_devices();
    assert!(num >= 1);

    let dev = cudf::rmm::device::current_device();
    // Device ID should be non-negative
    assert!(dev.value() >= 0);
}

// ===========================================================================
// Test: Multi-column join with aggregation pipeline
// ===========================================================================

#[test]
fn multi_column_join_then_aggregate() {
    // Orders table: (customer_id, product_id, quantity)
    let customers = Column::from_slice_i32(&[1, 1, 2, 2, 3]).call().unwrap();
    let products = Column::from_slice_i32(&[10, 20, 10, 30, 20])
        .call()
        .unwrap();
    let quantities = Column::from_slice_i32(&[5, 3, 2, 7, 1]).call().unwrap();
    let orders = build_table(vec![customers, products, quantities]);

    // Price lookup: (product_id, unit_price)
    let prod_ids = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
    let prices = Column::from_slice_i32(&[100, 200, 300]).call().unwrap();
    let price_table = build_table(vec![prod_ids, prices]);

    // Join orders with prices on product_id
    let joined = orders.inner_join(&price_table, &[1], &[0]).call().unwrap();
    assert_eq!(joined.len(), 5);

    // Compute total per row: quantity * unit_price
    let qty_col = joined.column(2).unwrap(); // quantities
    let price_col = joined.column(4).unwrap(); // unit_price
    let totals = qty_col.mul(&price_col, TypeId::INT32).call().unwrap();

    // Build new table: (customer_id, total)
    let cust_col = joined
        .column(0)
        .unwrap()
        .cast(TypeId::INT32)
        .call()
        .unwrap();
    let revenue = build_table(vec![cust_col, totals]);

    // GroupBy customer → SUM of revenue
    let result = revenue
        .groupby(&[0], 1, AggregationKind::SUM)
        .call()
        .unwrap();
    let result = sorted(&result);
    assert_eq!(table_col_i32(&result, 0), [1, 2, 3]);
    // Customer 1: 5*100 + 3*200 = 1100
    // Customer 2: 2*100 + 7*300 = 2300
    // Customer 3: 1*200 = 200
    assert_eq!(table_col_i64(&result, 1), [1100, 2300, 200]);
}

// ===========================================================================
// Test: Approx distinct count
// ===========================================================================

#[test]
fn approx_distinct_count_estimate() {
    let col = Column::from_slice_i32(&[1, 2, 3, 4, 5, 1, 2, 3])
        .call()
        .unwrap();
    let tbl = build_table(vec![col]);

    let approx = tbl.approx_distinct_count(10).call().unwrap();
    // HyperLogLog approximation — should be close to 5
    assert!((3..=7).contains(&approx));
}

// ===========================================================================
// Test: Explode list column
// ===========================================================================

#[test]
fn explode_list_column() {
    // Build a table with a list column using from_lists
    let offsets = Column::from_slice_i32(&[0, 2, 3, 5]).call().unwrap();
    let child = Column::from_slice_i32(&[10, 20, 30, 40, 50])
        .call()
        .unwrap();
    let list_col = Column::from_lists(3, offsets, child).call().unwrap();
    let keys = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
    let tbl = build_table(vec![keys, list_col]);

    let exploded = tbl.explode(1).call().unwrap();
    // List [[10,20], [30], [40,50]] explodes to 5 rows
    assert_eq!(exploded.len(), 5);
    assert_eq!(table_col_i32(&exploded, 0), [1, 1, 2, 3, 3]);
    assert_eq!(table_col_i32(&exploded, 1), [10, 20, 30, 40, 50]);
}
