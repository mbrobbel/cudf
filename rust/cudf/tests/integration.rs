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
use cudf::groupby::AggregationKind;
use cudf::scalar::Scalar;
use cudf::sorting::{NullOrder, Order};
use cudf::strings::StringExt;
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
    tbl.column(idx).unwrap().cast(TypeId::INT32).call().unwrap().to_vec_i32()
}

/// Extract a table column to host as i64.
fn table_col_i64(tbl: &Table, idx: usize) -> Vec<i64> {
    tbl.column(idx).unwrap().cast(TypeId::INT64).call().unwrap().to_vec_i64()
}

/// Extract a table column to host as f64.
fn table_col_f64(tbl: &Table, idx: usize) -> Vec<f64> {
    tbl.column(idx).unwrap().cast(TypeId::FLOAT64).call().unwrap().to_vec_f64()
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
    let a = Column::from_slice_i32(&[10, 20, 30, 40, 50]);
    let b = Column::from_slice_i32(&[1, 2, 3, 4, 5]);

    // Element-wise add
    let sum_col = a.view().add(&b.view(), TypeId::INT32).call().unwrap();
    assert_eq!(sum_col.to_vec_i32(), [11, 22, 33, 44, 55]);

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
    let ints = Column::from_slice_i32(&[1, 4, 9, 16, 25]);

    // Cast to f64, compute sqrt
    let floats = ints.view().cast(TypeId::FLOAT64).call().unwrap();
    let roots = floats.view().sqrt().call().unwrap();
    let result = roots.to_vec_f64();
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
    let ids = Column::from_slice_i32(&[3, 1, 4, 1, 5]);
    let vals = Column::from_slice_f64(&[30.0, 10.0, 40.0, 11.0, 50.0]);
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
    let values = Column::from_slice_i32(&[10, 25, 5, 30, 15]);
    let tbl = build_table(vec![values]);

    // Build mask: values > 12
    let threshold = Column::from_scalar(&Scalar::from_i32(12), 5);
    let mask = tbl
        .column(0)
        .unwrap()
        .gt(&threshold.view())
        .call()
        .unwrap();

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
    let keys = Column::from_slice_i32(&[1, 2, 1, 2, 1]);
    let vals = Column::from_slice_i32(&[10, 20, 30, 40, 50]);
    let tbl = build_table(vec![keys, vals]);

    // SUM by key
    let sum_result = tbl.groupby(&[0], 1, AggregationKind::SUM).call().unwrap();
    let sum_sorted = sorted(&sum_result);
    assert_eq!(table_col_i32(&sum_sorted, 0), [1, 2]); // keys
    assert_eq!(table_col_i64(&sum_sorted, 1), [90, 60]); // sums

    // COUNT by key
    let count_result = tbl
        .groupby(&[0], 1, AggregationKind::COUNT)
        .call()
        .unwrap();
    let count_sorted = sorted(&count_result);
    assert_eq!(table_col_i32(&count_sorted, 0), [1, 2]);
    assert_eq!(table_col_i32(&count_sorted, 1), [3, 2]);
}

// ===========================================================================
// Test: GroupBy with multiple aggregations
// ===========================================================================

#[test]
fn groupby_multi_aggregation() {
    let keys = Column::from_slice_i32(&[1, 2, 1, 2, 1]);
    let vals = Column::from_slice_i32(&[10, 20, 30, 40, 50]);
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
    let left_keys = Column::from_slice_i32(&[1, 2, 3, 4]);
    let left_vals = Column::from_slice_f64(&[10.0, 20.0, 30.0, 40.0]);
    let left = build_table(vec![left_keys, left_vals]);

    let right_keys = Column::from_slice_i32(&[2, 4, 5]);
    let right_vals = Column::from_slice_f64(&[200.0, 400.0, 500.0]);
    let right = build_table(vec![right_keys, right_vals]);

    let joined = left
        .inner_join(&right, &[0], &[0])
        .call()
        .unwrap();

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
    let left_keys = Column::from_slice_i32(&[1, 2, 3]);
    let left_vals = Column::from_slice_i32(&[10, 20, 30]);
    let left = build_table(vec![left_keys, left_vals]);

    let right_keys = Column::from_slice_i32(&[2]);
    let right_vals = Column::from_slice_i32(&[200]);
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
    let col = Column::from_strings(&["Hello", "WORLD", "foo Bar"]);

    let lower = col.view().to_lower().call().unwrap();
    assert_eq!(lower.to_vec_string(), ["hello", "world", "foo bar"]);

    let upper = col.view().to_upper().call().unwrap();
    assert_eq!(upper.to_vec_string(), ["HELLO", "WORLD", "FOO BAR"]);

    // Check which strings contain "o"
    let target = Scalar::from_string("o");
    let mask = col.view().str_contains(&target).call().unwrap();
    assert_eq!(mask.to_vec_bool(), [true, false, true]);
}

#[test]
fn string_split_and_length() {
    let col = Column::from_strings(&["a,b,c", "x,y", "hello"]);

    // Count characters
    let lengths = col.view().count_characters().call().unwrap();
    assert_eq!(lengths.to_vec_i32(), [5, 3, 5]);

    // Split by comma
    let delimiter = Scalar::from_string(",");
    let parts = col.view().str_split(&delimiter, -1).call().unwrap();
    assert_eq!(parts.columns_len(), 3); // max 3 parts
}

#[test]
fn string_regex_operations() {
    let col = Column::from_strings(&["abc123", "def", "456ghi", "789"]);

    // Check regex contains digits
    let has_digits = col.view().str_contains_re("\\d+").call().unwrap();
    assert_eq!(has_digits.to_vec_bool(), [true, false, true, true]);

    // Count digit sequences
    let counts = col.view().str_count_re("\\d+").call().unwrap();
    assert_eq!(counts.to_vec_i32(), [1, 0, 1, 1]);

    // Replace digits with "#"
    let replaced = col.view().str_replace_re("\\d+", "#").call().unwrap();
    assert_eq!(
        replaced.to_vec_string(),
        ["abc#", "def", "#ghi", "#"]
    );
}

// ===========================================================================
// Test: Null handling workflow
// ===========================================================================

#[test]
fn create_column_with_nulls_and_filter() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50]);
    // Create validity mask: positions 1 and 3 are null
    let validity = Column::from_slice_bool(&[true, false, true, false, true]);
    let with_nulls = col.with_null_mask_from_bools(&validity.view()).unwrap();

    assert_eq!(with_nulls.len(), 5);
    assert_eq!(with_nulls.null_count(), 2);
    assert!(with_nulls.has_nulls());

    // Verify the null mask round-trips
    assert_eq!(with_nulls.null_mask_to_host(), [true, false, true, false, true]);
}

#[test]
fn drop_null_rows_from_table() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40]);
    let validity = Column::from_slice_bool(&[true, false, true, false]);
    let with_nulls = col.with_null_mask_from_bools(&validity.view()).unwrap();
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
    let t1 = build_table(vec![Column::from_slice_i32(&[1, 2, 3])]);
    let t2 = build_table(vec![Column::from_slice_i32(&[4, 5])]);

    let combined = t1.concatenate(&t2).call().unwrap();
    assert_eq!(combined.len(), 5);
    assert_eq!(table_col_i32(&combined, 0), [1, 2, 3, 4, 5]);
}

// ===========================================================================
// Test: Partitioning round-trip
// ===========================================================================

#[test]
fn hash_partition_and_recombine() {
    let ids = Column::from_slice_i32(&[1, 2, 3, 4, 5, 6, 7, 8]);
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
    let keys = Column::from_slice_i32(&[1, 2, 2, 3, 3, 3]);
    let vals = Column::from_slice_i32(&[10, 20, 21, 30, 31, 32]);
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
    let regions = Column::from_slice_i32(&[1, 2, 1, 2, 1, 2, 1, 2]);
    let amounts = Column::from_slice_i32(&[100, 200, 150, 50, 300, 75, 25, 400]);
    let tbl = build_table(vec![regions, amounts]);

    // Step 1: Filter to amounts >= 100
    let threshold = Column::from_scalar(&Scalar::from_i32(100), tbl.len());
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
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50]);
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
    assert_eq!(seq.to_vec_i32(), [0, 10, 20, 30, 40]);
}

// ===========================================================================
// Test: Hashing produces consistent results
// ===========================================================================

#[test]
fn murmur3_hash_deterministic() {
    let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]);
    let tbl = build_table(vec![col]);

    // Same input should produce same hash
    let hash1 = tbl.murmur3(42).call();
    let hash2 = tbl.murmur3(42).call();
    assert_eq!(hash1.to_vec_u32(), hash2.to_vec_u32());

    // Different seed should produce different hash
    let hash3 = tbl.murmur3(99).call();
    assert_ne!(hash1.to_vec_u32(), hash3.to_vec_u32());
}

// ===========================================================================
// Test: Repeat and tile
// ===========================================================================

#[test]
fn repeat_and_tile_table() {
    let col = Column::from_slice_i32(&[1, 2, 3]);
    let tbl = build_table(vec![col]);

    // repeat duplicates each row N times (row-wise)
    let repeated = tbl.repeat(3).call().unwrap();
    assert_eq!(repeated.len(), 9);
    assert_eq!(
        table_col_i32(&repeated, 0),
        [1, 1, 1, 2, 2, 2, 3, 3, 3]
    );

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
    let col = Column::from_slice_i32(&[10, 20, 30]);
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
    let target = build_table(vec![Column::from_slice_i32(&[0, 0, 0, 0, 0])]);
    let source = build_table(vec![Column::from_slice_i32(&[99, 88])]);
    let indices = Column::from_slice_i32(&[1, 3]);

    let result = target
        .scatter(&source, &indices.view())
        .call()
        .unwrap();
    assert_eq!(table_col_i32(&result, 0), [0, 99, 0, 88, 0]);
}

// ===========================================================================
// Test: Interleave columns
// ===========================================================================

#[test]
fn interleave_table_columns() {
    let a = Column::from_slice_i32(&[1, 2, 3]);
    let b = Column::from_slice_i32(&[10, 20, 30]);
    let tbl = build_table(vec![a, b]);

    let interleaved = tbl.interleave_columns().call().unwrap();
    assert_eq!(interleaved.to_vec_i32(), [1, 10, 2, 20, 3, 30]);
}

// ===========================================================================
// Test: Search (lower_bound / upper_bound)
// ===========================================================================

#[test]
fn lower_and_upper_bound_search() {
    // Sorted haystack
    let haystack = build_table(vec![Column::from_slice_i32(&[10, 20, 20, 30, 40])]);
    // Needles to search for
    let needles = build_table(vec![Column::from_slice_i32(&[20, 35])]);

    let lower = haystack
        .lower_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
        .call()
        .unwrap();
    assert_eq!(lower.to_vec_i32(), [1, 4]); // first 20 at index 1, 35 would go at 4

    let upper = haystack
        .upper_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
        .call()
        .unwrap();
    assert_eq!(upper.to_vec_i32(), [3, 4]); // past last 20 at index 3
}

// ===========================================================================
// Test: Boolean comparison chaining
// ===========================================================================

#[test]
fn chained_boolean_comparisons() {
    let col = Column::from_slice_i32(&[5, 15, 25, 35, 45]);

    // 10 <= col <= 30
    let lo = Column::from_scalar(&Scalar::from_i32(10), 5);
    let hi = Column::from_scalar(&Scalar::from_i32(30), 5);

    let ge_lo = col.view().ge(&lo.view()).call().unwrap();
    let le_hi = col.view().le(&hi.view()).call().unwrap();

    // AND the two masks using binary op
    use cudf::ops::BinaryOperator;
    let in_range = ge_lo
        .view()
        .binary_op(&le_hi.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();
    assert_eq!(in_range.to_vec_bool(), [false, true, true, false, false]);
}

// ===========================================================================
// Test: String-to-integer conversion pipeline
// ===========================================================================

#[test]
fn string_to_integer_conversion() {
    let strings = Column::from_strings(&["100", "200", "300"]);

    // Convert to integers
    let ints = strings
        .view()
        .str_to_integers(TypeId::INT32)
        .call()
        .unwrap();
    assert_eq!(ints.to_vec_i32(), [100, 200, 300]);

    // Convert back to strings
    let back = ints.view().str_from_integers().call().unwrap();
    assert_eq!(back.to_vec_string(), ["100", "200", "300"]);
}

// ===========================================================================
// Test: Replace literal strings
// ===========================================================================

#[test]
fn string_replace_literal() {
    let col = Column::from_strings(&["hello world", "world peace", "no match"]);

    let replaced = col
        .view()
        .str_replace_literal("world", "earth", -1)
        .call()
        .unwrap();
    assert_eq!(
        replaced.to_vec_string(),
        ["hello earth", "earth peace", "no match"]
    );
}

// ===========================================================================
// Test: Cross join produces cartesian product
// ===========================================================================

#[test]
fn cross_join_cartesian_product() {
    let left = build_table(vec![Column::from_slice_i32(&[1, 2])]);
    let right = build_table(vec![Column::from_slice_i32(&[10, 20, 30])]);

    let result = left.cross_join(&right).call().unwrap();
    assert_eq!(result.len(), 6); // 2 * 3
    assert_eq!(result.columns_len(), 2);
}

// ===========================================================================
// Test: Merge two sorted tables
// ===========================================================================

#[test]
fn merge_two_sorted_tables() {
    let t1 = build_table(vec![Column::from_slice_i32(&[1, 3, 5])]);
    let t2 = build_table(vec![Column::from_slice_i32(&[2, 4, 6])]);

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
    let col = Column::from_scalar(&Scalar::from_i32(42), 5);
    assert_eq!(col.len(), 5);
    assert_eq!(col.to_vec_i32(), [42, 42, 42, 42, 42]);
    assert!(!col.has_nulls());
}

// ===========================================================================
// Test: Gather rows by index
// ===========================================================================

#[test]
fn gather_specific_rows() {
    let col = Column::from_slice_i32(&[10, 20, 30, 40, 50]);
    let tbl = build_table(vec![col]);

    let indices = Column::from_slice_i32(&[4, 2, 0]);
    let gathered = tbl.gather(&indices.view()).call().unwrap();

    assert_eq!(gathered.len(), 3);
    assert_eq!(table_col_i32(&gathered, 0), [50, 30, 10]);
}
