// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU operation benchmarks for cudf.
//!
//! Measures end-to-end latency of common GPU operations including kernel
//! launch overhead and synchronization. Reports throughput in elements/sec.
//!
//! Run with: `pixi run cargo bench`

use std::hint::black_box;
use std::mem;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use cudf::column::UnboundColumn as Column;
use cudf::data_type::TypeId;
use cudf::groupby::AggregationKind;
use cudf::ops::BinaryOperator;
use cudf::scalar::Scalar;
use cudf::sorting::{NullOrder, Order};
use cudf::stream::GpuOp;
use cudf::table::{TableBuilder, UnboundTable as Table};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_i32_column(n: usize) -> Column {
    let data: Vec<i32> = (0..n).map(|i| ((i * 7 + 13) % 1000) as i32).collect();
    Column::from_slice_i32(&data).call().unwrap()
}

fn make_f64_column(n: usize) -> Column {
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 1.1 + 0.5).collect();
    Column::from_slice_f64(&data).call().unwrap()
}

fn make_string_column(n: usize) -> Column {
    let data: Vec<String> = (0..n).map(|i| format!("str_{:06}", i % 100)).collect();
    let refs: Vec<&str> = data.iter().map(String::as_str).collect();
    Column::from_strings(&refs).call().unwrap()
}

fn make_table(columns: Vec<Column>) -> Table {
    let mut b = TableBuilder::new();
    for c in columns {
        b.push_column(c);
    }
    b.build().unwrap()
}

const SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];
const SIZES_SMALL: &[usize] = &[1_000, 10_000, 100_000];

/// Average bytes per string in `make_string_column` ("str_000000" = 10 bytes).
const AVG_STRING_BYTES: u64 = 10;

// ---------------------------------------------------------------------------
// Column creation
// ---------------------------------------------------------------------------

fn bench_column_from_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_from_scalar");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    let scalar = Scalar::from_i32(42);
    for &n in SIZES {
        group.throughput(Throughput::Bytes((n * mem::size_of::<i32>()) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                Column::from_scalar(black_box(&scalar), black_box(n))
                    .call()
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_column_from_slice_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_from_slice_i32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Bytes((n * mem::size_of::<i32>()) as u64));
        let data: Vec<i32> = (0..n as i32).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, data| {
            b.iter(|| Column::from_slice_i32(black_box(data)).call().unwrap());
        });
    }
    group.finish();
}

fn bench_column_from_strings(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_from_strings");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES_SMALL {
        group.throughput(Throughput::Bytes(n as u64 * AVG_STRING_BYTES));
        let data: Vec<String> = (0..n).map(|i| format!("value_{i}")).collect();
        let refs: Vec<&str> = data.iter().map(String::as_str).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &refs, |b, refs| {
            b.iter(|| Column::from_strings(black_box(refs)).call().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

fn bench_binary_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_op_add_f64");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a = make_f64_column(n);
        let b_col = make_f64_column(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                a.view()
                    .binary_op(&b_col.view(), BinaryOperator::ADD, TypeId::FLOAT64)
                    .call()
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_gt_i32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a = make_i32_column(n);
        let threshold = Column::from_scalar(&Scalar::from_i32(500), n)
            .call()
            .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| a.view().gt(&threshold.view()).call().unwrap());
        });
    }
    group.finish();
}

fn bench_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("cast_i32_to_f64");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let col = make_i32_column(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| col.view().cast(TypeId::FLOAT64).call().unwrap());
        });
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f64");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let col = make_f64_column(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| col.view().sum(TypeId::FLOAT64).call().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Table operations
// ---------------------------------------------------------------------------

fn bench_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_i32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let tbl = make_table(vec![make_i32_column(n)]);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                tbl.sort(&[Order::ASCENDING], &[NullOrder::BEFORE])
                    .call()
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_i32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let col = make_i32_column(n);
        let threshold = Column::from_scalar(&Scalar::from_i32(500), n)
            .call()
            .unwrap();
        let mask = col.view().gt(&threshold.view()).call().unwrap();
        let tbl = make_table(vec![col]);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| tbl.filter(&mask.view()).call().unwrap());
        });
    }
    group.finish();
}

fn bench_groupby(c: &mut Criterion) {
    let mut group = c.benchmark_group("groupby_sum_f64");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let keys: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();
        let tbl = make_table(vec![
            Column::from_slice_i32(&keys).call().unwrap(),
            make_f64_column(n),
        ]);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| tbl.groupby(&[0], 1, AggregationKind::SUM).call().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Joins
// ---------------------------------------------------------------------------

fn bench_inner_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_join_i32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES_SMALL {
        // Throughput = total rows across both sides.
        group.throughput(Throughput::Elements(2 * n as u64));
        let left_keys: Vec<i32> = (0..n as i32).collect();
        let right_keys: Vec<i32> = (0..n as i32).rev().collect();
        let tbl_l = make_table(vec![
            Column::from_slice_i32(&left_keys).call().unwrap(),
            make_f64_column(n),
        ]);
        let tbl_r = make_table(vec![
            Column::from_slice_i32(&right_keys).call().unwrap(),
            make_f64_column(n),
        ]);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| tbl_l.inner_join(&tbl_r, &[0], &[0]).call().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Host transfer
// ---------------------------------------------------------------------------

fn bench_to_vec_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec_i32");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES {
        group.throughput(Throughput::Bytes((n * mem::size_of::<i32>()) as u64));
        let col = make_i32_column(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| col.view().to_vec_i32().call().unwrap());
        });
    }
    group.finish();
}

fn bench_to_vec_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_vec_string");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    for &n in SIZES_SMALL {
        group.throughput(Throughput::Bytes(n as u64 * AVG_STRING_BYTES));
        let col = make_string_column(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| col.to_vec_string().call().unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    creation,
    bench_column_from_scalar,
    bench_column_from_slice_i32,
    bench_column_from_strings,
);

criterion_group!(
    compute,
    bench_binary_op,
    bench_comparison,
    bench_cast,
    bench_sum,
);

criterion_group!(table_ops, bench_sort, bench_filter, bench_groupby,);

criterion_group!(joins, bench_inner_join,);

criterion_group!(transfer, bench_to_vec_i32, bench_to_vec_string,);

criterion_main!(creation, compute, table_ops, joins, transfer);
