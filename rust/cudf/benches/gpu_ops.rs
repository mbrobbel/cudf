// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU operation benchmarks for cudf.
//!
//! Minimal benchmark harness with no external dependencies.
//! Reports median wall-clock time per iteration.
//!
//! Run with: `pixi run cargo bench`

use std::time::Instant;

use cudf::column::Column;
use cudf::data_type::TypeId;
use cudf::groupby::AggregationKind;
use cudf::ops::BinaryOperator;
use cudf::scalar::Scalar;
use cudf::sorting::{NullOrder, Order};
use cudf::table::{Table, TableBuilder};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const WARMUP_ITERS: u32 = 3;
const BENCH_ITERS: u32 = 20;

fn bench<F: FnMut()>(name: &str, mut f: F) {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        f();
    }

    // Timed iterations
    let mut times = Vec::with_capacity(BENCH_ITERS as usize);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    times.sort();
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    println!(
        "{name:<45} median: {:>10.3?}  min: {:>10.3?}  max: {:>10.3?}",
        median, min, max,
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_i32_column(n: usize) -> Column {
    let data: Vec<i32> = (0..n).map(|i| ((i * 7 + 13) % 1000) as i32).collect();
    Column::from_slice_i32(&data)
}

fn make_f64_column(n: usize) -> Column {
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 1.1 + 0.5).collect();
    Column::from_slice_f64(&data)
}

fn make_string_column(n: usize) -> Column {
    let data: Vec<String> = (0..n).map(|i| format!("str_{:06}", i % 100)).collect();
    let refs: Vec<&str> = data.iter().map(String::as_str).collect();
    Column::from_strings(&refs)
}

fn make_table(columns: Vec<Column>) -> Table {
    let mut b = TableBuilder::new();
    for c in columns {
        b.push_column(c);
    }
    b.build().unwrap()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_column_creation() {
    println!("\n--- Column Creation ---");
    let scalar = Scalar::from_i32(42);
    for n in [1_000, 10_000, 100_000, 1_000_000] {
        bench(&format!("from_scalar/i32/{n}"), || {
            let _ = Column::from_scalar(&scalar, n);
        });
    }

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data: Vec<i32> = (0..n as i32).collect();
        bench(&format!("from_slice_i32/{n}"), || {
            let _ = Column::from_slice_i32(&data);
        });
    }

    for n in [1_000, 10_000, 100_000] {
        let data: Vec<String> = (0..n).map(|i| format!("value_{i}")).collect();
        let refs: Vec<&str> = data.iter().map(String::as_str).collect();
        bench(&format!("from_strings/{n}"), || {
            let _ = Column::from_strings(&refs);
        });
    }
}

fn bench_compute() {
    println!("\n--- Compute ---");
    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let a = make_f64_column(n);
        let b = make_f64_column(n);
        bench(&format!("binary_op/add_f64/{n}"), || {
            let _ = a
                .view()
                .binary_op(&b.view(), BinaryOperator::ADD, TypeId::FLOAT64)
                .call()
                .unwrap();
        });
    }

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let a = make_i32_column(n);
        let threshold = Column::from_scalar(&Scalar::from_i32(500), n);
        bench(&format!("comparison/gt_i32/{n}"), || {
            let _ = a.view().gt(&threshold.view()).call().unwrap();
        });
    }

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let col = make_i32_column(n);
        bench(&format!("cast/i32_to_f64/{n}"), || {
            let _ = col.view().cast(TypeId::FLOAT64).call().unwrap();
        });
    }

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let col = make_f64_column(n);
        bench(&format!("sum/f64/{n}"), || {
            let _ = col.view().sum(TypeId::FLOAT64).call().unwrap();
        });
    }
}

fn bench_table_ops() {
    println!("\n--- Table Operations ---");
    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let tbl = make_table(vec![make_i32_column(n)]);
        bench(&format!("sort/i32/{n}"), || {
            let _ = tbl
                .sort(&[Order::ASCENDING], &[NullOrder::BEFORE])
                .call()
                .unwrap();
        });
    }

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let col = make_i32_column(n);
        let threshold = Column::from_scalar(&Scalar::from_i32(500), n);
        let mask = col.view().gt(&threshold.view()).call().unwrap();
        let tbl = make_table(vec![col]);
        bench(&format!("filter/i32/{n}"), || {
            let _ = tbl.filter(&mask.view()).call().unwrap();
        });
    }

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let keys: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();
        let tbl = make_table(vec![Column::from_slice_i32(&keys), make_f64_column(n)]);
        bench(&format!("groupby/sum_f64_100groups/{n}"), || {
            let _ = tbl
                .groupby(&[0], 1, AggregationKind::SUM)
                .call()
                .unwrap();
        });
    }
}

fn bench_joins() {
    println!("\n--- Joins ---");
    for n in [1_000, 10_000, 100_000] {
        let left_keys: Vec<i32> = (0..n as i32).collect();
        let right_keys: Vec<i32> = (0..n as i32).rev().collect();
        let tbl_l = make_table(vec![
            Column::from_slice_i32(&left_keys),
            make_f64_column(n),
        ]);
        let tbl_r = make_table(vec![
            Column::from_slice_i32(&right_keys),
            make_f64_column(n),
        ]);
        bench(&format!("inner_join/i32_unique/{n}"), || {
            let _ = tbl_l.inner_join(&tbl_r, &[0], &[0]).call().unwrap();
        });
    }
}

fn bench_transfer() {
    println!("\n--- Host Transfer ---");
    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let col = make_i32_column(n);
        bench(&format!("to_vec_i32/{n}"), || {
            let _ = col.view().to_vec_i32();
        });
    }

    for n in [1_000, 10_000, 100_000] {
        let col = make_string_column(n);
        bench(&format!("to_vec_string/{n}"), || {
            let _ = col.to_vec_string();
        });
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("cudf GPU operation benchmarks");
    println!("=============================");
    println!(
        "{} warmup iterations, {} bench iterations (reporting median)",
        WARMUP_ITERS, BENCH_ITERS,
    );

    bench_column_creation();
    bench_compute();
    bench_table_ops();
    bench_joins();
    bench_transfer();
}
