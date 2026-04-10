// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated SQL executor using DataFusion for query planning and cudf
//! for execution. Compares GPU vs CPU execution times on TPC-H queries.
//!
//! Supports two modes:
//! - `--path=/data/tpch/sf1/snappy` — load real TPC-H parquet files
//! - `--scale=10` — generate synthetic data (default: scale=1)

use std::sync::Arc;
use std::time::Instant;

use cudf_datafusion::executor::{GpuExecutor, gpu_table_to_df_record_batch};
use cudf_datafusion::tpch;
use datafusion::arrow::util::pretty::print_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;

type BoxError = Box<dyn std::error::Error>;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = parse_args();

    match args {
        Mode::Parquet(path) => run_parquet(&path).await,
        Mode::Synthetic(scale) => run_synthetic(scale).await,
    }
}

/// Runs the benchmark using real TPC-H parquet files.
async fn run_parquet(path: &str) -> Result<(), BoxError> {
    println!("TPC-H GPU vs CPU Benchmark (parquet: {path})\n");

    // Set up a pool memory resource using 90% of device memory.
    let total_mem = cudf::rmm::device::total_memory();
    let pool_size = (total_mem * 9 / 10) & !255; // 90%, aligned to 256
    let mut pool =
        cudf::rmm::memory_resource::PoolMemoryResource::with_limits(pool_size, pool_size)?;
    cudf::rmm::memory_resource::set_current_device_resource(&pool.as_ref());
    println!(
        "GPU memory pool: {:.1} GiB",
        pool.pool_size() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // GPU: load with cudf parquet reader.
    let gpu_start = Instant::now();
    let gpu_catalog = tpch::load_from_parquet(path)?;
    let gpu_load_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
    println!("GPU data loading: {gpu_load_ms:.1} ms");

    // Print table sizes.
    for (name, gpu_table) in gpu_catalog.iter() {
        println!(
            "  {name}: {} rows x {} cols",
            gpu_table.table.len(),
            gpu_table.table.columns_len()
        );
    }
    println!();

    // CPU: register parquet files directly with DataFusion for native execution.
    let cpu_ctx = SessionContext::new();
    for name in ["lineitem", "orders", "customer", "nation", "supplier"] {
        let filepath = format!("{path}/{name}.parquet");
        cpu_ctx
            .register_parquet(name, &filepath, ParquetReadOptions::default())
            .await?;
    }
    println!("CPU: registered parquet files with DataFusion\n");

    let gpu_queries = tpch::queries_parquet();
    let cpu_queries = tpch::queries_parquet_cpu();
    run_benchmark(&gpu_catalog, &cpu_ctx, &gpu_queries, &cpu_queries).await
}

/// Runs the benchmark using synthetic generated data.
async fn run_synthetic(scale: usize) -> Result<(), BoxError> {
    println!("TPC-H GPU vs CPU Benchmark (synthetic, scale={scale})\n");

    // Generate GPU tables.
    let gen_start = Instant::now();
    let gpu_catalog = tpch::create_catalog(scale)?;
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    println!("Data generation: {gen_ms:.1} ms\n");

    // Register tables with DataFusion for CPU execution.
    // Split into multiple partitions so DataFusion can use all cores.
    let num_partitions = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let cpu_ctx = SessionContext::new();
    for (name, gpu_table) in gpu_catalog.iter() {
        let batch = gpu_table_to_df_record_batch(&gpu_table.table, &gpu_table.column_names)?;
        let schema = Arc::clone(batch.schema_ref());
        let nrows = batch.num_rows();
        let chunk_size = (nrows + num_partitions - 1) / num_partitions;

        let mut partitions = Vec::new();
        let mut offset = 0;
        while offset < nrows {
            let len = chunk_size.min(nrows - offset);
            partitions.push(vec![batch.slice(offset, len)]);
            offset += len;
        }
        if partitions.is_empty() {
            partitions.push(vec![batch]);
        }
        let mem_table = MemTable::try_new(schema, partitions)?;
        cpu_ctx.register_table(name, Arc::new(mem_table))?;
    }
    println!("CPU partitions: {num_partitions}\n");

    let queries = tpch::queries();
    run_benchmark(&gpu_catalog, &cpu_ctx, &queries, &queries).await
}

/// Runs GPU vs CPU benchmark for a set of queries and prints results.
///
/// `gpu_queries` and `cpu_queries` may differ (e.g., integer dates vs DATE
/// literals) but must be the same length and correspond 1:1.
async fn run_benchmark(
    gpu_catalog: &cudf_datafusion::catalog::GpuCatalog,
    cpu_ctx: &SessionContext,
    gpu_queries: &[(&str, &str)],
    cpu_queries: &[(&str, &str)],
) -> Result<(), BoxError> {
    // Print header.
    println!(
        "{:<35} {:>10} {:>10} {:>8}",
        "Query", "GPU (ms)", "CPU (ms)", "Speedup"
    );
    println!("{}", "-".repeat(67));

    for (idx, ((name, gpu_sql), (_, cpu_sql))) in
        gpu_queries.iter().zip(cpu_queries.iter()).enumerate()
    {
        // GPU execution.
        let gpu_executor = GpuExecutor::new(gpu_catalog);
        let gpu_start = Instant::now();
        let gpu_batch = gpu_executor.run_sql(gpu_sql).await?;
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

        // CPU execution (DataFusion native).
        let cpu_start = Instant::now();
        let cpu_df = cpu_ctx.sql(cpu_sql).await?;
        let cpu_batches = cpu_df.collect().await?;
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms;
        println!("{name:<35} {gpu_ms:>9.1} {cpu_ms:>9.1} {speedup:>7.1}x");

        // Print first query results to show correctness.
        if idx == 0 {
            println!("\n  GPU result:");
            print_batches(&[gpu_batch])?;
            println!("\n  CPU result:");
            print_batches(&cpu_batches)?;
            println!();
        }
    }

    Ok(())
}

enum Mode {
    Parquet(String),
    Synthetic(usize),
}

/// Parses command-line arguments.
///
/// - `--path=<dir>` — load parquet files from the given directory
/// - `--scale=N` — generate synthetic data with scale factor N (default: 1)
fn parse_args() -> Mode {
    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--path=") {
            return Mode::Parquet(val.to_string());
        }
    }
    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--scale=") {
            if let Ok(n) = val.parse::<usize>() {
                return Mode::Synthetic(n);
            }
        }
    }
    Mode::Synthetic(1)
}
