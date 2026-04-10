// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Scalable TPC-H data generation and query definitions.
//!
//! Data is generated deterministically using modular arithmetic so results are
//! reproducible across runs with the same scale factor.

use cudf::column::Column;
use cudf::data_type::TypeId;
use cudf::stream::GpuOp;
use cudf::table::{Table, TableBuilder};

use crate::catalog::GpuCatalog;

type BoxError = Box<dyn std::error::Error>;

/// The 25 TPC-H nation names in standard order.
const NATION_NAMES: [&str; 25] = [
    "ALGERIA",
    "ARGENTINA",
    "BRAZIL",
    "CANADA",
    "EGYPT",
    "ETHIOPIA",
    "FRANCE",
    "GERMANY",
    "INDIA",
    "INDONESIA",
    "IRAN",
    "IRAQ",
    "JAPAN",
    "JORDAN",
    "KENYA",
    "MOROCCO",
    "MOZAMBIQUE",
    "PERU",
    "CHINA",
    "ROMANIA",
    "SAUDI ARABIA",
    "VIETNAM",
    "RUSSIA",
    "UNITED KINGDOM",
    "UNITED STATES",
];

/// Region keys for each nation (mapping to 5 regions, 0-4).
const NATION_REGIONKEYS: [i32; 25] = [
    0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2, 4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1,
];

const RETURN_FLAGS: [&str; 3] = ["A", "N", "R"];
const LINE_STATUSES: [&str; 2] = ["F", "O"];
const ORDER_STATUSES: [&str; 3] = ["F", "O", "P"];
const MARKET_SEGMENTS: [&str; 5] = [
    "AUTOMOBILE",
    "BUILDING",
    "FURNITURE",
    "HOUSEHOLD",
    "MACHINERY",
];

fn build_table(columns: Vec<Column>) -> Result<Table, BoxError> {
    let mut b = TableBuilder::new();
    for c in columns {
        b.push_column(c);
    }
    Ok(b.build()?)
}

fn generate_lineitem(n: usize) -> Result<(Table, Vec<String>), BoxError> {
    let mut orderkeys = Vec::with_capacity(n);
    let mut partkeys = Vec::with_capacity(n);
    let mut suppkeys = Vec::with_capacity(n);
    let mut quantities = Vec::with_capacity(n);
    let mut prices = Vec::with_capacity(n);
    let mut discounts = Vec::with_capacity(n);
    let mut taxes = Vec::with_capacity(n);
    let mut returnflags: Vec<&str> = Vec::with_capacity(n);
    let mut linestatuses: Vec<&str> = Vec::with_capacity(n);
    let mut shipdates = Vec::with_capacity(n);

    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            orderkeys.push((i % 2500 + 1) as i32);
            partkeys.push((i % 200 + 1) as i32);
            suppkeys.push((i % 40 + 1) as i32);
        }
        quantities.push((i % 50 + 1) as f64);
        prices.push((i % 1000) as f64 * 0.5 + 1.0);
        discounts.push((i % 11) as f64 * 0.01);
        taxes.push((i % 9) as f64 * 0.01);
        returnflags.push(RETURN_FLAGS[i % 3]);
        linestatuses.push(LINE_STATUSES[i % 2]);
        // Ship dates cycle through a range around 9000-9500
        #[allow(clippy::cast_possible_truncation)]
        shipdates.push(9000 + (i % 500) as i32);
    }

    let table = build_table(vec![
        Column::from_slice_i32(&orderkeys).call()?,
        Column::from_slice_i32(&partkeys).call()?,
        Column::from_slice_i32(&suppkeys).call()?,
        Column::from_slice_f64(&quantities).call()?,
        Column::from_slice_f64(&prices).call()?,
        Column::from_slice_f64(&discounts).call()?,
        Column::from_slice_f64(&taxes).call()?,
        Column::from_strings(&returnflags).call()?,
        Column::from_strings(&linestatuses).call()?,
        Column::from_slice_i32(&shipdates).call()?,
    ])?;

    let names = vec![
        "l_orderkey".into(),
        "l_partkey".into(),
        "l_suppkey".into(),
        "l_quantity".into(),
        "l_extendedprice".into(),
        "l_discount".into(),
        "l_tax".into(),
        "l_returnflag".into(),
        "l_linestatus".into(),
        "l_shipdate".into(),
    ];

    Ok((table, names))
}

fn generate_orders(n: usize) -> Result<(Table, Vec<String>), BoxError> {
    let mut orderkeys = Vec::with_capacity(n);
    let mut custkeys = Vec::with_capacity(n);
    let mut statuses: Vec<&str> = Vec::with_capacity(n);
    let mut totalprices = Vec::with_capacity(n);
    let mut orderdates = Vec::with_capacity(n);

    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            orderkeys.push((i + 1) as i32);
            custkeys.push((i % 500 + 1) as i32);
        }
        statuses.push(ORDER_STATUSES[i % 3]);
        totalprices.push((i % 5000) as f64 * 10.0 + 100.0);
        #[allow(clippy::cast_possible_truncation)]
        orderdates.push(9000 + (i % 400) as i32);
    }

    let table = build_table(vec![
        Column::from_slice_i32(&orderkeys).call()?,
        Column::from_slice_i32(&custkeys).call()?,
        Column::from_strings(&statuses).call()?,
        Column::from_slice_f64(&totalprices).call()?,
        Column::from_slice_i32(&orderdates).call()?,
    ])?;

    let names = vec![
        "o_orderkey".into(),
        "o_custkey".into(),
        "o_orderstatus".into(),
        "o_totalprice".into(),
        "o_orderdate".into(),
    ];

    Ok((table, names))
}

fn generate_customer(n: usize) -> Result<(Table, Vec<String>), BoxError> {
    let mut custkeys = Vec::with_capacity(n);
    let mut names: Vec<String> = Vec::with_capacity(n);
    let mut nationkeys = Vec::with_capacity(n);
    let mut acctbals = Vec::with_capacity(n);
    let mut segments: Vec<&str> = Vec::with_capacity(n);

    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            custkeys.push((i + 1) as i32);
            nationkeys.push((i % 25) as i32);
        }
        names.push(format!("Customer#{i:09}"));
        acctbals.push((i % 10000) as f64 * 0.5 - 500.0);
        segments.push(MARKET_SEGMENTS[i % 5]);
    }

    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let table = build_table(vec![
        Column::from_slice_i32(&custkeys).call()?,
        Column::from_strings(&name_refs).call()?,
        Column::from_slice_i32(&nationkeys).call()?,
        Column::from_slice_f64(&acctbals).call()?,
        Column::from_strings(&segments).call()?,
    ])?;

    let col_names = vec![
        "c_custkey".into(),
        "c_name".into(),
        "c_nationkey".into(),
        "c_acctbal".into(),
        "c_mktsegment".into(),
    ];

    Ok((table, col_names))
}

fn generate_nation() -> Result<(Table, Vec<String>), BoxError> {
    let n = NATION_NAMES.len();
    let mut nationkeys = Vec::with_capacity(n);
    let mut regionkeys = Vec::with_capacity(n);

    for (i, &rk) in NATION_REGIONKEYS.iter().enumerate().take(n) {
        #[allow(clippy::cast_possible_truncation)]
        nationkeys.push(i as i32);
        regionkeys.push(rk);
    }

    let table = build_table(vec![
        Column::from_slice_i32(&nationkeys).call()?,
        Column::from_strings(&NATION_NAMES).call()?,
        Column::from_slice_i32(&regionkeys).call()?,
    ])?;

    let names = vec!["n_nationkey".into(), "n_name".into(), "n_regionkey".into()];

    Ok((table, names))
}

fn generate_supplier(n: usize) -> Result<(Table, Vec<String>), BoxError> {
    let mut suppkeys = Vec::with_capacity(n);
    let mut names: Vec<String> = Vec::with_capacity(n);
    let mut nationkeys = Vec::with_capacity(n);

    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        {
            suppkeys.push((i + 1) as i32);
            nationkeys.push((i % 25) as i32);
        }
        names.push(format!("Supplier#{i:09}"));
    }

    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let table = build_table(vec![
        Column::from_slice_i32(&suppkeys).call()?,
        Column::from_strings(&name_refs).call()?,
        Column::from_slice_i32(&nationkeys).call()?,
    ])?;

    let col_names = vec!["s_suppkey".into(), "s_name".into(), "s_nationkey".into()];

    Ok((table, col_names))
}

/// Creates a catalog populated with scalable TPC-H tables.
///
/// The `scale` parameter controls table sizes:
/// - `scale=1`: ~10K lineitem, ~2.5K orders, ~500 customers, 25 nations, ~40 suppliers
/// - `scale=10`: ~100K lineitem, ~25K orders, ~5K customers, 25 nations, ~400 suppliers
pub fn create_catalog(scale: usize) -> Result<GpuCatalog, BoxError> {
    let mut catalog = GpuCatalog::new();

    let lineitem_rows = scale * 10_000;
    let orders_rows = scale * 2_500;
    let customer_rows = scale * 500;
    let supplier_rows = scale * 40;

    let (table, names) = generate_lineitem(lineitem_rows)?;
    catalog.register("lineitem", table, names)?;

    let (table, names) = generate_orders(orders_rows)?;
    catalog.register("orders", table, names)?;

    let (table, names) = generate_customer(customer_rows)?;
    catalog.register("customer", table, names)?;

    let (table, names) = generate_nation()?;
    catalog.register("nation", table, names)?;

    let (table, names) = generate_supplier(supplier_rows)?;
    catalog.register("supplier", table, names)?;

    Ok(catalog)
}

/// Loads TPC-H tables from parquet files at the given path.
///
/// The path should be like `/data/tpch/sf1/snappy/`.
///
/// Since cudf does not support decimal128, decimal columns are cast to float64
/// after loading.
/// Returns the columns needed for each table across all parquet queries.
fn needed_columns(table: &str) -> &[&str] {
    match table {
        "lineitem" => &[
            "l_orderkey",
            "l_suppkey",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_shipmode",
        ],
        "orders" => &["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate"],
        "customer" => &["c_custkey", "c_name", "c_nationkey", "c_acctbal", "c_mktsegment"],
        "nation" => &["n_nationkey", "n_name", "n_regionkey"],
        "supplier" => &["s_suppkey", "s_name", "s_nationkey"],
        _ => &[],
    }
}

pub fn load_from_parquet(path: &str) -> Result<GpuCatalog, BoxError> {
    let mut catalog = GpuCatalog::new();

    let tables = ["lineitem", "orders", "customer", "nation", "supplier"];
    for name in tables {
        let filepath = format!("{path}/{name}.parquet");
        let cols = needed_columns(name);
        let meta = cudf::io::parquet::read_columns(&filepath, cols, 0, -1)?;
        let col_names: Vec<String> = meta.metadata.column_names;

        let (table, col_names) = normalize_column_types(meta.table, &col_names)?;

        catalog.register(name, table, col_names)?;
    }
    Ok(catalog)
}

/// Casts unsupported column types to compatible ones:
/// - decimal128/64/32 → FLOAT64
/// - TIMESTAMP_DAYS (date32) → INT32 (days-since-epoch, same underlying representation)
fn normalize_column_types(
    table: Table,
    names: &[String],
) -> Result<(Table, Vec<String>), BoxError> {
    let mut columns = Vec::with_capacity(table.columns_len());
    for i in 0..table.columns_len() {
        let view = table.column(i)?;
        match view.type_id() {
            TypeId::DECIMAL32 | TypeId::DECIMAL64 | TypeId::DECIMAL128 => {
                columns.push(view.cast(TypeId::FLOAT64).call()?);
            }
            TypeId::TIMESTAMP_DAYS => {
                // TIMESTAMP_DAYS stores days-since-epoch as i32, same bits as INT32.
                // cudf won't cast directly, so read back and re-create as INT32.
                let vals = view.to_vec_i32().call()?;
                columns.push(Column::from_slice_i32(&vals).call()?);
            }
            _ => {
                columns.push(view.to_owned_column().call()?);
            }
        }
    }
    Ok((Table::from_columns(columns)?, names.to_vec()))
}

/// Returns TPC-H queries for parquet data.
///
/// Since dates are normalized to INT32 (days-since-epoch) on load, we use
/// integer comparisons instead of DATE literals. Key epoch values:
/// 1994-01-01=8766, 1995-01-01=9131, 1995-03-15=9204, 1995-09-01=9374,
/// 1995-10-01=9404, 1998-09-02=10471.
pub fn queries_parquet() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "Q6  - Forecasting Revenue",
            concat!(
                "SELECT SUM(l_extendedprice * l_discount) AS revenue ",
                "FROM lineitem ",
                "WHERE l_shipdate >= 8766 ",
                "AND l_shipdate < 9131 ",
                "AND l_discount BETWEEN 0.05 AND 0.07 ",
                "AND l_quantity < 24"
            ),
        ),
        (
            "Q1  - Pricing Summary",
            concat!(
                "SELECT l_returnflag, l_linestatus, ",
                "SUM(l_quantity) AS sum_qty, ",
                "SUM(l_extendedprice) AS sum_base_price, ",
                "SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, ",
                "COUNT(*) AS count_order ",
                "FROM lineitem ",
                "WHERE l_shipdate <= 10471 ",
                "GROUP BY l_returnflag, l_linestatus ",
                "ORDER BY l_returnflag, l_linestatus"
            ),
        ),
        (
            "Q3  - Shipping Priority",
            concat!(
                "SELECT l_orderkey, ",
                "SUM(l_extendedprice * (1 - l_discount)) AS revenue, ",
                "o_orderdate ",
                "FROM customer ",
                "INNER JOIN orders ON c_custkey = o_custkey ",
                "INNER JOIN lineitem ON l_orderkey = o_orderkey ",
                "WHERE c_mktsegment = 'BUILDING' ",
                "AND o_orderdate < 9204 ",
                "AND l_shipdate > 9204 ",
                "GROUP BY l_orderkey, o_orderdate ",
                "ORDER BY revenue DESC, o_orderdate ",
                "LIMIT 10"
            ),
        ),
        (
            "Q5  - Local Supplier Volume",
            concat!(
                "SELECT n_name, ",
                "SUM(l_extendedprice * (1 - l_discount)) AS revenue ",
                "FROM customer ",
                "INNER JOIN orders ON c_custkey = o_custkey ",
                "INNER JOIN lineitem ON l_orderkey = o_orderkey ",
                "INNER JOIN supplier ON l_suppkey = s_suppkey AND c_nationkey = s_nationkey ",
                "INNER JOIN nation ON s_nationkey = n_nationkey ",
                "WHERE n_regionkey = 3 ",
                "AND o_orderdate >= 8766 ",
                "AND o_orderdate < 9131 ",
                "GROUP BY n_name ",
                "ORDER BY revenue DESC"
            ),
        ),
        (
            "Q12 - Shipping Modes",
            concat!(
                "SELECT l_shipmode, COUNT(*) AS order_count ",
                "FROM orders ",
                "INNER JOIN lineitem ON o_orderkey = l_orderkey ",
                "WHERE l_shipdate >= 8766 AND l_shipdate < 9131 ",
                "GROUP BY l_shipmode ",
                "ORDER BY l_shipmode"
            ),
        ),
        (
            "Q14 - Promotion Effect",
            concat!(
                "SELECT SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue ",
                "FROM lineitem ",
                "WHERE l_shipdate >= 9374 AND l_shipdate < 9404"
            ),
        ),
    ]
}

/// CPU queries for parquet data — same as `queries_parquet` but with DATE
/// literals instead of integer epoch days (DataFusion reads Date32 natively).
pub fn queries_parquet_cpu() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Q6", concat!(
            "SELECT SUM(l_extendedprice * l_discount) AS revenue ",
            "FROM lineitem ",
            "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' ",
            "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24")),
        ("Q1", concat!(
            "SELECT l_returnflag, l_linestatus, ",
            "SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_base_price, ",
            "SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, ",
            "COUNT(*) AS count_order FROM lineitem ",
            "WHERE l_shipdate <= DATE '1998-09-02' ",
            "GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus")),
        ("Q3", concat!(
            "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate ",
            "FROM customer INNER JOIN orders ON c_custkey = o_custkey ",
            "INNER JOIN lineitem ON l_orderkey = o_orderkey ",
            "WHERE c_mktsegment = 'BUILDING' AND o_orderdate < DATE '1995-03-15' ",
            "AND l_shipdate > DATE '1995-03-15' ",
            "GROUP BY l_orderkey, o_orderdate ORDER BY revenue DESC, o_orderdate LIMIT 10")),
        ("Q5", concat!(
            "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue ",
            "FROM customer INNER JOIN orders ON c_custkey = o_custkey ",
            "INNER JOIN lineitem ON l_orderkey = o_orderkey ",
            "INNER JOIN supplier ON l_suppkey = s_suppkey AND c_nationkey = s_nationkey ",
            "INNER JOIN nation ON s_nationkey = n_nationkey ",
            "WHERE n_regionkey = 3 AND o_orderdate >= DATE '1994-01-01' ",
            "AND o_orderdate < DATE '1995-01-01' GROUP BY n_name ORDER BY revenue DESC")),
        ("Q12", concat!(
            "SELECT l_shipmode, COUNT(*) AS order_count FROM orders ",
            "INNER JOIN lineitem ON o_orderkey = l_orderkey ",
            "WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' ",
            "GROUP BY l_shipmode ORDER BY l_shipmode")),
        ("Q14", concat!(
            "SELECT SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem ",
            "WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01'")),
    ]
}

/// Returns a list of TPC-H queries (name, SQL) for synthetic data.
pub fn queries() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "Q6  - Forecasting Revenue",
            concat!(
                "SELECT SUM(l_extendedprice * l_discount) AS revenue ",
                "FROM lineitem ",
                "WHERE l_discount >= 0.05 ",
                "AND l_discount <= 0.07 ",
                "AND l_quantity < 24"
            ),
        ),
        (
            "Q1  - Pricing Summary",
            concat!(
                "SELECT l_returnflag, l_linestatus, ",
                "SUM(l_quantity) AS sum_qty, ",
                "SUM(l_extendedprice) AS sum_base_price, ",
                "SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, ",
                "COUNT(*) AS count_order ",
                "FROM lineitem ",
                "WHERE l_shipdate <= 9400 ",
                "GROUP BY l_returnflag, l_linestatus ",
                "ORDER BY l_returnflag, l_linestatus"
            ),
        ),
        (
            "Q3  - Shipping Priority",
            concat!(
                "SELECT l_orderkey, ",
                "SUM(l_extendedprice * (1 - l_discount)) AS revenue, ",
                "o_orderdate ",
                "FROM customer ",
                "INNER JOIN orders ON c_custkey = o_custkey ",
                "INNER JOIN lineitem ON l_orderkey = o_orderkey ",
                "WHERE c_mktsegment = 'BUILDING' ",
                "AND o_orderdate < 9200 ",
                "AND l_shipdate > 9100 ",
                "GROUP BY l_orderkey, o_orderdate ",
                "ORDER BY revenue DESC, o_orderdate ",
                "LIMIT 10"
            ),
        ),
        (
            "Q5  - Local Supplier Volume",
            concat!(
                "SELECT n_name, ",
                "SUM(l_extendedprice * (1 - l_discount)) AS revenue ",
                "FROM customer ",
                "INNER JOIN orders ON c_custkey = o_custkey ",
                "INNER JOIN lineitem ON l_orderkey = o_orderkey ",
                "INNER JOIN supplier ON l_suppkey = s_suppkey AND c_nationkey = s_nationkey ",
                "INNER JOIN nation ON s_nationkey = n_nationkey ",
                "WHERE n_regionkey = 3 ",
                "GROUP BY n_name ",
                "ORDER BY revenue DESC"
            ),
        ),
        (
            "Q12 - Shipping Modes",
            concat!(
                "SELECT l_shipdate, COUNT(*) AS order_count ",
                "FROM orders ",
                "INNER JOIN lineitem ON o_orderkey = l_orderkey ",
                "WHERE l_shipdate >= 9100 AND l_shipdate < 9300 ",
                "GROUP BY l_shipdate ",
                "ORDER BY l_shipdate"
            ),
        ),
        (
            "Q14 - Promotion Effect",
            concat!(
                "SELECT SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue ",
                "FROM lineitem ",
                "WHERE l_shipdate >= 9100 AND l_shipdate < 9200"
            ),
        ),
    ]
}
