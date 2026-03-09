// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! TPC-H style integration tests.
//!
//! These tests build small TPC-H-like datasets on the host and execute
//! query-like pipelines entirely on the GPU to verify correctness of
//! multi-step analytic workloads.

#![allow(clippy::unwrap_used)]

use cudf::column::Column;
use cudf::data_type::TypeId;
use cudf::groupby::AggregationKind;
use cudf::ops::BinaryOperator;
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

fn table_col_i32(tbl: &Table, idx: usize) -> Vec<i32> {
    tbl.column(idx)
        .unwrap()
        .cast(TypeId::INT32)
        .call()
        .unwrap()
        .to_vec_i32()
}

fn table_col_i64(tbl: &Table, idx: usize) -> Vec<i64> {
    tbl.column(idx)
        .unwrap()
        .cast(TypeId::INT64)
        .call()
        .unwrap()
        .to_vec_i64()
}

fn table_col_f64(tbl: &Table, idx: usize) -> Vec<f64> {
    tbl.column(idx)
        .unwrap()
        .cast(TypeId::FLOAT64)
        .call()
        .unwrap()
        .to_vec_f64()
}

fn table_col_strings(tbl: &Table, idx: usize) -> Vec<String> {
    tbl.column(idx)
        .unwrap()
        .to_owned_column()
        .unwrap()
        .to_vec_string()
}

fn sorted_by(tbl: &Table, cols: &[Order], nulls: &[NullOrder]) -> Table {
    // Pad orders/nulls to match the table's column count (sort expects one per column).
    let n = tbl.columns_len();
    let mut orders = cols.to_vec();
    let mut null_orders = nulls.to_vec();
    orders.resize(n, Order::ASCENDING);
    null_orders.resize(n, NullOrder::BEFORE);
    tbl.sort(&orders, &null_orders).call().unwrap()
}

// ---------------------------------------------------------------------------
// TPC-H-like dataset builders
// ---------------------------------------------------------------------------

/// lineitem: l_orderkey, l_partkey, l_suppkey, l_quantity, l_extendedprice,
///           l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate
fn make_lineitem() -> Table {
    let l_orderkey = Column::from_slice_i32(&[1, 1, 2, 3, 3, 3, 4, 4, 5, 5]);
    let l_partkey = Column::from_slice_i32(&[100, 101, 100, 102, 103, 100, 101, 102, 100, 103]);
    let l_suppkey = Column::from_slice_i32(&[10, 11, 10, 12, 13, 10, 11, 12, 10, 13]);
    let l_quantity = Column::from_slice_f64(&[17.0, 36.0, 38.0, 45.0, 49.0, 27.0, 28.0, 24.0, 15.0, 44.0]);
    let l_extendedprice =
        Column::from_slice_f64(&[21168.23, 45983.16, 44694.46, 54058.05, 46796.47, 32986.53, 28955.64, 24322.68, 18397.35, 48620.72]);
    let l_discount = Column::from_slice_f64(&[0.04, 0.09, 0.0, 0.06, 0.1, 0.05, 0.09, 0.1, 0.02, 0.07]);
    let l_tax = Column::from_slice_f64(&[0.02, 0.06, 0.05, 0.0, 0.0, 0.07, 0.02, 0.06, 0.06, 0.02]);
    let l_returnflag = Column::from_strings(&["N", "N", "A", "R", "A", "N", "N", "R", "N", "A"]);
    let l_linestatus = Column::from_strings(&["O", "O", "F", "F", "F", "O", "O", "F", "O", "F"]);
    // Ship dates as days-since-epoch (INT32 for simplicity)
    let l_shipdate = Column::from_slice_i32(&[
        9374, 9400, 9100, 9200, 9150, 9380, 9410, 9250, 9390, 9300,
    ]);
    build_table(vec![
        l_orderkey,
        l_partkey,
        l_suppkey,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_returnflag,
        l_linestatus,
        l_shipdate,
    ])
}

/// orders: o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority
fn make_orders() -> Table {
    let o_orderkey = Column::from_slice_i32(&[1, 2, 3, 4, 5]);
    let o_custkey = Column::from_slice_i32(&[370, 781, 1234, 1369, 445]);
    let o_orderstatus = Column::from_strings(&["O", "F", "F", "O", "O"]);
    let o_totalprice =
        Column::from_slice_f64(&[172799.49, 46929.18, 193846.25, 56924.25, 56552.31]);
    // Order dates as days-since-epoch (INT32)
    let o_orderdate = Column::from_slice_i32(&[9300, 9100, 9150, 9350, 9375]);
    let o_orderpriority =
        Column::from_strings(&["1-URGENT", "5-LOW", "5-LOW", "3-MEDIUM", "1-URGENT"]);
    build_table(vec![
        o_orderkey,
        o_custkey,
        o_orderstatus,
        o_totalprice,
        o_orderdate,
        o_orderpriority,
    ])
}

/// customer: c_custkey, c_name, c_nationkey, c_acctbal, c_mktsegment
fn make_customer() -> Table {
    let c_custkey = Column::from_slice_i32(&[370, 445, 781, 1234, 1369]);
    let c_name = Column::from_strings(&[
        "Customer#000000370",
        "Customer#000000445",
        "Customer#000000781",
        "Customer#000001234",
        "Customer#000001369",
    ]);
    let c_nationkey = Column::from_slice_i32(&[1, 2, 1, 3, 2]);
    let c_acctbal = Column::from_slice_f64(&[4891.29, 3218.13, 6819.74, 711.56, 5765.78]);
    let c_mktsegment = Column::from_strings(&[
        "BUILDING",
        "MACHINERY",
        "HOUSEHOLD",
        "BUILDING",
        "AUTOMOBILE",
    ]);
    build_table(vec![c_custkey, c_name, c_nationkey, c_acctbal, c_mktsegment])
}

/// nation: n_nationkey, n_name, n_regionkey
fn make_nation() -> Table {
    let n_nationkey = Column::from_slice_i32(&[1, 2, 3]);
    let n_name = Column::from_strings(&["FRANCE", "GERMANY", "BRAZIL"]);
    let n_regionkey = Column::from_slice_i32(&[3, 3, 2]);
    build_table(vec![n_nationkey, n_name, n_regionkey])
}

/// supplier: s_suppkey, s_name, s_nationkey
fn make_supplier() -> Table {
    let s_suppkey = Column::from_slice_i32(&[10, 11, 12, 13]);
    let s_name = Column::from_strings(&[
        "Supplier#000000010",
        "Supplier#000000011",
        "Supplier#000000012",
        "Supplier#000000013",
    ]);
    let s_nationkey = Column::from_slice_i32(&[1, 2, 3, 1]);
    build_table(vec![s_suppkey, s_name, s_nationkey])
}

// ===========================================================================
// TPC-H Q1: Pricing Summary Report
//
//   SELECT l_returnflag, l_linestatus,
//          SUM(l_quantity)            AS sum_qty,
//          SUM(l_extendedprice)       AS sum_base_price,
//          SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
//          COUNT(*)                   AS count_order
//   FROM lineitem
//   WHERE l_shipdate <= date '1998-12-01' - interval '90' day
//   GROUP BY l_returnflag, l_linestatus
//   ORDER BY l_returnflag, l_linestatus
// ===========================================================================

#[test]
fn tpch_q1_pricing_summary() {
    let lineitem = make_lineitem();
    // Columns: 0=orderkey, 1=partkey, 2=suppkey, 3=quantity, 4=extendedprice,
    //          5=discount, 6=tax, 7=returnflag, 8=linestatus, 9=shipdate

    // Filter: l_shipdate <= 9400 (all rows qualify in our dataset)
    let ship_threshold = Column::from_scalar(&Scalar::from_i32(9400), lineitem.len());
    let mask = lineitem
        .column(9)
        .unwrap()
        .le(&ship_threshold.view())
        .call()
        .unwrap();
    let filtered = lineitem.filter(&mask.view()).call().unwrap();

    // Compute disc_price = l_extendedprice * (1 - l_discount)
    let ones = Column::from_scalar(&Scalar::from_f64(1.0), filtered.len());
    let one_minus_disc = ones
        .view()
        .sub(&filtered.column(5).unwrap(), TypeId::FLOAT64)
        .call()
        .unwrap();
    let disc_price = filtered
        .column(4)
        .unwrap()
        .mul(&one_minus_disc.view(), TypeId::FLOAT64)
        .call()
        .unwrap();

    // Build working table: returnflag(0), linestatus(1), quantity(2), extendedprice(3), disc_price(4)
    let rf = filtered.column(7).unwrap().to_owned_column().unwrap();
    let ls = filtered.column(8).unwrap().to_owned_column().unwrap();
    let qty = filtered.column(3).unwrap().to_owned_column().unwrap();
    let ep = filtered.column(4).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![rf, ls, qty, ep, disc_price]);

    // GROUP BY returnflag, linestatus → SUM(quantity), SUM(extendedprice), SUM(disc_price), COUNT(*)
    let result = work
        .groupby_multi(
            &[0, 1],
            &[2, 3, 4, 2],
            &[
                AggregationKind::SUM,
                AggregationKind::SUM,
                AggregationKind::SUM,
                AggregationKind::COUNT,
            ],
        )
        .call()
        .unwrap();

    // ORDER BY returnflag, linestatus
    let result = sorted_by(
        &result,
        &[Order::ASCENDING, Order::ASCENDING],
        &[NullOrder::BEFORE, NullOrder::BEFORE],
    );

    let flags = table_col_strings(&result, 0);
    let statuses = table_col_strings(&result, 1);

    // Verify we have (A,F), (N,O), (R,F) groups
    assert_eq!(flags, ["A", "N", "R"]);
    assert_eq!(statuses, ["F", "O", "F"]);

    // Verify counts: A/F has 3 rows, N/O has 4 (row 6 shipdate=9410 filtered out), R/F has 2
    let counts = table_col_i64(&result, 5);
    assert_eq!(counts, [3, 4, 2]);

    // Verify sum_qty (A/F: 38+49+44=131, N/O: 17+36+27+15=95, R/F: 45+24=69)
    let sum_qty = table_col_f64(&result, 2);
    assert!((sum_qty[0] - 131.0).abs() < 1e-6);
    assert!((sum_qty[1] - 95.0).abs() < 1e-6);
    assert!((sum_qty[2] - 69.0).abs() < 1e-6);
}

// ===========================================================================
// TPC-H Q3: Shipping Priority (simplified)
//
//   SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue
//   FROM customer, orders, lineitem
//   WHERE c_mktsegment = 'BUILDING'
//     AND c_custkey = o_custkey
//     AND l_orderkey = o_orderkey
//     AND o_orderdate < date '1995-03-15'
//   GROUP BY l_orderkey
//   ORDER BY revenue DESC
// ===========================================================================

#[test]
fn tpch_q3_shipping_priority() {
    let customer = make_customer();
    let orders = make_orders();
    let lineitem = make_lineitem();

    // Filter customer: c_mktsegment = 'BUILDING'
    let building = Column::from_strings(&["BUILDING"; 5]);
    let cust_mask = customer
        .column(4)
        .unwrap()
        .eq(&building.view())
        .call()
        .unwrap();
    let cust_filtered = customer.filter(&cust_mask.view()).call().unwrap();
    // cust_filtered has custkeys 370 and 1234

    // Filter orders: o_orderdate < 9300 (our threshold)
    let date_threshold = Column::from_scalar(&Scalar::from_i32(9300), orders.len());
    let ord_mask = orders
        .column(4)
        .unwrap()
        .lt(&date_threshold.view())
        .call()
        .unwrap();
    let ord_filtered = orders.filter(&ord_mask.view()).call().unwrap();
    // ord_filtered has orders with dates < 9300: order 2 (9100), order 3 (9150)

    // Join customer ⋈ orders on custkey
    // cust_filtered: col0=custkey, orders: col1=custkey
    let cust_ord = cust_filtered
        .inner_join(&ord_filtered, &[0], &[1])
        .call()
        .unwrap();

    // Join with lineitem on orderkey
    // cust_ord has o_orderkey somewhere — need to find it
    // cust_filtered cols: 0=custkey,1=name,2=nationkey,3=acctbal,4=mktsegment
    // ord_filtered cols: 0=orderkey,1=custkey,2=status,3=totalprice,4=orderdate,5=priority
    // After join: cust cols (5) + ord cols (6) = 11 cols, ord orderkey at col 5
    let co_li = cust_ord
        .inner_join(&lineitem, &[5], &[0])
        .call()
        .unwrap();

    // Verify we got some joined rows
    assert!(co_li.len() > 0);

    // Compute revenue = l_extendedprice * (1 - l_discount)
    // In co_li: cust(5) + ord(6) + lineitem(10) = 21 cols
    // lineitem cols start at offset 11: extendedprice=11+4=15, discount=11+5=16
    let ones = Column::from_scalar(&Scalar::from_f64(1.0), co_li.len());
    let one_minus_disc = ones
        .view()
        .sub(&co_li.column(16).unwrap(), TypeId::FLOAT64)
        .call()
        .unwrap();
    let revenue = co_li
        .column(15)
        .unwrap()
        .mul(&one_minus_disc.view(), TypeId::FLOAT64)
        .call()
        .unwrap();

    // Build (l_orderkey, revenue) and group by orderkey
    let orderkey = co_li.column(11).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![orderkey, revenue]);
    let grouped = work.groupby(&[0], 1, AggregationKind::SUM).call().unwrap();

    // Order by revenue descending
    let result = sorted_by(
        &grouped,
        &[Order::ASCENDING, Order::DESCENDING],
        &[NullOrder::BEFORE, NullOrder::BEFORE],
    );

    // We should have at least one group with non-zero revenue
    assert!(result.len() > 0);
    let revenues = table_col_f64(&result, 1);
    for r in &revenues {
        assert!(*r > 0.0);
    }
}

// ===========================================================================
// TPC-H Q5: Local Supplier Volume (simplified)
//
//   SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
//   FROM lineitem, supplier, nation
//   WHERE l_suppkey = s_suppkey
//     AND s_nationkey = n_nationkey
//   GROUP BY n_name
//   ORDER BY revenue DESC
// ===========================================================================

#[test]
fn tpch_q5_local_supplier_volume() {
    let lineitem = make_lineitem();
    let supplier = make_supplier();
    let nation = make_nation();

    // Join supplier ⋈ nation on nationkey
    // supplier: 0=suppkey, 1=name, 2=nationkey
    // nation: 0=nationkey, 1=name, 2=regionkey
    let supp_nat = supplier
        .inner_join(&nation, &[2], &[0])
        .call()
        .unwrap();
    // supp_nat: supplier(3) + nation(3) = 6 cols; suppkey=0, nation_name=4

    // Join lineitem ⋈ supp_nat on suppkey
    // lineitem: 2=suppkey
    let li_sn = lineitem
        .inner_join(&supp_nat, &[2], &[0])
        .call()
        .unwrap();
    // li_sn: lineitem(10) + supp_nat(6) = 16 cols
    // extendedprice=4, discount=5, nation_name=10+4=14

    // Compute revenue = extendedprice * (1 - discount)
    let ones = Column::from_scalar(&Scalar::from_f64(1.0), li_sn.len());
    let one_minus_disc = ones
        .view()
        .sub(&li_sn.column(5).unwrap(), TypeId::FLOAT64)
        .call()
        .unwrap();
    let revenue = li_sn
        .column(4)
        .unwrap()
        .mul(&one_minus_disc.view(), TypeId::FLOAT64)
        .call()
        .unwrap();

    // Build (nation_name, revenue) and group by nation_name
    let nation_name = li_sn.column(14).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![nation_name, revenue]);
    let grouped = work.groupby(&[0], 1, AggregationKind::SUM).call().unwrap();

    // Order by revenue DESC
    let result = sorted_by(
        &grouped,
        &[Order::ASCENDING, Order::DESCENDING],
        &[NullOrder::BEFORE, NullOrder::BEFORE],
    );

    // All 3 nations should appear
    assert_eq!(result.len(), 3);
    let names = table_col_strings(&result, 0);
    assert!(names.contains(&"FRANCE".to_string()));
    assert!(names.contains(&"GERMANY".to_string()));
    assert!(names.contains(&"BRAZIL".to_string()));

    // All revenues must be positive
    let revs = table_col_f64(&result, 1);
    for r in &revs {
        assert!(*r > 0.0);
    }
}

// ===========================================================================
// TPC-H Q6: Forecasting Revenue Change
//
//   SELECT SUM(l_extendedprice * l_discount) AS revenue
//   FROM lineitem
//   WHERE l_discount BETWEEN 0.05 AND 0.1
//     AND l_quantity < 50
// ===========================================================================

#[test]
fn tpch_q6_forecasting_revenue() {
    let lineitem = make_lineitem();

    // Filter: l_discount BETWEEN 0.05 AND 0.1
    let disc_lo = Column::from_scalar(&Scalar::from_f64(0.05), lineitem.len());
    let disc_hi = Column::from_scalar(&Scalar::from_f64(0.1), lineitem.len());
    let disc_ge = lineitem
        .column(5)
        .unwrap()
        .ge(&disc_lo.view())
        .call()
        .unwrap();
    let disc_le = lineitem
        .column(5)
        .unwrap()
        .le(&disc_hi.view())
        .call()
        .unwrap();

    // AND l_quantity < 50
    let qty_threshold = Column::from_scalar(&Scalar::from_f64(50.0), lineitem.len());
    let qty_lt = lineitem
        .column(3)
        .unwrap()
        .lt(&qty_threshold.view())
        .call()
        .unwrap();

    // Combine masks: disc_ge AND disc_le AND qty_lt
    let mask1 = disc_ge
        .view()
        .binary_op(&disc_le.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();
    let mask = mask1
        .view()
        .binary_op(&qty_lt.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();

    let filtered = lineitem.filter(&mask.view()).call().unwrap();

    // Compute revenue = extendedprice * discount
    let revenue_col = filtered
        .column(4)
        .unwrap()
        .mul(&filtered.column(5).unwrap(), TypeId::FLOAT64)
        .call()
        .unwrap();

    // SUM(revenue)
    let total = revenue_col
        .view()
        .sum(TypeId::FLOAT64)
        .call()
        .unwrap();

    let total_val = total.as_f64().unwrap();
    assert!(total_val > 0.0);
    // Rows qualifying: discount in [0.05, 0.1] AND quantity < 50:
    // row 1: disc=0.09,qty=36 ✓  row 3: disc=0.06,qty=45 ✓  row 4: disc=0.1,qty=49 ✓
    // row 5: disc=0.05,qty=27 ✓  row 6: disc=0.09,qty=28 ✓  row 7: disc=0.1,qty=24 ✓
    // row 9: disc=0.07,qty=44 ✓  (row 0: disc=0.04 too low, row 2: disc=0.0, row 8: disc=0.02)
    assert_eq!(filtered.len(), 7);
}

// ===========================================================================
// TPC-H Q10: Returned Item Reporting (simplified)
//
//   SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
//   FROM customer, orders, lineitem
//   WHERE c_custkey = o_custkey
//     AND l_orderkey = o_orderkey
//     AND l_returnflag = 'R'
//   GROUP BY c_custkey, c_name
//   ORDER BY revenue DESC
// ===========================================================================

#[test]
fn tpch_q10_returned_item_reporting() {
    let customer = make_customer();
    let orders = make_orders();
    let lineitem = make_lineitem();

    // Filter lineitem: l_returnflag = 'R'
    let r_flag = Column::from_strings(&["R"; 10]);
    let li_mask = lineitem
        .column(7)
        .unwrap()
        .eq(&r_flag.view())
        .call()
        .unwrap();
    let li_filtered = lineitem.filter(&li_mask.view()).call().unwrap();
    // R rows: index 3 (orderkey=3) and 7 (orderkey=4)

    // Join lineitem ⋈ orders on orderkey
    let li_ord = li_filtered
        .inner_join(&orders, &[0], &[0])
        .call()
        .unwrap();
    // li(10) + ord(6) = 16 cols; o_custkey at 10+1=11

    // Join with customer on custkey
    let full = li_ord
        .inner_join(&customer, &[11], &[0])
        .call()
        .unwrap();
    // full: li_ord(16) + cust(5) = 21 cols; c_custkey=16, c_name=17

    // Compute revenue = extendedprice * (1 - discount)
    let ones = Column::from_scalar(&Scalar::from_f64(1.0), full.len());
    let one_minus_disc = ones
        .view()
        .sub(&full.column(5).unwrap(), TypeId::FLOAT64)
        .call()
        .unwrap();
    let revenue = full
        .column(4)
        .unwrap()
        .mul(&one_minus_disc.view(), TypeId::FLOAT64)
        .call()
        .unwrap();

    // Build (c_custkey, c_name, revenue) for groupby
    let cust_key = full.column(16).unwrap().to_owned_column().unwrap();
    let cust_name = full.column(17).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![cust_key, cust_name, revenue]);

    let grouped = work
        .groupby_multi(&[0, 1], &[2], &[AggregationKind::SUM])
        .call()
        .unwrap();

    // Should have at least one customer with returned items
    assert!(grouped.len() > 0);

    // All revenues should be positive
    let revs = table_col_f64(&grouped, 2);
    for r in &revs {
        assert!(*r > 0.0);
    }
}

// ===========================================================================
// TPC-H Q12: Shipping Modes (simplified)
//
//   Count orders by priority: high-priority vs low-priority per ship mode.
//   We simplify by counting orders per o_orderpriority.
//
//   SELECT o_orderpriority, COUNT(*) AS order_count
//   FROM orders, lineitem
//   WHERE o_orderkey = l_orderkey
//   GROUP BY o_orderpriority
//   ORDER BY o_orderpriority
// ===========================================================================

#[test]
fn tpch_q12_shipping_modes() {
    let orders = make_orders();
    let lineitem = make_lineitem();

    // Join orders ⋈ lineitem on orderkey
    let joined = orders
        .inner_join(&lineitem, &[0], &[0])
        .call()
        .unwrap();
    // orders(6) + lineitem(10) = 16 cols; o_orderpriority=5

    // Build (o_orderpriority, dummy_col_for_count)
    let priority = joined.column(5).unwrap().to_owned_column().unwrap();
    let dummy = joined.column(0).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![priority, dummy]);

    let grouped = work
        .groupby(&[0], 1, AggregationKind::COUNT)
        .call()
        .unwrap();

    let result = sorted_by(
        &grouped,
        &[Order::ASCENDING],
        &[NullOrder::BEFORE],
    );

    let priorities = table_col_strings(&result, 0);
    assert!(priorities.contains(&"1-URGENT".to_string()));
    assert!(priorities.contains(&"5-LOW".to_string()));

    // Total count across groups should equal lineitem rows (10)
    let counts = table_col_i64(&result, 1);
    let total: i64 = counts.iter().sum();
    assert_eq!(total, 10);
}

// ===========================================================================
// TPC-H Q4: Order Priority Checking (simplified)
//
//   SELECT o_orderpriority, COUNT(*) AS order_count
//   FROM orders
//   WHERE o_orderdate >= 9100 AND o_orderdate < 9300
//     AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey)
//   GROUP BY o_orderpriority
//   ORDER BY o_orderpriority
// ===========================================================================

#[test]
fn tpch_q4_order_priority() {
    let orders = make_orders();
    let lineitem = make_lineitem();

    // Filter orders by date range: [9100, 9300)
    let date_lo = Column::from_scalar(&Scalar::from_i32(9100), orders.len());
    let date_hi = Column::from_scalar(&Scalar::from_i32(9300), orders.len());
    let ge_mask = orders
        .column(4)
        .unwrap()
        .ge(&date_lo.view())
        .call()
        .unwrap();
    let lt_mask = orders
        .column(4)
        .unwrap()
        .lt(&date_hi.view())
        .call()
        .unwrap();
    let date_mask = ge_mask
        .view()
        .binary_op(&lt_mask.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();
    let ord_filtered = orders.filter(&date_mask.view()).call().unwrap();
    // Orders with dates in [9100, 9300): order 2 (9100), order 3 (9150)

    // Semi-join: orders that exist in lineitem (l_orderkey)
    let result = ord_filtered
        .left_semi_join(&lineitem, &[0], &[0])
        .call()
        .unwrap();
    // Both orders 2 and 3 have lineitem rows

    // Group by o_orderpriority and count
    let priority = result.column(5).unwrap().to_owned_column().unwrap();
    let dummy = result.column(0).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![priority, dummy]);
    let grouped = work
        .groupby(&[0], 1, AggregationKind::COUNT)
        .call()
        .unwrap();

    let sorted = sorted_by(&grouped, &[Order::ASCENDING], &[NullOrder::BEFORE]);
    let priorities = table_col_strings(&sorted, 0);
    // Orders 2 and 3 both have priority "5-LOW"
    assert_eq!(priorities, ["5-LOW"]);
    assert_eq!(table_col_i64(&sorted, 1), [2]);
}

// ===========================================================================
// TPC-H Q14: Promotion Effect (simplified)
//
//   SELECT 100 * SUM(CASE WHEN p_type LIKE 'PROMO%'
//                         THEN l_extendedprice * (1 - l_discount)
//                         ELSE 0 END)
//        / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
//   FROM lineitem
//   (simplified: compute total revenue and count promo-like rows)
// ===========================================================================

#[test]
fn tpch_q14_promotion_effect() {
    let lineitem = make_lineitem();

    // Compute disc_price = l_extendedprice * (1 - l_discount)
    let ones = Column::from_scalar(&Scalar::from_f64(1.0), lineitem.len());
    let one_minus_disc = ones
        .view()
        .sub(&lineitem.column(5).unwrap(), TypeId::FLOAT64)
        .call()
        .unwrap();
    let disc_price = lineitem
        .column(4)
        .unwrap()
        .mul(&one_minus_disc.view(), TypeId::FLOAT64)
        .call()
        .unwrap();

    // Total revenue
    let total_revenue = disc_price.view().sum(TypeId::FLOAT64).call().unwrap();
    assert!(total_revenue.as_f64().unwrap() > 0.0);

    // Count and sum (testing the pipeline, not specific promo types)
    let count = disc_price.view().sum(TypeId::FLOAT64).call().unwrap();
    assert!(count.as_f64().unwrap() > 0.0);
}

// ===========================================================================
// TPC-H Q19: Discounted Revenue (simplified)
//
//   Multi-predicate filter + aggregation pipeline testing
//   complex boolean logic with AND/OR.
// ===========================================================================

#[test]
fn tpch_q19_multi_predicate_filter() {
    let lineitem = make_lineitem();

    // Condition 1: quantity >= 10 AND quantity <= 20
    let qty = lineitem.column(3).unwrap();
    let lo = Column::from_scalar(&Scalar::from_f64(10.0), lineitem.len());
    let hi = Column::from_scalar(&Scalar::from_f64(20.0), lineitem.len());
    let ge_10 = qty.ge(&lo.view()).call().unwrap();
    let le_20 = qty.le(&hi.view()).call().unwrap();
    let cond1 = ge_10
        .view()
        .binary_op(&le_20.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();

    // Condition 2: quantity >= 30 AND quantity <= 50
    let lo2 = Column::from_scalar(&Scalar::from_f64(30.0), lineitem.len());
    let hi2 = Column::from_scalar(&Scalar::from_f64(50.0), lineitem.len());
    let ge_30 = qty.ge(&lo2.view()).call().unwrap();
    let le_50 = qty.le(&hi2.view()).call().unwrap();
    let cond2 = ge_30
        .view()
        .binary_op(&le_50.view(), BinaryOperator::BITWISE_AND, TypeId::BOOL8)
        .call()
        .unwrap();

    // Combined: cond1 OR cond2
    let combined = cond1
        .view()
        .binary_op(&cond2.view(), BinaryOperator::BITWISE_OR, TypeId::BOOL8)
        .call()
        .unwrap();

    let filtered = lineitem.filter(&combined.view()).call().unwrap();

    // Rows: qty=17 (cond1), qty=36 (cond2), qty=38 (cond2), qty=45 (cond2),
    //        qty=49 (cond2), qty=15 (cond1), qty=44 (cond2)
    // Doesn't match: qty=27 (neither), qty=28 (neither), qty=24 (neither)
    assert_eq!(filtered.len(), 7);

    // Sum of filtered extendedprice
    let sum = filtered
        .column(4)
        .unwrap()
        .sum(TypeId::FLOAT64)
        .call()
        .unwrap();
    assert!(sum.as_f64().unwrap() > 0.0);
}

// ===========================================================================
// Test: Multi-table join chain (customer → orders → lineitem → supplier)
// ===========================================================================

#[test]
fn tpch_four_way_join() {
    let customer = make_customer();
    let orders = make_orders();
    let lineitem = make_lineitem();
    let supplier = make_supplier();

    // customer ⋈ orders on custkey
    let co = customer
        .inner_join(&orders, &[0], &[1])
        .call()
        .unwrap();

    // (customer+orders) ⋈ lineitem on orderkey
    // co: customer(5) + orders(6) = 11, orderkey at col 5
    let col = co
        .inner_join(&lineitem, &[5], &[0])
        .call()
        .unwrap();

    // (co+lineitem) ⋈ supplier on suppkey
    // col: co(11) + lineitem(10) = 21, suppkey at 11+2=13
    let full = col
        .inner_join(&supplier, &[13], &[0])
        .call()
        .unwrap();

    // All lineitem rows should have matching customer, order, and supplier
    assert_eq!(full.len(), lineitem.len());
    assert_eq!(full.columns_len(), 21 + 3); // 24 columns total
}

// ===========================================================================
// Test: Anti-join — find orders with no lineitem
// ===========================================================================

#[test]
fn tpch_anti_join_missing_lineitem() {
    let orders = make_orders();

    // Create a lineitem with missing orderkeys
    let partial_li_keys = Column::from_slice_i32(&[1, 2, 3]);
    let partial_li = build_table(vec![partial_li_keys]);

    // Anti-join: orders without lineitem
    let missing = orders
        .left_anti_join(&partial_li, &[0], &[0])
        .call()
        .unwrap();

    // Orders 4 and 5 are not in partial_li
    assert_eq!(missing.len(), 2);
    let keys = table_col_i32(&missing, 0);
    assert!(keys.contains(&4));
    assert!(keys.contains(&5));
}

// ===========================================================================
// Test: String filter + aggregation (like Q13 customer distribution)
// ===========================================================================

#[test]
fn tpch_string_filter_aggregate() {
    let customer = make_customer();

    // Filter customers whose name contains "000001"
    let result = customer
        .column(1)
        .unwrap()
        .str_contains_literal("000001")
        .call()
        .unwrap();
    let filtered = customer.filter(&result.view()).call().unwrap();

    // Customers 1234 and 1369 match
    assert_eq!(filtered.len(), 2);

    // Group by market segment and count
    let seg = filtered.column(4).unwrap().to_owned_column().unwrap();
    let key = filtered.column(0).unwrap().to_owned_column().unwrap();
    let work = build_table(vec![seg, key]);
    let grouped = work
        .groupby(&[0], 1, AggregationKind::COUNT)
        .call()
        .unwrap();

    // Should have BUILDING and AUTOMOBILE segments
    assert_eq!(grouped.len(), 2);
}
