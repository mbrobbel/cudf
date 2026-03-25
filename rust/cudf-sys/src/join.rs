// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Table = crate::ffi::Table;

        // -- Join operations --

        /// Inner join: returns a table with all columns from left and right
        /// for matching rows.
        fn inner_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left join: returns all rows from left, with matching right rows
        /// (nulls for unmatched).
        fn left_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Full outer join: returns all rows from both sides.
        fn full_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left semi join: returns rows from left that have matches in right.
        fn left_semi_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left anti join: returns rows from left that have NO matches in right.
        fn left_anti_join(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Index-only join variants --

        /// Inner join returning raw gather map indices (2-column table:
        /// left indices, right indices) without materializing gathered rows.
        fn inner_join_indices(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left join returning raw gather map indices (2-column table:
        /// left indices, right indices) without materializing gathered rows.
        fn left_join_indices(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Full outer join returning raw gather map indices (2-column table:
        /// left indices, right indices) without materializing gathered rows.
        fn full_join_indices(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left semi join returning raw gather map indices (single-column
        /// table: left indices) without materializing gathered rows.
        fn left_semi_join_indices(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Left anti join returning raw gather map indices (single-column
        /// table: left indices) without materializing gathered rows.
        fn left_anti_join_indices(
            left: &Table,
            right: &Table,
            left_on: &[i32],
            right_on: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Compute the cross join (Cartesian product) of two tables.
        fn cross_join(left: &Table, right: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- MarkJoin --

        /// Stateful hash join that builds a hash table once for repeated probes.
        type MarkJoin;

        /// Creates a new MarkJoin from a build table's key columns.
        fn mark_join_new(
            tbl: &Table,
            keys: &[i32],
            compare_nulls_equal: bool,
            stream: usize,
        ) -> Result<UniquePtr<MarkJoin>>;

        /// Semi join probe: returns build rows that have matches in probe.
        fn mark_join_semi(
            joiner: &MarkJoin,
            build: &Table,
            probe: &Table,
            probe_keys: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Anti join probe: returns build rows that have NO matches in probe.
        fn mark_join_anti(
            joiner: &MarkJoin,
            build: &Table,
            probe: &Table,
            probe_keys: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
