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

        /// Compute the cross join (Cartesian product) of two tables.
        fn cross_join(left: &Table, right: &Table, stream: usize) -> Result<UniquePtr<Table>>;
    }
}
