// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        type Scalar = crate::ffi::Scalar;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Search operations --

        /// Checks if a scalar value exists in a column.
        fn contains_scalar(haystack: &column_view, needle: &Scalar, stream: usize) -> Result<bool>;

        /// Checks which values from needles exist in haystack.
        fn contains_column(
            haystack: &column_view,
            needles: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds lower bound insertion points in a sorted table.
        fn lower_bound(
            haystack: &Table,
            needles: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds upper bound insertion points in a sorted table.
        fn upper_bound(
            haystack: &Table,
            needles: &Table,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
