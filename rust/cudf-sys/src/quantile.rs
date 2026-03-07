// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Quantile operations --

        /// Computes quantiles of a column.
        fn quantile_column(
            col: &column_view,
            quantiles: &[f64],
            interp: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Quantiles (new) --

        /// Table-level quantile rows.
        fn quantiles_table(
            tbl: &Table,
            quantiles: &[f64],
            interp: i32,
            is_input_sorted: bool,
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Percentile approx --

        /// Computes approximate percentiles from a t-digest column.
        fn percentile_approx(
            tdigest_col: &column_view,
            percentiles: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
