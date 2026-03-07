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

        // -- Rolling window --

        /// Fixed-size rolling window aggregation.
        fn rolling_window(
            col: &column_view,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Grouped fixed-size rolling window aggregation.
        fn grouped_rolling_window(
            group_keys: &Table,
            col: &column_view,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Rolling window with defaults --

        /// Rolling window aggregation with default output values (for LEAD/LAG).
        fn rolling_window_with_defaults(
            col: &column_view,
            default_outputs: &column_view,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Grouped rolling window with defaults --

        /// Grouped rolling window with default output values (for LEAD/LAG).
        fn grouped_rolling_window_with_defaults(
            group_keys: &Table,
            col: &column_view,
            default_outputs: &column_view,
            preceding: i32,
            following: i32,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
