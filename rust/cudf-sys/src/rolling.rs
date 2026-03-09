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

        /// Opaque range window bounds for grouped range rolling windows.
        type RangeWindowBounds;

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

        // -- Range window bounds --

        /// Creates bounded range window bounds from a scalar value.
        fn range_window_bounds_get(
            boundary: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<RangeWindowBounds>>;

        /// Creates range window bounds matching the current row.
        fn range_window_bounds_current_row(
            type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<RangeWindowBounds>>;

        /// Creates unbounded range window bounds.
        fn range_window_bounds_unbounded(
            type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<RangeWindowBounds>>;

        /// Grouped range-based rolling window aggregation.
        fn grouped_range_rolling_window(
            group_keys: &Table,
            orderby: &column_view,
            order: i32,
            input: &column_view,
            preceding: &RangeWindowBounds,
            following: &RangeWindowBounds,
            min_periods: i32,
            agg_kind: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
