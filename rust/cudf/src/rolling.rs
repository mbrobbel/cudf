// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Rolling window aggregation operations.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use cudf_sys::ffi::AggregationKind;
use crate::stream::Stream;
use crate::table::Table;

impl ColumnView<'_> {
    /// Applies a fixed-size rolling window aggregation.
    ///
    /// `preceding` and `following` define the window around each element.
    /// Total window size = preceding + following.
    /// Element `i` uses elements `[i - preceding + 1, i + following]`.
    pub fn rolling_window(
        &self,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> Result<Column> {
        self.rolling_window_on(preceding, following, min_periods, agg_kind, Stream::default_stream())
    }

    /// rolling_window on a custom CUDA stream.
    pub fn rolling_window_on(
        &self,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
        stream: Stream,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::rolling_window(
            self.0,
            preceding,
            following,
            min_periods,
            agg_kind.repr,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Applies a grouped fixed-size rolling window aggregation.
    ///
    /// Elements are grouped by `group_keys` (must be pre-sorted).
    /// The window does not cross group boundaries.
    pub fn grouped_rolling_window(
        &self,
        group_keys: &Table,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> Result<Column> {
        self.grouped_rolling_window_on(
            group_keys, preceding, following, min_periods, agg_kind, Stream::default_stream(),
        )
    }

    /// grouped_rolling_window on a custom CUDA stream.
    pub fn grouped_rolling_window_on(
        &self,
        group_keys: &Table,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
        stream: Stream,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::grouped_rolling_window(
            &group_keys.0,
            self.0,
            preceding,
            following,
            min_periods,
            agg_kind.repr,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}
