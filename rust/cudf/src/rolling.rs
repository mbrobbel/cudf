// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Rolling window aggregation operations.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::stream::Stream;
use crate::table::Table;
use cudf_sys::ffi::AggregationKind;

/// Builder for a fixed-size rolling window aggregation.
///
/// Created by [`ColumnView::rolling_window`].
/// Call [`.call()`](RollingWindow::call) to execute.
pub struct RollingWindow<'a> {
    view: &'a ColumnView<'a>,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl RollingWindow<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the rolling window aggregation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::rolling::ffi::rolling_window(
            self.view.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for a grouped fixed-size rolling window aggregation.
///
/// Created by [`ColumnView::grouped_rolling_window`].
/// Call [`.call()`](GroupedRollingWindow::call) to execute.
pub struct GroupedRollingWindow<'a> {
    view: &'a ColumnView<'a>,
    group_keys: &'a Table,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl GroupedRollingWindow<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the grouped rolling window aggregation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::rolling::ffi::grouped_rolling_window(
            &self.group_keys.0,
            self.view.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for a rolling window with default output values.
///
/// Created by [`ColumnView::rolling_window_with_defaults`].
/// Call [`.call()`](RollingWindowWithDefaults::call) to execute.
pub struct RollingWindowWithDefaults<'a> {
    view: &'a ColumnView<'a>,
    default_outputs: &'a ColumnView<'a>,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl RollingWindowWithDefaults<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the rolling window aggregation with defaults.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::rolling::ffi::rolling_window_with_defaults(
            self.view.0,
            self.default_outputs.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for a grouped rolling window with default output values.
///
/// Created by [`ColumnView::grouped_rolling_window_with_defaults`].
/// Call [`.call()`](GroupedRollingWindowWithDefaults::call) to execute.
pub struct GroupedRollingWindowWithDefaults<'a> {
    view: &'a ColumnView<'a>,
    group_keys: &'a Table,
    default_outputs: &'a ColumnView<'a>,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl GroupedRollingWindowWithDefaults<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the grouped rolling window aggregation with defaults.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::rolling::ffi::grouped_rolling_window_with_defaults(
            &self.group_keys.0,
            self.view.0,
            self.default_outputs.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

impl<'a> ColumnView<'a> {
    /// Applies a fixed-size rolling window aggregation.
    ///
    /// `preceding` and `following` define the window around each element.
    /// Total window size = preceding + following.
    /// Element `i` uses elements `[i - preceding + 1, i + following]`.
    ///
    /// Returns a [`RollingWindow`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    pub fn rolling_window(
        &'a self,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> RollingWindow<'a> {
        RollingWindow {
            view: self,
            preceding,
            following,
            min_periods,
            agg_kind,
            stream: Stream::default_stream(),
        }
    }

    /// Applies a grouped fixed-size rolling window aggregation.
    ///
    /// Elements are grouped by `group_keys` (must be pre-sorted).
    /// The window does not cross group boundaries.
    ///
    /// Returns a [`GroupedRollingWindow`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn grouped_rolling_window(
        &'a self,
        group_keys: &'a Table,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> GroupedRollingWindow<'a> {
        GroupedRollingWindow {
            view: self,
            group_keys,
            preceding,
            following,
            min_periods,
            agg_kind,
            stream: Stream::default_stream(),
        }
    }

    /// Rolling window with default output values (for LEAD/LAG aggregations).
    ///
    /// When the window extends beyond column boundaries, values from
    /// `default_outputs` are used instead of null.
    ///
    /// Returns a [`RollingWindowWithDefaults`] builder. Use `.stream()` to set
    /// a custom CUDA stream, then `.call()` to execute.
    pub fn rolling_window_with_defaults(
        &'a self,
        default_outputs: &'a ColumnView<'a>,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> RollingWindowWithDefaults<'a> {
        RollingWindowWithDefaults {
            view: self,
            default_outputs,
            preceding,
            following,
            min_periods,
            agg_kind,
            stream: Stream::default_stream(),
        }
    }

    /// Grouped rolling window with default output values (for LEAD/LAG).
    ///
    /// Returns a [`GroupedRollingWindowWithDefaults`] builder. Use `.stream()`
    /// to set a custom CUDA stream, then `.call()` to execute.
    pub fn grouped_rolling_window_with_defaults(
        &'a self,
        group_keys: &'a Table,
        default_outputs: &'a ColumnView<'a>,
        preceding: i32,
        following: i32,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> GroupedRollingWindowWithDefaults<'a> {
        GroupedRollingWindowWithDefaults {
            view: self,
            group_keys,
            default_outputs,
            preceding,
            following,
            min_periods,
            agg_kind,
            stream: Stream::default_stream(),
        }
    }
}
