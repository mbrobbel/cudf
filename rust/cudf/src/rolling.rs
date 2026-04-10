// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Rolling window aggregation operations.
//!
//! Rolling windows compute an aggregate over a sliding window of rows around
//! each element. Available as methods on
//! [`ColumnView`]:
//!
//! * [`ColumnView::rolling_window`](crate::column::ColumnView::rolling_window) --
//!   fixed-size window on the entire column.
//! * [`ColumnView::grouped_rolling_window`](crate::column::ColumnView::grouped_rolling_window) --
//!   fixed-size window that resets at group boundaries.
//! * [`ColumnView::rolling_window_with_defaults`](crate::column::ColumnView::rolling_window_with_defaults) --
//!   fixed-size window with fallback default values (for LEAD/LAG).
//! * [`ColumnView::grouped_rolling_window_with_defaults`](crate::column::ColumnView::grouped_rolling_window_with_defaults) --
//!   grouped variant with default values.
//! * [`ColumnView::grouped_range_rolling_window`](crate::column::ColumnView::grouped_range_rolling_window) --
//!   window defined by a value range rather than a fixed row count.

use cxx::UniquePtr;

use crate::column::ColumnView;
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::sorting::Order;
use crate::stream::Stream;
use crate::table::UnboundTable;
use cudf_sys::ffi::AggregationKind;

/// Builder for a fixed-size rolling window aggregation.
///
/// Created by [`ColumnView::rolling_window`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
///
/// For element *i* the window covers rows
/// `[i - preceding + 1, i + following]` (inclusive). If the window contains
/// fewer than `min_periods` non-null values the output element is null.
pub struct RollingWindow<'a> {
    view: &'a ColumnView<'a>,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl crate::stream::GpuOp for RollingWindow<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::rolling::ffi::rolling_window(
            self.view.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for a grouped fixed-size rolling window aggregation.
///
/// Created by [`ColumnView::grouped_rolling_window`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
///
/// The window does not cross group boundaries defined by `group_keys`.
/// The input data must be pre-sorted by the group key columns.
pub struct GroupedRollingWindow<'a> {
    view: &'a ColumnView<'a>,
    group_keys: &'a UnboundTable,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl crate::stream::GpuOp for GroupedRollingWindow<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::rolling::ffi::grouped_rolling_window(
            &self.group_keys.0,
            self.view.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for a rolling window with default output values.
///
/// Created by [`ColumnView::rolling_window_with_defaults`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
///
/// When the window extends beyond column boundaries (e.g. for `LEAD` or `LAG`
/// aggregations), values from `default_outputs` are used instead of null.
/// `default_outputs` must have the same length as the input column.
pub struct RollingWindowWithDefaults<'a> {
    view: &'a ColumnView<'a>,
    default_outputs: &'a ColumnView<'a>,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl crate::stream::GpuOp for RollingWindowWithDefaults<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::rolling::ffi::rolling_window_with_defaults(
            self.view.0,
            self.default_outputs.0,
            self.preceding,
            self.following,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for a grouped rolling window with default output values.
///
/// Created by [`ColumnView::grouped_rolling_window_with_defaults`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
///
/// Combines grouped semantics (window resets at group boundaries) with default
/// output values for out-of-bounds positions.
pub struct GroupedRollingWindowWithDefaults<'a> {
    view: &'a ColumnView<'a>,
    group_keys: &'a UnboundTable,
    default_outputs: &'a ColumnView<'a>,
    preceding: i32,
    following: i32,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl crate::stream::GpuOp for GroupedRollingWindowWithDefaults<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
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
        Ok(crate::column::RawColumn(c))
    }
}

impl<'a> ColumnView<'a> {
    /// Applies a fixed-size rolling window aggregation.
    ///
    /// For element *i*, the window covers rows
    /// `[i - preceding + 1, i + following]` (inclusive on both ends).
    /// The total window size is `preceding + following`.
    ///
    /// # Arguments
    ///
    /// * `preceding` -- Number of rows before the current row (inclusive of
    ///   the current row) in the window. Must be >= 1 to include the current
    ///   row itself.
    /// * `following` -- Number of rows after the current row in the window.
    ///   Use 0 for a trailing window.
    /// * `min_periods` -- Minimum number of non-null values in the window
    ///   required to produce a non-null output. If fewer non-null values are
    ///   present the output element is null.
    /// * `agg_kind` -- The aggregation to apply (e.g. `SUM`, `MEAN`, `MIN`,
    ///   `MAX`, `COUNT_VALID`, `COUNT_ALL`, `LEAD`, `LAG`, etc.).
    ///
    /// # Returns
    ///
    /// A [`Column`] of the same length as `self` containing the aggregated
    /// values. The output type depends on the aggregation kind.
    ///
    /// # Errors
    ///
    /// Returns an error if the aggregation kind is incompatible with the
    /// column type, or if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    /// use cudf_sys::ffi::AggregationKind;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// // 3-element trailing window sum
    /// let result = col.view().rolling_window(3, 0, 1, AggregationKind::SUM).call()?;
    /// assert_eq!(result.len(), 5);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
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
    /// Elements are grouped by `group_keys`. The window does not cross group
    /// boundaries -- it resets at each group start. The input data **must**
    /// be pre-sorted by the group key columns.
    ///
    /// # Arguments
    ///
    /// * `group_keys` -- A [`Table`] whose columns define the groups.
    ///   Must have the same number of rows as `self` and be sorted by the
    ///   group key columns.
    /// * `preceding` -- Number of preceding rows in the window (see
    ///   [`rolling_window`](ColumnView::rolling_window) for semantics).
    /// * `following` -- Number of following rows in the window.
    /// * `min_periods` -- Minimum non-null values required for a non-null
    ///   output.
    /// * `agg_kind` -- The aggregation to apply.
    ///
    /// # Returns
    ///
    /// A [`Column`] of the same length as `self` with per-group aggregated
    /// values.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// Returns a [`GroupedRollingWindow`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn grouped_rolling_window(
        &'a self,
        group_keys: &'a UnboundTable,
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
    /// `default_outputs` are used instead of null. This is primarily useful
    /// for `LEAD` and `LAG` aggregations where you want a specific fill
    /// value at the edges.
    ///
    /// # Arguments
    ///
    /// * `default_outputs` -- Column of default values to use when the window
    ///   extends past the column boundaries. Must have the same length as
    ///   `self` and a compatible type.
    /// * `preceding` -- Number of preceding rows in the window.
    /// * `following` -- Number of following rows in the window.
    /// * `min_periods` -- Minimum non-null values required.
    /// * `agg_kind` -- The aggregation to apply.
    ///
    /// # Returns
    ///
    /// A [`Column`] of the same length as `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
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
    /// Combines grouped semantics (window resets at group boundaries) with
    /// default output values for out-of-bounds positions. The input must be
    /// pre-sorted by the group key columns.
    ///
    /// # Arguments
    ///
    /// * `group_keys` -- A [`Table`] defining the groups (sorted).
    /// * `default_outputs` -- Column of default values. Must have the same
    ///   length as `self`.
    /// * `preceding`, `following`, `min_periods`, `agg_kind` -- See
    ///   [`rolling_window`](ColumnView::rolling_window).
    ///
    /// # Returns
    ///
    /// A [`Column`] of the same length as `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// Returns a [`GroupedRollingWindowWithDefaults`] builder. Use `.stream()`
    /// to set a custom CUDA stream, then `.call()` to execute.
    pub fn grouped_rolling_window_with_defaults(
        &'a self,
        group_keys: &'a UnboundTable,
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

    /// Grouped range-based rolling window aggregation.
    ///
    /// Instead of a fixed number of rows, the window is defined by a **value
    /// range** on the `orderby` column. For each row, all rows within the
    /// group whose `orderby` value falls within
    /// `[current - preceding, current + following]` are included. The input
    /// must be pre-sorted by both `group_keys` and `orderby`.
    ///
    /// # Arguments
    ///
    /// * `group_keys` -- A [`Table`] defining the groups (sorted).
    /// * `orderby` -- The column whose values define the range window.
    ///   Must be a numeric or timestamp type, sorted within each group.
    /// * `order` -- Sort order of `orderby` (`ASCENDING` or `DESCENDING`).
    /// * `preceding` -- [`RangeWindowBounds`] for the preceding boundary.
    /// * `following` -- [`RangeWindowBounds`] for the following boundary.
    /// * `min_periods` -- Minimum non-null values in the window required
    ///   for a non-null output.
    /// * `agg_kind` -- The aggregation to apply.
    ///
    /// # Returns
    ///
    /// A [`Column`] of the same length as `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// Returns a [`GroupedRangeRollingWindow`] builder. Use `.stream()` to set
    /// a custom CUDA stream, then `.call()` to execute.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_range_rolling_window(
        &'a self,
        group_keys: &'a UnboundTable,
        orderby: &'a ColumnView<'a>,
        order: Order,
        preceding: &'a RangeWindowBounds,
        following: &'a RangeWindowBounds,
        min_periods: i32,
        agg_kind: AggregationKind,
    ) -> GroupedRangeRollingWindow<'a> {
        GroupedRangeRollingWindow {
            view: self,
            group_keys,
            orderby,
            order,
            preceding,
            following,
            min_periods,
            agg_kind,
            stream: Stream::default_stream(),
        }
    }
}

// ---------------------------------------------------------------------------
// Range window bounds
// ---------------------------------------------------------------------------

/// Defines the bounds for a range-based rolling window.
///
/// Range window bounds specify how far the window extends based on the
/// **value** of an order-by column, rather than a fixed number of rows.
///
/// Created via the factory methods [`bounded`](RangeWindowBounds::bounded),
/// [`current_row`](RangeWindowBounds::current_row), or
/// [`unbounded`](RangeWindowBounds::unbounded).
///
/// # Examples
///
/// ```no_run
/// use cudf::rolling::RangeWindowBounds;
/// use cudf::scalar::Scalar;
/// use cudf::data_type::TypeId;
///
/// // Window extends 5 units before/after the current row's orderby value
/// let preceding = RangeWindowBounds::bounded(&Scalar::from_i32(5))?;
/// let following = RangeWindowBounds::bounded(&Scalar::from_i32(5))?;
///
/// // Window extends to the beginning/end of the group
/// let unbounded = RangeWindowBounds::unbounded(TypeId::INT32)?;
/// # let _ = (preceding, following, unbounded);
/// # Ok::<(), cudf::error::Error>(())
/// ```
#[doc(alias = "range_window_bounds")]
pub struct RangeWindowBounds(UniquePtr<cudf_sys::rolling::ffi::RangeWindowBounds>);

impl RangeWindowBounds {
    /// Creates bounded range window bounds from a scalar value.
    ///
    /// The scalar `boundary` specifies how far the window extends from the
    /// current row's order-by value. The scalar type must match the order-by
    /// column's type.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn bounded(boundary: &Scalar) -> Result<Self> {
        let ffi = crate::scalar::scalar_to_ffi(boundary)?;
        let b = cudf_sys::rolling::ffi::range_window_bounds_get(
            &ffi,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Self(b))
    }

    /// Creates range window bounds matching only the current row's value.
    ///
    /// The `type_id` must match the order-by column's type.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn current_row(type_id: TypeId) -> Result<Self> {
        let b = cudf_sys::rolling::ffi::range_window_bounds_current_row(
            type_id.repr,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Self(b))
    }

    /// Creates unbounded range window bounds that extend to the entire group.
    ///
    /// Effectively includes all rows in the group. The `type_id` must match
    /// the order-by column's type.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn unbounded(type_id: TypeId) -> Result<Self> {
        let b = cudf_sys::rolling::ffi::range_window_bounds_unbounded(
            type_id.repr,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Self(b))
    }
}

/// Builder for a grouped range-based rolling window aggregation.
///
/// Created by [`ColumnView::grouped_range_rolling_window`].
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
///
/// The window for each element is defined by a value range on the `orderby`
/// column rather than a fixed row count. The window does not cross group
/// boundaries.
pub struct GroupedRangeRollingWindow<'a> {
    view: &'a ColumnView<'a>,
    group_keys: &'a UnboundTable,
    orderby: &'a ColumnView<'a>,
    order: Order,
    preceding: &'a RangeWindowBounds,
    following: &'a RangeWindowBounds,
    min_periods: i32,
    agg_kind: AggregationKind,
    stream: Stream,
}

impl crate::stream::GpuOp for GroupedRangeRollingWindow<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::rolling::ffi::grouped_range_rolling_window(
            &self.group_keys.0,
            self.orderby.0,
            self.order.repr,
            self.view.0,
            &self.preceding.0,
            &self.following.0,
            self.min_periods,
            self.agg_kind.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}
