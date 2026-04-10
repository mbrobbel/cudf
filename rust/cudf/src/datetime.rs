// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Datetime operations on timestamp columns.
//!
//! The [`DatetimeExt`] trait provides component extraction (year, month, day,
//! hour, etc.), calendar queries (leap year, days in month), and rounding
//! operations (ceil, floor, round) for timestamp columns.
//!
//! All methods operate on columns with a `TIMESTAMP_*` type
//! (`TIMESTAMP_DAYS`, `TIMESTAMP_SECONDS`,
//! `TIMESTAMP_MILLISECONDS`, `TIMESTAMP_MICROSECONDS`, or
//! `TIMESTAMP_NANOSECONDS`).
//!
//! # Examples
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::datetime::DatetimeExt;
//! use cudf::stream::GpuOp;
//!
//! let ts = Column::from_timestamps_s(&[1718443845]).call()?;
//! let years = ts.view().extract_year().call()?;
//! let months = ts.view().extract_month().call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

use crate::column::ColumnView;
use crate::error::Result;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

#[path = "datetime/ext_impl.rs"]
mod ext_impl;

/// Extension trait for datetime operations on timestamp columns.
///
/// Provides component extraction, calendar queries, rounding, and month
/// arithmetic for columns with a `TIMESTAMP_*` type. This trait is implemented
/// for [`ColumnView`] and is sealed -- it cannot be implemented outside this
/// crate.
///
/// All methods return builder structs that implement [`GpuOp`](crate::stream::GpuOp).
/// Call `.call()` to execute the operation, or chain `.stream(s)` first to run
/// on a non-default CUDA stream.
///
/// # Errors
///
/// Methods return an error if the input column does not have a timestamp type.
pub trait DatetimeExt: private::Sealed {
    /// Extracts the year component from each timestamp.
    ///
    /// Returns an `INT16` column containing the year (e.g. `2024`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let years = ts.view().extract_year().call()?;
    /// // years contains [2024]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_year(&self) -> ExtractYear<'_>;

    /// Extracts the month component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `1..=12`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let months = ts.view().extract_month().call()?;
    /// // months contains values 1-12
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_month(&self) -> ExtractMonth<'_>;

    /// Extracts the day-of-month component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `1..=31`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let days = ts.view().extract_day().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_day(&self) -> ExtractDay<'_>;

    /// Extracts the weekday component from each timestamp.
    ///
    /// Returns an `INT16` column where Monday is `0` and Sunday is `6`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let weekdays = ts.view().extract_weekday().call()?;
    /// // 0=Monday, 1=Tuesday, ..., 6=Sunday
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_weekday(&self) -> ExtractWeekday<'_>;

    /// Extracts the hour component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `0..=23`.
    /// For `TIMESTAMP_DAYS` columns the result is always `0`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let hours = ts.view().extract_hour().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_hour(&self) -> ExtractHour<'_>;

    /// Extracts the minute component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `0..=59`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let minutes = ts.view().extract_minute().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_minute(&self) -> ExtractMinute<'_>;

    /// Extracts the second component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `0..=59`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let seconds = ts.view().extract_second().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_second(&self) -> ExtractSecond<'_>;

    /// Returns the 1-based day of the year for each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `1..=366`.
    /// January 1 is day `1`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let doy = ts.view().day_of_year().call()?;
    /// // Jan 1 => 1, Feb 1 => 32, etc.
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn day_of_year(&self) -> DayOfYear<'_>;

    /// Tests whether each timestamp falls in a leap year.
    ///
    /// Returns a `BOOL8` column: `true` if the year is a leap year, `false`
    /// otherwise.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let leap = ts.view().is_leap_year().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn is_leap_year(&self) -> IsLeapYear<'_>;

    /// Returns the number of days in the month for each timestamp.
    ///
    /// Returns an `INT16` column (e.g. `28`, `29`, `30`, or `31`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let dim = ts.view().days_in_month().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn days_in_month(&self) -> DaysInMonth<'_>;

    /// Returns the last day of the month for each timestamp.
    ///
    /// Returns a `TIMESTAMP_DAYS` column where each value is set to the last
    /// calendar day of that row's month. For example, any date in June yields
    /// June 30.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let last = ts.view().last_day_of_month().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn last_day_of_month(&self) -> LastDayOfMonth<'_>;

    /// Extracts the quarter for each timestamp.
    ///
    /// Returns an `INT16` column with values `1` (Jan-Mar), `2` (Apr-Jun),
    /// `3` (Jul-Sep), or `4` (Oct-Dec).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let quarters = ts.view().extract_quarter().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_quarter(&self) -> ExtractQuarter<'_>;

    /// Rounds each timestamp up (ceiling) to the given frequency.
    ///
    /// The `freq` parameter specifies the unit boundary to ceil to (e.g.
    /// [`RoundingFrequency::Day`], [`RoundingFrequency::Hour`]). The output
    /// column has the same timestamp type as the input.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::{DatetimeExt, RoundingFrequency};
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let ceiled = ts.view().dt_ceil(RoundingFrequency::Hour).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dt_ceil(&self, freq: RoundingFrequency) -> DtCeil<'_>;

    /// Rounds each timestamp down (floor) to the given frequency.
    ///
    /// The `freq` parameter specifies the unit boundary to floor to (e.g.
    /// [`RoundingFrequency::Day`], [`RoundingFrequency::Second`]). The output
    /// column has the same timestamp type as the input.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::{DatetimeExt, RoundingFrequency};
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let floored = ts.view().dt_floor(RoundingFrequency::Minute).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dt_floor(&self, freq: RoundingFrequency) -> DtFloor<'_>;

    /// Rounds each timestamp to the nearest frequency boundary.
    ///
    /// The `freq` parameter specifies the unit to round to. Values exactly at
    /// the midpoint are rounded up. The output column has the same timestamp
    /// type as the input.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::{DatetimeExt, RoundingFrequency};
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let rounded = ts.view().dt_round(RoundingFrequency::Second).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dt_round(&self, freq: RoundingFrequency) -> DtRound<'_>;

    /// Adds months to each timestamp using per-row values from a column.
    ///
    /// The `months` column must be an integer type (`INT16` or `INT32`)
    /// with the same number of rows. Negative values subtract months. If the
    /// resulting day exceeds the target month's length, it is clamped to the
    /// last valid day (e.g. Jan 31 + 1 month = Feb 28/29).
    ///
    /// Returns a timestamp column of the same type as the input.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845, 1718443845, 1718443845]).call()?;
    /// let months = Column::from_slice_i32(&[1, -2, 3]).call()?;
    /// let shifted = ts.view().dt_add_months(&months.view()).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dt_add_months<'a>(&'a self, months: &'a ColumnView<'a>) -> DtAddMonths<'a>;

    /// Adds a scalar number of months to every timestamp in the column.
    ///
    /// The `months` scalar must be an integer type. Negative values subtract
    /// months. Day clamping applies as in [`dt_add_months`](DatetimeExt::dt_add_months).
    ///
    /// Returns a timestamp column of the same type as the input.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_s(&[1718443845]).call()?;
    /// let months = Scalar::from_i32(6);
    /// let shifted = ts.view().dt_add_months_scalar(&months).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dt_add_months_scalar<'a>(
        &'a self,
        months: &'a crate::scalar::Scalar,
    ) -> DtAddMonthsScalar<'a>;

    /// Extracts the millisecond fraction component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `0..=999`.
    /// This is the sub-second millisecond part, not the total milliseconds
    /// since epoch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_ms(&[1718443845123]).call()?;
    /// let ms = ts.view().extract_millisecond().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_millisecond(&self) -> ExtractMillisecond<'_>;

    /// Extracts the microsecond fraction component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `0..=999`.
    /// This is the sub-millisecond microsecond part.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_us(&[1718443845123456]).call()?;
    /// let us = ts.view().extract_microsecond().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_microsecond(&self) -> ExtractMicrosecond<'_>;

    /// Extracts the nanosecond fraction component from each timestamp.
    ///
    /// Returns an `INT16` column with values in the range `0..=999`.
    /// This is the sub-microsecond nanosecond part.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::datetime::DatetimeExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let ts = Column::from_timestamps_ns(&[1718443845123456789]).call()?;
    /// let ns = ts.view().extract_nanosecond().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn extract_nanosecond(&self) -> ExtractNanosecond<'_>;
}

#[doc(alias = "rounding_frequency")]
/// Specifies the time unit boundary for datetime rounding operations.
///
/// Used with [`DatetimeExt::dt_ceil`], [`DatetimeExt::dt_floor`], and
/// [`DatetimeExt::dt_round`] to control the granularity of rounding.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingFrequency {
    /// Round to day boundary.
    Day = 0,
    /// Round to hour boundary.
    Hour = 1,
    /// Round to minute boundary.
    Minute = 2,
    /// Round to second boundary.
    Second = 3,
    /// Round to millisecond boundary.
    Millisecond = 4,
    /// Round to microsecond boundary.
    Microsecond = 5,
    /// Round to nanosecond boundary.
    Nanosecond = 6,
}

#[allow(clippy::as_conversions)]
impl From<RoundingFrequency> for i32 {
    fn from(freq: RoundingFrequency) -> Self {
        freq as Self
    }
}

// ---------------------------------------------------------------------------
// Builder structs
// ---------------------------------------------------------------------------

/// Builder for [`DatetimeExt::extract_year`].
///
/// Created by [`DatetimeExt::extract_year`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractYear<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractYear<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_year(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_month`].
///
/// Created by [`DatetimeExt::extract_month`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractMonth<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractMonth<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_month(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_day`].
///
/// Created by [`DatetimeExt::extract_day`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractDay<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractDay<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_day(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_weekday`].
///
/// Created by [`DatetimeExt::extract_weekday`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractWeekday<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractWeekday<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_weekday(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_hour`].
///
/// Created by [`DatetimeExt::extract_hour`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractHour<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractHour<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_hour(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_minute`].
///
/// Created by [`DatetimeExt::extract_minute`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractMinute<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractMinute<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_minute(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_second`].
///
/// Created by [`DatetimeExt::extract_second`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractSecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractSecond<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_second(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::day_of_year`].
///
/// Created by [`DatetimeExt::day_of_year`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DayOfYear<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DayOfYear<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_day_of_year(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::is_leap_year`].
///
/// Created by [`DatetimeExt::is_leap_year`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct IsLeapYear<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for IsLeapYear<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_is_leap_year(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::days_in_month`].
///
/// Created by [`DatetimeExt::days_in_month`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DaysInMonth<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DaysInMonth<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_days_in_month(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::last_day_of_month`].
///
/// Created by [`DatetimeExt::last_day_of_month`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct LastDayOfMonth<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for LastDayOfMonth<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::datetime::ffi::datetime_last_day_of_month(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_quarter`].
///
/// Created by [`DatetimeExt::extract_quarter`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractQuarter<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractQuarter<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_quarter(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::dt_ceil`].
///
/// Created by [`DatetimeExt::dt_ceil`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DtCeil<'a> {
    view: &'a ColumnView<'a>,
    freq: RoundingFrequency,
    stream: Stream,
}

impl crate::stream::GpuOp for DtCeil<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_ceil(
            self.view.0,
            i32::from(self.freq),
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::dt_floor`].
///
/// Created by [`DatetimeExt::dt_floor`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DtFloor<'a> {
    view: &'a ColumnView<'a>,
    freq: RoundingFrequency,
    stream: Stream,
}

impl crate::stream::GpuOp for DtFloor<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_floor(
            self.view.0,
            i32::from(self.freq),
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::dt_round`].
///
/// Created by [`DatetimeExt::dt_round`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DtRound<'a> {
    view: &'a ColumnView<'a>,
    freq: RoundingFrequency,
    stream: Stream,
}

impl crate::stream::GpuOp for DtRound<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_round(
            self.view.0,
            i32::from(self.freq),
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::dt_add_months`].
///
/// Created by [`DatetimeExt::dt_add_months`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DtAddMonths<'a> {
    view: &'a ColumnView<'a>,
    months: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DtAddMonths<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_add_months(
            self.view.0,
            self.months.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::dt_add_months_scalar`].
///
/// Created by [`DatetimeExt::dt_add_months_scalar`]. Call `.call()` to execute,
/// or chain `.stream(s)` to run on a specific CUDA stream.
pub struct DtAddMonthsScalar<'a> {
    view: &'a ColumnView<'a>,
    months: &'a crate::scalar::Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for DtAddMonthsScalar<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi_scalar = crate::scalar::scalar_to_ffi(self.months)?;
        let c = cudf_sys::datetime::ffi::datetime_add_months_scalar(
            self.view.0,
            &ffi_scalar,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_millisecond`].
///
/// Created by [`DatetimeExt::extract_millisecond`]. Call `.call()` to execute,
/// or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractMillisecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractMillisecond<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_millisecond(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_microsecond`].
///
/// Created by [`DatetimeExt::extract_microsecond`]. Call `.call()` to execute,
/// or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractMicrosecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractMicrosecond<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_microsecond(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DatetimeExt::extract_nanosecond`].
///
/// Created by [`DatetimeExt::extract_nanosecond`]. Call `.call()` to execute,
/// or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ExtractNanosecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ExtractNanosecond<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::datetime::ffi::datetime_extract_nanosecond(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::data_type::TypeId;
    use crate::stream::GpuOp;

    /// Helper: create a timestamp column from epoch seconds.
    fn make_timestamp_seconds(epochs: &[i64]) -> crate::column::UnboundColumn {
        Column::from_timestamps_s(epochs).call().unwrap()
    }

    const EPOCH_2024_06_15_10_30_45: i64 = 1718443845;
    const EPOCH_2024_01_01_00_00_00: i64 = 1704067200;
    const EPOCH_2021_01_01_00_00_00: i64 = 1609459200;

    #[test]
    fn extract_year_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().extract_year().call().unwrap();
        assert_eq!(result.type_id(), TypeId::INT16);
        assert_eq!(result.to_vec_i16().call().unwrap(), vec![2024, 2024, 2021]);
    }

    #[test]
    fn extract_month_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().extract_month().call().unwrap();
        assert_eq!(result.to_vec_i16().call().unwrap(), vec![6, 1, 1]);
    }

    #[test]
    fn extract_day_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().extract_day().call().unwrap();
        assert_eq!(result.to_vec_i16().call().unwrap(), vec![15, 1, 1]);
    }

    #[test]
    fn extract_hour_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45, EPOCH_2024_01_01_00_00_00]);
        let result = ts.view().extract_hour().call().unwrap();
        assert_eq!(result.to_vec_i16().call().unwrap(), vec![9, 0]);
    }

    #[test]
    fn extract_minute_second() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let minutes = ts.view().extract_minute().call().unwrap();
        let seconds = ts.view().extract_second().call().unwrap();
        assert_eq!(minutes.to_vec_i16().call().unwrap(), vec![30]);
        assert_eq!(seconds.to_vec_i16().call().unwrap(), vec![45]);
    }

    #[test]
    fn day_of_year_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().day_of_year().call().unwrap();
        let vals = result.to_vec_i16().call().unwrap();
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], 167);
    }

    #[test]
    fn is_leap_year_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2021_01_01_00_00_00]);
        let result = ts.view().is_leap_year().call().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool().call().unwrap(), vec![true, false]);
    }

    #[test]
    fn days_in_month_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().days_in_month().call().unwrap();
        assert_eq!(result.to_vec_i16().call().unwrap(), vec![31, 30]);
    }

    #[test]
    fn extract_quarter_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().extract_quarter().call().unwrap();
        assert_eq!(result.to_vec_i16().call().unwrap(), vec![1, 2]);
    }

    #[test]
    fn last_day_of_month_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().last_day_of_month().call().unwrap();
        assert_eq!(result.type_id(), TypeId::TIMESTAMP_DAYS);
        assert_eq!(result.len(), 1);
    }
}
