// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Datetime operations on timestamp columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

/// Extension trait for datetime operations on timestamp columns.
pub trait DatetimeExt: private::Sealed {
    /// Extracts the year component (returns INT16).
    fn extract_year(&self) -> ExtractYear<'_>;
    /// Extracts the month component (returns INT16).
    fn extract_month(&self) -> ExtractMonth<'_>;
    /// Extracts the day component (returns INT16).
    fn extract_day(&self) -> ExtractDay<'_>;
    /// Extracts the weekday component (returns INT16).
    fn extract_weekday(&self) -> ExtractWeekday<'_>;
    /// Extracts the hour component (returns INT16).
    fn extract_hour(&self) -> ExtractHour<'_>;
    /// Extracts the minute component (returns INT16).
    fn extract_minute(&self) -> ExtractMinute<'_>;
    /// Extracts the second component (returns INT16).
    fn extract_second(&self) -> ExtractSecond<'_>;
    /// Returns the day of year (1-366) (returns INT16).
    fn day_of_year(&self) -> DayOfYear<'_>;
    /// Returns whether each year is a leap year (returns BOOL8).
    fn is_leap_year(&self) -> IsLeapYear<'_>;
    /// Returns the number of days in the month (returns INT16).
    fn days_in_month(&self) -> DaysInMonth<'_>;
    /// Returns the last day of the month (returns `TIMESTAMP_DAYS`).
    fn last_day_of_month(&self) -> LastDayOfMonth<'_>;
    /// Returns the quarter (1-4) (returns INT16).
    fn extract_quarter(&self) -> ExtractQuarter<'_>;
    /// Ceil datetimes to given frequency.
    fn dt_ceil(&self, freq: RoundingFrequency) -> DtCeil<'_>;
    /// Floor datetimes to given frequency.
    fn dt_floor(&self, freq: RoundingFrequency) -> DtFloor<'_>;
    /// Round datetimes to given frequency.
    fn dt_round(&self, freq: RoundingFrequency) -> DtRound<'_>;
    /// Add months (from another column) to timestamps.
    fn dt_add_months<'a>(&'a self, months: &'a ColumnView<'a>) -> DtAddMonths<'a>;
    /// Add months (from a scalar) to timestamps.
    fn dt_add_months_scalar<'a>(
        &'a self,
        months: &'a crate::scalar::Scalar,
    ) -> DtAddMonthsScalar<'a>;
    /// Extracts the millisecond fraction component (returns INT16).
    fn extract_millisecond(&self) -> ExtractMillisecond<'_>;
    /// Extracts the microsecond fraction component (returns INT16).
    fn extract_microsecond(&self) -> ExtractMicrosecond<'_>;
    /// Extracts the nanosecond fraction component (returns INT16).
    fn extract_nanosecond(&self) -> ExtractNanosecond<'_>;
}

#[doc(alias = "rounding_frequency")]
/// Datetime rounding frequency.
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
pub struct ExtractYear<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractYear<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_year(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_month`].
pub struct ExtractMonth<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractMonth<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_month(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_day`].
pub struct ExtractDay<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractDay<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_day(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_weekday`].
pub struct ExtractWeekday<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractWeekday<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_weekday(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_hour`].
pub struct ExtractHour<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractHour<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_hour(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_minute`].
pub struct ExtractMinute<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractMinute<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_minute(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_second`].
pub struct ExtractSecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractSecond<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_second(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::day_of_year`].
pub struct DayOfYear<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl DayOfYear<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_day_of_year(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::is_leap_year`].
pub struct IsLeapYear<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl IsLeapYear<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_is_leap_year(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::days_in_month`].
pub struct DaysInMonth<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl DaysInMonth<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_days_in_month(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::last_day_of_month`].
pub struct LastDayOfMonth<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl LastDayOfMonth<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::datetime::ffi::datetime_last_day_of_month(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_quarter`].
pub struct ExtractQuarter<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractQuarter<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::datetime::ffi::datetime_extract_quarter(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::dt_ceil`].
pub struct DtCeil<'a> {
    view: &'a ColumnView<'a>,
    freq: RoundingFrequency,
    stream: Stream,
}

impl DtCeil<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_ceil(
            self.view.0,
            i32::from(self.freq),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::dt_floor`].
pub struct DtFloor<'a> {
    view: &'a ColumnView<'a>,
    freq: RoundingFrequency,
    stream: Stream,
}

impl DtFloor<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_floor(
            self.view.0,
            i32::from(self.freq),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::dt_round`].
pub struct DtRound<'a> {
    view: &'a ColumnView<'a>,
    freq: RoundingFrequency,
    stream: Stream,
}

impl DtRound<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_round(
            self.view.0,
            i32::from(self.freq),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::dt_add_months`].
pub struct DtAddMonths<'a> {
    view: &'a ColumnView<'a>,
    months: &'a ColumnView<'a>,
    stream: Stream,
}

impl DtAddMonths<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_add_months(
            self.view.0,
            self.months.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::dt_add_months_scalar`].
pub struct DtAddMonthsScalar<'a> {
    view: &'a ColumnView<'a>,
    months: &'a crate::scalar::Scalar,
    stream: Stream,
}

impl DtAddMonthsScalar<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let ffi_scalar = crate::scalar::scalar_to_ffi(self.months);
        let c = cudf_sys::datetime::ffi::datetime_add_months_scalar(
            self.view.0,
            &ffi_scalar,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_millisecond`].
pub struct ExtractMillisecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractMillisecond<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_millisecond(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_microsecond`].
pub struct ExtractMicrosecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractMicrosecond<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_microsecond(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DatetimeExt::extract_nanosecond`].
pub struct ExtractNanosecond<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ExtractNanosecond<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::datetime::ffi::datetime_extract_nanosecond(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

// ---------------------------------------------------------------------------
// Sealed impl + trait impl
// ---------------------------------------------------------------------------

impl private::Sealed for ColumnView<'_> {}

impl DatetimeExt for ColumnView<'_> {
    fn extract_year(&self) -> ExtractYear<'_> {
        ExtractYear {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_month(&self) -> ExtractMonth<'_> {
        ExtractMonth {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_day(&self) -> ExtractDay<'_> {
        ExtractDay {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_weekday(&self) -> ExtractWeekday<'_> {
        ExtractWeekday {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_hour(&self) -> ExtractHour<'_> {
        ExtractHour {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_minute(&self) -> ExtractMinute<'_> {
        ExtractMinute {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_second(&self) -> ExtractSecond<'_> {
        ExtractSecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn day_of_year(&self) -> DayOfYear<'_> {
        DayOfYear {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn is_leap_year(&self) -> IsLeapYear<'_> {
        IsLeapYear {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn days_in_month(&self) -> DaysInMonth<'_> {
        DaysInMonth {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn last_day_of_month(&self) -> LastDayOfMonth<'_> {
        LastDayOfMonth {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_quarter(&self) -> ExtractQuarter<'_> {
        ExtractQuarter {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn dt_ceil(&self, freq: RoundingFrequency) -> DtCeil<'_> {
        DtCeil {
            view: self,
            freq,
            stream: Stream::default_stream(),
        }
    }
    fn dt_floor(&self, freq: RoundingFrequency) -> DtFloor<'_> {
        DtFloor {
            view: self,
            freq,
            stream: Stream::default_stream(),
        }
    }
    fn dt_round(&self, freq: RoundingFrequency) -> DtRound<'_> {
        DtRound {
            view: self,
            freq,
            stream: Stream::default_stream(),
        }
    }
    fn dt_add_months<'a>(&'a self, months: &'a ColumnView<'a>) -> DtAddMonths<'a> {
        DtAddMonths {
            view: self,
            months,
            stream: Stream::default_stream(),
        }
    }
    fn dt_add_months_scalar<'a>(
        &'a self,
        months: &'a crate::scalar::Scalar,
    ) -> DtAddMonthsScalar<'a> {
        DtAddMonthsScalar {
            view: self,
            months,
            stream: Stream::default_stream(),
        }
    }
    fn extract_millisecond(&self) -> ExtractMillisecond<'_> {
        ExtractMillisecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_microsecond(&self) -> ExtractMicrosecond<'_> {
        ExtractMicrosecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_nanosecond(&self) -> ExtractNanosecond<'_> {
        ExtractNanosecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::data_type::TypeId;

    /// Helper: create a timestamp column from epoch seconds.
    fn make_timestamp_seconds(epochs: &[i64]) -> Column {
        Column::from_timestamps_s(epochs)
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
        assert_eq!(result.to_vec_i16(), vec![2024, 2024, 2021]);
    }

    #[test]
    fn extract_month_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().extract_month().call().unwrap();
        assert_eq!(result.to_vec_i16(), vec![6, 1, 1]);
    }

    #[test]
    fn extract_day_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().extract_day().call().unwrap();
        assert_eq!(result.to_vec_i16(), vec![15, 1, 1]);
    }

    #[test]
    fn extract_hour_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45, EPOCH_2024_01_01_00_00_00]);
        let result = ts.view().extract_hour().call().unwrap();
        assert_eq!(result.to_vec_i16(), vec![9, 0]);
    }

    #[test]
    fn extract_minute_second() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let minutes = ts.view().extract_minute().call().unwrap();
        let seconds = ts.view().extract_second().call().unwrap();
        assert_eq!(minutes.to_vec_i16(), vec![30]);
        assert_eq!(seconds.to_vec_i16(), vec![45]);
    }

    #[test]
    fn day_of_year_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().day_of_year().call().unwrap();
        let vals = result.to_vec_i16();
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], 167);
    }

    #[test]
    fn is_leap_year_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2021_01_01_00_00_00]);
        let result = ts.view().is_leap_year().call().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![true, false]);
    }

    #[test]
    fn days_in_month_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().days_in_month().call().unwrap();
        assert_eq!(result.to_vec_i16(), vec![31, 30]);
    }

    #[test]
    fn extract_quarter_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_01_01_00_00_00, EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().extract_quarter().call().unwrap();
        assert_eq!(result.to_vec_i16(), vec![1, 2]);
    }

    #[test]
    fn last_day_of_month_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().last_day_of_month().call().unwrap();
        assert_eq!(result.type_id(), TypeId::TIMESTAMP_DAYS);
        assert_eq!(result.len(), 1);
    }
}
