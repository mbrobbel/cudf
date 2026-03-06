// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Datetime operations on timestamp columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::stream::Stream;

/// Extension trait for datetime operations on timestamp columns.
pub trait DatetimeExt {
    /// Extracts the year component (returns INT16).
    fn extract_year(&self) -> Result<Column>;
    /// Extracts the year on a custom CUDA stream.
    fn extract_year_on(&self, stream: Stream) -> Result<Column>;
    /// Extracts the month component (returns INT16).
    fn extract_month(&self) -> Result<Column>;
    /// Extracts the month on a custom CUDA stream.
    fn extract_month_on(&self, stream: Stream) -> Result<Column>;
    /// Extracts the day component (returns INT16).
    fn extract_day(&self) -> Result<Column>;
    /// Extracts the day on a custom CUDA stream.
    fn extract_day_on(&self, stream: Stream) -> Result<Column>;
    /// Extracts the weekday component (returns INT16).
    fn extract_weekday(&self) -> Result<Column>;
    /// Extracts the weekday on a custom CUDA stream.
    fn extract_weekday_on(&self, stream: Stream) -> Result<Column>;
    /// Extracts the hour component (returns INT16).
    fn extract_hour(&self) -> Result<Column>;
    /// Extracts the hour on a custom CUDA stream.
    fn extract_hour_on(&self, stream: Stream) -> Result<Column>;
    /// Extracts the minute component (returns INT16).
    fn extract_minute(&self) -> Result<Column>;
    /// Extracts the minute on a custom CUDA stream.
    fn extract_minute_on(&self, stream: Stream) -> Result<Column>;
    /// Extracts the second component (returns INT16).
    fn extract_second(&self) -> Result<Column>;
    /// Extracts the second on a custom CUDA stream.
    fn extract_second_on(&self, stream: Stream) -> Result<Column>;
    /// Returns the day of year (1-366) (returns INT16).
    fn day_of_year(&self) -> Result<Column>;
    /// Day of year on a custom CUDA stream.
    fn day_of_year_on(&self, stream: Stream) -> Result<Column>;
    /// Returns whether each year is a leap year (returns BOOL8).
    fn is_leap_year(&self) -> Result<Column>;
    /// is_leap_year on a custom CUDA stream.
    fn is_leap_year_on(&self, stream: Stream) -> Result<Column>;
    /// Returns the number of days in the month (returns INT16).
    fn days_in_month(&self) -> Result<Column>;
    /// days_in_month on a custom CUDA stream.
    fn days_in_month_on(&self, stream: Stream) -> Result<Column>;
    /// Returns the last day of the month (returns TIMESTAMP_DAYS).
    fn last_day_of_month(&self) -> Result<Column>;
    /// last_day_of_month on a custom CUDA stream.
    fn last_day_of_month_on(&self, stream: Stream) -> Result<Column>;
    /// Returns the quarter (1-4) (returns INT16).
    fn extract_quarter(&self) -> Result<Column>;
    /// extract_quarter on a custom CUDA stream.
    fn extract_quarter_on(&self, stream: Stream) -> Result<Column>;
    /// Ceil datetimes to given frequency.
    fn dt_ceil(&self, freq: RoundingFrequency) -> Result<Column>;
    /// Floor datetimes to given frequency.
    fn dt_floor(&self, freq: RoundingFrequency) -> Result<Column>;
    /// Round datetimes to given frequency.
    fn dt_round(&self, freq: RoundingFrequency) -> Result<Column>;
    /// Add months (from another column) to timestamps.
    fn dt_add_months(&self, months: &ColumnView<'_>) -> Result<Column>;
}

/// Datetime rounding frequency.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingFrequency {
    Day = 0,
    Hour = 1,
    Minute = 2,
    Second = 3,
    Millisecond = 4,
    Microsecond = 5,
    Nanosecond = 6,
}

impl DatetimeExt for ColumnView<'_> {
    fn extract_year(&self) -> Result<Column> {
        self.extract_year_on(Stream::default_stream())
    }
    fn extract_year_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_year(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_month(&self) -> Result<Column> {
        self.extract_month_on(Stream::default_stream())
    }
    fn extract_month_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_month(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_day(&self) -> Result<Column> {
        self.extract_day_on(Stream::default_stream())
    }
    fn extract_day_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_day(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_weekday(&self) -> Result<Column> {
        self.extract_weekday_on(Stream::default_stream())
    }
    fn extract_weekday_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_weekday(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_hour(&self) -> Result<Column> {
        self.extract_hour_on(Stream::default_stream())
    }
    fn extract_hour_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_hour(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_minute(&self) -> Result<Column> {
        self.extract_minute_on(Stream::default_stream())
    }
    fn extract_minute_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_minute(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_second(&self) -> Result<Column> {
        self.extract_second_on(Stream::default_stream())
    }
    fn extract_second_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_second(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn day_of_year(&self) -> Result<Column> {
        self.day_of_year_on(Stream::default_stream())
    }
    fn day_of_year_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_day_of_year(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn is_leap_year(&self) -> Result<Column> {
        self.is_leap_year_on(Stream::default_stream())
    }
    fn is_leap_year_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_is_leap_year(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn days_in_month(&self) -> Result<Column> {
        self.days_in_month_on(Stream::default_stream())
    }
    fn days_in_month_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_days_in_month(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn last_day_of_month(&self) -> Result<Column> {
        self.last_day_of_month_on(Stream::default_stream())
    }
    fn last_day_of_month_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_last_day_of_month(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn extract_quarter(&self) -> Result<Column> {
        self.extract_quarter_on(Stream::default_stream())
    }
    fn extract_quarter_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_extract_quarter(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn dt_ceil(&self, freq: RoundingFrequency) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_ceil(self.0, freq as i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn dt_floor(&self, freq: RoundingFrequency) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_floor(self.0, freq as i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn dt_round(&self, freq: RoundingFrequency) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_round(self.0, freq as i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn dt_add_months(&self, months: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::datetime_add_months(self.0, months.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
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
        let result = ts.view().extract_year().unwrap();
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
        let result = ts.view().extract_month().unwrap();
        assert_eq!(result.to_vec_i16(), vec![6, 1, 1]);
    }

    #[test]
    fn extract_day_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().extract_day().unwrap();
        assert_eq!(result.to_vec_i16(), vec![15, 1, 1]);
    }

    #[test]
    fn extract_hour_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
        ]);
        let result = ts.view().extract_hour().unwrap();
        assert_eq!(result.to_vec_i16(), vec![9, 0]);
    }

    #[test]
    fn extract_minute_second() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let minutes = ts.view().extract_minute().unwrap();
        let seconds = ts.view().extract_second().unwrap();
        assert_eq!(minutes.to_vec_i16(), vec![30]);
        assert_eq!(seconds.to_vec_i16(), vec![45]);
    }

    #[test]
    fn day_of_year_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2024_06_15_10_30_45,
        ]);
        let result = ts.view().day_of_year().unwrap();
        let vals = result.to_vec_i16();
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], 167);
    }

    #[test]
    fn is_leap_year_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = ts.view().is_leap_year().unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![true, false]);
    }

    #[test]
    fn days_in_month_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2024_06_15_10_30_45,
        ]);
        let result = ts.view().days_in_month().unwrap();
        assert_eq!(result.to_vec_i16(), vec![31, 30]);
    }

    #[test]
    fn extract_quarter_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2024_06_15_10_30_45,
        ]);
        let result = ts.view().extract_quarter().unwrap();
        assert_eq!(result.to_vec_i16(), vec![1, 2]);
    }

    #[test]
    fn last_day_of_month_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let result = ts.view().last_day_of_month().unwrap();
        assert_eq!(result.type_id(), TypeId::TIMESTAMP_DAYS);
        assert_eq!(result.len(), 1);
    }
}
