// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Datetime operations on timestamp columns.

use crate::column::{Column, ColumnView};
use crate::error::Result;

/// Extracts the year component from a timestamp column (returns INT16).
pub fn extract_year(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_year(col.0)?;
    Ok(Column(c))
}

/// Extracts the month component from a timestamp column (returns INT16).
pub fn extract_month(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_month(col.0)?;
    Ok(Column(c))
}

/// Extracts the day component from a timestamp column (returns INT16).
pub fn extract_day(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_day(col.0)?;
    Ok(Column(c))
}

/// Extracts the weekday component from a timestamp column (returns INT16).
pub fn extract_weekday(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_weekday(col.0)?;
    Ok(Column(c))
}

/// Extracts the hour component from a timestamp column (returns INT16).
pub fn extract_hour(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_hour(col.0)?;
    Ok(Column(c))
}

/// Extracts the minute component from a timestamp column (returns INT16).
pub fn extract_minute(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_minute(col.0)?;
    Ok(Column(c))
}

/// Extracts the second component from a timestamp column (returns INT16).
pub fn extract_second(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_second(col.0)?;
    Ok(Column(c))
}

/// Returns the day of year (1-366) for each timestamp (returns INT16).
pub fn day_of_year(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_day_of_year(col.0)?;
    Ok(Column(c))
}

/// Returns whether each timestamp's year is a leap year (returns BOOL8).
pub fn is_leap_year(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_is_leap_year(col.0)?;
    Ok(Column(c))
}

/// Returns the number of days in the month for each timestamp (returns INT16).
pub fn days_in_month(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_days_in_month(col.0)?;
    Ok(Column(c))
}

/// Returns the last day of the month for each timestamp (returns TIMESTAMP_DAYS).
pub fn last_day_of_month(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_last_day_of_month(col.0)?;
    Ok(Column(c))
}

/// Returns the quarter (1-4) for each timestamp (returns INT16).
pub fn extract_quarter(col: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::datetime_extract_quarter(col.0)?;
    Ok(Column(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::TypeId;
    use crate::ops;

    /// Helper: create a timestamp column from epoch seconds.
    fn make_timestamp_seconds(epochs: &[i64]) -> Column {
        let int_col = Column::from_slice_i64(epochs);
        let ts_col = ops::unary::cast(&int_col.view(), TypeId::TIMESTAMP_SECONDS).unwrap();
        ts_col
    }

    // Test dates (as epoch seconds):
    // 2024-06-15 10:30:45 UTC = 1718444445 + 300 + 45 = 1718444445
    //   Actually: let's compute precisely.
    //   2024-01-01 00:00:00 UTC = 1704067200
    //   2024-06-15 10:30:45 UTC = 1718443845
    //   2021-01-01 00:00:00 UTC = 1609459200
    //   2023-03-15 00:00:00 UTC = 1678838400
    //
    // Using well-known epoch values:
    //   2024-06-15 10:30:45 = 1718443845
    //   2024-01-01 00:00:00 = 1704067200
    //   2021-01-01 00:00:00 = 1609459200

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
        let result = extract_year(&ts.view()).unwrap();
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
        let result = extract_month(&ts.view()).unwrap();
        assert_eq!(result.to_vec_i16(), vec![6, 1, 1]);
    }

    #[test]
    fn extract_day_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = extract_day(&ts.view()).unwrap();
        assert_eq!(result.to_vec_i16(), vec![15, 1, 1]);
    }

    #[test]
    fn extract_hour_basic() {
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_06_15_10_30_45,
            EPOCH_2024_01_01_00_00_00,
        ]);
        let result = extract_hour(&ts.view()).unwrap();
        assert_eq!(result.to_vec_i16(), vec![10, 0]);
    }

    #[test]
    fn extract_minute_second() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let minutes = extract_minute(&ts.view()).unwrap();
        let seconds = extract_second(&ts.view()).unwrap();
        assert_eq!(minutes.to_vec_i16(), vec![30]);
        assert_eq!(seconds.to_vec_i16(), vec![45]);
    }

    #[test]
    fn day_of_year_basic() {
        // 2024-01-01 -> day 1
        // 2024-06-15 -> day 167 (2024 is leap year: 31+29+31+30+31+15 = 167)
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2024_06_15_10_30_45,
        ]);
        let result = day_of_year(&ts.view()).unwrap();
        let vals = result.to_vec_i16();
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], 167);
    }

    #[test]
    fn is_leap_year_basic() {
        // 2024 is a leap year, 2021 is not
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2021_01_01_00_00_00,
        ]);
        let result = is_leap_year(&ts.view()).unwrap();
        assert_eq!(result.type_id(), TypeId::BOOL8);
        assert_eq!(result.to_vec_bool(), vec![true, false]);
    }

    #[test]
    fn days_in_month_basic() {
        // Jan 2024 -> 31 days, Jun 2024 -> 30 days
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2024_06_15_10_30_45,
        ]);
        let result = days_in_month(&ts.view()).unwrap();
        assert_eq!(result.to_vec_i16(), vec![31, 30]);
    }

    #[test]
    fn extract_quarter_basic() {
        // Jan -> Q1, Jun -> Q2
        let ts = make_timestamp_seconds(&[
            EPOCH_2024_01_01_00_00_00,
            EPOCH_2024_06_15_10_30_45,
        ]);
        let result = extract_quarter(&ts.view()).unwrap();
        assert_eq!(result.to_vec_i16(), vec![1, 2]);
    }

    #[test]
    fn last_day_of_month_basic() {
        let ts = make_timestamp_seconds(&[EPOCH_2024_06_15_10_30_45]);
        let result = last_day_of_month(&ts.view()).unwrap();
        assert_eq!(result.type_id(), TypeId::TIMESTAMP_DAYS);
        assert_eq!(result.len(), 1);
    }
}
