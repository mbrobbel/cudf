// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Scalar = crate::ffi::Scalar;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Datetime operations --

        /// Extracts the year component from a timestamp column (returns INT16).
        fn datetime_extract_year(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the month component from a timestamp column (returns INT16).
        fn datetime_extract_month(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the day component from a timestamp column (returns INT16).
        fn datetime_extract_day(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the weekday component from a timestamp column (returns INT16).
        fn datetime_extract_weekday(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the hour component from a timestamp column (returns INT16).
        fn datetime_extract_hour(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the minute component from a timestamp column (returns INT16).
        fn datetime_extract_minute(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Extracts the second component from a timestamp column (returns INT16).
        fn datetime_extract_second(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the day of year (1-366) for each timestamp (returns INT16).
        fn datetime_day_of_year(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns whether each timestamp's year is a leap year (returns BOOL8).
        fn datetime_is_leap_year(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the number of days in the month for each timestamp (returns INT16).
        fn datetime_days_in_month(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the last day of the month for each timestamp (returns TIMESTAMP_DAYS).
        fn datetime_last_day_of_month(
            col: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns the quarter (1-4) for each timestamp (returns INT16).
        fn datetime_extract_quarter(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Datetime operations (new) --

        /// Ceil datetimes to frequency (DAY=0, HOUR=1, MINUTE=2, SECOND=3, MS=4, US=5, NS=6).
        fn datetime_ceil(col: &column_view, freq: i32, stream: usize) -> Result<UniquePtr<Column>>;
        /// Floor datetimes to frequency.
        fn datetime_floor(col: &column_view, freq: i32, stream: usize)
        -> Result<UniquePtr<Column>>;
        /// Round datetimes to frequency.
        fn datetime_round(col: &column_view, freq: i32, stream: usize)
        -> Result<UniquePtr<Column>>;
        /// Add months column to timestamp column.
        fn datetime_add_months(
            timestamps: &column_view,
            months: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Datetime: fractional seconds --

        /// Extract millisecond fraction from timestamp.
        fn datetime_extract_millisecond(
            col: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Extract microsecond fraction from timestamp.
        fn datetime_extract_microsecond(
            col: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Extract nanosecond fraction from timestamp.
        fn datetime_extract_nanosecond(
            col: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Datetime: add months with scalar --

        /// Adds months (scalar) to a timestamp column.
        fn datetime_add_months_scalar(
            timestamps: &column_view,
            months: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
