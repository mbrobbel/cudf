// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/datetime.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/datetime.hpp>

namespace cudf_sys {

std::unique_ptr<Column> datetime_extract_year(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::YEAR, S(stream)));
}

std::unique_ptr<Column> datetime_extract_month(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MONTH, S(stream)));
}

std::unique_ptr<Column> datetime_extract_day(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::DAY, S(stream)));
}

std::unique_ptr<Column> datetime_extract_weekday(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::WEEKDAY, S(stream)));
}

std::unique_ptr<Column> datetime_extract_hour(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::HOUR, S(stream)));
}

std::unique_ptr<Column> datetime_extract_minute(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MINUTE, S(stream)));
}

std::unique_ptr<Column> datetime_extract_second(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::SECOND, S(stream)));
}

std::unique_ptr<Column> datetime_day_of_year(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::day_of_year(col, S(stream)));
}

std::unique_ptr<Column> datetime_is_leap_year(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::is_leap_year(col, S(stream)));
}

std::unique_ptr<Column> datetime_days_in_month(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::days_in_month(col, S(stream)));
}

std::unique_ptr<Column> datetime_last_day_of_month(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::last_day_of_month(col, S(stream)));
}

std::unique_ptr<Column> datetime_extract_quarter(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_quarter(col, S(stream)));
}

std::unique_ptr<Column> datetime_ceil(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  return COL(cudf::datetime::ceil_datetimes(col, ENUM<cudf::datetime::rounding_frequency>(freq), S(stream)));
}

std::unique_ptr<Column> datetime_floor(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  return COL(cudf::datetime::floor_datetimes(col, ENUM<cudf::datetime::rounding_frequency>(freq), S(stream)));
}

std::unique_ptr<Column> datetime_round(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  return COL(cudf::datetime::round_datetimes(col, ENUM<cudf::datetime::rounding_frequency>(freq), S(stream)));
}

std::unique_ptr<Column> datetime_add_months(cudf::column_view const& timestamps, cudf::column_view const& months, std::size_t stream) {
  return COL(cudf::datetime::add_calendrical_months(timestamps, months, S(stream)));
}

std::unique_ptr<Column> datetime_extract_millisecond(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MILLISECOND, S(stream)));
}

std::unique_ptr<Column> datetime_extract_microsecond(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MICROSECOND, S(stream)));
}

std::unique_ptr<Column> datetime_extract_nanosecond(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::NANOSECOND, S(stream)));
}

std::unique_ptr<Column> datetime_add_months_scalar(cudf::column_view const& timestamps, Scalar const& months, std::size_t stream) {
  return COL(cudf::datetime::add_calendrical_months(timestamps, months.inner(), S(stream)));
}

}  // namespace cudf_sys
