// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/datetime.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/datetime.hpp>

namespace cudf_sys {

// -- Datetime operations --

std::unique_ptr<Column> datetime_extract_year(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::YEAR, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_month(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MONTH, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_day(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::DAY, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_weekday(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::WEEKDAY, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_hour(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::HOUR, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_minute(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MINUTE, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_second(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::SECOND, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_day_of_year(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::day_of_year(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_is_leap_year(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::is_leap_year(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_days_in_month(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::days_in_month(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_last_day_of_month(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::last_day_of_month(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_quarter(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_quarter(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Datetime operations (new) --

std::unique_ptr<Column> datetime_ceil(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::ceil_datetimes(col, static_cast<cudf::datetime::rounding_frequency>(freq), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_floor(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::floor_datetimes(col, static_cast<cudf::datetime::rounding_frequency>(freq), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_round(cudf::column_view const& col, int32_t freq, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::round_datetimes(col, static_cast<cudf::datetime::rounding_frequency>(freq), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_add_months(cudf::column_view const& timestamps, cudf::column_view const& months, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::add_calendrical_months(timestamps, months, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Datetime: fractional seconds --

std::unique_ptr<Column> datetime_extract_millisecond(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MILLISECOND, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_microsecond(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::MICROSECOND, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> datetime_extract_nanosecond(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::extract_datetime_component(col, cudf::datetime::datetime_component::NANOSECOND, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Datetime: add months with scalar --

std::unique_ptr<Column> datetime_add_months_scalar(cudf::column_view const& timestamps, Scalar const& months, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::datetime::add_calendrical_months(timestamps, months.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
