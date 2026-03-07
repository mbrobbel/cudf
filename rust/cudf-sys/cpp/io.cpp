// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/io.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/interop.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/parquet.hpp>

namespace cudf_sys {

// -- CSV I/O --

std::unique_ptr<Table> read_csv(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::csv_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_csv(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

std::unique_ptr<Table> read_csv_with_options(
    rust::Str filepath,
    uint8_t delimiter,
    bool header,
    int32_t skip_rows,
    int32_t num_rows) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::csv_reader_options::builder(cudf::io::source_info{path});
  builder.delimiter(static_cast<char>(delimiter));
  if (!header) {
    builder.header(-1);
  }
  builder.skiprows(skip_rows);
  if (num_rows >= 0) {
    builder.nrows(num_rows);
  }
  auto opts = builder.build();
  auto result = cudf::io::read_csv(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_csv(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::csv_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_csv(opts);
}

void write_csv_with_options(
    Table const& tbl,
    rust::Str filepath,
    uint8_t delimiter,
    bool include_header,
    rust::Str na_rep) {
  std::string path(filepath.data(), filepath.size());
  std::string na_str(na_rep.data(), na_rep.size());
  auto builder = cudf::io::csv_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view());
  builder.inter_column_delimiter(static_cast<char>(delimiter));
  builder.include_header(include_header);
  builder.na_rep(na_str);
  auto opts = builder.build();
  cudf::io::write_csv(opts);
}

// -- Parquet I/O --

std::unique_ptr<Table> read_parquet(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_parquet(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_parquet(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_parquet(opts);
}

// -- ORC I/O --

std::unique_ptr<Table> read_orc(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_orc(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_orc(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::orc_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_orc(opts);
}

// -- JSON I/O --

std::unique_ptr<Table> read_json(rust::Str filepath, bool json_lines) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::json_reader_options::builder(cudf::io::source_info{path});
  builder.lines(json_lines);
  auto opts = builder.build();
  auto result = cudf::io::read_json(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

void write_json(Table const& tbl, rust::Str filepath, bool json_lines) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::json_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view());
  builder.lines(json_lines);
  auto opts = builder.build();
  cudf::io::write_json(opts);
}

// -- Avro I/O --

std::unique_ptr<Table> read_avro(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::avro_reader_options::builder(cudf::io::source_info{path}).build();
  auto result = cudf::io::read_avro(opts);
  return std::make_unique<Table>(std::move(result.tbl));
}

// -- DLPack interop --

std::unique_ptr<Table> from_dlpack(std::size_t managed_tensor_ptr, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto* tensor = reinterpret_cast<DLManagedTensor const*>(managed_tensor_ptr);
  auto result = cudf::from_dlpack(tensor, s);
  return std::make_unique<Table>(std::move(result));
}

std::size_t to_dlpack(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto* tensor = cudf::to_dlpack(tbl.cached_view(), s);
  return reinterpret_cast<std::size_t>(tensor);
}

}  // namespace cudf_sys
