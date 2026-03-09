// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/io.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

#include <cudf/interop.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/parquet.hpp>

namespace cudf_sys {

// -- TableWithMetadata --

std::unique_ptr<Table> table_with_metadata_take_table(TableWithMetadata& twm) {
  return std::make_unique<Table>(std::move(twm.twm.tbl));
}

rust::Vec<rust::String> table_with_metadata_column_names(TableWithMetadata const& twm) {
  rust::Vec<rust::String> out;
  for (auto const& info : twm.twm.metadata.schema_info) {
    out.push_back(rust::String(info.name));
  }
  return out;
}

int32_t table_with_metadata_num_columns(TableWithMetadata const& twm) {
  return static_cast<int32_t>(twm.twm.metadata.schema_info.size());
}

rust::String table_with_metadata_column_name(TableWithMetadata const& twm, int32_t index) {
  return rust::String(twm.twm.metadata.schema_info.at(index).name);
}

// -- CSV I/O --

std::unique_ptr<Table> read_csv(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::csv_reader_options::builder(cudf::io::source_info{path}).build();
  return TBL(cudf::io::read_csv(opts).tbl);
}

std::unique_ptr<TableWithMetadata> read_csv_meta(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::csv_reader_options::builder(cudf::io::source_info{path}).build();
  return std::make_unique<TableWithMetadata>(cudf::io::read_csv(opts));
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
  return TBL(cudf::io::read_csv(builder.build()).tbl);
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
  cudf::io::write_csv(builder.build());
}

// -- Parquet I/O --

std::unique_ptr<Table> read_parquet(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path}).build();
  return TBL(cudf::io::read_parquet(opts).tbl);
}

std::unique_ptr<TableWithMetadata> read_parquet_meta(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path}).build();
  return std::make_unique<TableWithMetadata>(cudf::io::read_parquet(opts));
}

std::unique_ptr<TableWithMetadata> read_parquet_with_columns(
    rust::Str filepath,
    rust::Slice<rust::Str const> columns,
    int64_t skip_rows,
    int64_t num_rows) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::parquet_reader_options::builder(cudf::io::source_info{path});
  if (!columns.empty()) {
    std::vector<std::string> col_names;
    col_names.reserve(columns.size());
    for (auto const& c : columns) {
      col_names.emplace_back(c.data(), c.size());
    }
    builder.column_names(std::move(col_names));
  }
  if (skip_rows > 0) {
    builder.skip_rows(skip_rows);
  }
  if (num_rows >= 0) {
    builder.num_rows(num_rows);
  }
  return std::make_unique<TableWithMetadata>(cudf::io::read_parquet(builder.build()));
}

void write_parquet(Table const& tbl, rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view()).build();
  cudf::io::write_parquet(opts);
}

void write_parquet_with_names(
    Table const& tbl,
    rust::Str filepath,
    rust::Slice<rust::Str const> column_names) {
  std::string path(filepath.data(), filepath.size());
  auto view = tbl.cached_view();
  cudf::io::table_input_metadata meta(view);
  for (std::size_t i = 0; i < column_names.size() && i < meta.column_metadata.size(); ++i) {
    meta.column_metadata[i].set_name(std::string(column_names[i].data(), column_names[i].size()));
  }
  auto builder = cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info{path}, view);
  builder.metadata(std::move(meta));
  cudf::io::write_parquet(builder.build());
}

// -- ORC I/O --

std::unique_ptr<Table> read_orc(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{path}).build();
  return TBL(cudf::io::read_orc(opts).tbl);
}

std::unique_ptr<TableWithMetadata> read_orc_meta(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::orc_reader_options::builder(cudf::io::source_info{path}).build();
  return std::make_unique<TableWithMetadata>(cudf::io::read_orc(opts));
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
  return TBL(cudf::io::read_json(builder.build()).tbl);
}

std::unique_ptr<TableWithMetadata> read_json_meta(rust::Str filepath, bool json_lines) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::json_reader_options::builder(cudf::io::source_info{path});
  builder.lines(json_lines);
  return std::make_unique<TableWithMetadata>(cudf::io::read_json(builder.build()));
}

void write_json(Table const& tbl, rust::Str filepath, bool json_lines) {
  std::string path(filepath.data(), filepath.size());
  auto builder = cudf::io::json_writer_options::builder(
      cudf::io::sink_info{path}, tbl.cached_view());
  builder.lines(json_lines);
  cudf::io::write_json(builder.build());
}

// -- Avro I/O --

std::unique_ptr<Table> read_avro(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::avro_reader_options::builder(cudf::io::source_info{path}).build();
  return TBL(cudf::io::read_avro(opts).tbl);
}

std::unique_ptr<TableWithMetadata> read_avro_meta(rust::Str filepath) {
  std::string path(filepath.data(), filepath.size());
  auto opts = cudf::io::avro_reader_options::builder(cudf::io::source_info{path}).build();
  return std::make_unique<TableWithMetadata>(cudf::io::read_avro(opts));
}

// -- DLPack interop --

std::unique_ptr<Table> from_dlpack(std::size_t managed_tensor_ptr, std::size_t stream) {
  auto* tensor = reinterpret_cast<DLManagedTensor const*>(managed_tensor_ptr);
  return TBL(cudf::from_dlpack(tensor, S(stream)));
}

std::size_t to_dlpack(Table const& tbl, std::size_t stream) {
  return reinterpret_cast<std::size_t>(cudf::to_dlpack(tbl.cached_view(), S(stream)));
}

}  // namespace cudf_sys
