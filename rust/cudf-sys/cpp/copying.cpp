// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/copying.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>

namespace cudf_sys {

// -- Gather --

std::unique_ptr<Table> gather_table(
    Table const& tbl,
    cudf::column_view const& indices,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::gather(tbl.cached_view(), indices,
      cudf::out_of_bounds_policy::DONT_CHECK, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> gather_table_checked(
    Table const& tbl,
    cudf::column_view const& gather_map,
    bool nullify_oob,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto policy = nullify_oob ? cudf::out_of_bounds_policy::NULLIFY
                            : cudf::out_of_bounds_policy::DONT_CHECK;
  auto result = cudf::gather(tbl.cached_view(), gather_map, policy, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Empty / Allocate --

std::unique_ptr<Column> empty_like_column(cudf::column_view const& col) {
  auto result = cudf::empty_like(col);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> empty_like_table(Table const& tbl) {
  auto result = cudf::empty_like(tbl.cached_view());
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> allocate_like_column(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::allocate_like(col, cudf::mask_allocation_policy::RETAIN, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Scatter --

std::unique_ptr<Table> scatter_table(Table const& source, cudf::column_view const& scatter_map, Table const& target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::scatter(source.cached_view(), scatter_map, target.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> scatter_scalars(ScalarList& sources, cudf::column_view const& indices, Table const& target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto refs = sources.refs();
  auto result = cudf::scatter(refs, indices, target.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> boolean_mask_scatter_table(Table const& source, Table const& target, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::boolean_mask_scatter(source.cached_view(), target.cached_view(), mask, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> boolean_mask_scatter_scalars(ScalarList& sources, Table const& target, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto refs = sources.refs();
  auto result = cudf::boolean_mask_scatter(refs, target.cached_view(), mask, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Reverse --

std::unique_ptr<Column> reverse_column(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  // cudf::reverse takes a table_view, so wrap single column
  cudf::table_view tv{{col}};
  auto result = cudf::reverse(tv, s);
  auto cols = result->release();
  return std::make_unique<Column>(std::move(cols[0]));
}

std::unique_ptr<Table> reverse_table(Table const& tbl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::reverse(tbl.cached_view(), s);
  return std::make_unique<Table>(std::move(result));
}

// -- Shift --

std::unique_ptr<Column> shift_column(cudf::column_view const& col, int32_t offset, Scalar const& fill_value, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::shift(col, offset, fill_value.inner(), s);
  return std::make_unique<Column>(std::move(result));
}

// -- Get element --

std::unique_ptr<Scalar> get_element(cudf::column_view const& col, int32_t index, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::get_element(col, index, s);
  return std::make_unique<Scalar>(std::move(result));
}

// -- Copy-if-else --

std::unique_ptr<Column> copy_if_else_columns(cudf::column_view const& lhs, cudf::column_view const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs, rhs, mask, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_scalar_column(Scalar const& lhs, cudf::column_view const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs.inner(), rhs, mask, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_column_scalar(cudf::column_view const& lhs, Scalar const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs, rhs.inner(), mask, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> copy_if_else_scalars(Scalar const& lhs, Scalar const& rhs, cudf::column_view const& mask, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_if_else(lhs.inner(), rhs.inner(), mask, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Slice --

std::unique_ptr<Column> slice_column(cudf::column_view const& col, int32_t begin, int32_t end, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> indices = {begin, end};
  auto views = cudf::slice(col, indices);
  // Copy the view into an owned column
  auto result = std::make_unique<cudf::column>(views[0], s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> slice_table(Table const& tbl, int32_t begin, int32_t end, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  std::vector<cudf::size_type> indices = {begin, end};
  auto views = cudf::slice(tbl.cached_view(), indices);
  // Copy the view into an owned table
  auto result = std::make_unique<cudf::table>(views[0], s);
  return std::make_unique<Table>(std::move(result));
}

// -- Sample --

std::unique_ptr<Table> sample_table(Table const& tbl, int32_t n, bool with_replacement, int64_t seed, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto policy = with_replacement ? cudf::sample_with_replacement::TRUE : cudf::sample_with_replacement::FALSE;
  auto result = cudf::sample(tbl.cached_view(), n, policy, seed, s);
  return std::make_unique<Table>(std::move(result));
}

// -- Copy range --

std::unique_ptr<Column> copy_range(
    cudf::column_view const& source,
    cudf::column_view const& target,
    int32_t source_begin,
    int32_t source_end,
    int32_t target_begin,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::copy_range(source, target, source_begin, source_end, target_begin, s);
  return std::make_unique<Column>(std::move(result));
}

// -- Null inspection --

bool has_nonempty_nulls(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  return cudf::has_nonempty_nulls(col, s);
}

bool may_have_nonempty_nulls(cudf::column_view const& col) {
  return cudf::may_have_nonempty_nulls(col);
}

std::unique_ptr<Column> purge_nonempty_nulls(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::purge_nonempty_nulls(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- In-place operations --

void fill_in_place(Column& col, int32_t begin, int32_t end, Scalar const& value, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto mcv = col.mutable_inner().mutable_view();
  cudf::fill_in_place(mcv, begin, end, value.inner(), s);
}

void copy_range_in_place(Column& dest, cudf::column_view const& source, int32_t source_begin, int32_t source_end, int32_t dest_begin, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto mcv = dest.mutable_inner().mutable_view();
  cudf::copy_range_in_place(source, mcv, source_begin, source_end, dest_begin, s);
}

}  // namespace cudf_sys
