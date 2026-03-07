// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/strings.rs.h"
#include "cudf-sys/src/lib.rs.h"

#include <cudf/strings/attributes.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/convert/convert_lists.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/convert/int_cast.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/strings/wrap.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cudf_sys {

// -- String case operations --

std::unique_ptr<Column> strings_to_lower(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_lower(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_upper(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_upper(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_swapcase(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::swapcase(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_capitalize(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::capitalize(scv, cudf::string_scalar("", true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_title(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::title(scv, cudf::strings::string_character_types::ALPHA, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_title(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_title(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String find / contains --

std::unique_ptr<Column> strings_contains(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::contains(scv, str_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_starts_with(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::starts_with(scv, str_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_ends_with(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::ends_with(scv, str_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find(
    cudf::column_view const& col,
    Scalar const& target,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(target.inner());
  auto result = cudf::strings::find(scv, str_scalar, 0, -1, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_contains_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::contains(scv, cudf::string_scalar(tgt, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_starts_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::starts_with(scv, cudf::string_scalar(tgt, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_ends_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::ends_with(scv, cudf::string_scalar(tgt, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find_str(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::find(scv, cudf::string_scalar(tgt, true, s), start, stop, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rfind(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto result = cudf::strings::rfind(scv, cudf::string_scalar(tgt, true, s), start, stop, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_contains_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view tgts(targets);
  auto result = cudf::strings::contains_multiple(scv, tgts, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_find_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view tgts(targets);
  auto result = cudf::strings::find_multiple(scv, tgts, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find_instance(cudf::column_view const& col, rust::Str target, int32_t instance, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::string target_str(target.data(), target.size());
  cudf::string_scalar tgt(target_str);
  auto result = cudf::strings::find_instance(scv, tgt, instance, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String replace --

std::unique_ptr<Column> strings_replace(
    cudf::column_view const& col,
    Scalar const& target,
    Scalar const& replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& tgt = static_cast<cudf::string_scalar const&>(target.inner());
  auto const& repl = static_cast<cudf::string_scalar const&>(replacement.inner());
  auto result = cudf::strings::replace(scv, tgt, repl, -1, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_literal(cudf::column_view const& col, rust::Str target, rust::Str repl, int32_t maxrepl, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto tgt = std::string(target.data(), target.size());
  auto rep = std::string(repl.data(), repl.size());
  auto result = cudf::strings::replace(scv, cudf::string_scalar(tgt, true, s), cudf::string_scalar(rep, true, s), maxrepl, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_slice(
    cudf::column_view const& col,
    rust::Str repl,
    int32_t start,
    int32_t stop,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto repl_str = std::string(repl.data(), repl.size());
  cudf::string_scalar repl_scalar(repl_str, true, s);
  auto result = cudf::strings::replace_slice(scv, repl_scalar, start, stop, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_multiple(
    cudf::column_view const& col,
    cudf::column_view const& targets,
    cudf::column_view const& repls,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view tcv(targets);
  cudf::strings_column_view rcv(repls);
  auto result = cudf::strings::replace_multiple(scv, tcv, rcv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String strip / pad --

std::unique_ptr<Column> strings_strip(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::strip(scv, cudf::strings::side_type::BOTH,
      cudf::string_scalar("", true), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_lstrip(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::strip(scv, cudf::strings::side_type::LEFT,
      cudf::string_scalar("", true), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rstrip(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::strip(scv, cudf::strings::side_type::RIGHT,
      cudf::string_scalar("", true), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_strip_chars(cudf::column_view const& col, int32_t side, rust::Str to_strip, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto chars = std::string(to_strip.data(), to_strip.size());
  auto result = cudf::strings::strip(scv, static_cast<cudf::strings::side_type>(side), cudf::string_scalar(chars, true, s), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_pad(cudf::column_view const& col, int32_t width, int32_t side, rust::Str fill_char, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::string fc(fill_char.data(), fill_char.size());
  auto result = cudf::strings::pad(scv, width,
      static_cast<cudf::strings::side_type>(side), fc, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_zfill(cudf::column_view const& col, int32_t width, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::zfill(scv, width, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_zfill_by_widths(
    cudf::column_view const& col,
    cudf::column_view const& widths,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::zfill_by_widths(scv, widths, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String attributes --

std::unique_ptr<Column> strings_count_characters(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::count_characters(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_count_bytes(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::count_bytes(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_code_points(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::code_points(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String character types --

std::unique_ptr<Column> strings_all_characters_of_type(cudf::column_view const& col, uint32_t types, uint32_t verify_types, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::all_characters_of_type(scv, static_cast<cudf::strings::string_character_types>(types), static_cast<cudf::strings::string_character_types>(verify_types), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_filter_characters_of_type(cudf::column_view const& col, uint32_t types_to_remove, rust::Str replacement, uint32_t types_to_keep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto repl = std::string(replacement.data(), replacement.size());
  cudf::string_scalar repl_scalar(repl, true, s);
  auto result = cudf::strings::filter_characters_of_type(
      scv,
      static_cast<cudf::strings::string_character_types>(types_to_remove),
      repl_scalar,
      static_cast<cudf::strings::string_character_types>(types_to_keep),
      s);
  return std::make_unique<Column>(std::move(result));
}

// -- String slice / substring --

std::unique_ptr<Column> strings_slice(cudf::column_view const& col, int32_t start, int32_t stop, int32_t step, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto start_scalar = cudf::numeric_scalar<cudf::size_type>(start, true, s);
  auto stop_scalar = cudf::numeric_scalar<cudf::size_type>(stop, true, s);
  auto step_scalar = cudf::numeric_scalar<cudf::size_type>(step, true, s);
  auto result = cudf::strings::slice_strings(scv, start_scalar, stop_scalar, step_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_slice_column(
    cudf::column_view const& col,
    cudf::column_view const& starts,
    cudf::column_view const& stops,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::slice_strings(scv, starts, stops, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String repeat --

std::unique_ptr<Column> strings_repeat(cudf::column_view const& col, int32_t repeat_times, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::repeat_strings(scv, repeat_times, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_repeat_column(cudf::column_view const& col, cudf::column_view const& repeat_times, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::repeat_strings(scv, repeat_times, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Scalar> repeat_string_scalar(Scalar const& input, int32_t repeat_times, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(input.inner());
  auto result = cudf::strings::repeat_string(str_scalar, repeat_times, s);
  return std::make_unique<Scalar>(std::move(result));
}

// -- String reverse / wrap --

std::unique_ptr<Column> strings_reverse(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::reverse(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_wrap(cudf::column_view const& col, int32_t width, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::wrap(scv, width, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String split --

std::unique_ptr<Table> strings_split_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::split(scv, str_scalar, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> strings_rsplit_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::rsplit(scv, str_scalar, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_split_part(cudf::column_view const& col, Scalar const& delimiter, int32_t index, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::split_part(scv, str_scalar, index, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_split_record(
    cudf::column_view const& col,
    Scalar const& delimiter,
    int32_t maxsplit,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::split_record(scv, str_scalar, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rsplit_record(
    cudf::column_view const& col,
    Scalar const& delimiter,
    int32_t maxsplit,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& str_scalar = static_cast<cudf::string_scalar const&>(delimiter.inner());
  auto result = cudf::strings::rsplit_record(scv, str_scalar, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_partition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto delim = std::string(delimiter.data(), delimiter.size());
  cudf::string_scalar delim_scalar(delim, true, s);
  auto result = cudf::strings::partition(scv, delim_scalar, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> strings_rpartition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto delim = std::string(delimiter.data(), delimiter.size());
  cudf::string_scalar delim_scalar(delim, true, s);
  auto result = cudf::strings::rpartition(scv, delim_scalar, s);
  return std::make_unique<Table>(std::move(result));
}

// -- String join / concatenate --

std::unique_ptr<Column> strings_join(cudf::column_view const& col, Scalar const& separator, Scalar const& narep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto const& sep = static_cast<cudf::string_scalar const&>(separator.inner());
  auto const& na = static_cast<cudf::string_scalar const&>(narep.inner());
  auto result = cudf::strings::join_strings(scv, sep, na, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_join_strings(cudf::column_view const& col, rust::Str separator, rust::Str narep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::string sep_str(separator.data(), separator.size());
  std::string narep_str(narep.data(), narep.size());
  cudf::string_scalar sep(sep_str);
  cudf::string_scalar na(narep_str);
  auto result = cudf::strings::join_strings(scv, sep, na, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_concatenate_columns(Table const& tbl, Scalar const& separator, Scalar const& narep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto const& sep = static_cast<cudf::string_scalar const&>(separator.inner());
  auto const& na = static_cast<cudf::string_scalar const&>(narep.inner());
  auto result = cudf::strings::concatenate(tbl.cached_view(), sep, na,
      cudf::strings::separator_on_nulls::YES, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_concatenate_columns_sep_col(
    Table const& tbl,
    cudf::column_view const& separators,
    rust::Str separator_narep,
    rust::Str col_narep,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view sep_col(separators);
  auto sep_na_str = std::string(separator_narep.data(), separator_narep.size());
  auto col_na_str = std::string(col_narep.data(), col_narep.size());
  cudf::string_scalar sep_na(sep_na_str, !sep_na_str.empty(), s);
  cudf::string_scalar col_na(col_na_str, !col_na_str.empty(), s);
  auto result = cudf::strings::concatenate(tbl.cached_view(), sep_col, sep_na, col_na,
      cudf::strings::separator_on_nulls::YES, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_join_list_elements(
    cudf::column_view const& col,
    rust::Str separator,
    rust::Str narep,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto sep_str = std::string(separator.data(), separator.size());
  auto na_str = std::string(narep.data(), narep.size());
  cudf::string_scalar sep_scalar(sep_str, true, s);
  cudf::string_scalar na_scalar(na_str, !na_str.empty(), s);
  auto result = cudf::strings::join_list_elements(lcv, sep_scalar, na_scalar,
      cudf::strings::separator_on_nulls::YES,
      cudf::strings::output_if_empty_list::EMPTY_STRING, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_join_list_elements_column(
    cudf::column_view const& col,
    cudf::column_view const& separators,
    rust::Str separator_narep,
    rust::Str string_narep,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  cudf::strings_column_view sep_col(separators);
  auto sep_na_str = std::string(separator_narep.data(), separator_narep.size());
  auto str_na_str = std::string(string_narep.data(), string_narep.size());
  cudf::string_scalar sep_na(sep_na_str, !sep_na_str.empty(), s);
  cudf::string_scalar str_na(str_na_str, !str_na_str.empty(), s);
  auto result = cudf::strings::join_list_elements(lcv, sep_col, sep_na, str_na,
      cudf::strings::separator_on_nulls::YES,
      cudf::strings::output_if_empty_list::EMPTY_STRING, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String regex operations --

std::unique_ptr<Column> strings_like(cudf::column_view const& col, rust::Str pattern, rust::Str escape_char, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string_view(pattern.data(), pattern.size());
  auto esc = std::string_view(escape_char.data(), escape_char.size());
  auto result = cudf::strings::like(scv, pat, esc, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_like_column(cudf::column_view const& col, cudf::column_view const& patterns, rust::Str escape_char, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  cudf::strings_column_view patterns_scv(patterns);
  std::string esc_str(escape_char.data(), escape_char.size());
  cudf::string_scalar esc(esc_str);
  auto result = cudf::strings::like(scv, patterns_scv, esc, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_contains_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::contains_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_matches_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::matches_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_count_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::count_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_re(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto repl = std::string(replacement.data(), replacement.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::replace_re(scv, *prog, cudf::string_scalar(repl, true, s), std::nullopt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_replace_with_backrefs(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto repl = std::string(replacement.data(), replacement.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::replace_with_backrefs(scv, *prog, repl, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_extract(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::extract(scv, *prog, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_extract_single(
    cudf::column_view const& col,
    rust::Str pattern,
    int32_t group_index,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::extract_single(scv, *prog, group_index, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_extract_all_record(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::extract_all_record(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_findall(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::findall(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_find_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::find_re(scv, *prog, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Table> strings_split_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::split_re(scv, *prog, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Table> strings_rsplit_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::rsplit_re(scv, *prog, maxsplit, s);
  return std::make_unique<Table>(std::move(result));
}

std::unique_ptr<Column> strings_split_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::split_record_re(scv, *prog, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_rsplit_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto pat = std::string(pattern.data(), pattern.size());
  auto prog = cudf::strings::regex_program::create(pat);
  auto result = cudf::strings::rsplit_record_re(scv, *prog, maxsplit, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: integers / floats --

std::unique_ptr<Column> strings_from_integers(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::from_integers(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_integers(scv,
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_floats(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::from_floats(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_to_floats(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_floats(scv,
      cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: timestamps --

std::unique_ptr<Column> strings_to_timestamps(cudf::column_view const& col, int32_t timestamp_type_id, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::to_timestamps(scv, cudf::data_type{static_cast<cudf::type_id>(timestamp_type_id)}, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_timestamps(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::from_timestamps(col, fmt, cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_timestamp(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::is_timestamp(scv, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: booleans --

std::unique_ptr<Column> strings_to_booleans(cudf::column_view const& col, rust::Str true_string, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto ts = std::string(true_string.data(), true_string.size());
  cudf::string_scalar true_scalar(ts, true, s);
  auto result = cudf::strings::to_booleans(scv, true_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_booleans(cudf::column_view const& col, rust::Str true_string, rust::Str false_string, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto ts = std::string(true_string.data(), true_string.size());
  auto fs = std::string(false_string.data(), false_string.size());
  cudf::string_scalar true_scalar(ts, true, s);
  cudf::string_scalar false_scalar(fs, true, s);
  auto result = cudf::strings::from_booleans(col, true_scalar, false_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: durations --

std::unique_ptr<Column> strings_to_durations(cudf::column_view const& col, int32_t duration_type_id, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::to_durations(scv, cudf::data_type{static_cast<cudf::type_id>(duration_type_id)}, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_durations(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto fmt = std::string(format.data(), format.size());
  auto result = cudf::strings::from_durations(col, fmt, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: fixed-point --

std::unique_ptr<Column> strings_to_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::to_fixed_point(scv, cudf::data_type{static_cast<cudf::type_id>(type_id), scale}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_from_fixed_point(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::from_fixed_point(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_fixed_point(scv, cudf::data_type{static_cast<cudf::type_id>(type_id), scale}, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: URLs --

std::unique_ptr<Column> strings_url_encode(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::url_encode(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_url_decode(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::url_decode(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: IPv4 --

std::unique_ptr<Column> strings_ipv4_to_integers(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::ipv4_to_integers(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_integers_to_ipv4(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::integers_to_ipv4(col, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_ipv4(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_ipv4(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String validation --

std::unique_ptr<Column> strings_is_integer(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_integer(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_integer_with_type(cudf::column_view const& col, int32_t int_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_integer(scv, cudf::data_type{static_cast<cudf::type_id>(int_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_float(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_float(scv, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: hex --

std::unique_ptr<Column> strings_hex_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::hex_to_integers(scv, cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_is_hex(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto result = cudf::strings::is_hex(scv, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_integers_to_hex(cudf::column_view const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto result = cudf::strings::integers_to_hex(col, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String conversions: integer binary cast --

std::unique_ptr<Column> strings_cast_to_integer(
    cudf::column_view const& col,
    int32_t output_type_id,
    bool big_endian,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto swap = big_endian ? cudf::strings::endian::BIG
                         : cudf::strings::endian::LITTLE;
  auto result = cudf::strings::cast_to_integer(
      scv, cudf::data_type{static_cast<cudf::type_id>(output_type_id)}, swap, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_cast_from_integer(
    cudf::column_view const& col,
    bool big_endian,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto swap = big_endian ? cudf::strings::endian::BIG
                         : cudf::strings::endian::LITTLE;
  auto result = cudf::strings::cast_from_integer(col, swap, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String translate / filter --

std::unique_ptr<Column> strings_translate(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> table;
  table.reserve(from_chars.size());
  for (size_t i = 0; i < from_chars.size(); ++i) {
    table.emplace_back(static_cast<cudf::char_utf8>(from_chars[i]),
                       static_cast<cudf::char_utf8>(to_chars[i]));
  }
  auto result = cudf::strings::translate(scv, table, s);
  return std::make_unique<Column>(std::move(result));
}

std::unique_ptr<Column> strings_filter_characters(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    bool keep,
    rust::Str replacement,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> ranges;
  ranges.reserve(from_chars.size());
  for (size_t i = 0; i < from_chars.size(); ++i) {
    ranges.emplace_back(static_cast<cudf::char_utf8>(from_chars[i]),
                        static_cast<cudf::char_utf8>(to_chars[i]));
  }
  auto repl = std::string(replacement.data(), replacement.size());
  cudf::string_scalar repl_scalar(repl, true, s);
  auto ft = keep ? cudf::strings::filter_type::KEEP
                 : cudf::strings::filter_type::REMOVE;
  auto result = cudf::strings::filter_characters(scv, ranges, ft, repl_scalar, s);
  return std::make_unique<Column>(std::move(result));
}

// -- String formatting --

std::unique_ptr<Column> strings_format_list_column(cudf::column_view const& col, rust::Str na_rep, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::lists_column_view lcv(col);
  auto na = std::string(na_rep.data(), na_rep.size());
  cudf::string_scalar na_scalar(na, true, s);
  auto result = cudf::strings::format_list_column(lcv, na_scalar, cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), s);
  return std::make_unique<Column>(std::move(result));
}

// -- String column construction / extraction --

std::unique_ptr<Column> make_string_column(rust::Vec<rust::String> strings, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  if (strings.empty()) {
    return std::make_unique<Column>(
        cudf::make_empty_column(cudf::type_id::STRING));
  }

  std::vector<cudf::column_view> views;
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(strings.size());

  for (auto const& str : strings) {
    auto sv = std::string_view(str.data(), str.size());
    auto scalar = cudf::string_scalar(sv, true, s);
    auto col = cudf::make_column_from_scalar(scalar, 1, s);
    views.push_back(col->view());
    cols.push_back(std::move(col));
  }

  auto result = cudf::concatenate(views, s);
  return std::make_unique<Column>(std::move(result));
}

rust::Vec<rust::String> column_to_host_strings(Column const& col, std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  auto view = col.cached_view();
  auto size = view.size();
  rust::Vec<rust::String> result;
  result.reserve(size);

  if (size == 0) return result;

  cudf::strings_column_view scv(view);

  // Get offsets to host
  auto offsets_view = scv.offsets();
  auto num_offsets = size + 1;
  auto offset_begin = scv.offset();
  std::vector<int32_t> host_offsets(num_offsets);
  cudaMemcpyAsync(host_offsets.data(),
             offsets_view.data<int32_t>() + offset_begin,
             num_offsets * sizeof(int32_t), cudaMemcpyDeviceToHost, s.value());
  s.synchronize();

  // Get chars to host
  auto chars_begin_ptr = scv.chars_begin(s);
  auto total_bytes = host_offsets[size] - host_offsets[0];
  std::vector<char> host_chars(total_bytes);
  if (total_bytes > 0) {
    cudaMemcpyAsync(host_chars.data(),
               chars_begin_ptr + host_offsets[0],
               total_bytes, cudaMemcpyDeviceToHost, s.value());
    s.synchronize();
  }

  auto base_offset = host_offsets[0];
  for (int32_t i = 0; i < size; ++i) {
    auto start = host_offsets[i] - base_offset;
    auto end = host_offsets[i + 1] - base_offset;
    auto str = std::string(host_chars.data() + start, end - start);
    result.push_back(rust::String(str));
  }
  return result;
}

// -- JSON path extraction --

std::unique_ptr<Column> get_json_object(
    cudf::column_view const& col,
    rust::Str json_path,
    bool allow_single_quotes,
    bool strip_quotes,
    bool missing_fields_as_nulls,
    std::size_t stream) {
  rmm::cuda_stream_view s{reinterpret_cast<cudaStream_t>(stream)};
  cudf::strings_column_view scv(col);
  auto path_str = std::string(json_path.data(), json_path.size());
  cudf::string_scalar path_scalar(path_str, true, s);
  cudf::get_json_object_options opts;
  opts.set_allow_single_quotes(allow_single_quotes);
  opts.set_strip_quotes_from_single_strings(strip_quotes);
  opts.set_missing_fields_as_nulls(missing_fields_as_nulls);
  auto result = cudf::get_json_object(scv, path_scalar, opts, s);
  return std::make_unique<Column>(std::move(result));
}

}  // namespace cudf_sys
