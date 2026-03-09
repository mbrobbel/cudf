// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "cudf-sys/src/strings.rs.h"
#include "cudf-sys/src/lib.rs.h"
#include "cudf-sys/helpers.hpp"

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

// Convenience: rust::Str → std::string
static std::string STR(rust::Str s) { return {s.data(), s.size()}; }

// -- String case operations --

std::unique_ptr<Column> strings_to_lower(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::to_lower(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_to_upper(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::to_upper(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_swapcase(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::swapcase(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_capitalize(cudf::column_view const& col, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::capitalize(cudf::strings_column_view(col), cudf::string_scalar("", true, s), s));
}
std::unique_ptr<Column> strings_title(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::title(cudf::strings_column_view(col), cudf::strings::string_character_types::ALPHA, S(stream)));
}
std::unique_ptr<Column> strings_is_title(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::is_title(cudf::strings_column_view(col), S(stream)));
}

// -- String find / contains --

std::unique_ptr<Column> strings_contains(cudf::column_view const& col, Scalar const& target, std::size_t stream) {
  return COL(cudf::strings::contains(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(target.inner()), S(stream)));
}
std::unique_ptr<Column> strings_starts_with(cudf::column_view const& col, Scalar const& target, std::size_t stream) {
  return COL(cudf::strings::starts_with(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(target.inner()), S(stream)));
}
std::unique_ptr<Column> strings_ends_with(cudf::column_view const& col, Scalar const& target, std::size_t stream) {
  return COL(cudf::strings::ends_with(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(target.inner()), S(stream)));
}
std::unique_ptr<Column> strings_find(cudf::column_view const& col, Scalar const& target, std::size_t stream) {
  return COL(cudf::strings::find(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(target.inner()), 0, -1, S(stream)));
}
std::unique_ptr<Column> strings_contains_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::contains(cudf::strings_column_view(col), cudf::string_scalar(STR(target), true, s), s));
}
std::unique_ptr<Column> strings_starts_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::starts_with(cudf::strings_column_view(col), cudf::string_scalar(STR(target), true, s), s));
}
std::unique_ptr<Column> strings_ends_with_str(cudf::column_view const& col, rust::Str target, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::ends_with(cudf::strings_column_view(col), cudf::string_scalar(STR(target), true, s), s));
}
std::unique_ptr<Column> strings_find_str(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::find(cudf::strings_column_view(col), cudf::string_scalar(STR(target), true, s), start, stop, s));
}
std::unique_ptr<Column> strings_rfind(cudf::column_view const& col, rust::Str target, int32_t start, int32_t stop, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::rfind(cudf::strings_column_view(col), cudf::string_scalar(STR(target), true, s), start, stop, s));
}
std::unique_ptr<Table> strings_contains_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream) {
  return TBL(cudf::strings::contains_multiple(cudf::strings_column_view(col), cudf::strings_column_view(targets), S(stream)));
}
std::unique_ptr<Column> strings_find_multiple(cudf::column_view const& col, cudf::column_view const& targets, std::size_t stream) {
  return COL(cudf::strings::find_multiple(cudf::strings_column_view(col), cudf::strings_column_view(targets), S(stream)));
}
std::unique_ptr<Column> strings_find_instance(cudf::column_view const& col, rust::Str target, int32_t instance, std::size_t stream) {
  cudf::string_scalar tgt(STR(target));
  return COL(cudf::strings::find_instance(cudf::strings_column_view(col), tgt, instance, S(stream)));
}

// -- String replace --

std::unique_ptr<Column> strings_replace(cudf::column_view const& col, Scalar const& target, Scalar const& replacement, std::size_t stream) {
  return COL(cudf::strings::replace(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(target.inner()),
      static_cast<cudf::string_scalar const&>(replacement.inner()), -1, S(stream)));
}
std::unique_ptr<Column> strings_replace_literal(cudf::column_view const& col, rust::Str target, rust::Str repl, int32_t maxrepl, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::replace(cudf::strings_column_view(col),
      cudf::string_scalar(STR(target), true, s), cudf::string_scalar(STR(repl), true, s), maxrepl, s));
}
std::unique_ptr<Column> strings_replace_slice(cudf::column_view const& col, rust::Str repl, int32_t start, int32_t stop, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::replace_slice(cudf::strings_column_view(col),
      cudf::string_scalar(STR(repl), true, s), start, stop, s));
}
std::unique_ptr<Column> strings_replace_multiple(cudf::column_view const& col, cudf::column_view const& targets, cudf::column_view const& repls, std::size_t stream) {
  return COL(cudf::strings::replace_multiple(cudf::strings_column_view(col),
      cudf::strings_column_view(targets), cudf::strings_column_view(repls), S(stream)));
}

// -- String strip / pad --

std::unique_ptr<Column> strings_strip(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::strip(cudf::strings_column_view(col),
      cudf::strings::side_type::BOTH, cudf::string_scalar("", true), S(stream)));
}
std::unique_ptr<Column> strings_lstrip(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::strip(cudf::strings_column_view(col),
      cudf::strings::side_type::LEFT, cudf::string_scalar("", true), S(stream)));
}
std::unique_ptr<Column> strings_rstrip(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::strip(cudf::strings_column_view(col),
      cudf::strings::side_type::RIGHT, cudf::string_scalar("", true), S(stream)));
}
std::unique_ptr<Column> strings_strip_chars(cudf::column_view const& col, int32_t side, rust::Str to_strip, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::strip(cudf::strings_column_view(col),
      ENUM<cudf::strings::side_type>(side), cudf::string_scalar(STR(to_strip), true, s), s));
}
std::unique_ptr<Column> strings_pad(cudf::column_view const& col, int32_t width, int32_t side, rust::Str fill_char, std::size_t stream) {
  return COL(cudf::strings::pad(cudf::strings_column_view(col), width,
      ENUM<cudf::strings::side_type>(side), STR(fill_char), S(stream)));
}
std::unique_ptr<Column> strings_zfill(cudf::column_view const& col, int32_t width, std::size_t stream) {
  return COL(cudf::strings::zfill(cudf::strings_column_view(col), width, S(stream)));
}
std::unique_ptr<Column> strings_zfill_by_widths(cudf::column_view const& col, cudf::column_view const& widths, std::size_t stream) {
  return COL(cudf::strings::zfill_by_widths(cudf::strings_column_view(col), widths, S(stream)));
}

// -- String attributes --

std::unique_ptr<Column> strings_count_characters(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::count_characters(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_count_bytes(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::count_bytes(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_code_points(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::code_points(cudf::strings_column_view(col), S(stream)));
}

// -- String character types --

std::unique_ptr<Column> strings_all_characters_of_type(cudf::column_view const& col, uint32_t types, uint32_t verify_types, std::size_t stream) {
  return COL(cudf::strings::all_characters_of_type(cudf::strings_column_view(col),
      static_cast<cudf::strings::string_character_types>(types),
      static_cast<cudf::strings::string_character_types>(verify_types), S(stream)));
}
std::unique_ptr<Column> strings_filter_characters_of_type(cudf::column_view const& col, uint32_t types_to_remove, rust::Str replacement, uint32_t types_to_keep, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::filter_characters_of_type(cudf::strings_column_view(col),
      static_cast<cudf::strings::string_character_types>(types_to_remove),
      cudf::string_scalar(STR(replacement), true, s),
      static_cast<cudf::strings::string_character_types>(types_to_keep), s));
}

// -- String slice / substring --

std::unique_ptr<Column> strings_slice(cudf::column_view const& col, int32_t start, int32_t stop, int32_t step, std::size_t stream) {
  auto s = S(stream);
  auto start_scalar = cudf::numeric_scalar<cudf::size_type>(start, true, s);
  auto stop_scalar = cudf::numeric_scalar<cudf::size_type>(stop, true, s);
  auto step_scalar = cudf::numeric_scalar<cudf::size_type>(step, true, s);
  return COL(cudf::strings::slice_strings(cudf::strings_column_view(col), start_scalar, stop_scalar, step_scalar, s));
}
std::unique_ptr<Column> strings_slice_column(cudf::column_view const& col, cudf::column_view const& starts, cudf::column_view const& stops, std::size_t stream) {
  return COL(cudf::strings::slice_strings(cudf::strings_column_view(col), starts, stops, S(stream)));
}

// -- String repeat --

std::unique_ptr<Column> strings_repeat(cudf::column_view const& col, int32_t repeat_times, std::size_t stream) {
  return COL(cudf::strings::repeat_strings(cudf::strings_column_view(col), repeat_times, S(stream)));
}
std::unique_ptr<Column> strings_repeat_column(cudf::column_view const& col, cudf::column_view const& repeat_times, std::size_t stream) {
  return COL(cudf::strings::repeat_strings(cudf::strings_column_view(col), repeat_times, S(stream)));
}
std::unique_ptr<Scalar> repeat_string_scalar(Scalar const& input, int32_t repeat_times, std::size_t stream) {
  return std::make_unique<Scalar>(cudf::strings::repeat_string(
      static_cast<cudf::string_scalar const&>(input.inner()), repeat_times, S(stream)));
}

// -- String reverse / wrap --

std::unique_ptr<Column> strings_reverse(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::reverse(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_wrap(cudf::column_view const& col, int32_t width, std::size_t stream) {
  return COL(cudf::strings::wrap(cudf::strings_column_view(col), width, S(stream)));
}

// -- String split --

std::unique_ptr<Table> strings_split_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  return TBL(cudf::strings::split(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(delimiter.inner()), maxsplit, S(stream)));
}
std::unique_ptr<Table> strings_rsplit_to_table(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  return TBL(cudf::strings::rsplit(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(delimiter.inner()), maxsplit, S(stream)));
}
std::unique_ptr<Column> strings_split_part(cudf::column_view const& col, Scalar const& delimiter, int32_t index, std::size_t stream) {
  return COL(cudf::strings::split_part(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(delimiter.inner()), index, S(stream)));
}
std::unique_ptr<Column> strings_split_record(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  return COL(cudf::strings::split_record(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(delimiter.inner()), maxsplit, S(stream)));
}
std::unique_ptr<Column> strings_rsplit_record(cudf::column_view const& col, Scalar const& delimiter, int32_t maxsplit, std::size_t stream) {
  return COL(cudf::strings::rsplit_record(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(delimiter.inner()), maxsplit, S(stream)));
}
std::unique_ptr<Table> strings_partition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream) {
  auto s = S(stream);
  return TBL(cudf::strings::partition(cudf::strings_column_view(col),
      cudf::string_scalar(STR(delimiter), true, s), s));
}
std::unique_ptr<Table> strings_rpartition(cudf::column_view const& col, rust::Str delimiter, std::size_t stream) {
  auto s = S(stream);
  return TBL(cudf::strings::rpartition(cudf::strings_column_view(col),
      cudf::string_scalar(STR(delimiter), true, s), s));
}

// -- String join / concatenate --

std::unique_ptr<Column> strings_join(cudf::column_view const& col, Scalar const& separator, Scalar const& narep, std::size_t stream) {
  return COL(cudf::strings::join_strings(cudf::strings_column_view(col),
      static_cast<cudf::string_scalar const&>(separator.inner()),
      static_cast<cudf::string_scalar const&>(narep.inner()), S(stream)));
}
std::unique_ptr<Column> strings_join_strings(cudf::column_view const& col, rust::Str separator, rust::Str narep, std::size_t stream) {
  cudf::string_scalar sep(STR(separator));
  cudf::string_scalar na(STR(narep));
  return COL(cudf::strings::join_strings(cudf::strings_column_view(col), sep, na, S(stream)));
}
std::unique_ptr<Column> strings_concatenate_columns(Table const& tbl, Scalar const& separator, Scalar const& narep, std::size_t stream) {
  return COL(cudf::strings::concatenate(tbl.cached_view(),
      static_cast<cudf::string_scalar const&>(separator.inner()),
      static_cast<cudf::string_scalar const&>(narep.inner()),
      cudf::strings::separator_on_nulls::YES, S(stream)));
}
std::unique_ptr<Column> strings_concatenate_columns_sep_col(
    Table const& tbl, cudf::column_view const& separators,
    rust::Str separator_narep, rust::Str col_narep, std::size_t stream) {
  auto s = S(stream);
  auto sep_na_str = STR(separator_narep);
  auto col_na_str = STR(col_narep);
  cudf::string_scalar sep_na(sep_na_str, !sep_na_str.empty(), s);
  cudf::string_scalar col_na(col_na_str, !col_na_str.empty(), s);
  return COL(cudf::strings::concatenate(tbl.cached_view(),
      cudf::strings_column_view(separators), sep_na, col_na,
      cudf::strings::separator_on_nulls::YES, s));
}
std::unique_ptr<Column> strings_join_list_elements(
    cudf::column_view const& col, rust::Str separator, rust::Str narep, std::size_t stream) {
  auto s = S(stream);
  auto na_str = STR(narep);
  cudf::string_scalar sep_scalar(STR(separator), true, s);
  cudf::string_scalar na_scalar(na_str, !na_str.empty(), s);
  return COL(cudf::strings::join_list_elements(cudf::lists_column_view(col), sep_scalar, na_scalar,
      cudf::strings::separator_on_nulls::YES,
      cudf::strings::output_if_empty_list::EMPTY_STRING, s));
}
std::unique_ptr<Column> strings_join_list_elements_column(
    cudf::column_view const& col, cudf::column_view const& separators,
    rust::Str separator_narep, rust::Str string_narep, std::size_t stream) {
  auto s = S(stream);
  auto sep_na_str = STR(separator_narep);
  auto str_na_str = STR(string_narep);
  cudf::string_scalar sep_na(sep_na_str, !sep_na_str.empty(), s);
  cudf::string_scalar str_na(str_na_str, !str_na_str.empty(), s);
  return COL(cudf::strings::join_list_elements(cudf::lists_column_view(col),
      cudf::strings_column_view(separators), sep_na, str_na,
      cudf::strings::separator_on_nulls::YES,
      cudf::strings::output_if_empty_list::EMPTY_STRING, s));
}

// -- String regex operations --

std::unique_ptr<Column> strings_like(cudf::column_view const& col, rust::Str pattern, rust::Str escape_char, std::size_t stream) {
  return COL(cudf::strings::like(cudf::strings_column_view(col),
      std::string_view(pattern.data(), pattern.size()),
      std::string_view(escape_char.data(), escape_char.size()), S(stream)));
}
std::unique_ptr<Column> strings_like_column(cudf::column_view const& col, cudf::column_view const& patterns, rust::Str escape_char, std::size_t stream) {
  cudf::string_scalar esc(STR(escape_char));
  return COL(cudf::strings::like(cudf::strings_column_view(col),
      cudf::strings_column_view(patterns), esc, S(stream)));
}
std::unique_ptr<Column> strings_contains_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::contains_re(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Column> strings_matches_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::matches_re(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Column> strings_count_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::count_re(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Column> strings_replace_re(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream) {
  auto s = S(stream);
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::replace_re(cudf::strings_column_view(col), *prog,
      cudf::string_scalar(STR(replacement), true, s), std::nullopt, s));
}
std::unique_ptr<Column> strings_replace_with_backrefs(cudf::column_view const& col, rust::Str pattern, rust::Str replacement, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::replace_with_backrefs(cudf::strings_column_view(col), *prog, STR(replacement), S(stream)));
}
std::unique_ptr<Table> strings_extract(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return TBL(cudf::strings::extract(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Column> strings_extract_single(cudf::column_view const& col, rust::Str pattern, int32_t group_index, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::extract_single(cudf::strings_column_view(col), *prog, group_index, S(stream)));
}
std::unique_ptr<Column> strings_extract_all_record(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::extract_all_record(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Column> strings_findall(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::findall(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Column> strings_find_re(cudf::column_view const& col, rust::Str pattern, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::find_re(cudf::strings_column_view(col), *prog, S(stream)));
}
std::unique_ptr<Table> strings_split_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return TBL(cudf::strings::split_re(cudf::strings_column_view(col), *prog, maxsplit, S(stream)));
}
std::unique_ptr<Table> strings_rsplit_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return TBL(cudf::strings::rsplit_re(cudf::strings_column_view(col), *prog, maxsplit, S(stream)));
}
std::unique_ptr<Column> strings_split_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::split_record_re(cudf::strings_column_view(col), *prog, maxsplit, S(stream)));
}
std::unique_ptr<Column> strings_rsplit_record_re(cudf::column_view const& col, rust::Str pattern, int32_t maxsplit, std::size_t stream) {
  auto prog = cudf::strings::regex_program::create(STR(pattern));
  return COL(cudf::strings::rsplit_record_re(cudf::strings_column_view(col), *prog, maxsplit, S(stream)));
}

// -- String conversions: integers / floats --

std::unique_ptr<Column> strings_from_integers(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::from_integers(col, S(stream)));
}
std::unique_ptr<Column> strings_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  return COL(cudf::strings::to_integers(cudf::strings_column_view(col), DT(output_type_id), S(stream)));
}
std::unique_ptr<Column> strings_from_floats(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::from_floats(col, S(stream)));
}
std::unique_ptr<Column> strings_to_floats(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  return COL(cudf::strings::to_floats(cudf::strings_column_view(col), DT(output_type_id), S(stream)));
}

// -- String conversions: timestamps --

std::unique_ptr<Column> strings_to_timestamps(cudf::column_view const& col, int32_t timestamp_type_id, rust::Str format, std::size_t stream) {
  return COL(cudf::strings::to_timestamps(cudf::strings_column_view(col), DT(timestamp_type_id), STR(format), S(stream)));
}
std::unique_ptr<Column> strings_from_timestamps(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::from_timestamps(col, STR(format),
      cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), s));
}
std::unique_ptr<Column> strings_is_timestamp(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  return COL(cudf::strings::is_timestamp(cudf::strings_column_view(col), STR(format), S(stream)));
}

// -- String conversions: booleans --

std::unique_ptr<Column> strings_to_booleans(cudf::column_view const& col, rust::Str true_string, std::size_t stream) {
  auto s = S(stream);
  return COL(cudf::strings::to_booleans(cudf::strings_column_view(col),
      cudf::string_scalar(STR(true_string), true, s), s));
}
std::unique_ptr<Column> strings_from_booleans(cudf::column_view const& col, rust::Str true_string, rust::Str false_string, std::size_t stream) {
  auto s = S(stream);
  cudf::string_scalar true_scalar(STR(true_string), true, s);
  cudf::string_scalar false_scalar(STR(false_string), true, s);
  return COL(cudf::strings::from_booleans(col, true_scalar, false_scalar, s));
}

// -- String conversions: durations --

std::unique_ptr<Column> strings_to_durations(cudf::column_view const& col, int32_t duration_type_id, rust::Str format, std::size_t stream) {
  return COL(cudf::strings::to_durations(cudf::strings_column_view(col), DT(duration_type_id), STR(format), S(stream)));
}
std::unique_ptr<Column> strings_from_durations(cudf::column_view const& col, rust::Str format, std::size_t stream) {
  return COL(cudf::strings::from_durations(col, STR(format), S(stream)));
}

// -- String conversions: fixed-point --

std::unique_ptr<Column> strings_to_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream) {
  return COL(cudf::strings::to_fixed_point(cudf::strings_column_view(col),
      cudf::data_type{static_cast<cudf::type_id>(type_id), scale}, S(stream)));
}
std::unique_ptr<Column> strings_from_fixed_point(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::from_fixed_point(col, S(stream)));
}
std::unique_ptr<Column> strings_is_fixed_point(cudf::column_view const& col, int32_t type_id, int32_t scale, std::size_t stream) {
  return COL(cudf::strings::is_fixed_point(cudf::strings_column_view(col),
      cudf::data_type{static_cast<cudf::type_id>(type_id), scale}, S(stream)));
}

// -- String conversions: URLs --

std::unique_ptr<Column> strings_url_encode(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::url_encode(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_url_decode(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::url_decode(cudf::strings_column_view(col), S(stream)));
}

// -- String conversions: IPv4 --

std::unique_ptr<Column> strings_ipv4_to_integers(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::ipv4_to_integers(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_integers_to_ipv4(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::integers_to_ipv4(col, S(stream)));
}
std::unique_ptr<Column> strings_is_ipv4(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::is_ipv4(cudf::strings_column_view(col), S(stream)));
}

// -- String validation --

std::unique_ptr<Column> strings_is_integer(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::is_integer(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_is_integer_with_type(cudf::column_view const& col, int32_t int_type_id, std::size_t stream) {
  return COL(cudf::strings::is_integer(cudf::strings_column_view(col), DT(int_type_id), S(stream)));
}
std::unique_ptr<Column> strings_is_float(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::is_float(cudf::strings_column_view(col), S(stream)));
}

// -- String conversions: hex --

std::unique_ptr<Column> strings_hex_to_integers(cudf::column_view const& col, int32_t output_type_id, std::size_t stream) {
  return COL(cudf::strings::hex_to_integers(cudf::strings_column_view(col), DT(output_type_id), S(stream)));
}
std::unique_ptr<Column> strings_is_hex(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::is_hex(cudf::strings_column_view(col), S(stream)));
}
std::unique_ptr<Column> strings_integers_to_hex(cudf::column_view const& col, std::size_t stream) {
  return COL(cudf::strings::integers_to_hex(col, S(stream)));
}

// -- String conversions: integer binary cast --

std::unique_ptr<Column> strings_cast_to_integer(cudf::column_view const& col, int32_t output_type_id, bool big_endian, std::size_t stream) {
  auto swap = big_endian ? cudf::strings::endian::BIG : cudf::strings::endian::LITTLE;
  return COL(cudf::strings::cast_to_integer(cudf::strings_column_view(col), DT(output_type_id), swap, S(stream)));
}
std::unique_ptr<Column> strings_cast_from_integer(cudf::column_view const& col, bool big_endian, std::size_t stream) {
  auto swap = big_endian ? cudf::strings::endian::BIG : cudf::strings::endian::LITTLE;
  return COL(cudf::strings::cast_from_integer(col, swap, S(stream)));
}

// -- String translate / filter --

std::unique_ptr<Column> strings_translate(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    std::size_t stream) {
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> table;
  table.reserve(from_chars.size());
  for (size_t i = 0; i < from_chars.size(); ++i) {
    table.emplace_back(static_cast<cudf::char_utf8>(from_chars[i]),
                       static_cast<cudf::char_utf8>(to_chars[i]));
  }
  return COL(cudf::strings::translate(cudf::strings_column_view(col), table, S(stream)));
}
std::unique_ptr<Column> strings_filter_characters(
    cudf::column_view const& col,
    rust::Slice<uint32_t const> from_chars,
    rust::Slice<uint32_t const> to_chars,
    bool keep,
    rust::Str replacement,
    std::size_t stream) {
  auto s = S(stream);
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> ranges;
  ranges.reserve(from_chars.size());
  for (size_t i = 0; i < from_chars.size(); ++i) {
    ranges.emplace_back(static_cast<cudf::char_utf8>(from_chars[i]),
                        static_cast<cudf::char_utf8>(to_chars[i]));
  }
  cudf::string_scalar repl_scalar(STR(replacement), true, s);
  auto ft = keep ? cudf::strings::filter_type::KEEP : cudf::strings::filter_type::REMOVE;
  return COL(cudf::strings::filter_characters(cudf::strings_column_view(col), ranges, ft, repl_scalar, s));
}

// -- String formatting --

std::unique_ptr<Column> strings_format_list_column(cudf::column_view const& col, rust::Str na_rep, std::size_t stream) {
  auto s = S(stream);
  cudf::string_scalar na_scalar(STR(na_rep), true, s);
  return COL(cudf::strings::format_list_column(cudf::lists_column_view(col), na_scalar,
      cudf::strings_column_view(cudf::column_view{cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}), s));
}

// -- String column construction / extraction --

std::unique_ptr<Column> make_string_column(rust::Vec<rust::String> strings, std::size_t stream) {
  auto s = S(stream);
  if (strings.empty()) {
    return COL(cudf::make_empty_column(cudf::type_id::STRING));
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

  return COL(cudf::concatenate(views, s));
}

static rust::Vec<rust::String> strings_to_host(cudf::column_view const& view, std::size_t stream) {
  auto s = S(stream);
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
    result.push_back(rust::String(std::string(host_chars.data() + start, end - start)));
  }
  return result;
}

rust::Vec<rust::String> column_to_host_strings(Column const& col, std::size_t stream) {
  return strings_to_host(col.cached_view(), stream);
}

rust::Vec<rust::String> view_to_host_strings(cudf::column_view const& col, std::size_t stream) {
  return strings_to_host(col, stream);
}

// -- JSON path extraction --

std::unique_ptr<Column> get_json_object(
    cudf::column_view const& col,
    rust::Str json_path,
    bool allow_single_quotes,
    bool strip_quotes,
    bool missing_fields_as_nulls,
    std::size_t stream) {
  auto s = S(stream);
  cudf::string_scalar path_scalar(STR(json_path), true, s);
  cudf::get_json_object_options opts;
  opts.set_allow_single_quotes(allow_single_quotes);
  opts.set_strip_quotes_from_single_strings(strip_quotes);
  opts.set_missing_fields_as_nulls(missing_fields_as_nulls);
  return COL(cudf::get_json_object(cudf::strings_column_view(col), path_scalar, opts, s));
}

}  // namespace cudf_sys
