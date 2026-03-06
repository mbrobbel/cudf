// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! String operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;
use crate::table::Table;

pub use cudf_sys::ffi::SideType;

/// Extension trait for string operations on GPU columns.
pub trait StringExt {
    /// Converts each string to lower case.
    fn to_lower(&self) -> Result<Column>;
    /// to_lower on a custom CUDA stream.
    fn to_lower_on(&self, stream: Stream) -> Result<Column>;
    /// Converts each string to upper case.
    fn to_upper(&self) -> Result<Column>;
    /// to_upper on a custom CUDA stream.
    fn to_upper_on(&self, stream: Stream) -> Result<Column>;
    /// Returns a BOOL8 column indicating whether each string contains the target.
    fn str_contains(&self, target: &Scalar) -> Result<Column>;
    /// str_contains on a custom CUDA stream.
    fn str_contains_on(&self, target: &Scalar, stream: Stream) -> Result<Column>;
    /// Returns a BOOL8 column indicating whether each string starts with the target.
    fn str_starts_with(&self, target: &Scalar) -> Result<Column>;
    /// str_starts_with on a custom CUDA stream.
    fn str_starts_with_on(&self, target: &Scalar, stream: Stream) -> Result<Column>;
    /// Returns a BOOL8 column indicating whether each string ends with the target.
    fn str_ends_with(&self, target: &Scalar) -> Result<Column>;
    /// str_ends_with on a custom CUDA stream.
    fn str_ends_with_on(&self, target: &Scalar, stream: Stream) -> Result<Column>;
    /// Returns an INT32 column with the position of the first occurrence of the target.
    fn str_find(&self, target: &Scalar) -> Result<Column>;
    /// str_find on a custom CUDA stream.
    fn str_find_on(&self, target: &Scalar, stream: Stream) -> Result<Column>;
    /// Replaces all occurrences of `target` with `replacement`.
    fn str_replace(&self, target: &Scalar, replacement: &Scalar) -> Result<Column>;
    /// str_replace on a custom CUDA stream.
    fn str_replace_on(&self, target: &Scalar, replacement: &Scalar, stream: Stream) -> Result<Column>;
    /// Strips whitespace from both sides.
    fn str_strip(&self) -> Result<Column>;
    /// str_strip on a custom CUDA stream.
    fn str_strip_on(&self, stream: Stream) -> Result<Column>;
    /// Strips whitespace from the left side.
    fn str_lstrip(&self) -> Result<Column>;
    /// str_lstrip on a custom CUDA stream.
    fn str_lstrip_on(&self, stream: Stream) -> Result<Column>;
    /// Strips whitespace from the right side.
    fn str_rstrip(&self) -> Result<Column>;
    /// str_rstrip on a custom CUDA stream.
    fn str_rstrip_on(&self, stream: Stream) -> Result<Column>;
    /// Returns an INT32 column with the character count.
    fn count_characters(&self) -> Result<Column>;
    /// count_characters on a custom CUDA stream.
    fn count_characters_on(&self, stream: Stream) -> Result<Column>;
    /// Returns an INT32 column with the byte count.
    fn count_bytes(&self) -> Result<Column>;
    /// count_bytes on a custom CUDA stream.
    fn count_bytes_on(&self, stream: Stream) -> Result<Column>;
    /// Converts an integer column to a string column.
    fn str_from_integers(&self) -> Result<Column>;
    /// str_from_integers on a custom CUDA stream.
    fn str_from_integers_on(&self, stream: Stream) -> Result<Column>;
    /// Converts a string column to an integer column.
    fn str_to_integers(&self, output_type: TypeId) -> Result<Column>;
    /// str_to_integers on a custom CUDA stream.
    fn str_to_integers_on(&self, output_type: TypeId, stream: Stream) -> Result<Column>;
    /// Converts a float column to a string column.
    fn str_from_floats(&self) -> Result<Column>;
    /// str_from_floats on a custom CUDA stream.
    fn str_from_floats_on(&self, stream: Stream) -> Result<Column>;
    /// Converts a string column to a float column.
    fn str_to_floats(&self, output_type: TypeId) -> Result<Column>;
    /// str_to_floats on a custom CUDA stream.
    fn str_to_floats_on(&self, output_type: TypeId, stream: Stream) -> Result<Column>;

    // -- New string operations --

    /// Pads strings to a minimum width.
    fn str_pad(&self, width: usize, side: SideType, fill_char: &str) -> Result<Column>;
    /// Zero-fills strings to a minimum width.
    fn str_zfill(&self, width: usize) -> Result<Column>;
    /// Zero-fills strings using per-row widths from a column.
    fn str_zfill_by_widths(&self, widths: &ColumnView<'_>) -> Result<Column>;
    /// Extracts a substring [start, stop) with optional step.
    fn str_slice(&self, start: i32, stop: i32, step: i32) -> Result<Column>;
    /// Repeats each string N times.
    fn str_repeat(&self, times: usize) -> Result<Column>;
    /// Splits strings by a delimiter into a table of columns.
    fn str_split(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Table>;
    /// Right-splits strings by a delimiter into a table of columns.
    fn str_rsplit(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Table>;
    /// Returns the Nth part after splitting by delimiter.
    fn str_split_part(&self, delimiter: &Scalar, index: i32) -> Result<Column>;
    /// Joins all strings into a single string.
    fn str_join(&self, separator: &Scalar, narep: &Scalar) -> Result<Column>;
    /// SQL LIKE pattern matching.
    fn str_like(&self, pattern: &str, escape_char: &str) -> Result<Column>;
    /// Regex contains check.
    fn str_contains_re(&self, pattern: &str) -> Result<Column>;
    /// Regex match from start of string.
    fn str_matches_re(&self, pattern: &str) -> Result<Column>;
    /// Counts regex matches per string.
    fn str_count_re(&self, pattern: &str) -> Result<Column>;
    /// Replaces regex matches.
    fn str_replace_re(&self, pattern: &str, replacement: &str) -> Result<Column>;
    /// Swap case (upper→lower, lower→upper).
    fn swapcase(&self) -> Result<Column>;
    /// Strip specified characters from sides of strings.
    fn str_strip_chars(&self, side: SideType, to_strip: &str) -> Result<Column>;
    /// Replace literal target with replacement (max `maxrepl` times, -1 = all).
    fn str_replace_literal(&self, target: &str, repl: &str, maxrepl: i32) -> Result<Column>;
    /// Find first position of literal target in range [start, stop).
    fn str_find_str(&self, target: &str, start: i32, stop: i32) -> Result<Column>;
    /// Find last position of literal target (reverse find).
    fn str_rfind(&self, target: &str, start: i32, stop: i32) -> Result<Column>;
    /// Check if string contains literal target.
    fn str_contains_literal(&self, target: &str) -> Result<Column>;
    /// Check if string starts with literal target.
    fn str_starts_with_str(&self, target: &str) -> Result<Column>;
    /// Check if string ends with literal target.
    fn str_ends_with_str(&self, target: &str) -> Result<Column>;
    /// Reverse characters within each string.
    fn str_reverse(&self) -> Result<Column>;
    /// Regex extract groups into a Table (one column per group).
    fn str_extract(&self, pattern: &str) -> Result<Table>;
    /// Regex extract all matches into a lists column.
    fn str_extract_all(&self, pattern: &str) -> Result<Column>;
    /// Find all regex matches as a lists column.
    fn str_findall(&self, pattern: &str) -> Result<Column>;
    /// Find first regex match position.
    fn str_find_re(&self, pattern: &str) -> Result<Column>;
    /// Capitalizes the first character of each string.
    fn capitalize(&self) -> Result<Column>;
    /// Title-cases each string.
    fn title(&self) -> Result<Column>;
    /// Returns BOOL8 column indicating whether each string is title-cased.
    fn is_title(&self) -> Result<Column>;
    /// Wraps strings onto multiple lines shorter than `width`.
    fn wrap(&self, width: i32) -> Result<Column>;

    // -- String conversions --

    /// Convert strings to timestamps using format (e.g. "%Y-%m-%d").
    fn str_to_timestamps(&self, timestamp_type: TypeId, format: &str) -> Result<Column>;
    /// Convert timestamps to strings using format.
    fn str_from_timestamps(&self, format: &str) -> Result<Column>;
    /// Check if strings are valid timestamps with given format.
    fn str_is_timestamp(&self, format: &str) -> Result<Column>;
    /// Convert strings to booleans (matching `true_string` → true, else false).
    fn str_to_booleans(&self, true_string: &str) -> Result<Column>;
    /// Convert booleans to strings.
    fn str_from_booleans(&self, true_string: &str, false_string: &str) -> Result<Column>;
    /// Convert strings to durations using format.
    fn str_to_durations(&self, duration_type: TypeId, format: &str) -> Result<Column>;
    /// Convert durations to strings.
    fn str_from_durations(&self, format: &str) -> Result<Column>;
    /// Convert strings to fixed-point decimals.
    fn str_to_fixed_point(&self, type_id: TypeId, scale: i32) -> Result<Column>;
    /// Convert fixed-point decimals to strings.
    fn str_from_fixed_point(&self) -> Result<Column>;
    /// Check if strings are valid fixed-point.
    fn str_is_fixed_point(&self, type_id: TypeId, scale: i32) -> Result<Column>;
    /// URL-encode each string.
    fn url_encode(&self) -> Result<Column>;
    /// URL-decode each string.
    fn url_decode(&self) -> Result<Column>;
    /// Convert IPv4 strings to UINT32.
    fn ipv4_to_integers(&self) -> Result<Column>;
    /// Convert UINT32 to IPv4 strings.
    fn integers_to_ipv4(&self) -> Result<Column>;
    /// Check if strings are valid IPv4.
    fn is_ipv4(&self) -> Result<Column>;

    // -- More string operations --

    /// Regex split to table of columns.
    fn str_split_re(&self, pattern: &str, maxsplit: i32) -> Result<Table>;
    /// Regex reverse split to table of columns.
    fn str_rsplit_re(&self, pattern: &str, maxsplit: i32) -> Result<Table>;
    /// Regex split to lists column.
    fn str_split_record_re(&self, pattern: &str, maxsplit: i32) -> Result<Column>;
    /// Regex reverse split to lists column.
    fn str_rsplit_record_re(&self, pattern: &str, maxsplit: i32) -> Result<Column>;
    /// Partition around first delimiter into 3-column table.
    fn str_partition(&self, delimiter: &str) -> Result<Table>;
    /// Partition around last delimiter into 3-column table.
    fn str_rpartition(&self, delimiter: &str) -> Result<Table>;
    /// Regex replace with back-reference template.
    fn str_replace_with_backrefs(&self, pattern: &str, replacement: &str) -> Result<Column>;
    /// Repeat each string by count in another column.
    fn str_repeat_column(&self, repeat_times: &ColumnView<'_>) -> Result<Column>;
    /// Check if strings contain multiple targets (Table of BOOL8 columns).
    fn str_contains_multiple(&self, targets: &ColumnView<'_>) -> Result<Table>;
    /// Find positions of multiple targets in each string (lists column).
    fn str_find_multiple(&self, targets: &ColumnView<'_>) -> Result<Column>;
    /// Check if all characters match the given type bitmask.
    ///
    /// Types are bitmasks: DECIMAL=1, NUMERIC=2, DIGIT=4, ALPHA=8, SPACE=16, UPPER=32, LOWER=64.
    /// `verify_types` restricts which types are checked (default ALL_TYPES=127).
    fn all_characters_of_type(&self, types: u32, verify_types: u32) -> Result<Column>;
    /// Filter characters by type, replacing removed chars with replacement string.
    fn filter_characters_of_type(&self, types_to_remove: u32, replacement: &str, types_to_keep: u32) -> Result<Column>;
    /// Check if all chars in each string are valid integers.
    fn str_is_integer(&self) -> Result<Column>;
    /// Check if all chars are valid integers within the given type range.
    fn str_is_integer_with_type(&self, int_type: TypeId) -> Result<Column>;
    /// Check if all chars in each string are valid floats.
    fn str_is_float(&self) -> Result<Column>;
    /// Convert hex strings to integers of given type.
    fn str_hex_to_integers(&self, output_type: TypeId) -> Result<Column>;
    /// Check if strings are valid hex format.
    fn str_is_hex(&self) -> Result<Column>;
    /// Convert integers to hex strings.
    fn str_integers_to_hex(&self) -> Result<Column>;

    // -- Replace slice / multiple --

    /// Replace substring at positions [start, stop) with replacement string.
    fn str_replace_slice(&self, repl: &str, start: i32, stop: i32) -> Result<Column>;
    /// Replace multiple targets with corresponding replacement strings.
    fn str_replace_multiple(&self, targets: &ColumnView<'_>, repls: &ColumnView<'_>) -> Result<Column>;

    // -- Split to lists column (non-regex) --

    /// Split strings by delimiter into a lists column (left-to-right).
    fn str_split_record(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Column>;
    /// Split strings by delimiter into a lists column (right-to-left).
    fn str_rsplit_record(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Column>;

    // -- Join list elements --

    /// Join lists of strings into a single string per row with separator.
    fn str_join_list_elements(&self, separator: &str, narep: &str) -> Result<Column>;

    // -- Translate / filter characters --

    /// Translate individual characters using from→to mapping.
    /// `from_chars[i]` is replaced by `to_chars[i]`. Use 0 in to_chars to remove a character.
    fn str_translate(&self, from_chars: &[u32], to_chars: &[u32]) -> Result<Column>;
    /// Filter character ranges. `keep=true` keeps only characters in ranges, `false` removes them.
    /// Each range is `[from_chars[i], to_chars[i]]`.
    fn str_filter_characters(&self, from_chars: &[u32], to_chars: &[u32], keep: bool, replacement: &str) -> Result<Column>;

    // -- Code points --

    /// Returns an INT32 column of Unicode code points for all characters (concatenated).
    fn code_points(&self) -> Result<Column>;

    // -- Binary string↔integer encoding --

    /// Encode strings as integers (binary byte representation).
    fn str_cast_to_integer(&self, output_type: TypeId, big_endian: bool) -> Result<Column>;
    /// Decode integer-encoded bytes back to strings.
    fn str_cast_from_integer(&self, big_endian: bool) -> Result<Column>;

    // -- Slice by column / extract_single --

    /// Slice strings using per-row start/stop columns (INT32).
    fn str_slice_column(&self, starts: &ColumnView<'_>, stops: &ColumnView<'_>) -> Result<Column>;
    /// Extract a single regex capture group from each string.
    fn str_extract_single(&self, pattern: &str, group_index: i32) -> Result<Column>;
    /// Join lists of strings with per-row separator column.
    fn str_join_list_elements_column(&self, separators: &ColumnView<'_>, separator_narep: &str, string_narep: &str) -> Result<Column>;
    /// Join all strings in a column into a single-row column using a string separator.
    fn str_join_strings(&self, separator: &str, narep: &str) -> Result<Column>;
    /// Find the Nth occurrence of a target substring, returning INT32 positions.
    fn str_find_instance(&self, target: &str, instance: i32) -> Result<Column>;
    /// SQL LIKE pattern matching with per-row patterns from another column.
    fn str_like_column(&self, patterns: &ColumnView<'_>, escape_char: &str) -> Result<Column>;
}

impl StringExt for ColumnView<'_> {
    fn to_lower(&self) -> Result<Column> {
        self.to_lower_on(Stream::default_stream())
    }
    fn to_lower_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_lower(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn to_upper(&self) -> Result<Column> {
        self.to_upper_on(Stream::default_stream())
    }
    fn to_upper_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_upper(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_contains(&self, target: &Scalar) -> Result<Column> {
        self.str_contains_on(target, Stream::default_stream())
    }
    fn str_contains_on(&self, target: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(target);
        let c = cudf_sys::ffi::strings_contains(self.0, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_starts_with(&self, target: &Scalar) -> Result<Column> {
        self.str_starts_with_on(target, Stream::default_stream())
    }
    fn str_starts_with_on(&self, target: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(target);
        let c = cudf_sys::ffi::strings_starts_with(self.0, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_ends_with(&self, target: &Scalar) -> Result<Column> {
        self.str_ends_with_on(target, Stream::default_stream())
    }
    fn str_ends_with_on(&self, target: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(target);
        let c = cudf_sys::ffi::strings_ends_with(self.0, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_find(&self, target: &Scalar) -> Result<Column> {
        self.str_find_on(target, Stream::default_stream())
    }
    fn str_find_on(&self, target: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(target);
        let c = cudf_sys::ffi::strings_find(self.0, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_replace(&self, target: &Scalar, replacement: &Scalar) -> Result<Column> {
        self.str_replace_on(target, replacement, Stream::default_stream())
    }
    fn str_replace_on(
        &self,
        target: &Scalar,
        replacement: &Scalar,
        stream: Stream,
    ) -> Result<Column> {
        let target_ffi = crate::scalar::scalar_to_ffi(target);
        let replacement_ffi = crate::scalar::scalar_to_ffi(replacement);
        let c = cudf_sys::ffi::strings_replace(
            self.0,
            &target_ffi,
            &replacement_ffi,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }
    fn str_strip(&self) -> Result<Column> {
        self.str_strip_on(Stream::default_stream())
    }
    fn str_strip_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_strip(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_lstrip(&self) -> Result<Column> {
        self.str_lstrip_on(Stream::default_stream())
    }
    fn str_lstrip_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_lstrip(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_rstrip(&self) -> Result<Column> {
        self.str_rstrip_on(Stream::default_stream())
    }
    fn str_rstrip_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_rstrip(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn count_characters(&self) -> Result<Column> {
        self.count_characters_on(Stream::default_stream())
    }
    fn count_characters_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_count_characters(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn count_bytes(&self) -> Result<Column> {
        self.count_bytes_on(Stream::default_stream())
    }
    fn count_bytes_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_count_bytes(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_from_integers(&self) -> Result<Column> {
        self.str_from_integers_on(Stream::default_stream())
    }
    fn str_from_integers_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_from_integers(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_to_integers(&self, output_type: TypeId) -> Result<Column> {
        self.str_to_integers_on(output_type, Stream::default_stream())
    }
    fn str_to_integers_on(&self, output_type: TypeId, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_integers(self.0, output_type.repr, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_from_floats(&self) -> Result<Column> {
        self.str_from_floats_on(Stream::default_stream())
    }
    fn str_from_floats_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_from_floats(self.0, stream.as_raw())?;
        Ok(Column(c))
    }
    fn str_to_floats(&self, output_type: TypeId) -> Result<Column> {
        self.str_to_floats_on(output_type, Stream::default_stream())
    }
    fn str_to_floats_on(&self, output_type: TypeId, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_floats(self.0, output_type.repr, stream.as_raw())?;
        Ok(Column(c))
    }

    fn str_pad(&self, width: usize, side: SideType, fill_char: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_pad(self.0, width as i32, side.repr, fill_char, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_zfill(&self, width: usize) -> Result<Column> {
        let c = cudf_sys::ffi::strings_zfill(self.0, width as i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_zfill_by_widths(&self, widths: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::strings_zfill_by_widths(self.0, widths.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_slice(&self, start: i32, stop: i32, step: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_slice(self.0, start, stop, step, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_repeat(&self, times: usize) -> Result<Column> {
        let c = cudf_sys::ffi::strings_repeat(self.0, times as i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_split(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Table> {
        let ffi = crate::scalar::scalar_to_ffi(delimiter);
        let t = cudf_sys::ffi::strings_split_to_table(self.0, &ffi, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_rsplit(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Table> {
        let ffi = crate::scalar::scalar_to_ffi(delimiter);
        let t = cudf_sys::ffi::strings_rsplit_to_table(self.0, &ffi, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_split_part(&self, delimiter: &Scalar, index: i32) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(delimiter);
        let c = cudf_sys::ffi::strings_split_part(self.0, &ffi, index, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_join(&self, separator: &Scalar, narep: &Scalar) -> Result<Column> {
        let sep_ffi = crate::scalar::scalar_to_ffi(separator);
        let na_ffi = crate::scalar::scalar_to_ffi(narep);
        let c = cudf_sys::ffi::strings_join(self.0, &sep_ffi, &na_ffi, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_like(&self, pattern: &str, escape_char: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_like(self.0, pattern, escape_char, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_contains_re(&self, pattern: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_contains_re(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_matches_re(&self, pattern: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_matches_re(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_count_re(&self, pattern: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_count_re(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_replace_re(&self, pattern: &str, replacement: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_replace_re(self.0, pattern, replacement, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn swapcase(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_swapcase(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_strip_chars(&self, side: SideType, to_strip: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_strip_chars(self.0, side.repr, to_strip, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_replace_literal(&self, target: &str, repl: &str, maxrepl: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_replace_literal(self.0, target, repl, maxrepl, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_find_str(&self, target: &str, start: i32, stop: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_find_str(self.0, target, start, stop, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_rfind(&self, target: &str, start: i32, stop: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_rfind(self.0, target, start, stop, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_contains_literal(&self, target: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_contains_str(self.0, target, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_starts_with_str(&self, target: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_starts_with_str(self.0, target, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_ends_with_str(&self, target: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_ends_with_str(self.0, target, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_reverse(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_reverse(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_extract(&self, pattern: &str) -> Result<Table> {
        let t = cudf_sys::ffi::strings_extract(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_extract_all(&self, pattern: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_extract_all_record(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_findall(&self, pattern: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_findall(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_find_re(&self, pattern: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_find_re(self.0, pattern, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn capitalize(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_capitalize(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn title(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_title(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn is_title(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_title(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn wrap(&self, width: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_wrap(self.0, width, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_to_timestamps(&self, timestamp_type: TypeId, format: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_timestamps(self.0, timestamp_type.repr, format, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_from_timestamps(&self, format: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_from_timestamps(self.0, format, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_is_timestamp(&self, format: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_timestamp(self.0, format, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_to_booleans(&self, true_string: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_booleans(self.0, true_string, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_from_booleans(&self, true_string: &str, false_string: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_from_booleans(self.0, true_string, false_string, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_to_durations(&self, duration_type: TypeId, format: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_durations(self.0, duration_type.repr, format, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_from_durations(&self, format: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_from_durations(self.0, format, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_to_fixed_point(&self, type_id: TypeId, scale: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_to_fixed_point(self.0, type_id.repr, scale, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_from_fixed_point(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_from_fixed_point(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_is_fixed_point(&self, type_id: TypeId, scale: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_fixed_point(self.0, type_id.repr, scale, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn url_encode(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_url_encode(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn url_decode(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_url_decode(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn ipv4_to_integers(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_ipv4_to_integers(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn integers_to_ipv4(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_integers_to_ipv4(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn is_ipv4(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_ipv4(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_split_re(&self, pattern: &str, maxsplit: i32) -> Result<Table> {
        let t = cudf_sys::ffi::strings_split_re(self.0, pattern, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_rsplit_re(&self, pattern: &str, maxsplit: i32) -> Result<Table> {
        let t = cudf_sys::ffi::strings_rsplit_re(self.0, pattern, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_split_record_re(&self, pattern: &str, maxsplit: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_split_record_re(self.0, pattern, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_rsplit_record_re(&self, pattern: &str, maxsplit: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_rsplit_record_re(self.0, pattern, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_partition(&self, delimiter: &str) -> Result<Table> {
        let t = cudf_sys::ffi::strings_partition(self.0, delimiter, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_rpartition(&self, delimiter: &str) -> Result<Table> {
        let t = cudf_sys::ffi::strings_rpartition(self.0, delimiter, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_replace_with_backrefs(&self, pattern: &str, replacement: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_replace_with_backrefs(self.0, pattern, replacement, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_repeat_column(&self, repeat_times: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::strings_repeat_column(self.0, repeat_times.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_contains_multiple(&self, targets: &ColumnView<'_>) -> Result<Table> {
        let t = cudf_sys::ffi::strings_contains_multiple(self.0, targets.0, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }
    fn str_find_multiple(&self, targets: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::strings_find_multiple(self.0, targets.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn all_characters_of_type(&self, types: u32, verify_types: u32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_all_characters_of_type(self.0, types, verify_types, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn filter_characters_of_type(&self, types_to_remove: u32, replacement: &str, types_to_keep: u32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_filter_characters_of_type(self.0, types_to_remove, replacement, types_to_keep, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_is_integer(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_integer(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_is_integer_with_type(&self, int_type: TypeId) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_integer_with_type(self.0, int_type.repr, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_is_float(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_float(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_hex_to_integers(&self, output_type: TypeId) -> Result<Column> {
        let c = cudf_sys::ffi::strings_hex_to_integers(self.0, output_type.repr, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_is_hex(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_is_hex(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_integers_to_hex(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_integers_to_hex(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_replace_slice(&self, repl: &str, start: i32, stop: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_replace_slice(self.0, repl, start, stop, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_replace_multiple(&self, targets: &ColumnView<'_>, repls: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::strings_replace_multiple(self.0, targets.0, repls.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_split_record(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(delimiter);
        let c = cudf_sys::ffi::strings_split_record(self.0, &ffi, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_rsplit_record(&self, delimiter: &Scalar, maxsplit: i32) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(delimiter);
        let c = cudf_sys::ffi::strings_rsplit_record(self.0, &ffi, maxsplit, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_join_list_elements(&self, separator: &str, narep: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_join_list_elements(self.0, separator, narep, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_translate(&self, from_chars: &[u32], to_chars: &[u32]) -> Result<Column> {
        let c = cudf_sys::ffi::strings_translate(self.0, from_chars, to_chars, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_filter_characters(&self, from_chars: &[u32], to_chars: &[u32], keep: bool, replacement: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_filter_characters(self.0, from_chars, to_chars, keep, replacement, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn code_points(&self) -> Result<Column> {
        let c = cudf_sys::ffi::strings_code_points(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_cast_to_integer(&self, output_type: TypeId, big_endian: bool) -> Result<Column> {
        let c = cudf_sys::ffi::strings_cast_to_integer(self.0, output_type.repr, big_endian, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_cast_from_integer(&self, big_endian: bool) -> Result<Column> {
        let c = cudf_sys::ffi::strings_cast_from_integer(self.0, big_endian, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_slice_column(&self, starts: &ColumnView<'_>, stops: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::strings_slice_column(self.0, starts.0, stops.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_extract_single(&self, pattern: &str, group_index: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_extract_single(self.0, pattern, group_index, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_join_list_elements_column(&self, separators: &ColumnView<'_>, separator_narep: &str, string_narep: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_join_list_elements_column(self.0, separators.0, separator_narep, string_narep, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_join_strings(&self, separator: &str, narep: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_join_strings(self.0, separator, narep, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_find_instance(&self, target: &str, instance: i32) -> Result<Column> {
        let c = cudf_sys::ffi::strings_find_instance(self.0, target, instance, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn str_like_column(&self, patterns: &ColumnView<'_>, escape_char: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_like_column(self.0, patterns.0, escape_char, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
}

/// Concatenate string columns row-wise with per-row separator column.
pub fn concatenate_strings_with_separator(
    tbl: &crate::table::Table,
    separators: &ColumnView<'_>,
    separator_narep: &str,
    col_narep: &str,
) -> Result<Column> {
    let c = cudf_sys::ffi::strings_concatenate_columns_sep_col(&tbl.0, separators.0, separator_narep, col_narep, Stream::default_stream().as_raw())?;
    Ok(Column(c))
}

/// Extract values from JSON strings using a JSONPath expression.
///
/// Each row in the input column must be a valid JSON string. The JSONPath
/// expression is applied to every row, returning the matched values as strings.
pub fn get_json_object(
    col: &ColumnView<'_>,
    json_path: &str,
    allow_single_quotes: bool,
    strip_quotes: bool,
    missing_fields_as_nulls: bool,
) -> Result<Column> {
    let c = cudf_sys::ffi::get_json_object(
        col.0,
        json_path,
        allow_single_quotes,
        strip_quotes,
        missing_fields_as_nulls,
        Stream::default_stream().as_raw(),
    )?;
    Ok(Column(c))
}

/// String character type bitmask constants.
pub mod char_types {
    pub const DECIMAL: u32 = 1;
    pub const NUMERIC: u32 = 2;
    pub const DIGIT: u32 = 4;
    pub const ALPHA: u32 = 8;
    pub const SPACE: u32 = 16;
    pub const UPPER: u32 = 32;
    pub const LOWER: u32 = 64;
    pub const ALPHANUM: u32 = DECIMAL | NUMERIC | DIGIT | ALPHA;
    pub const CASE_TYPES: u32 = UPPER | LOWER;
    pub const ALL_TYPES: u32 = ALPHANUM | CASE_TYPES | SPACE;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, Scalar};

    fn make_string_col(values: &[&str]) -> Column {
        Column::from_strings(values)
    }

    #[test]
    fn to_lower_basic() {
        let col = make_string_col(&["HELLO", "World", "foo"]);
        let result = col.view().to_lower().unwrap();
        assert_eq!(result.to_vec_string(), vec!["hello", "world", "foo"]);
    }

    #[test]
    fn to_upper_basic() {
        let col = make_string_col(&["hello", "World", "FOO"]);
        let result = col.view().to_upper().unwrap();
        assert_eq!(result.to_vec_string(), vec!["HELLO", "WORLD", "FOO"]);
    }

    #[test]
    fn contains_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("abc");
        let result = col.view().str_contains(&target).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, false, true]);
    }

    #[test]
    fn starts_with_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("ab");
        let result = col.view().str_starts_with(&target).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, false, true]);
    }

    #[test]
    fn ends_with_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("ef");
        let result = col.view().str_ends_with(&target).unwrap();
        assert_eq!(result.to_vec_bool(), vec![false, true, true]);
    }

    #[test]
    fn find_basic() {
        let col = make_string_col(&["hello world", "goodbye", "hello"]);
        let target = Scalar::from_string("lo");
        let result = col.view().str_find(&target).unwrap();
        assert_eq!(result.to_vec_i32(), vec![3, -1, 3]);
    }

    #[test]
    fn replace_basic() {
        let col = make_string_col(&["hello", "world"]);
        let target = Scalar::from_string("o");
        let replacement = Scalar::from_string("0");
        let result = col.view().str_replace(&target, &replacement).unwrap();
        assert_eq!(result.to_vec_string(), vec!["hell0", "w0rld"]);
    }

    #[test]
    fn strip_basic() {
        let col = make_string_col(&["  hello  ", " world", "foo "]);
        let result = col.view().str_strip().unwrap();
        assert_eq!(result.to_vec_string(), vec!["hello", "world", "foo"]);
    }

    #[test]
    fn lstrip_basic() {
        let col = make_string_col(&["  hello  ", " world"]);
        let result = col.view().str_lstrip().unwrap();
        assert_eq!(result.to_vec_string(), vec!["hello  ", "world"]);
    }

    #[test]
    fn rstrip_basic() {
        let col = make_string_col(&["  hello  ", "world "]);
        let result = col.view().str_rstrip().unwrap();
        assert_eq!(result.to_vec_string(), vec!["  hello", "world"]);
    }

    #[test]
    fn count_characters_basic() {
        let col = make_string_col(&["hello", "ab", ""]);
        let result = col.view().count_characters().unwrap();
        assert_eq!(result.to_vec_i32(), vec![5, 2, 0]);
    }

    #[test]
    fn count_bytes_basic() {
        let col = make_string_col(&["hello", "ab", ""]);
        let result = col.view().count_bytes().unwrap();
        assert_eq!(result.to_vec_i32(), vec![5, 2, 0]);
    }

    #[test]
    fn strings_from_integers_basic() {
        let col = Column::from_strings(&["1", "2", "3"]);
        let int_col = col.view().str_to_integers(TypeId::INT32).unwrap();
        assert_eq!(int_col.to_vec_i32(), vec![1, 2, 3]);

        let back = int_col.view().str_from_integers().unwrap();
        assert_eq!(back.to_vec_string(), vec!["1", "2", "3"]);
    }

    #[test]
    fn strings_to_integers_basic() {
        let col = make_string_col(&["10", "20", "30"]);
        let result = col.view().str_to_integers(TypeId::INT32).unwrap();
        assert_eq!(result.to_vec_i32(), vec![10, 20, 30]);
    }

    #[test]
    fn strings_from_floats_basic() {
        let col = Scalar::from_f64(1.5);
        let float_col = Column::from_scalar(&col, 2);
        let result = float_col.view().str_from_floats().unwrap();
        assert_eq!(result.len(), 2);
        let strs = result.to_vec_string();
        assert!(strs.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn strings_to_floats_basic() {
        let col = make_string_col(&["1.5", "2.5", "3.0"]);
        let result = col.view().str_to_floats(TypeId::FLOAT64).unwrap();
        let vals = result.to_vec_f64();
        assert!((vals[0] - 1.5).abs() < 1e-9);
        assert!((vals[1] - 2.5).abs() < 1e-9);
        assert!((vals[2] - 3.0).abs() < 1e-9);
    }
}
