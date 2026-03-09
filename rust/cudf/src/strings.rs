// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! String operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;
use crate::table::Table;

#[doc(alias = "side_type")]
pub use cudf_sys::ffi::SideType;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for crate::column::ColumnView<'_> {}

#[doc(alias = "strings")]
/// Extension trait for string operations on GPU columns.
pub trait StringExt: private::Sealed {
    /// Converts each string to lower case.
    fn to_lower(&self) -> ToLower<'_>;
    /// Converts each string to upper case.
    fn to_upper(&self) -> ToUpper<'_>;
    /// Returns a BOOL8 column indicating whether each string contains the target.
    fn str_contains<'a>(&'a self, target: &'a Scalar) -> StrContains<'a>;
    /// Returns a BOOL8 column indicating whether each string starts with the target.
    fn str_starts_with<'a>(&'a self, target: &'a Scalar) -> StrStartsWith<'a>;
    /// Returns a BOOL8 column indicating whether each string ends with the target.
    fn str_ends_with<'a>(&'a self, target: &'a Scalar) -> StrEndsWith<'a>;
    /// Returns an INT32 column with the position of the first occurrence of the target.
    fn str_find<'a>(&'a self, target: &'a Scalar) -> StrFind<'a>;
    /// Replaces all occurrences of `target` with `replacement`.
    fn str_replace<'a>(&'a self, target: &'a Scalar, replacement: &'a Scalar) -> StrReplace<'a>;
    /// Strips whitespace from both sides.
    fn str_strip(&self) -> StrStrip<'_>;
    /// Strips whitespace from the left side.
    fn str_lstrip(&self) -> StrLstrip<'_>;
    /// Strips whitespace from the right side.
    fn str_rstrip(&self) -> StrRstrip<'_>;
    /// Returns an INT32 column with the character count.
    fn count_characters(&self) -> CountCharacters<'_>;
    /// Returns an INT32 column with the byte count.
    fn count_bytes(&self) -> CountBytes<'_>;
    /// Converts an integer column to a string column.
    fn str_from_integers(&self) -> StrFromIntegers<'_>;
    /// Converts a string column to an integer column.
    fn str_to_integers(&self, output_type: TypeId) -> StrToIntegers<'_>;
    /// Converts a float column to a string column.
    fn str_from_floats(&self) -> StrFromFloats<'_>;
    /// Converts a string column to a float column.
    fn str_to_floats(&self, output_type: TypeId) -> StrToFloats<'_>;
    /// Pads strings to a minimum width.
    fn str_pad<'a>(&'a self, width: usize, side: SideType, fill_char: &'a str) -> StrPad<'a>;
    /// Zero-fills strings to a minimum width.
    fn str_zfill(&self, width: usize) -> StrZfill<'_>;
    /// Zero-fills strings using per-row widths from a column.
    fn str_zfill_by_widths<'a>(&'a self, widths: &'a ColumnView<'a>) -> StrZfillByWidths<'a>;
    /// Extracts a substring [start, stop) with optional step.
    fn str_slice(&self, start: i32, stop: i32, step: i32) -> StrSlice<'_>;
    /// Repeats each string N times.
    fn str_repeat(&self, times: usize) -> StrRepeat<'_>;
    /// Splits strings by a delimiter into a table of columns.
    fn str_split<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplit<'a>;
    /// Right-splits strings by a delimiter into a table of columns.
    fn str_rsplit<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrRsplit<'a>;
    /// Returns the Nth part after splitting by delimiter.
    fn str_split_part<'a>(&'a self, delimiter: &'a Scalar, index: i32) -> StrSplitPart<'a>;
    /// Joins all strings into a single string.
    fn str_join<'a>(&'a self, separator: &'a Scalar, narep: &'a Scalar) -> StrJoin<'a>;
    /// SQL LIKE pattern matching.
    fn str_like<'a>(&'a self, pattern: &'a str, escape_char: &'a str) -> StrLike<'a>;
    /// Regex contains check.
    fn str_contains_re<'a>(&'a self, pattern: &'a str) -> StrContainsRe<'a>;
    /// Regex match from start of string.
    fn str_matches_re<'a>(&'a self, pattern: &'a str) -> StrMatchesRe<'a>;
    /// Counts regex matches per string.
    fn str_count_re<'a>(&'a self, pattern: &'a str) -> StrCountRe<'a>;
    /// Replaces regex matches.
    fn str_replace_re<'a>(&'a self, pattern: &'a str, replacement: &'a str) -> StrReplaceRe<'a>;
    /// Swap case (upper to lower, lower to upper).
    fn swapcase(&self) -> Swapcase<'_>;
    /// Strip specified characters from sides of strings.
    fn str_strip_chars<'a>(&'a self, side: SideType, to_strip: &'a str) -> StrStripChars<'a>;
    /// Replace literal target with replacement (max `maxrepl` times, -1 = all).
    fn str_replace_literal<'a>(
        &'a self,
        target: &'a str,
        repl: &'a str,
        maxrepl: i32,
    ) -> StrReplaceLiteral<'a>;
    /// Find first position of literal target in range [start, stop).
    fn str_find_str<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrFindStr<'a>;
    /// Find last position of literal target (reverse find).
    fn str_rfind<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrRfind<'a>;
    /// Check if string contains literal target.
    fn str_contains_literal<'a>(&'a self, target: &'a str) -> StrContainsLiteral<'a>;
    /// Check if string starts with literal target.
    fn str_starts_with_str<'a>(&'a self, target: &'a str) -> StrStartsWithStr<'a>;
    /// Check if string ends with literal target.
    fn str_ends_with_str<'a>(&'a self, target: &'a str) -> StrEndsWithStr<'a>;
    /// Reverse characters within each string.
    fn str_reverse(&self) -> StrReverse<'_>;
    /// Regex extract groups into a Table (one column per group).
    fn str_extract<'a>(&'a self, pattern: &'a str) -> StrExtract<'a>;
    /// Regex extract all matches into a lists column.
    fn str_extract_all<'a>(&'a self, pattern: &'a str) -> StrExtractAll<'a>;
    /// Find all regex matches as a lists column.
    fn str_findall<'a>(&'a self, pattern: &'a str) -> StrFindall<'a>;
    /// Find first regex match position.
    fn str_find_re<'a>(&'a self, pattern: &'a str) -> StrFindRe<'a>;
    /// Capitalizes the first character of each string.
    fn capitalize(&self) -> Capitalize<'_>;
    /// Title-cases each string.
    fn title(&self) -> Title<'_>;
    /// Returns BOOL8 column indicating whether each string is title-cased.
    fn is_title(&self) -> IsTitle<'_>;
    /// Wraps strings onto multiple lines shorter than `width`.
    fn wrap(&self, width: i32) -> Wrap<'_>;
    /// Convert strings to timestamps using format (e.g. "%Y-%m-%d").
    fn str_to_timestamps<'a>(
        &'a self,
        timestamp_type: TypeId,
        format: &'a str,
    ) -> StrToTimestamps<'a>;
    /// Convert timestamps to strings using format.
    fn str_from_timestamps<'a>(&'a self, format: &'a str) -> StrFromTimestamps<'a>;
    /// Check if strings are valid timestamps with given format.
    fn str_is_timestamp<'a>(&'a self, format: &'a str) -> StrIsTimestamp<'a>;
    /// Convert strings to booleans (matching `true_string` to true, else false).
    fn str_to_booleans<'a>(&'a self, true_string: &'a str) -> StrToBooleans<'a>;
    /// Convert booleans to strings.
    fn str_from_booleans<'a>(
        &'a self,
        true_string: &'a str,
        false_string: &'a str,
    ) -> StrFromBooleans<'a>;
    /// Convert strings to durations using format.
    fn str_to_durations<'a>(&'a self, duration_type: TypeId, format: &'a str)
    -> StrToDurations<'a>;
    /// Convert durations to strings.
    fn str_from_durations<'a>(&'a self, format: &'a str) -> StrFromDurations<'a>;
    /// Convert strings to fixed-point decimals.
    fn str_to_fixed_point(&self, type_id: TypeId, scale: i32) -> StrToFixedPoint<'_>;
    /// Convert fixed-point decimals to strings.
    fn str_from_fixed_point(&self) -> StrFromFixedPoint<'_>;
    /// Check if strings are valid fixed-point.
    fn str_is_fixed_point(&self, type_id: TypeId, scale: i32) -> StrIsFixedPoint<'_>;
    /// URL-encode each string.
    fn url_encode(&self) -> UrlEncode<'_>;
    /// URL-decode each string.
    fn url_decode(&self) -> UrlDecode<'_>;
    /// Convert IPv4 strings to UINT32.
    fn ipv4_to_integers(&self) -> Ipv4ToIntegers<'_>;
    /// Convert UINT32 to IPv4 strings.
    fn integers_to_ipv4(&self) -> IntegersToIpv4<'_>;
    /// Check if strings are valid IPv4.
    fn is_ipv4(&self) -> IsIpv4<'_>;
    /// Regex split to table of columns.
    fn str_split_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRe<'a>;
    /// Regex reverse split to table of columns.
    fn str_rsplit_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrRsplitRe<'a>;
    /// Regex split to lists column.
    fn str_split_record_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRecordRe<'a>;
    /// Regex reverse split to lists column.
    fn str_rsplit_record_re<'a>(&'a self, pattern: &'a str, maxsplit: i32)
    -> StrRsplitRecordRe<'a>;
    /// Partition around first delimiter into 3-column table.
    fn str_partition<'a>(&'a self, delimiter: &'a str) -> StrPartition<'a>;
    /// Partition around last delimiter into 3-column table.
    fn str_rpartition<'a>(&'a self, delimiter: &'a str) -> StrRpartition<'a>;
    /// Regex replace with back-reference template.
    fn str_replace_with_backrefs<'a>(
        &'a self,
        pattern: &'a str,
        replacement: &'a str,
    ) -> StrReplaceWithBackrefs<'a>;
    /// Repeat each string by count in another column.
    fn str_repeat_column<'a>(&'a self, repeat_times: &'a ColumnView<'a>) -> StrRepeatColumn<'a>;
    /// Check if strings contain multiple targets (Table of BOOL8 columns).
    fn str_contains_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrContainsMultiple<'a>;
    /// Find positions of multiple targets in each string (lists column).
    fn str_find_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrFindMultiple<'a>;
    /// Check if all characters match the given type bitmask.
    ///
    /// Types are bitmasks: DECIMAL=1, NUMERIC=2, DIGIT=4, ALPHA=8, SPACE=16, UPPER=32, LOWER=64.
    /// `verify_types` restricts which types are checked (default `ALL_TYPES=127`).
    fn all_characters_of_type(&self, types: u32, verify_types: u32) -> AllCharactersOfType<'_>;
    /// Filter characters by type, replacing removed chars with replacement string.
    fn filter_characters_of_type<'a>(
        &'a self,
        types_to_remove: u32,
        replacement: &'a str,
        types_to_keep: u32,
    ) -> FilterCharactersOfType<'a>;
    /// Check if all chars in each string are valid integers.
    fn str_is_integer(&self) -> StrIsInteger<'_>;
    /// Check if all chars are valid integers within the given type range.
    fn str_is_integer_with_type(&self, int_type: TypeId) -> StrIsIntegerWithType<'_>;
    /// Check if all chars in each string are valid floats.
    fn str_is_float(&self) -> StrIsFloat<'_>;
    /// Convert hex strings to integers of given type.
    fn str_hex_to_integers(&self, output_type: TypeId) -> StrHexToIntegers<'_>;
    /// Check if strings are valid hex format.
    fn str_is_hex(&self) -> StrIsHex<'_>;
    /// Convert integers to hex strings.
    fn str_integers_to_hex(&self) -> StrIntegersToHex<'_>;
    /// Replace substring at positions [start, stop) with replacement string.
    fn str_replace_slice<'a>(&'a self, repl: &'a str, start: i32, stop: i32)
    -> StrReplaceSlice<'a>;
    /// Replace multiple targets with corresponding replacement strings.
    fn str_replace_multiple<'a>(
        &'a self,
        targets: &'a ColumnView<'a>,
        repls: &'a ColumnView<'a>,
    ) -> StrReplaceMultiple<'a>;
    /// Split strings by delimiter into a lists column (left-to-right).
    fn str_split_record<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplitRecord<'a>;
    /// Split strings by delimiter into a lists column (right-to-left).
    fn str_rsplit_record<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32)
    -> StrRsplitRecord<'a>;
    /// Join lists of strings into a single string per row with separator.
    fn str_join_list_elements<'a>(
        &'a self,
        separator: &'a str,
        narep: &'a str,
    ) -> StrJoinListElements<'a>;
    /// Translate individual characters using from-to mapping.
    fn str_translate<'a>(&'a self, from_chars: &'a [u32], to_chars: &'a [u32]) -> StrTranslate<'a>;
    /// Filter character ranges. `keep=true` keeps only characters in ranges, `false` removes them.
    fn str_filter_characters<'a>(
        &'a self,
        from_chars: &'a [u32],
        to_chars: &'a [u32],
        keep: bool,
        replacement: &'a str,
    ) -> StrFilterCharacters<'a>;
    /// Returns an INT32 column of Unicode code points for all characters (concatenated).
    fn code_points(&self) -> CodePoints<'_>;
    /// Encode strings as integers (binary byte representation).
    fn str_cast_to_integer(&self, output_type: TypeId, big_endian: bool) -> StrCastToInteger<'_>;
    /// Decode integer-encoded bytes back to strings.
    fn str_cast_from_integer(&self, big_endian: bool) -> StrCastFromInteger<'_>;
    /// Slice strings using per-row start/stop columns (INT32).
    fn str_slice_column<'a>(
        &'a self,
        starts: &'a ColumnView<'a>,
        stops: &'a ColumnView<'a>,
    ) -> StrSliceColumn<'a>;
    /// Extract a single regex capture group from each string.
    fn str_extract_single<'a>(&'a self, pattern: &'a str, group_index: i32)
    -> StrExtractSingle<'a>;
    /// Join lists of strings with per-row separator column.
    fn str_join_list_elements_column<'a>(
        &'a self,
        separators: &'a ColumnView<'a>,
        separator_narep: &'a str,
        string_narep: &'a str,
    ) -> StrJoinListElementsColumn<'a>;
    /// Join all strings in a column into a single-row column using a string separator.
    fn str_join_strings<'a>(&'a self, separator: &'a str, narep: &'a str) -> StrJoinStrings<'a>;
    /// Find the Nth occurrence of a target substring, returning INT32 positions.
    fn str_find_instance<'a>(&'a self, target: &'a str, instance: i32) -> StrFindInstance<'a>;
    /// SQL LIKE pattern matching with per-row patterns from another column.
    fn str_like_column<'a>(
        &'a self,
        patterns: &'a ColumnView<'a>,
        escape_char: &'a str,
    ) -> StrLikeColumn<'a>;
}

// ---------------------------------------------------------------------------
// Builder structs + macro
// ---------------------------------------------------------------------------

macro_rules! simple_builder {
    ($name:ident -> $ret:ident, $body:expr) => {
        impl crate::stream::GpuOp for $name<'_> {
            type Output = $ret;
            fn stream(mut self, stream: Stream) -> Self {
                self.stream = stream;
                self
            }
            fn call(self) -> Result<Self::Output> {
                $body(self)
            }
        }
    };
}

/// Builder for [`StringExt::to_lower`].
pub struct ToLower<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(ToLower -> Column, |s: ToLower<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_lower(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::to_upper`].
pub struct ToUpper<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(ToUpper -> Column, |s: ToUpper<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_upper(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_contains`].
pub struct StrContains<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrContains -> Column, |s: StrContains<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target);
    let c = cudf_sys::strings::ffi::strings_contains(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_starts_with`].
pub struct StrStartsWith<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrStartsWith -> Column, |s: StrStartsWith<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target);
    let c = cudf_sys::strings::ffi::strings_starts_with(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_ends_with`].
pub struct StrEndsWith<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrEndsWith -> Column, |s: StrEndsWith<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target);
    let c = cudf_sys::strings::ffi::strings_ends_with(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_find`].
pub struct StrFind<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrFind -> Column, |s: StrFind<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target);
    let c = cudf_sys::strings::ffi::strings_find(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_replace`].
pub struct StrReplace<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    replacement: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrReplace -> Column, |s: StrReplace<'_>| {
    let target_ffi = crate::scalar::scalar_to_ffi(s.target);
    let replacement_ffi = crate::scalar::scalar_to_ffi(s.replacement);
    let c = cudf_sys::strings::ffi::strings_replace(
        s.view.0, &target_ffi, &replacement_ffi, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_strip`].
pub struct StrStrip<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrStrip -> Column, |s: StrStrip<'_>| {
    let c = cudf_sys::strings::ffi::strings_strip(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_lstrip`].
pub struct StrLstrip<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrLstrip -> Column, |s: StrLstrip<'_>| {
    let c = cudf_sys::strings::ffi::strings_lstrip(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_rstrip`].
pub struct StrRstrip<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrRstrip -> Column, |s: StrRstrip<'_>| {
    let c = cudf_sys::strings::ffi::strings_rstrip(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::count_characters`].
pub struct CountCharacters<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(CountCharacters -> Column, |s: CountCharacters<'_>| {
    let c = cudf_sys::strings::ffi::strings_count_characters(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::count_bytes`].
pub struct CountBytes<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(CountBytes -> Column, |s: CountBytes<'_>| {
    let c = cudf_sys::strings::ffi::strings_count_bytes(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_from_integers`].
pub struct StrFromIntegers<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFromIntegers -> Column, |s: StrFromIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_integers(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_to_integers`].
pub struct StrToIntegers<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}
simple_builder!(StrToIntegers -> Column, |s: StrToIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_integers(s.view.0, s.output_type.repr, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_from_floats`].
pub struct StrFromFloats<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFromFloats -> Column, |s: StrFromFloats<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_floats(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_to_floats`].
pub struct StrToFloats<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}
simple_builder!(StrToFloats -> Column, |s: StrToFloats<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_floats(s.view.0, s.output_type.repr, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_pad`].
pub struct StrPad<'a> {
    view: &'a ColumnView<'a>,
    width: usize,
    side: SideType,
    fill_char: &'a str,
    stream: Stream,
}
simple_builder!(StrPad -> Column, |s: StrPad<'_>| {
    let c = cudf_sys::strings::ffi::strings_pad(
        s.view.0, crate::usize_to_i32(s.width), s.side.repr, s.fill_char, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_zfill`].
pub struct StrZfill<'a> {
    view: &'a ColumnView<'a>,
    width: usize,
    stream: Stream,
}
simple_builder!(StrZfill -> Column, |s: StrZfill<'_>| {
    let c = cudf_sys::strings::ffi::strings_zfill(s.view.0, crate::usize_to_i32(s.width), s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_zfill_by_widths`].
pub struct StrZfillByWidths<'a> {
    view: &'a ColumnView<'a>,
    widths: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrZfillByWidths -> Column, |s: StrZfillByWidths<'_>| {
    let c = cudf_sys::strings::ffi::strings_zfill_by_widths(s.view.0, s.widths.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_slice`].
pub struct StrSlice<'a> {
    view: &'a ColumnView<'a>,
    start: i32,
    stop: i32,
    step: i32,
    stream: Stream,
}
simple_builder!(StrSlice -> Column, |s: StrSlice<'_>| {
    let c = cudf_sys::strings::ffi::strings_slice(s.view.0, s.start, s.stop, s.step, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_repeat`].
pub struct StrRepeat<'a> {
    view: &'a ColumnView<'a>,
    times: usize,
    stream: Stream,
}
simple_builder!(StrRepeat -> Column, |s: StrRepeat<'_>| {
    let c = cudf_sys::strings::ffi::strings_repeat(s.view.0, crate::usize_to_i32(s.times), s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_split`].
pub struct StrSplit<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplit -> Table, |s: StrSplit<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter);
    let t = cudf_sys::strings::ffi::strings_split_to_table(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_rsplit`].
pub struct StrRsplit<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplit -> Table, |s: StrRsplit<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter);
    let t = cudf_sys::strings::ffi::strings_rsplit_to_table(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_split_part`].
pub struct StrSplitPart<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    index: i32,
    stream: Stream,
}
simple_builder!(StrSplitPart -> Column, |s: StrSplitPart<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter);
    let c = cudf_sys::strings::ffi::strings_split_part(s.view.0, &ffi, s.index, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_join`].
pub struct StrJoin<'a> {
    view: &'a ColumnView<'a>,
    separator: &'a Scalar,
    narep: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrJoin -> Column, |s: StrJoin<'_>| {
    let sep_ffi = crate::scalar::scalar_to_ffi(s.separator);
    let na_ffi = crate::scalar::scalar_to_ffi(s.narep);
    let c = cudf_sys::strings::ffi::strings_join(s.view.0, &sep_ffi, &na_ffi, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_like`].
pub struct StrLike<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    escape_char: &'a str,
    stream: Stream,
}
simple_builder!(StrLike -> Column, |s: StrLike<'_>| {
    let c = cudf_sys::strings::ffi::strings_like(s.view.0, s.pattern, s.escape_char, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_contains_re`].
pub struct StrContainsRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrContainsRe -> Column, |s: StrContainsRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_contains_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_matches_re`].
pub struct StrMatchesRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrMatchesRe -> Column, |s: StrMatchesRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_matches_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_count_re`].
pub struct StrCountRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrCountRe -> Column, |s: StrCountRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_count_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_replace_re`].
pub struct StrReplaceRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    replacement: &'a str,
    stream: Stream,
}
simple_builder!(StrReplaceRe -> Column, |s: StrReplaceRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_re(s.view.0, s.pattern, s.replacement, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::swapcase`].
pub struct Swapcase<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Swapcase -> Column, |s: Swapcase<'_>| {
    let c = cudf_sys::strings::ffi::strings_swapcase(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_strip_chars`].
pub struct StrStripChars<'a> {
    view: &'a ColumnView<'a>,
    side: SideType,
    to_strip: &'a str,
    stream: Stream,
}
simple_builder!(StrStripChars -> Column, |s: StrStripChars<'_>| {
    let c = cudf_sys::strings::ffi::strings_strip_chars(s.view.0, s.side.repr, s.to_strip, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_replace_literal`].
pub struct StrReplaceLiteral<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    repl: &'a str,
    maxrepl: i32,
    stream: Stream,
}
simple_builder!(StrReplaceLiteral -> Column, |s: StrReplaceLiteral<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_literal(s.view.0, s.target, s.repl, s.maxrepl, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_find_str`].
pub struct StrFindStr<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    start: i32,
    stop: i32,
    stream: Stream,
}
simple_builder!(StrFindStr -> Column, |s: StrFindStr<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_str(s.view.0, s.target, s.start, s.stop, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_rfind`].
pub struct StrRfind<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    start: i32,
    stop: i32,
    stream: Stream,
}
simple_builder!(StrRfind -> Column, |s: StrRfind<'_>| {
    let c = cudf_sys::strings::ffi::strings_rfind(s.view.0, s.target, s.start, s.stop, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_contains_literal`].
pub struct StrContainsLiteral<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    stream: Stream,
}
simple_builder!(StrContainsLiteral -> Column, |s: StrContainsLiteral<'_>| {
    let c = cudf_sys::strings::ffi::strings_contains_str(s.view.0, s.target, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_starts_with_str`].
pub struct StrStartsWithStr<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    stream: Stream,
}
simple_builder!(StrStartsWithStr -> Column, |s: StrStartsWithStr<'_>| {
    let c = cudf_sys::strings::ffi::strings_starts_with_str(s.view.0, s.target, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_ends_with_str`].
pub struct StrEndsWithStr<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    stream: Stream,
}
simple_builder!(StrEndsWithStr -> Column, |s: StrEndsWithStr<'_>| {
    let c = cudf_sys::strings::ffi::strings_ends_with_str(s.view.0, s.target, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_reverse`].
pub struct StrReverse<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrReverse -> Column, |s: StrReverse<'_>| {
    let c = cudf_sys::strings::ffi::strings_reverse(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_extract`].
pub struct StrExtract<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrExtract -> Table, |s: StrExtract<'_>| {
    let t = cudf_sys::strings::ffi::strings_extract(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_extract_all`].
pub struct StrExtractAll<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrExtractAll -> Column, |s: StrExtractAll<'_>| {
    let c = cudf_sys::strings::ffi::strings_extract_all_record(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_findall`].
pub struct StrFindall<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrFindall -> Column, |s: StrFindall<'_>| {
    let c = cudf_sys::strings::ffi::strings_findall(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_find_re`].
pub struct StrFindRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrFindRe -> Column, |s: StrFindRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::capitalize`].
pub struct Capitalize<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Capitalize -> Column, |s: Capitalize<'_>| {
    let c = cudf_sys::strings::ffi::strings_capitalize(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::title`].
pub struct Title<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Title -> Column, |s: Title<'_>| {
    let c = cudf_sys::strings::ffi::strings_title(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::is_title`].
pub struct IsTitle<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(IsTitle -> Column, |s: IsTitle<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_title(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::wrap`].
pub struct Wrap<'a> {
    view: &'a ColumnView<'a>,
    width: i32,
    stream: Stream,
}
simple_builder!(Wrap -> Column, |s: Wrap<'_>| {
    let c = cudf_sys::strings::ffi::strings_wrap(s.view.0, s.width, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_to_timestamps`].
pub struct StrToTimestamps<'a> {
    view: &'a ColumnView<'a>,
    timestamp_type: TypeId,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrToTimestamps -> Column, |s: StrToTimestamps<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_timestamps(
        s.view.0, s.timestamp_type.repr, s.format, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_from_timestamps`].
pub struct StrFromTimestamps<'a> {
    view: &'a ColumnView<'a>,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrFromTimestamps -> Column, |s: StrFromTimestamps<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_timestamps(s.view.0, s.format, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_is_timestamp`].
pub struct StrIsTimestamp<'a> {
    view: &'a ColumnView<'a>,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrIsTimestamp -> Column, |s: StrIsTimestamp<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_timestamp(s.view.0, s.format, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_to_booleans`].
pub struct StrToBooleans<'a> {
    view: &'a ColumnView<'a>,
    true_string: &'a str,
    stream: Stream,
}
simple_builder!(StrToBooleans -> Column, |s: StrToBooleans<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_booleans(s.view.0, s.true_string, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_from_booleans`].
pub struct StrFromBooleans<'a> {
    view: &'a ColumnView<'a>,
    true_string: &'a str,
    false_string: &'a str,
    stream: Stream,
}
simple_builder!(StrFromBooleans -> Column, |s: StrFromBooleans<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_booleans(
        s.view.0, s.true_string, s.false_string, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_to_durations`].
pub struct StrToDurations<'a> {
    view: &'a ColumnView<'a>,
    duration_type: TypeId,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrToDurations -> Column, |s: StrToDurations<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_durations(
        s.view.0, s.duration_type.repr, s.format, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_from_durations`].
pub struct StrFromDurations<'a> {
    view: &'a ColumnView<'a>,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrFromDurations -> Column, |s: StrFromDurations<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_durations(s.view.0, s.format, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_to_fixed_point`].
pub struct StrToFixedPoint<'a> {
    view: &'a ColumnView<'a>,
    type_id: TypeId,
    scale: i32,
    stream: Stream,
}
simple_builder!(StrToFixedPoint -> Column, |s: StrToFixedPoint<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_fixed_point(
        s.view.0, s.type_id.repr, s.scale, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_from_fixed_point`].
pub struct StrFromFixedPoint<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFromFixedPoint -> Column, |s: StrFromFixedPoint<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_fixed_point(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_is_fixed_point`].
pub struct StrIsFixedPoint<'a> {
    view: &'a ColumnView<'a>,
    type_id: TypeId,
    scale: i32,
    stream: Stream,
}
simple_builder!(StrIsFixedPoint -> Column, |s: StrIsFixedPoint<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_fixed_point(
        s.view.0, s.type_id.repr, s.scale, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::url_encode`].
pub struct UrlEncode<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(UrlEncode -> Column, |s: UrlEncode<'_>| {
    let c = cudf_sys::strings::ffi::strings_url_encode(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::url_decode`].
pub struct UrlDecode<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(UrlDecode -> Column, |s: UrlDecode<'_>| {
    let c = cudf_sys::strings::ffi::strings_url_decode(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::ipv4_to_integers`].
pub struct Ipv4ToIntegers<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Ipv4ToIntegers -> Column, |s: Ipv4ToIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_ipv4_to_integers(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::integers_to_ipv4`].
pub struct IntegersToIpv4<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(IntegersToIpv4 -> Column, |s: IntegersToIpv4<'_>| {
    let c = cudf_sys::strings::ffi::strings_integers_to_ipv4(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::is_ipv4`].
pub struct IsIpv4<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(IsIpv4 -> Column, |s: IsIpv4<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_ipv4(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_split_re`].
pub struct StrSplitRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplitRe -> Table, |s: StrSplitRe<'_>| {
    let t = cudf_sys::strings::ffi::strings_split_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_rsplit_re`].
pub struct StrRsplitRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplitRe -> Table, |s: StrRsplitRe<'_>| {
    let t = cudf_sys::strings::ffi::strings_rsplit_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_split_record_re`].
pub struct StrSplitRecordRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplitRecordRe -> Column, |s: StrSplitRecordRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_split_record_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_rsplit_record_re`].
pub struct StrRsplitRecordRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplitRecordRe -> Column, |s: StrRsplitRecordRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_rsplit_record_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_partition`].
pub struct StrPartition<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a str,
    stream: Stream,
}
simple_builder!(StrPartition -> Table, |s: StrPartition<'_>| {
    let t = cudf_sys::strings::ffi::strings_partition(s.view.0, s.delimiter, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_rpartition`].
pub struct StrRpartition<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a str,
    stream: Stream,
}
simple_builder!(StrRpartition -> Table, |s: StrRpartition<'_>| {
    let t = cudf_sys::strings::ffi::strings_rpartition(s.view.0, s.delimiter, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_replace_with_backrefs`].
pub struct StrReplaceWithBackrefs<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    replacement: &'a str,
    stream: Stream,
}
simple_builder!(StrReplaceWithBackrefs -> Column, |s: StrReplaceWithBackrefs<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_with_backrefs(
        s.view.0, s.pattern, s.replacement, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_repeat_column`].
pub struct StrRepeatColumn<'a> {
    view: &'a ColumnView<'a>,
    repeat_times: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrRepeatColumn -> Column, |s: StrRepeatColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_repeat_column(s.view.0, s.repeat_times.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_contains_multiple`].
pub struct StrContainsMultiple<'a> {
    view: &'a ColumnView<'a>,
    targets: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrContainsMultiple -> Table, |s: StrContainsMultiple<'_>| {
    let t = cudf_sys::strings::ffi::strings_contains_multiple(s.view.0, s.targets.0, s.stream.as_raw())?;
    Ok(Table(t))
});

/// Builder for [`StringExt::str_find_multiple`].
pub struct StrFindMultiple<'a> {
    view: &'a ColumnView<'a>,
    targets: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFindMultiple -> Column, |s: StrFindMultiple<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_multiple(s.view.0, s.targets.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::all_characters_of_type`].
pub struct AllCharactersOfType<'a> {
    view: &'a ColumnView<'a>,
    types: u32,
    verify_types: u32,
    stream: Stream,
}
simple_builder!(AllCharactersOfType -> Column, |s: AllCharactersOfType<'_>| {
    let c = cudf_sys::strings::ffi::strings_all_characters_of_type(
        s.view.0, s.types, s.verify_types, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::filter_characters_of_type`].
pub struct FilterCharactersOfType<'a> {
    view: &'a ColumnView<'a>,
    types_to_remove: u32,
    replacement: &'a str,
    types_to_keep: u32,
    stream: Stream,
}
simple_builder!(FilterCharactersOfType -> Column, |s: FilterCharactersOfType<'_>| {
    let c = cudf_sys::strings::ffi::strings_filter_characters_of_type(
        s.view.0, s.types_to_remove, s.replacement, s.types_to_keep, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_is_integer`].
pub struct StrIsInteger<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIsInteger -> Column, |s: StrIsInteger<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_integer(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_is_integer_with_type`].
pub struct StrIsIntegerWithType<'a> {
    view: &'a ColumnView<'a>,
    int_type: TypeId,
    stream: Stream,
}
simple_builder!(StrIsIntegerWithType -> Column, |s: StrIsIntegerWithType<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_integer_with_type(s.view.0, s.int_type.repr, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_is_float`].
pub struct StrIsFloat<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIsFloat -> Column, |s: StrIsFloat<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_float(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_hex_to_integers`].
pub struct StrHexToIntegers<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}
simple_builder!(StrHexToIntegers -> Column, |s: StrHexToIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_hex_to_integers(s.view.0, s.output_type.repr, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_is_hex`].
pub struct StrIsHex<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIsHex -> Column, |s: StrIsHex<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_hex(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_integers_to_hex`].
pub struct StrIntegersToHex<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIntegersToHex -> Column, |s: StrIntegersToHex<'_>| {
    let c = cudf_sys::strings::ffi::strings_integers_to_hex(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_replace_slice`].
pub struct StrReplaceSlice<'a> {
    view: &'a ColumnView<'a>,
    repl: &'a str,
    start: i32,
    stop: i32,
    stream: Stream,
}
simple_builder!(StrReplaceSlice -> Column, |s: StrReplaceSlice<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_slice(
        s.view.0, s.repl, s.start, s.stop, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_replace_multiple`].
pub struct StrReplaceMultiple<'a> {
    view: &'a ColumnView<'a>,
    targets: &'a ColumnView<'a>,
    repls: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrReplaceMultiple -> Column, |s: StrReplaceMultiple<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_multiple(
        s.view.0, s.targets.0, s.repls.0, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_split_record`].
pub struct StrSplitRecord<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplitRecord -> Column, |s: StrSplitRecord<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter);
    let c = cudf_sys::strings::ffi::strings_split_record(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_rsplit_record`].
pub struct StrRsplitRecord<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplitRecord -> Column, |s: StrRsplitRecord<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter);
    let c = cudf_sys::strings::ffi::strings_rsplit_record(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_join_list_elements`].
pub struct StrJoinListElements<'a> {
    view: &'a ColumnView<'a>,
    separator: &'a str,
    narep: &'a str,
    stream: Stream,
}
simple_builder!(StrJoinListElements -> Column, |s: StrJoinListElements<'_>| {
    let c = cudf_sys::strings::ffi::strings_join_list_elements(
        s.view.0, s.separator, s.narep, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_translate`].
pub struct StrTranslate<'a> {
    view: &'a ColumnView<'a>,
    from_chars: &'a [u32],
    to_chars: &'a [u32],
    stream: Stream,
}
simple_builder!(StrTranslate -> Column, |s: StrTranslate<'_>| {
    let c = cudf_sys::strings::ffi::strings_translate(
        s.view.0, s.from_chars, s.to_chars, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_filter_characters`].
pub struct StrFilterCharacters<'a> {
    view: &'a ColumnView<'a>,
    from_chars: &'a [u32],
    to_chars: &'a [u32],
    keep: bool,
    replacement: &'a str,
    stream: Stream,
}
simple_builder!(StrFilterCharacters -> Column, |s: StrFilterCharacters<'_>| {
    let c = cudf_sys::strings::ffi::strings_filter_characters(
        s.view.0, s.from_chars, s.to_chars, s.keep, s.replacement, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::code_points`].
pub struct CodePoints<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(CodePoints -> Column, |s: CodePoints<'_>| {
    let c = cudf_sys::strings::ffi::strings_code_points(s.view.0, s.stream.as_raw())?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_cast_to_integer`].
pub struct StrCastToInteger<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    big_endian: bool,
    stream: Stream,
}
simple_builder!(StrCastToInteger -> Column, |s: StrCastToInteger<'_>| {
    let c = cudf_sys::strings::ffi::strings_cast_to_integer(
        s.view.0, s.output_type.repr, s.big_endian, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_cast_from_integer`].
pub struct StrCastFromInteger<'a> {
    view: &'a ColumnView<'a>,
    big_endian: bool,
    stream: Stream,
}
simple_builder!(StrCastFromInteger -> Column, |s: StrCastFromInteger<'_>| {
    let c = cudf_sys::strings::ffi::strings_cast_from_integer(
        s.view.0, s.big_endian, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_slice_column`].
pub struct StrSliceColumn<'a> {
    view: &'a ColumnView<'a>,
    starts: &'a ColumnView<'a>,
    stops: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrSliceColumn -> Column, |s: StrSliceColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_slice_column(
        s.view.0, s.starts.0, s.stops.0, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_extract_single`].
pub struct StrExtractSingle<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    group_index: i32,
    stream: Stream,
}
simple_builder!(StrExtractSingle -> Column, |s: StrExtractSingle<'_>| {
    let c = cudf_sys::strings::ffi::strings_extract_single(
        s.view.0, s.pattern, s.group_index, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_join_list_elements_column`].
pub struct StrJoinListElementsColumn<'a> {
    view: &'a ColumnView<'a>,
    separators: &'a ColumnView<'a>,
    separator_narep: &'a str,
    string_narep: &'a str,
    stream: Stream,
}
simple_builder!(StrJoinListElementsColumn -> Column, |s: StrJoinListElementsColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_join_list_elements_column(
        s.view.0, s.separators.0, s.separator_narep, s.string_narep, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_join_strings`].
pub struct StrJoinStrings<'a> {
    view: &'a ColumnView<'a>,
    separator: &'a str,
    narep: &'a str,
    stream: Stream,
}
simple_builder!(StrJoinStrings -> Column, |s: StrJoinStrings<'_>| {
    let c = cudf_sys::strings::ffi::strings_join_strings(
        s.view.0, s.separator, s.narep, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_find_instance`].
pub struct StrFindInstance<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    instance: i32,
    stream: Stream,
}
simple_builder!(StrFindInstance -> Column, |s: StrFindInstance<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_instance(
        s.view.0, s.target, s.instance, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`StringExt::str_like_column`].
pub struct StrLikeColumn<'a> {
    view: &'a ColumnView<'a>,
    patterns: &'a ColumnView<'a>,
    escape_char: &'a str,
    stream: Stream,
}
simple_builder!(StrLikeColumn -> Column, |s: StrLikeColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_like_column(
        s.view.0, s.patterns.0, s.escape_char, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`concatenate_strings_with_separator`].
pub struct ConcatenateStringsWithSeparator<'a> {
    tbl: &'a Table,
    separators: &'a ColumnView<'a>,
    separator_narep: &'a str,
    col_narep: &'a str,
    stream: Stream,
}
simple_builder!(ConcatenateStringsWithSeparator -> Column, |s: ConcatenateStringsWithSeparator<'_>| {
    let c = cudf_sys::strings::ffi::strings_concatenate_columns_sep_col(
        &s.tbl.0, s.separators.0, s.separator_narep, s.col_narep, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

/// Builder for [`get_json_object`].
pub struct GetJsonObject<'a> {
    col: &'a ColumnView<'a>,
    json_path: &'a str,
    allow_single_quotes: bool,
    strip_quotes: bool,
    missing_fields_as_nulls: bool,
    stream: Stream,
}
simple_builder!(GetJsonObject -> Column, |s: GetJsonObject<'_>| {
    let c = cudf_sys::strings::ffi::get_json_object(
        s.col.0, s.json_path, s.allow_single_quotes, s.strip_quotes,
        s.missing_fields_as_nulls, s.stream.as_raw(),
    )?;
    Ok(Column(c))
});

// ---------------------------------------------------------------------------
// Trait implementation
// ---------------------------------------------------------------------------

impl StringExt for ColumnView<'_> {
    fn to_lower(&self) -> ToLower<'_> {
        ToLower {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn to_upper(&self) -> ToUpper<'_> {
        ToUpper {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains<'a>(&'a self, target: &'a Scalar) -> StrContains<'a> {
        StrContains {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_starts_with<'a>(&'a self, target: &'a Scalar) -> StrStartsWith<'a> {
        StrStartsWith {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_ends_with<'a>(&'a self, target: &'a Scalar) -> StrEndsWith<'a> {
        StrEndsWith {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_find<'a>(&'a self, target: &'a Scalar) -> StrFind<'a> {
        StrFind {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace<'a>(&'a self, target: &'a Scalar, replacement: &'a Scalar) -> StrReplace<'a> {
        StrReplace {
            view: self,
            target,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn str_strip(&self) -> StrStrip<'_> {
        StrStrip {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_lstrip(&self) -> StrLstrip<'_> {
        StrLstrip {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_rstrip(&self) -> StrRstrip<'_> {
        StrRstrip {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn count_characters(&self) -> CountCharacters<'_> {
        CountCharacters {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn count_bytes(&self) -> CountBytes<'_> {
        CountBytes {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_integers(&self) -> StrFromIntegers<'_> {
        StrFromIntegers {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_integers(&self, output_type: TypeId) -> StrToIntegers<'_> {
        StrToIntegers {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_floats(&self) -> StrFromFloats<'_> {
        StrFromFloats {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_floats(&self, output_type: TypeId) -> StrToFloats<'_> {
        StrToFloats {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_pad<'a>(&'a self, width: usize, side: SideType, fill_char: &'a str) -> StrPad<'a> {
        StrPad {
            view: self,
            width,
            side,
            fill_char,
            stream: Stream::default_stream(),
        }
    }
    fn str_zfill(&self, width: usize) -> StrZfill<'_> {
        StrZfill {
            view: self,
            width,
            stream: Stream::default_stream(),
        }
    }
    fn str_zfill_by_widths<'a>(&'a self, widths: &'a ColumnView<'a>) -> StrZfillByWidths<'a> {
        StrZfillByWidths {
            view: self,
            widths,
            stream: Stream::default_stream(),
        }
    }
    fn str_slice(&self, start: i32, stop: i32, step: i32) -> StrSlice<'_> {
        StrSlice {
            view: self,
            start,
            stop,
            step,
            stream: Stream::default_stream(),
        }
    }
    fn str_repeat(&self, times: usize) -> StrRepeat<'_> {
        StrRepeat {
            view: self,
            times,
            stream: Stream::default_stream(),
        }
    }
    fn str_split<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplit<'a> {
        StrSplit {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrRsplit<'a> {
        StrRsplit {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_part<'a>(&'a self, delimiter: &'a Scalar, index: i32) -> StrSplitPart<'a> {
        StrSplitPart {
            view: self,
            delimiter,
            index,
            stream: Stream::default_stream(),
        }
    }
    fn str_join<'a>(&'a self, separator: &'a Scalar, narep: &'a Scalar) -> StrJoin<'a> {
        StrJoin {
            view: self,
            separator,
            narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_like<'a>(&'a self, pattern: &'a str, escape_char: &'a str) -> StrLike<'a> {
        StrLike {
            view: self,
            pattern,
            escape_char,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains_re<'a>(&'a self, pattern: &'a str) -> StrContainsRe<'a> {
        StrContainsRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_matches_re<'a>(&'a self, pattern: &'a str) -> StrMatchesRe<'a> {
        StrMatchesRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_count_re<'a>(&'a self, pattern: &'a str) -> StrCountRe<'a> {
        StrCountRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_re<'a>(&'a self, pattern: &'a str, replacement: &'a str) -> StrReplaceRe<'a> {
        StrReplaceRe {
            view: self,
            pattern,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn swapcase(&self) -> Swapcase<'_> {
        Swapcase {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_strip_chars<'a>(&'a self, side: SideType, to_strip: &'a str) -> StrStripChars<'a> {
        StrStripChars {
            view: self,
            side,
            to_strip,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_literal<'a>(
        &'a self,
        target: &'a str,
        repl: &'a str,
        maxrepl: i32,
    ) -> StrReplaceLiteral<'a> {
        StrReplaceLiteral {
            view: self,
            target,
            repl,
            maxrepl,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_str<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrFindStr<'a> {
        StrFindStr {
            view: self,
            target,
            start,
            stop,
            stream: Stream::default_stream(),
        }
    }
    fn str_rfind<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrRfind<'a> {
        StrRfind {
            view: self,
            target,
            start,
            stop,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains_literal<'a>(&'a self, target: &'a str) -> StrContainsLiteral<'a> {
        StrContainsLiteral {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_starts_with_str<'a>(&'a self, target: &'a str) -> StrStartsWithStr<'a> {
        StrStartsWithStr {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_ends_with_str<'a>(&'a self, target: &'a str) -> StrEndsWithStr<'a> {
        StrEndsWithStr {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_reverse(&self) -> StrReverse<'_> {
        StrReverse {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_extract<'a>(&'a self, pattern: &'a str) -> StrExtract<'a> {
        StrExtract {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_extract_all<'a>(&'a self, pattern: &'a str) -> StrExtractAll<'a> {
        StrExtractAll {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_findall<'a>(&'a self, pattern: &'a str) -> StrFindall<'a> {
        StrFindall {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_re<'a>(&'a self, pattern: &'a str) -> StrFindRe<'a> {
        StrFindRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn capitalize(&self) -> Capitalize<'_> {
        Capitalize {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn title(&self) -> Title<'_> {
        Title {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn is_title(&self) -> IsTitle<'_> {
        IsTitle {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn wrap(&self, width: i32) -> Wrap<'_> {
        Wrap {
            view: self,
            width,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_timestamps<'a>(
        &'a self,
        timestamp_type: TypeId,
        format: &'a str,
    ) -> StrToTimestamps<'a> {
        StrToTimestamps {
            view: self,
            timestamp_type,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_timestamps<'a>(&'a self, format: &'a str) -> StrFromTimestamps<'a> {
        StrFromTimestamps {
            view: self,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_timestamp<'a>(&'a self, format: &'a str) -> StrIsTimestamp<'a> {
        StrIsTimestamp {
            view: self,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_booleans<'a>(&'a self, true_string: &'a str) -> StrToBooleans<'a> {
        StrToBooleans {
            view: self,
            true_string,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_booleans<'a>(
        &'a self,
        true_string: &'a str,
        false_string: &'a str,
    ) -> StrFromBooleans<'a> {
        StrFromBooleans {
            view: self,
            true_string,
            false_string,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_durations<'a>(
        &'a self,
        duration_type: TypeId,
        format: &'a str,
    ) -> StrToDurations<'a> {
        StrToDurations {
            view: self,
            duration_type,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_durations<'a>(&'a self, format: &'a str) -> StrFromDurations<'a> {
        StrFromDurations {
            view: self,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_fixed_point(&self, type_id: TypeId, scale: i32) -> StrToFixedPoint<'_> {
        StrToFixedPoint {
            view: self,
            type_id,
            scale,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_fixed_point(&self) -> StrFromFixedPoint<'_> {
        StrFromFixedPoint {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_fixed_point(&self, type_id: TypeId, scale: i32) -> StrIsFixedPoint<'_> {
        StrIsFixedPoint {
            view: self,
            type_id,
            scale,
            stream: Stream::default_stream(),
        }
    }
    fn url_encode(&self) -> UrlEncode<'_> {
        UrlEncode {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn url_decode(&self) -> UrlDecode<'_> {
        UrlDecode {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn ipv4_to_integers(&self) -> Ipv4ToIntegers<'_> {
        Ipv4ToIntegers {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn integers_to_ipv4(&self) -> IntegersToIpv4<'_> {
        IntegersToIpv4 {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn is_ipv4(&self) -> IsIpv4<'_> {
        IsIpv4 {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRe<'a> {
        StrSplitRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrRsplitRe<'a> {
        StrRsplitRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_record_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRecordRe<'a> {
        StrSplitRecordRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit_record_re<'a>(
        &'a self,
        pattern: &'a str,
        maxsplit: i32,
    ) -> StrRsplitRecordRe<'a> {
        StrRsplitRecordRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_partition<'a>(&'a self, delimiter: &'a str) -> StrPartition<'a> {
        StrPartition {
            view: self,
            delimiter,
            stream: Stream::default_stream(),
        }
    }
    fn str_rpartition<'a>(&'a self, delimiter: &'a str) -> StrRpartition<'a> {
        StrRpartition {
            view: self,
            delimiter,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_with_backrefs<'a>(
        &'a self,
        pattern: &'a str,
        replacement: &'a str,
    ) -> StrReplaceWithBackrefs<'a> {
        StrReplaceWithBackrefs {
            view: self,
            pattern,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn str_repeat_column<'a>(&'a self, repeat_times: &'a ColumnView<'a>) -> StrRepeatColumn<'a> {
        StrRepeatColumn {
            view: self,
            repeat_times,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrContainsMultiple<'a> {
        StrContainsMultiple {
            view: self,
            targets,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrFindMultiple<'a> {
        StrFindMultiple {
            view: self,
            targets,
            stream: Stream::default_stream(),
        }
    }
    fn all_characters_of_type(&self, types: u32, verify_types: u32) -> AllCharactersOfType<'_> {
        AllCharactersOfType {
            view: self,
            types,
            verify_types,
            stream: Stream::default_stream(),
        }
    }
    fn filter_characters_of_type<'a>(
        &'a self,
        types_to_remove: u32,
        replacement: &'a str,
        types_to_keep: u32,
    ) -> FilterCharactersOfType<'a> {
        FilterCharactersOfType {
            view: self,
            types_to_remove,
            replacement,
            types_to_keep,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_integer(&self) -> StrIsInteger<'_> {
        StrIsInteger {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_integer_with_type(&self, int_type: TypeId) -> StrIsIntegerWithType<'_> {
        StrIsIntegerWithType {
            view: self,
            int_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_float(&self) -> StrIsFloat<'_> {
        StrIsFloat {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_hex_to_integers(&self, output_type: TypeId) -> StrHexToIntegers<'_> {
        StrHexToIntegers {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_hex(&self) -> StrIsHex<'_> {
        StrIsHex {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_integers_to_hex(&self) -> StrIntegersToHex<'_> {
        StrIntegersToHex {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_slice<'a>(
        &'a self,
        repl: &'a str,
        start: i32,
        stop: i32,
    ) -> StrReplaceSlice<'a> {
        StrReplaceSlice {
            view: self,
            repl,
            start,
            stop,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_multiple<'a>(
        &'a self,
        targets: &'a ColumnView<'a>,
        repls: &'a ColumnView<'a>,
    ) -> StrReplaceMultiple<'a> {
        StrReplaceMultiple {
            view: self,
            targets,
            repls,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_record<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplitRecord<'a> {
        StrSplitRecord {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit_record<'a>(
        &'a self,
        delimiter: &'a Scalar,
        maxsplit: i32,
    ) -> StrRsplitRecord<'a> {
        StrRsplitRecord {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_join_list_elements<'a>(
        &'a self,
        separator: &'a str,
        narep: &'a str,
    ) -> StrJoinListElements<'a> {
        StrJoinListElements {
            view: self,
            separator,
            narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_translate<'a>(&'a self, from_chars: &'a [u32], to_chars: &'a [u32]) -> StrTranslate<'a> {
        StrTranslate {
            view: self,
            from_chars,
            to_chars,
            stream: Stream::default_stream(),
        }
    }
    fn str_filter_characters<'a>(
        &'a self,
        from_chars: &'a [u32],
        to_chars: &'a [u32],
        keep: bool,
        replacement: &'a str,
    ) -> StrFilterCharacters<'a> {
        StrFilterCharacters {
            view: self,
            from_chars,
            to_chars,
            keep,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn code_points(&self) -> CodePoints<'_> {
        CodePoints {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_cast_to_integer(&self, output_type: TypeId, big_endian: bool) -> StrCastToInteger<'_> {
        StrCastToInteger {
            view: self,
            output_type,
            big_endian,
            stream: Stream::default_stream(),
        }
    }
    fn str_cast_from_integer(&self, big_endian: bool) -> StrCastFromInteger<'_> {
        StrCastFromInteger {
            view: self,
            big_endian,
            stream: Stream::default_stream(),
        }
    }
    fn str_slice_column<'a>(
        &'a self,
        starts: &'a ColumnView<'a>,
        stops: &'a ColumnView<'a>,
    ) -> StrSliceColumn<'a> {
        StrSliceColumn {
            view: self,
            starts,
            stops,
            stream: Stream::default_stream(),
        }
    }
    fn str_extract_single<'a>(
        &'a self,
        pattern: &'a str,
        group_index: i32,
    ) -> StrExtractSingle<'a> {
        StrExtractSingle {
            view: self,
            pattern,
            group_index,
            stream: Stream::default_stream(),
        }
    }
    fn str_join_list_elements_column<'a>(
        &'a self,
        separators: &'a ColumnView<'a>,
        separator_narep: &'a str,
        string_narep: &'a str,
    ) -> StrJoinListElementsColumn<'a> {
        StrJoinListElementsColumn {
            view: self,
            separators,
            separator_narep,
            string_narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_join_strings<'a>(&'a self, separator: &'a str, narep: &'a str) -> StrJoinStrings<'a> {
        StrJoinStrings {
            view: self,
            separator,
            narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_instance<'a>(&'a self, target: &'a str, instance: i32) -> StrFindInstance<'a> {
        StrFindInstance {
            view: self,
            target,
            instance,
            stream: Stream::default_stream(),
        }
    }
    fn str_like_column<'a>(
        &'a self,
        patterns: &'a ColumnView<'a>,
        escape_char: &'a str,
    ) -> StrLikeColumn<'a> {
        StrLikeColumn {
            view: self,
            patterns,
            escape_char,
            stream: Stream::default_stream(),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Concatenate string columns row-wise with per-row separator column.
///
/// Returns a [`ConcatenateStringsWithSeparator`] builder. Use `.stream()` to set
/// a custom CUDA stream, then `.call()` to execute.
pub fn concatenate_strings_with_separator<'a>(
    tbl: &'a Table,
    separators: &'a ColumnView<'a>,
    separator_narep: &'a str,
    col_narep: &'a str,
) -> ConcatenateStringsWithSeparator<'a> {
    ConcatenateStringsWithSeparator {
        tbl,
        separators,
        separator_narep,
        col_narep,
        stream: Stream::default_stream(),
    }
}

/// Extract values from JSON strings using a `JSONPath` expression.
///
/// Each row in the input column must be a valid JSON string. The `JSONPath`
/// expression is applied to every row, returning the matched values as strings.
///
/// Returns a [`GetJsonObject`] builder. Use `.stream()` to set a custom CUDA
/// stream, then `.call()` to execute.
pub fn get_json_object<'a>(
    col: &'a ColumnView<'a>,
    json_path: &'a str,
    allow_single_quotes: bool,
    strip_quotes: bool,
    missing_fields_as_nulls: bool,
) -> GetJsonObject<'a> {
    GetJsonObject {
        col,
        json_path,
        allow_single_quotes,
        strip_quotes,
        missing_fields_as_nulls,
        stream: Stream::default_stream(),
    }
}

/// String character type bitmask constants.
pub mod char_types {
    /// Decimal characters (e.g. `0`–`9` in various scripts).
    pub const DECIMAL: u32 = 1;
    /// Numeric characters (includes fractions, subscripts, etc.).
    pub const NUMERIC: u32 = 2;
    /// Digit characters (ASCII `0`–`9`).
    pub const DIGIT: u32 = 4;
    /// Alphabetic characters.
    pub const ALPHA: u32 = 8;
    /// Whitespace characters.
    pub const SPACE: u32 = 16;
    /// Uppercase characters.
    pub const UPPER: u32 = 32;
    /// Lowercase characters.
    pub const LOWER: u32 = 64;
    /// Alphanumeric: `DECIMAL | NUMERIC | DIGIT | ALPHA`.
    pub const ALPHANUM: u32 = DECIMAL | NUMERIC | DIGIT | ALPHA;
    /// Case types: `UPPER | LOWER`.
    pub const CASE_TYPES: u32 = UPPER | LOWER;
    /// All character types combined.
    pub const ALL_TYPES: u32 = ALPHANUM | CASE_TYPES | SPACE;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;

    fn make_string_col(values: &[&str]) -> Column {
        Column::from_strings(values).call().unwrap()
    }

    #[test]
    fn to_lower_basic() {
        let col = make_string_col(&["HELLO", "World", "foo"]);
        let result = col.view().to_lower().call().unwrap();
        assert_eq!(
            result.to_vec_string().call().unwrap(),
            vec!["hello", "world", "foo"]
        );
    }

    #[test]
    fn to_upper_basic() {
        let col = make_string_col(&["hello", "World", "FOO"]);
        let result = col.view().to_upper().call().unwrap();
        assert_eq!(
            result.to_vec_string().call().unwrap(),
            vec!["HELLO", "WORLD", "FOO"]
        );
    }

    #[test]
    fn contains_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("abc");
        let result = col.view().str_contains(&target).call().unwrap();
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![true, false, true]
        );
    }

    #[test]
    fn starts_with_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("ab");
        let result = col.view().str_starts_with(&target).call().unwrap();
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![true, false, true]
        );
    }

    #[test]
    fn ends_with_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("ef");
        let result = col.view().str_ends_with(&target).call().unwrap();
        assert_eq!(
            result.to_vec_bool().call().unwrap(),
            vec![false, true, true]
        );
    }

    #[test]
    fn find_basic() {
        let col = make_string_col(&["hello world", "goodbye", "hello"]);
        let target = Scalar::from_string("lo");
        let result = col.view().str_find(&target).call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![3, -1, 3]);
    }

    #[test]
    fn replace_basic() {
        let col = make_string_col(&["hello", "world"]);
        let target = Scalar::from_string("o");
        let replacement = Scalar::from_string("0");
        let result = col
            .view()
            .str_replace(&target, &replacement)
            .call()
            .unwrap();
        assert_eq!(
            result.to_vec_string().call().unwrap(),
            vec!["hell0", "w0rld"]
        );
    }

    #[test]
    fn strip_basic() {
        let col = make_string_col(&["  hello  ", " world", "foo "]);
        let result = col.view().str_strip().call().unwrap();
        assert_eq!(
            result.to_vec_string().call().unwrap(),
            vec!["hello", "world", "foo"]
        );
    }

    #[test]
    fn lstrip_basic() {
        let col = make_string_col(&["  hello  ", " world"]);
        let result = col.view().str_lstrip().call().unwrap();
        assert_eq!(
            result.to_vec_string().call().unwrap(),
            vec!["hello  ", "world"]
        );
    }

    #[test]
    fn rstrip_basic() {
        let col = make_string_col(&["  hello  ", "world "]);
        let result = col.view().str_rstrip().call().unwrap();
        assert_eq!(
            result.to_vec_string().call().unwrap(),
            vec!["  hello", "world"]
        );
    }

    #[test]
    fn count_characters_basic() {
        let col = make_string_col(&["hello", "ab", ""]);
        let result = col.view().count_characters().call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![5, 2, 0]);
    }

    #[test]
    fn count_bytes_basic() {
        let col = make_string_col(&["hello", "ab", ""]);
        let result = col.view().count_bytes().call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![5, 2, 0]);
    }

    #[test]
    fn strings_from_integers_basic() {
        let col = Column::from_strings(&["1", "2", "3"]).call().unwrap();
        let int_col = col.view().str_to_integers(TypeId::INT32).call().unwrap();
        assert_eq!(int_col.to_vec_i32().call().unwrap(), vec![1, 2, 3]);

        let back = int_col.view().str_from_integers().call().unwrap();
        assert_eq!(back.to_vec_string().call().unwrap(), vec!["1", "2", "3"]);
    }

    #[test]
    fn strings_to_integers_basic() {
        let col = make_string_col(&["10", "20", "30"]);
        let result = col.view().str_to_integers(TypeId::INT32).call().unwrap();
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn strings_from_floats_basic() {
        let col = Scalar::from_f64(1.5);
        let float_col = Column::from_scalar(&col, 2).call().unwrap();
        let result = float_col.view().str_from_floats().call().unwrap();
        assert_eq!(result.len(), 2);
        let strs = result.to_vec_string().call().unwrap();
        assert!(strs.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn strings_to_floats_basic() {
        let col = make_string_col(&["1.5", "2.5", "3.0"]);
        let result = col.view().str_to_floats(TypeId::FLOAT64).call().unwrap();
        let vals = result.to_vec_f64().call().unwrap();
        assert!((vals[0] - 1.5).abs() < 1e-9);
        assert!((vals[1] - 2.5).abs() < 1e-9);
        assert!((vals[2] - 3.0).abs() < 1e-9);
    }
}
