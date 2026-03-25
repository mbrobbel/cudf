// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        type Scalar = crate::ffi::Scalar;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- String case operations --

        /// Converts strings to lower case.
        fn strings_to_lower(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts strings to upper case.
        fn strings_to_upper(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Swap case (upper->lower, lower->upper).
        fn strings_swapcase(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Capitalizes the first character of each string.
        fn strings_capitalize(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Title-cases each string (first char of each word uppercase).
        fn strings_title(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns BOOL8 column indicating whether each string is title-cased.
        fn strings_is_title(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String find / contains --

        /// Returns BOOL8 column indicating whether each string contains the target.
        fn strings_contains(
            col: &column_view,
            target: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns BOOL8 column indicating whether each string starts with the target.
        fn strings_starts_with(
            col: &column_view,
            target: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns BOOL8 column indicating whether each string ends with the target.
        fn strings_ends_with(
            col: &column_view,
            target: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns INT32 column with position of first occurrence of target in each string.
        fn strings_find(
            col: &column_view,
            target: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Check if string contains literal target.
        fn strings_contains_str(
            col: &column_view,
            target: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Check if string starts with literal target.
        fn strings_starts_with_str(
            col: &column_view,
            target: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Check if string ends with literal target.
        fn strings_ends_with_str(
            col: &column_view,
            target: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Find first position of target in range [start, stop).
        fn strings_find_str(
            col: &column_view,
            target: &str,
            start: i32,
            stop: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Find last position of target (reverse find).
        fn strings_rfind(
            col: &column_view,
            target: &str,
            start: i32,
            stop: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Check if strings contain multiple targets (returns Table of BOOL8 columns).
        fn strings_contains_multiple(
            col: &column_view,
            targets: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Find positions of multiple targets in each string (returns lists column).
        fn strings_find_multiple(
            col: &column_view,
            targets: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds the nth instance of target in each string, returns position (-1 if not found).
        fn strings_find_instance(
            col: &column_view,
            target: &str,
            instance: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String replace --

        /// Replaces occurrences of target with replacement in each string.
        fn strings_replace(
            col: &column_view,
            target: &Scalar,
            replacement: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Literal string replace with maxrepl.
        fn strings_replace_literal(
            col: &column_view,
            target: &str,
            repl: &str,
            maxrepl: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replace substring positions [start, stop) with replacement string.
        fn strings_replace_slice(
            col: &column_view,
            repl: &str,
            start: i32,
            stop: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replace multiple target strings with corresponding replacements.
        fn strings_replace_multiple(
            col: &column_view,
            targets: &column_view,
            repls: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String strip / pad --

        /// Strips whitespace from both sides of each string.
        fn strings_strip(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Strips whitespace from the left side of each string.
        fn strings_lstrip(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Strips whitespace from the right side of each string.
        fn strings_rstrip(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Strip specified characters from sides of strings.
        fn strings_strip_chars(
            col: &column_view,
            side: i32,
            to_strip: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Pads strings to a minimum width.
        fn strings_pad(
            col: &column_view,
            width: i32,
            side: i32,
            fill_char: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Zero-fills strings to a minimum width.
        fn strings_zfill(col: &column_view, width: i32, stream: usize)
        -> Result<UniquePtr<Column>>;

        /// Zero-fill strings using per-row widths column.
        fn strings_zfill_by_widths(
            col: &column_view,
            widths: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String attributes --

        /// Returns INT32 column with character count of each string.
        fn strings_count_characters(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns INT32 column with byte count of each string.
        fn strings_count_bytes(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns INT32 column of Unicode code points for all characters.
        fn strings_code_points(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String character types --

        /// Check if all characters in each string match the given type bitmask.
        fn strings_all_characters_of_type(
            col: &column_view,
            types: u32,
            verify_types: u32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Filter characters by type, replacing removed chars with replacement string.
        fn strings_filter_characters_of_type(
            col: &column_view,
            types_to_remove: u32,
            replacement: &str,
            types_to_keep: u32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String slice / substring --

        /// Extracts a substring [start, stop) with optional step.
        fn strings_slice(
            col: &column_view,
            start: i32,
            stop: i32,
            step: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Slice strings using per-row start/stop columns.
        fn strings_slice_column(
            col: &column_view,
            starts: &column_view,
            stops: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String repeat --

        /// Repeats each string N times.
        fn strings_repeat(
            col: &column_view,
            repeat_times: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Repeat each string by count in another column.
        fn strings_repeat_column(
            col: &column_view,
            repeat_times: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Repeats a string scalar N times.
        fn repeat_string_scalar(
            input: &Scalar,
            repeat_times: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        // -- String reverse / wrap --

        /// Reverse characters within each string.
        fn strings_reverse(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Wraps strings onto multiple lines shorter than `width`.
        fn strings_wrap(col: &column_view, width: i32, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String split --

        /// Splits strings by a delimiter into a table of columns.
        fn strings_split_to_table(
            col: &column_view,
            delimiter: &Scalar,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Right-splits strings by a delimiter into a table of columns.
        fn strings_rsplit_to_table(
            col: &column_view,
            delimiter: &Scalar,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Returns the Nth part after splitting by delimiter.
        fn strings_split_part(
            col: &column_view,
            delimiter: &Scalar,
            index: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Split strings by delimiter into lists column (left-to-right).
        fn strings_split_record(
            col: &column_view,
            delimiter: &Scalar,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Split strings by delimiter into lists column (right-to-left).
        fn strings_rsplit_record(
            col: &column_view,
            delimiter: &Scalar,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Partition around first delimiter into 3 columns.
        fn strings_partition(
            col: &column_view,
            delimiter: &str,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Partition around last delimiter into 3 columns.
        fn strings_rpartition(
            col: &column_view,
            delimiter: &str,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- String join / concatenate --

        /// Joins all strings in a column into a single string column of size 1.
        fn strings_join(
            col: &column_view,
            separator: &Scalar,
            narep: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Joins all strings in a column with a separator into a single-row column.
        fn strings_join_strings(
            col: &column_view,
            separator: &str,
            narep: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Concatenates string columns element-wise with a separator.
        fn strings_concatenate_columns(
            tbl: &Table,
            separator: &Scalar,
            narep: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Row-wise concatenation of string columns with per-row separator column.
        fn strings_concatenate_columns_sep_col(
            tbl: &Table,
            separators: &column_view,
            separator_narep: &str,
            col_narep: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Join lists of strings into a single string per row with separator.
        fn strings_join_list_elements(
            col: &column_view,
            separator: &str,
            narep: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Join lists of strings with per-row separator column.
        fn strings_join_list_elements_column(
            col: &column_view,
            separators: &column_view,
            separator_narep: &str,
            string_narep: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String regex operations --

        /// SQL LIKE pattern matching.
        fn strings_like(
            col: &column_view,
            pattern: &str,
            escape_char: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// SQL LIKE with per-row patterns from a strings column.
        fn strings_like_column(
            col: &column_view,
            patterns: &column_view,
            escape_char: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex contains check.
        fn strings_contains_re(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex match from start of string.
        fn strings_matches_re(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Counts regex matches per string.
        fn strings_count_re(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces regex matches with a replacement string.
        fn strings_replace_re(
            col: &column_view,
            pattern: &str,
            replacement: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex replace with back-reference template.
        fn strings_replace_with_backrefs(
            col: &column_view,
            pattern: &str,
            replacement: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex extract groups into a Table (one column per group).
        fn strings_extract(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Extract a single regex group from each string.
        fn strings_extract_single(
            col: &column_view,
            pattern: &str,
            group_index: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex extract all matches into a lists column.
        fn strings_extract_all_record(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Find all regex matches as a lists column.
        fn strings_findall(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Find first regex match position.
        fn strings_find_re(
            col: &column_view,
            pattern: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex split to table of columns.
        fn strings_split_re(
            col: &column_view,
            pattern: &str,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Regex reverse split to table of columns.
        fn strings_rsplit_re(
            col: &column_view,
            pattern: &str,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Regex split to lists column.
        fn strings_split_record_re(
            col: &column_view,
            pattern: &str,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Regex reverse split to lists column.
        fn strings_rsplit_record_re(
            col: &column_view,
            pattern: &str,
            maxsplit: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String conversions: integers / floats --

        /// Converts an integer column to a string column.
        fn strings_from_integers(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts a string column to an integer column.
        fn strings_to_integers(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Converts a float column to a string column.
        fn strings_from_floats(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts a string column to a float column.
        fn strings_to_floats(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String conversions: timestamps --

        /// Converts strings to timestamps using format pattern.
        fn strings_to_timestamps(
            col: &column_view,
            timestamp_type_id: i32,
            format: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Converts timestamp column to strings using format pattern.
        fn strings_from_timestamps(
            col: &column_view,
            format: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Verifies strings can be parsed as timestamps with given format.
        fn strings_is_timestamp(
            col: &column_view,
            format: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String conversions: booleans --

        /// Converts strings to booleans using true_string match.
        fn strings_to_booleans(
            col: &column_view,
            true_string: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Converts boolean column to strings.
        fn strings_from_booleans(
            col: &column_view,
            true_string: &str,
            false_string: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String conversions: durations --

        /// Converts strings to durations using format pattern.
        fn strings_to_durations(
            col: &column_view,
            duration_type_id: i32,
            format: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Converts duration column to strings using format pattern.
        fn strings_from_durations(
            col: &column_view,
            format: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String conversions: fixed-point --

        /// Converts strings to fixed-point decimal.
        fn strings_to_fixed_point(
            col: &column_view,
            type_id: i32,
            scale: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Converts fixed-point column to strings.
        fn strings_from_fixed_point(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Verifies strings can be parsed as fixed-point.
        fn strings_is_fixed_point(
            col: &column_view,
            type_id: i32,
            scale: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String conversions: URLs --

        /// URL-encodes each string.
        fn strings_url_encode(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// URL-decodes each string.
        fn strings_url_decode(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String conversions: IPv4 --

        /// Converts IPv4 strings to integers.
        fn strings_ipv4_to_integers(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Converts integers to IPv4 strings.
        fn strings_integers_to_ipv4(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Verifies strings are valid IPv4 format.
        fn strings_is_ipv4(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String validation --

        /// Check if all chars in each string are valid integers.
        fn strings_is_integer(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Check if all chars are valid integers within given int type range.
        fn strings_is_integer_with_type(
            col: &column_view,
            int_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Check if all chars in each string are valid floats.
        fn strings_is_float(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String conversions: hex --

        /// Convert hex strings to integers.
        fn strings_hex_to_integers(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Check if strings are valid hex format.
        fn strings_is_hex(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Convert integers to hex strings.
        fn strings_integers_to_hex(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- String conversions: integer binary cast --

        /// Encode strings as integers (binary byte representation). big_endian controls byte order.
        fn strings_cast_to_integer(
            col: &column_view,
            output_type_id: i32,
            big_endian: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Decode integer-encoded bytes back to strings.
        fn strings_cast_from_integer(
            col: &column_view,
            big_endian: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String translate / filter --

        /// Translate individual characters using from->to mapping (flat arrays of char_utf8 as u32).
        fn strings_translate(
            col: &column_view,
            from_chars: &[u32],
            to_chars: &[u32],
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Filter character ranges. keep=true keeps ranges, false removes them.
        fn strings_filter_characters(
            col: &column_view,
            from_chars: &[u32],
            to_chars: &[u32],
            keep: bool,
            replacement: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String formatting --

        /// Format a list-of-strings column into a single formatted strings column.
        fn strings_format_list_column(
            col: &column_view,
            na_rep: &str,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- String column construction / extraction --

        /// Creates a string column from a vector of strings.
        fn make_string_column(strings: Vec<String>, stream: usize) -> UniquePtr<Column>;

        /// Creates a string column from pre-built chars and offsets buffers.
        ///
        /// `chars` is the concatenation of all string bytes.
        /// `offsets` is an `i32` array of length `n + 1` where `offsets[i]` is the
        /// byte start of string `i` and `offsets[n]` equals `chars.len()`.
        ///
        /// This performs two host-to-device copies (chars + offsets) and builds
        /// the column in one shot, avoiding per-string GPU allocations.
        fn make_string_column_from_offsets(
            chars: &[u8],
            offsets: &[i32],
            stream: usize,
        ) -> UniquePtr<Column>;

        /// Copies string column data to a host vector of strings.
        fn column_to_host_strings(col: &Column, stream: usize) -> Vec<String>;

        /// Copies string column_view data to a host vector (avoids deep copy).
        fn view_to_host_strings(col: &column_view, stream: usize) -> Vec<String>;

        // -- JSON path extraction --

        /// Extract values from JSON strings using a JSONPath expression.
        fn get_json_object(
            col: &column_view,
            json_path: &str,
            allow_single_quotes: bool,
            strip_quotes: bool,
            missing_fields_as_nulls: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
