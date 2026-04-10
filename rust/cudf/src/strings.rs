// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! String operations on GPU columns.
//!
//! The [`StringExt`] trait provides a comprehensive set of per-element string
//! operations: case conversion, searching, replacing, splitting, joining,
//! padding, regex matching, type conversions, encoding, and more.
//!
//! Free functions in this module provide multi-column string operations:
//! [`concatenate_strings_with_separator`] and [`get_json_object`].
//!
//! The [`char_types`] submodule defines bitmask constants for character type
//! classification.
//!
//! # Examples
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::strings::StringExt;
//! use cudf::stream::GpuOp;
//!
//! let col = Column::from_strings(&["Hello", "World"]).call()?;
//! let lower = col.view().to_lower().call()?;
//! let lengths = col.view().count_characters().call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

use crate::column::ColumnView;
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;
use crate::table::UnboundTable;

#[doc(alias = "side_type")]
/// Specifies which side(s) of a string to operate on for padding and stripping.
///
/// Re-exported from `cudf_sys`. Used with [`StringExt::str_pad`] and
/// [`StringExt::str_strip_chars`].
pub use cudf_sys::ffi::SideType;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for crate::column::ColumnView<'_> {}

#[doc(alias = "strings")]
/// Extension trait for string operations on GPU columns.
///
/// Provides a comprehensive set of per-element string operations for columns
/// with type `STRING`. This trait is implemented for [`ColumnView`] and is
/// sealed -- it cannot be implemented outside this crate.
///
/// All methods return builder structs that implement [`GpuOp`](crate::stream::GpuOp).
/// Call `.call()` to execute the operation, or chain `.stream(s)` first to run
/// on a non-default CUDA stream.
///
/// # Errors
///
/// Methods generally return an error if the input column does not have the
/// expected type (e.g. `STRING` for most methods, integer for `str_from_integers`).
pub trait StringExt: private::Sealed {
    /// Converts each string element to lower case.
    ///
    /// Returns a `STRING` column with all characters converted to their
    /// lowercase equivalents. Supports full Unicode case mapping.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_strings(&["HELLO", "World"]).call()?;
    /// let lower = col.view().to_lower().call()?;
    /// // ["hello", "world"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn to_lower(&self) -> ToLower<'_>;

    /// Converts each string element to upper case.
    ///
    /// Returns a `STRING` column with all characters converted to their
    /// uppercase equivalents. Supports full Unicode case mapping.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", "World"]).call()?;
    /// let upper = col.view().to_upper().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn to_upper(&self) -> ToUpper<'_>;

    /// Tests whether each string contains the `target` substring.
    ///
    /// The `target` must be a string [`Scalar`]. Returns a `BOOL8` column:
    /// `true` if the string contains `target`, `false` otherwise. Null rows
    /// produce null results.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["abc", "def"]).call()?;
    /// let target = Scalar::from_string("abc");
    /// let found = col.view().str_contains(&target).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_contains<'a>(&'a self, target: &'a Scalar) -> StrContains<'a>;

    /// Tests whether each string starts with the `target` prefix.
    ///
    /// The `target` must be a string [`Scalar`]. Returns a `BOOL8` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["http://example.com", "file.txt"]).call()?;
    /// let prefix = Scalar::from_string("http");
    /// let result = col.view().str_starts_with(&prefix).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_starts_with<'a>(&'a self, target: &'a Scalar) -> StrStartsWith<'a>;

    /// Tests whether each string ends with the `target` suffix.
    ///
    /// The `target` must be a string [`Scalar`]. Returns a `BOOL8` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["notes.txt", "image.png"]).call()?;
    /// let suffix = Scalar::from_string(".txt");
    /// let result = col.view().str_ends_with(&suffix).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_ends_with<'a>(&'a self, target: &'a Scalar) -> StrEndsWith<'a>;

    /// Finds the first position of the `target` substring in each string.
    ///
    /// The `target` must be a string [`Scalar`]. Returns an `INT32` column
    /// containing the 0-based character position of the first occurrence,
    /// or `-1` if the target is not found.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello world", "goodbye"]).call()?;
    /// let target = Scalar::from_string("lo");
    /// let positions = col.view().str_find(&target).call()?;
    /// // "hello world" => 3, "goodbye" => -1
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_find<'a>(&'a self, target: &'a Scalar) -> StrFind<'a>;

    /// Replaces all occurrences of `target` with `replacement` in each string.
    ///
    /// Both `target` and `replacement` must be string [`Scalar`] values.
    /// Returns a `STRING` column with all substitutions applied.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", "world"]).call()?;
    /// let target = Scalar::from_string("o");
    /// let repl = Scalar::from_string("0");
    /// let result = col.view().str_replace(&target, &repl).call()?;
    /// // "hello" => "hell0"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_replace<'a>(&'a self, target: &'a Scalar, replacement: &'a Scalar) -> StrReplace<'a>;

    /// Strips leading and trailing whitespace from each string.
    ///
    /// Returns a `STRING` column with whitespace removed from both sides.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["  hello  ", " world"]).call()?;
    /// let stripped = col.view().str_strip().call()?;
    /// // "  hello  " => "hello"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_strip(&self) -> StrStrip<'_>;

    /// Strips leading whitespace from each string.
    ///
    /// Returns a `STRING` column with whitespace removed from the left side only.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["  hello  ", " world"]).call()?;
    /// let stripped = col.view().str_lstrip().call()?;
    /// // "  hello  " => "hello  "
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_lstrip(&self) -> StrLstrip<'_>;

    /// Strips trailing whitespace from each string.
    ///
    /// Returns a `STRING` column with whitespace removed from the right side only.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["  hello  ", " world"]).call()?;
    /// let stripped = col.view().str_rstrip().call()?;
    /// // "  hello  " => "  hello"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_rstrip(&self) -> StrRstrip<'_>;

    /// Returns the number of characters in each string.
    ///
    /// Returns an `INT32` column containing Unicode character counts (not byte
    /// counts). For multi-byte UTF-8 characters, this differs from
    /// [`count_bytes`](StringExt::count_bytes).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", ""]).call()?;
    /// let lengths = col.view().count_characters().call()?;
    /// // "hello" => 5, "" => 0
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn count_characters(&self) -> CountCharacters<'_>;

    /// Returns the number of bytes in each string's UTF-8 encoding.
    ///
    /// Returns an `INT32` column containing byte lengths. For ASCII-only
    /// strings this equals the character count; for multi-byte UTF-8 it
    /// may be larger.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", "caf\u{e9}"]).call()?;
    /// let byte_lens = col.view().count_bytes().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn count_bytes(&self) -> CountBytes<'_>;

    /// Converts an integer column to its decimal string representation.
    ///
    /// The input column must be an integer type (`INT8`, `INT16`, `INT32`,
    /// `INT64`, etc.). Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let int_col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let strings = int_col.view().str_from_integers().call()?;
    /// // [1, 2, 3] => ["1", "2", "3"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_from_integers(&self) -> StrFromIntegers<'_>;

    /// Parses strings as integers of the specified `output_type`.
    ///
    /// The `output_type` determines the target integer type (e.g.
    /// [`TypeId::INT32`], [`TypeId::INT64`]). Strings that cannot be parsed
    /// produce zero or undefined values.
    ///
    /// Returns a column of the specified integer type.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["10", "20"]).call()?;
    /// let ints = col.view().str_to_integers(TypeId::INT32).call()?;
    /// // ["10", "20"] => [10, 20]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_to_integers(&self, output_type: TypeId) -> StrToIntegers<'_>;

    /// Converts a floating-point column to its string representation.
    ///
    /// The input column must be a float type (`FLOAT32` or `FLOAT64`).
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let float_col = Column::from_slice_f64(&[1.5, 2.5]).call()?;
    /// let strings = float_col.view().str_from_floats().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_from_floats(&self) -> StrFromFloats<'_>;

    /// Parses strings as floating-point values of the specified `output_type`.
    ///
    /// The `output_type` must be [`TypeId::FLOAT32`] or [`TypeId::FLOAT64`].
    /// Strings that cannot be parsed produce NaN.
    ///
    /// Returns a column of the specified float type.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["1.5", "2.5"]).call()?;
    /// let floats = col.view().str_to_floats(TypeId::FLOAT64).call()?;
    /// // ["1.5", "2.5"] => [1.5, 2.5]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_to_floats(&self, output_type: TypeId) -> StrToFloats<'_>;

    /// Pads each string to a minimum `width` using a fill character.
    ///
    /// The `side` parameter controls where padding is added: left, right, or
    /// both sides (see [`SideType`]). The `fill_char` is a single-character
    /// string used for padding. Strings already at or above `width` are
    /// unchanged.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::{SideType, StringExt};
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hi", "hello"]).call()?;
    /// let padded = col.view().str_pad(10, SideType::LEFT, " ").call()?;
    /// // "hi" => "        hi"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_pad<'a>(&'a self, width: usize, side: SideType, fill_char: &'a str) -> StrPad<'a>;

    /// Pads numeric strings with leading zeros to reach `width` characters.
    ///
    /// A leading sign character (`+` or `-`) is preserved before the zeros.
    /// Non-numeric strings are left-padded with zeros. Strings already at or
    /// above `width` are unchanged.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["42", "-7"]).call()?;
    /// let zfilled = col.view().str_zfill(5).call()?;
    /// // "42" => "00042", "-7" => "-0007"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_zfill(&self, width: usize) -> StrZfill<'_>;

    /// Pads numeric strings with leading zeros using per-row widths from a column.
    ///
    /// The `widths` column must be an `INT32` column with the same number of
    /// rows. Behavior is otherwise identical to [`str_zfill`](StringExt::str_zfill).
    ///
    /// Returns a `STRING` column.
    fn str_zfill_by_widths<'a>(&'a self, widths: &'a ColumnView<'a>) -> StrZfillByWidths<'a>;

    /// Extracts a substring from each string using character indices.
    ///
    /// Extracts characters in the half-open range `[start, stop)` with the
    /// given `step`. A `step` of `1` extracts every character. Negative
    /// `start`/`stop` values are not supported (they are treated as 0).
    /// A `stop` of `-1` means through the end of the string.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", "world"]).call()?;
    /// let sliced = col.view().str_slice(0, 3, 1).call()?;
    /// // "hello" => "hel"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_slice(&self, start: i32, stop: i32, step: i32) -> StrSlice<'_>;

    /// Repeats each string `times` times by concatenating it with itself.
    ///
    /// A `times` value of `0` produces empty strings. Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["ab", "cd"]).call()?;
    /// let repeated = col.view().str_repeat(3).call()?;
    /// // "ab" => "ababab"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_repeat(&self, times: usize) -> StrRepeat<'_>;

    /// Splits each string by a delimiter into a [`Table`] of string columns.
    ///
    /// The `delimiter` must be a string [`Scalar`]. Each output column
    /// corresponds to a split part. The `maxsplit` parameter limits the number
    /// of splits (`-1` for unlimited). Splitting proceeds left-to-right.
    ///
    /// Returns a [`Table`] with one column per split part.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["a,b,c", "x,y"]).call()?;
    /// let delim = Scalar::from_string(",");
    /// let parts = col.view().str_split(&delim, -1).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_split<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplit<'a>;

    /// Splits each string by a delimiter right-to-left into a [`Table`] of string columns.
    ///
    /// Behaves like [`str_split`](StringExt::str_split) but splitting proceeds
    /// from the right end of each string.
    fn str_rsplit<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrRsplit<'a>;

    /// Returns the Nth part after splitting each string by a delimiter.
    ///
    /// The `delimiter` must be a string [`Scalar`]. The `index` is 0-based
    /// and selects which split part to return. Negative `index` counts from
    /// the last part (`-1` = last part).
    ///
    /// Returns a `STRING` column. If a row has fewer parts than `index`,
    /// the result for that row is null.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["a/b/c", "d/e"]).call()?;
    /// let delim = Scalar::from_string("/");
    /// let part = col.view().str_split_part(&delim, 1).call()?;
    /// // "a/b/c" => "b"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_split_part<'a>(&'a self, delimiter: &'a Scalar, index: i32) -> StrSplitPart<'a>;

    /// Joins all strings in the column into a single string.
    ///
    /// The `separator` scalar is placed between consecutive strings. The
    /// `narep` scalar is substituted for null elements. If `narep` is a null
    /// scalar, null elements are skipped.
    ///
    /// Returns a single-row `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["a", "b", "c"]).call()?;
    /// let sep = Scalar::from_string(", ");
    /// let na = Scalar::from_string("");
    /// let joined = col.view().str_join(&sep, &na).call()?;
    /// // ["a", "b", "c"] => "a, b, c"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_join<'a>(&'a self, separator: &'a Scalar, narep: &'a Scalar) -> StrJoin<'a>;

    /// Performs SQL `LIKE` pattern matching on each string.
    ///
    /// The `pattern` uses SQL `LIKE` wildcards: `%` matches zero or more
    /// characters, `_` matches exactly one character. The `escape_char` is
    /// used to escape wildcards in the pattern (typically `"\\"`).
    ///
    /// Returns a `BOOL8` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", "well hello there"]).call()?;
    /// let matched = col.view().str_like("%hello%", "\\").call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_like<'a>(&'a self, pattern: &'a str, escape_char: &'a str) -> StrLike<'a>;

    /// Tests whether each string contains a match for a regular expression.
    ///
    /// The `pattern` is a regular expression in libcudf regex syntax. Returns
    /// a `BOOL8` column: `true` if the pattern matches anywhere within the
    /// string.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["abc123", "def"]).call()?;
    /// let found = col.view().str_contains_re("\\d+").call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_contains_re<'a>(&'a self, pattern: &'a str) -> StrContainsRe<'a>;

    /// Tests whether each string matches a regular expression from the beginning.
    ///
    /// The `pattern` is anchored to the start of the string (like `^pattern`).
    /// Returns a `BOOL8` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["Apple", "banana"]).call()?;
    /// let matched = col.view().str_matches_re("[A-Z].*").call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_matches_re<'a>(&'a self, pattern: &'a str) -> StrMatchesRe<'a>;

    /// Counts the number of non-overlapping regex matches in each string.
    ///
    /// Returns an `INT32` column containing the match count per string.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["a1b2c3", "x"]).call()?;
    /// let counts = col.view().str_count_re("\\d").call()?;
    /// // "a1b2c3" => 3
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_count_re<'a>(&'a self, pattern: &'a str) -> StrCountRe<'a>;

    /// Replaces all regex matches in each string with a replacement string.
    ///
    /// The `pattern` is a regular expression and `replacement` is a literal
    /// string (no back-references; use
    /// [`str_replace_with_backrefs`](StringExt::str_replace_with_backrefs) for
    /// that). Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["abc123def", "456"]).call()?;
    /// let result = col.view().str_replace_re("\\d+", "NUM").call()?;
    /// // "abc123def" => "abcNUMdef"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_replace_re<'a>(&'a self, pattern: &'a str, replacement: &'a str) -> StrReplaceRe<'a>;

    /// Swaps the case of every character in each string.
    ///
    /// Uppercase characters become lowercase and vice versa. Returns a
    /// `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["Hello"]).call()?;
    /// let swapped = col.view().swapcase().call()?;
    /// // "Hello" => "hELLO"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn swapcase(&self) -> Swapcase<'_>;

    /// Strips the specified characters from one or both sides of each string.
    ///
    /// The `side` parameter controls which side(s) to strip (see [`SideType`]).
    /// The `to_strip` string lists the characters to remove (each character is
    /// removed independently, not as a substring).
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::{SideType, StringExt};
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["xyzhelloxyz", "xyzworld"]).call()?;
    /// let stripped = col.view().str_strip_chars(SideType::BOTH, "xyz").call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_strip_chars<'a>(&'a self, side: SideType, to_strip: &'a str) -> StrStripChars<'a>;

    /// Replaces occurrences of a literal `target` with `repl` in each string.
    ///
    /// The `maxrepl` parameter limits the number of replacements per string;
    /// `-1` means replace all occurrences. Unlike [`str_replace`](StringExt::str_replace),
    /// this takes string slices instead of scalars.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["foo baz", "no match"]).call()?;
    /// let result = col.view().str_replace_literal("foo", "bar", -1).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_replace_literal<'a>(
        &'a self,
        target: &'a str,
        repl: &'a str,
        maxrepl: i32,
    ) -> StrReplaceLiteral<'a>;

    /// Finds the first position of a literal `target` within the character range `[start, stop)`.
    ///
    /// Returns an `INT32` column with 0-based positions, or `-1` if not found.
    /// The `start` and `stop` are character positions (not byte offsets).
    /// Use `stop = -1` to search through the end of the string.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["xxabcxx", "def"]).call()?;
    /// let pos = col.view().str_find_str("abc", 0, -1).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_find_str<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrFindStr<'a>;

    /// Finds the last position of a literal `target` within `[start, stop)` (reverse find).
    ///
    /// Returns an `INT32` column with 0-based positions, or `-1` if not found.
    /// Searches from the end of the string toward the beginning.
    fn str_rfind<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrRfind<'a>;

    /// Tests whether each string contains a literal `target` substring.
    ///
    /// Unlike [`str_contains`](StringExt::str_contains), this takes a string
    /// slice instead of a [`Scalar`]. Returns a `BOOL8` column.
    fn str_contains_literal<'a>(&'a self, target: &'a str) -> StrContainsLiteral<'a>;

    /// Tests whether each string starts with a literal `target` prefix.
    ///
    /// Unlike [`str_starts_with`](StringExt::str_starts_with), this takes a
    /// string slice instead of a [`Scalar`]. Returns a `BOOL8` column.
    fn str_starts_with_str<'a>(&'a self, target: &'a str) -> StrStartsWithStr<'a>;

    /// Tests whether each string ends with a literal `target` suffix.
    ///
    /// Unlike [`str_ends_with`](StringExt::str_ends_with), this takes a
    /// string slice instead of a [`Scalar`]. Returns a `BOOL8` column.
    fn str_ends_with_str<'a>(&'a self, target: &'a str) -> StrEndsWithStr<'a>;

    /// Reverses the characters within each string.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello", "world"]).call()?;
    /// let reversed = col.view().str_reverse().call()?;
    /// // "hello" => "olleh"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_reverse(&self) -> StrReverse<'_>;

    /// Extracts regex capture groups into a [`Table`], one column per group.
    ///
    /// The `pattern` must contain one or more capture groups (parenthesized
    /// sub-expressions). Each group becomes a `STRING` column in the output
    /// table. If a row does not match, all group columns for that row are null.
    ///
    /// Returns a [`Table`] with as many columns as capture groups.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["user@host", "admin@site"]).call()?;
    /// let parts = col.view().str_extract("(\\w+)@(\\w+)").call()?;
    /// // Two columns: user part and domain part
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_extract<'a>(&'a self, pattern: &'a str) -> StrExtract<'a>;

    /// Extracts all regex matches into a `LIST(STRING)` column.
    ///
    /// Each row contains a list of all non-overlapping matches of `pattern`.
    /// If there are no matches, the list is empty.
    ///
    /// Returns a `LIST` column of strings.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["a1b2c3", "no digits"]).call()?;
    /// let all_matches = col.view().str_extract_all("\\d+").call()?;
    /// // "a1b2c3" => ["1", "2", "3"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_extract_all<'a>(&'a self, pattern: &'a str) -> StrExtractAll<'a>;

    /// Finds all regex matches in each string as a `LIST(STRING)` column.
    ///
    /// Similar to [`str_extract_all`](StringExt::str_extract_all) but returns
    /// the full match rather than capture groups.
    fn str_findall<'a>(&'a self, pattern: &'a str) -> StrFindall<'a>;

    /// Finds the character position of the first regex match in each string.
    ///
    /// Returns an `INT32` column with the 0-based starting position of the
    /// first match, or `-1` if no match is found.
    fn str_find_re<'a>(&'a self, pattern: &'a str) -> StrFindRe<'a>;

    /// Capitalizes the first character of each string.
    ///
    /// The first character is upper-cased and the rest are lower-cased.
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello WORLD"]).call()?;
    /// let result = col.view().capitalize().call()?;
    /// // "hello WORLD" => "Hello world"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn capitalize(&self) -> Capitalize<'_>;

    /// Title-cases each string (first letter of each word uppercased).
    ///
    /// Word boundaries are defined by non-alphanumeric characters. Returns a
    /// `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["hello world"]).call()?;
    /// let titled = col.view().title().call()?;
    /// // "hello world" => "Hello World"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn title(&self) -> Title<'_>;

    /// Tests whether each string is title-cased.
    ///
    /// Returns a `BOOL8` column: `true` if every word in the string starts with
    /// an uppercase character followed by lowercase characters.
    fn is_title(&self) -> IsTitle<'_>;

    /// Wraps each string onto multiple lines, each no longer than `width` characters.
    ///
    /// Long words that exceed `width` are not broken. Returns a `STRING` column
    /// with newline characters inserted at wrap points.
    fn wrap(&self, width: i32) -> Wrap<'_>;

    /// Parses timestamp strings into a timestamp column using a `strftime`-style format.
    ///
    /// The `timestamp_type` specifies the output resolution (e.g.
    /// [`TypeId::TIMESTAMP_SECONDS`], [`TypeId::TIMESTAMP_MILLISECONDS`]).
    /// The `format` string uses `strftime` directives (e.g. `"%Y-%m-%d %H:%M:%S"`).
    ///
    /// Returns a timestamp column of the specified type.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["2026-04-10", "2026-04-11"]).call()?;
    /// let ts = col.view().str_to_timestamps(
    ///     TypeId::TIMESTAMP_SECONDS, "%Y-%m-%d",
    /// ).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_to_timestamps<'a>(
        &'a self,
        timestamp_type: TypeId,
        format: &'a str,
    ) -> StrToTimestamps<'a>;

    /// Formats timestamp values as strings using a `strftime`-style format.
    ///
    /// The input column must be a `TIMESTAMP_*` type. The `format` string uses
    /// `strftime` directives (e.g. `"%Y-%m-%d"`). Returns a `STRING` column.
    fn str_from_timestamps<'a>(&'a self, format: &'a str) -> StrFromTimestamps<'a>;

    /// Tests whether each string is a valid timestamp matching the given format.
    ///
    /// Returns a `BOOL8` column: `true` if the string can be parsed with
    /// the specified `strftime` `format`.
    fn str_is_timestamp<'a>(&'a self, format: &'a str) -> StrIsTimestamp<'a>;

    /// Converts string values to booleans.
    ///
    /// Strings matching `true_string` (case-sensitive) produce `true`; all
    /// others produce `false`. Null strings remain null. Returns a `BOOL8`
    /// column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["True", "False", "True"]).call()?;
    /// let bools = col.view().str_to_booleans("True").call()?;
    /// // ["True", "False", "True"] => [true, false, true]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_to_booleans<'a>(&'a self, true_string: &'a str) -> StrToBooleans<'a>;

    /// Converts a boolean column to strings.
    ///
    /// `true` values become `true_string`, `false` values become
    /// `false_string`. Null values remain null. Returns a `STRING` column.
    fn str_from_booleans<'a>(
        &'a self,
        true_string: &'a str,
        false_string: &'a str,
    ) -> StrFromBooleans<'a>;

    /// Parses duration strings into a duration column using a format string.
    ///
    /// The `duration_type` specifies the output resolution (e.g.
    /// [`TypeId::DURATION_SECONDS`]). The `format` string defines the expected
    /// pattern. Returns a duration column of the specified type.
    fn str_to_durations<'a>(&'a self, duration_type: TypeId, format: &'a str)
    -> StrToDurations<'a>;

    /// Formats duration values as strings using a format string.
    ///
    /// The input column must be a `DURATION_*` type. Returns a `STRING` column.
    fn str_from_durations<'a>(&'a self, format: &'a str) -> StrFromDurations<'a>;

    /// Converts strings to fixed-point (decimal) values.
    ///
    /// The `type_id` must be a decimal type (e.g. [`TypeId::DECIMAL32`]) and
    /// `scale` controls the number of decimal places. Returns a column of the
    /// specified decimal type.
    fn str_to_fixed_point(&self, type_id: TypeId, scale: i32) -> StrToFixedPoint<'_>;

    /// Converts fixed-point (decimal) values to their string representation.
    ///
    /// The input column must be a decimal type. Returns a `STRING` column.
    fn str_from_fixed_point(&self) -> StrFromFixedPoint<'_>;

    /// Tests whether each string is a valid fixed-point number for the given type and scale.
    ///
    /// Returns a `BOOL8` column: `true` if the string can be parsed as a
    /// decimal of the specified `type_id` and `scale`.
    fn str_is_fixed_point(&self, type_id: TypeId, scale: i32) -> StrIsFixedPoint<'_>;

    /// URL-encodes each string (percent-encoding).
    ///
    /// Replaces unsafe characters with `%XX` hex sequences. Returns a `STRING`
    /// column.
    fn url_encode(&self) -> UrlEncode<'_>;

    /// URL-decodes each string (reverses percent-encoding).
    ///
    /// Replaces `%XX` hex sequences with their corresponding characters.
    /// Returns a `STRING` column.
    fn url_decode(&self) -> UrlDecode<'_>;

    /// Converts IPv4 address strings to their `UINT32` integer representation.
    ///
    /// Each string must be a dotted-decimal IPv4 address (e.g. `"192.168.1.1"`).
    /// Returns a `UINT32` column.
    fn ipv4_to_integers(&self) -> Ipv4ToIntegers<'_>;

    /// Converts `UINT32` integers to IPv4 address strings.
    ///
    /// The input column must be `UINT32`. Returns a `STRING` column with
    /// dotted-decimal notation (e.g. `"192.168.1.1"`).
    fn integers_to_ipv4(&self) -> IntegersToIpv4<'_>;

    /// Tests whether each string is a valid IPv4 address.
    ///
    /// Returns a `BOOL8` column.
    fn is_ipv4(&self) -> IsIpv4<'_>;

    /// Splits each string by a regex pattern into a [`Table`] of string columns.
    ///
    /// The `pattern` is a regular expression used as the delimiter. The
    /// `maxsplit` parameter limits the number of splits (`-1` for unlimited).
    /// Splitting proceeds left-to-right.
    fn str_split_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRe<'a>;

    /// Splits each string by a regex pattern right-to-left into a [`Table`].
    ///
    /// Behaves like [`str_split_re`](StringExt::str_split_re) but splitting
    /// proceeds from the right.
    fn str_rsplit_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrRsplitRe<'a>;

    /// Splits each string by a regex pattern into a `LIST(STRING)` column.
    ///
    /// Each row becomes a list of the split parts. The `maxsplit` parameter
    /// limits the number of splits (`-1` for unlimited). Splitting proceeds
    /// left-to-right.
    fn str_split_record_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRecordRe<'a>;

    /// Splits each string by a regex pattern right-to-left into a `LIST(STRING)` column.
    ///
    /// Behaves like [`str_split_record_re`](StringExt::str_split_record_re)
    /// but splitting proceeds from the right.
    fn str_rsplit_record_re<'a>(&'a self, pattern: &'a str, maxsplit: i32)
    -> StrRsplitRecordRe<'a>;

    /// Partitions each string around the first occurrence of `delimiter`.
    ///
    /// Returns a 3-column [`Table`]: the part before the delimiter, the
    /// delimiter itself, and the part after. If the delimiter is not found, the
    /// first column contains the original string and the other two are empty.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["user@host", "admin@site"]).call()?;
    /// let parts = col.view().str_partition("@").call()?;
    /// // "user@host" => ["user", "@", "host"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_partition<'a>(&'a self, delimiter: &'a str) -> StrPartition<'a>;

    /// Partitions each string around the last occurrence of `delimiter`.
    ///
    /// Returns a 3-column [`Table`]: the part before the last delimiter, the
    /// delimiter, and the part after. If the delimiter is not found, the
    /// last column contains the original string and the first two are empty.
    fn str_rpartition<'a>(&'a self, delimiter: &'a str) -> StrRpartition<'a>;

    /// Replaces regex matches using a back-reference replacement template.
    ///
    /// The `replacement` string may contain back-references like `\\1`, `\\2`,
    /// etc. to refer to captured groups in `pattern`. Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["user@host"]).call()?;
    /// let result = col.view().str_replace_with_backrefs(
    ///     "(\\w+)@(\\w+)", "\\2@\\1",
    /// ).call()?;
    /// // "user@host" => "host@user"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_replace_with_backrefs<'a>(
        &'a self,
        pattern: &'a str,
        replacement: &'a str,
    ) -> StrReplaceWithBackrefs<'a>;

    /// Repeats each string a per-row number of times given by another column.
    ///
    /// The `repeat_times` column must be an integer type (`INT32`) with the
    /// same number of rows. Returns a `STRING` column.
    fn str_repeat_column<'a>(&'a self, repeat_times: &'a ColumnView<'a>) -> StrRepeatColumn<'a>;

    /// Tests each string against multiple targets simultaneously.
    ///
    /// The `targets` column is a `STRING` column of target strings. Returns a
    /// [`Table`] with one `BOOL8` column per target. Each column indicates
    /// whether the corresponding target is contained in each source string.
    fn str_contains_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrContainsMultiple<'a>;

    /// Finds the positions of multiple targets in each string.
    ///
    /// The `targets` column is a `STRING` column of target strings. Returns a
    /// `LIST(INT32)` column where each list contains the positions of the first
    /// occurrence of each target (`-1` if not found).
    fn str_find_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrFindMultiple<'a>;

    /// Tests whether all characters in each string match the given type bitmask.
    ///
    /// Character types are bitmask constants from [`char_types`]:
    /// `DECIMAL=1`, `NUMERIC=2`, `DIGIT=4`, `ALPHA=8`, `SPACE=16`, `UPPER=32`,
    /// `LOWER=64`. Combine with bitwise OR for multiple types.
    ///
    /// The `types` parameter specifies which character types must be present.
    /// The `verify_types` parameter restricts which types are checked (use
    /// [`char_types::ALL_TYPES`] to check all).
    ///
    /// Returns a `BOOL8` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::strings::char_types;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["abc", "xyz"]).call()?;
    /// let is_alpha = col.view().all_characters_of_type(
    ///     char_types::ALPHA, char_types::ALL_TYPES,
    /// ).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn all_characters_of_type(&self, types: u32, verify_types: u32) -> AllCharactersOfType<'_>;

    /// Filters characters by type, replacing removed characters with a replacement string.
    ///
    /// Characters matching `types_to_remove` (but not `types_to_keep`) are
    /// replaced with the `replacement` string. See [`char_types`] for bitmask
    /// constants.
    ///
    /// Returns a `STRING` column.
    fn filter_characters_of_type<'a>(
        &'a self,
        types_to_remove: u32,
        replacement: &'a str,
        types_to_keep: u32,
    ) -> FilterCharactersOfType<'a>;

    /// Tests whether each string represents a valid integer.
    ///
    /// Returns a `BOOL8` column: `true` if the entire string can be parsed as
    /// any integer type.
    fn str_is_integer(&self) -> StrIsInteger<'_>;

    /// Tests whether each string is a valid integer within the range of `int_type`.
    ///
    /// The `int_type` specifies the target integer type (e.g. [`TypeId::INT32`]).
    /// Returns a `BOOL8` column: `true` if the string can be parsed as an
    /// integer that fits in the specified type.
    fn str_is_integer_with_type(&self, int_type: TypeId) -> StrIsIntegerWithType<'_>;

    /// Tests whether each string represents a valid floating-point number.
    ///
    /// Returns a `BOOL8` column.
    fn str_is_float(&self) -> StrIsFloat<'_>;

    /// Converts hexadecimal strings to integers of the given type.
    ///
    /// The `output_type` specifies the target integer type. Strings may
    /// optionally include a `0x` prefix. Returns a column of the specified
    /// integer type.
    fn str_hex_to_integers(&self, output_type: TypeId) -> StrHexToIntegers<'_>;

    /// Tests whether each string is a valid hexadecimal number.
    ///
    /// Returns a `BOOL8` column.
    fn str_is_hex(&self) -> StrIsHex<'_>;

    /// Converts integer values to hexadecimal string representations.
    ///
    /// The input column must be an integer type. Returns a `STRING` column.
    fn str_integers_to_hex(&self) -> StrIntegersToHex<'_>;

    /// Replaces a character-position slice `[start, stop)` in each string with `repl`.
    ///
    /// Positions are 0-based character indices. Use `stop = -1` to replace
    /// through the end of the string. Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["abcde"]).call()?;
    /// let result = col.view().str_replace_slice("XY", 1, 3).call()?;
    /// // "abcde" => "aXYde"
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_replace_slice<'a>(&'a self, repl: &'a str, start: i32, stop: i32)
    -> StrReplaceSlice<'a>;

    /// Replaces multiple target strings with corresponding replacements.
    ///
    /// The `targets` and `repls` columns must be `STRING` columns of equal
    /// length. Each occurrence of `targets[i]` in the source strings is
    /// replaced with `repls[i]`. Replacements are applied sequentially.
    ///
    /// Returns a `STRING` column.
    fn str_replace_multiple<'a>(
        &'a self,
        targets: &'a ColumnView<'a>,
        repls: &'a ColumnView<'a>,
    ) -> StrReplaceMultiple<'a>;

    /// Splits each string by a delimiter into a `LIST(STRING)` column (left-to-right).
    ///
    /// The `delimiter` must be a string [`Scalar`]. Each row becomes a list of
    /// split parts. The `maxsplit` parameter limits the number of splits
    /// (`-1` for unlimited).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["a,b,c", "x,y"]).call()?;
    /// let delim = Scalar::from_string(",");
    /// let lists = col.view().str_split_record(&delim, -1).call()?;
    /// // "a,b,c" => ["a", "b", "c"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_split_record<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplitRecord<'a>;

    /// Splits each string by a delimiter into a `LIST(STRING)` column (right-to-left).
    ///
    /// Behaves like [`str_split_record`](StringExt::str_split_record) but
    /// splitting proceeds from the right end of each string.
    fn str_rsplit_record<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32)
    -> StrRsplitRecord<'a>;

    /// Joins a `LIST(STRING)` column's lists into single strings per row.
    ///
    /// The input must be a `LIST(STRING)` column. Elements within each list
    /// are joined using `separator`. Null elements are replaced with `narep`.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3]).call()?;
    /// # let values = Column::from_strings(&["a", "b", "c"]).call()?;
    /// # let list_col = Column::from_lists(1, offsets, values).call()?;
    /// let joined = list_col.view().str_join_list_elements(", ", "NULL").call()?;
    /// // [["a","b","c"]] => ["a, b, c"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_join_list_elements<'a>(
        &'a self,
        separator: &'a str,
        narep: &'a str,
    ) -> StrJoinListElements<'a>;

    /// Translates individual characters in each string using a character mapping.
    ///
    /// The `from_chars` and `to_chars` slices must have the same length. Each
    /// occurrence of `from_chars[i]` (as a Unicode code point) is replaced
    /// with `to_chars[i]`. Returns a `STRING` column.
    fn str_translate<'a>(&'a self, from_chars: &'a [u32], to_chars: &'a [u32]) -> StrTranslate<'a>;

    /// Filters characters from each string by Unicode code point ranges.
    ///
    /// The `from_chars` and `to_chars` slices define ranges: characters with
    /// code points in `[from_chars[i], to_chars[i]]` are selected. When
    /// `keep` is `true`, only characters within the ranges are kept; when
    /// `false`, characters within the ranges are removed. Removed characters
    /// are replaced with `replacement`.
    ///
    /// Returns a `STRING` column.
    fn str_filter_characters<'a>(
        &'a self,
        from_chars: &'a [u32],
        to_chars: &'a [u32],
        keep: bool,
        replacement: &'a str,
    ) -> StrFilterCharacters<'a>;

    /// Returns the Unicode code points for all characters, concatenated across rows.
    ///
    /// Returns a single `INT32` column containing the code points of every
    /// character in every row, concatenated end-to-end. Use with
    /// [`count_characters`](StringExt::count_characters) to determine row
    /// boundaries.
    fn code_points(&self) -> CodePoints<'_>;

    /// Encodes each string as a binary integer value.
    ///
    /// The string's raw bytes are reinterpreted as an integer of `output_type`.
    /// When `big_endian` is `true`, bytes are in big-endian order; otherwise
    /// little-endian. Returns a column of the specified integer type.
    fn str_cast_to_integer(&self, output_type: TypeId, big_endian: bool) -> StrCastToInteger<'_>;

    /// Decodes integer-encoded binary values back to strings.
    ///
    /// The inverse of [`str_cast_to_integer`](StringExt::str_cast_to_integer).
    /// When `big_endian` is `true`, bytes are read in big-endian order.
    /// Returns a `STRING` column.
    fn str_cast_from_integer(&self, big_endian: bool) -> StrCastFromInteger<'_>;

    /// Extracts substrings using per-row start and stop positions from columns.
    ///
    /// The `starts` and `stops` columns must be `INT32` columns with the same
    /// number of rows. Each row is sliced at `[starts[i], stops[i])`.
    ///
    /// Returns a `STRING` column.
    fn str_slice_column<'a>(
        &'a self,
        starts: &'a ColumnView<'a>,
        stops: &'a ColumnView<'a>,
    ) -> StrSliceColumn<'a>;

    /// Extracts a single regex capture group from each string.
    ///
    /// The `pattern` must contain at least `group_index + 1` capture groups.
    /// The `group_index` is 0-based. Returns a `STRING` column containing the
    /// matched group text, or null for non-matching rows.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::strings::StringExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let col = Column::from_strings(&["user@host", "admin@site"]).call()?;
    /// let domains = col.view().str_extract_single("@(\\w+)", 0).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn str_extract_single<'a>(&'a self, pattern: &'a str, group_index: i32)
    -> StrExtractSingle<'a>;

    /// Joins a `LIST(STRING)` column's lists using per-row separators from a column.
    ///
    /// The `separators` column must be a `STRING` column with the same number
    /// of rows. The `separator_narep` is used when a separator is null. The
    /// `string_narep` is used when a list element is null.
    ///
    /// Returns a `STRING` column.
    fn str_join_list_elements_column<'a>(
        &'a self,
        separators: &'a ColumnView<'a>,
        separator_narep: &'a str,
        string_narep: &'a str,
    ) -> StrJoinListElementsColumn<'a>;

    /// Joins all strings in the column into a single-row string using a literal separator.
    ///
    /// The `separator` is placed between consecutive strings. The `narep` is
    /// substituted for null elements. Returns a single-row `STRING` column.
    fn str_join_strings<'a>(&'a self, separator: &'a str, narep: &'a str) -> StrJoinStrings<'a>;

    /// Finds the Nth occurrence of a target substring in each string.
    ///
    /// The `instance` parameter is 0-based: `0` finds the first occurrence,
    /// `1` the second, etc. Returns an `INT32` column with 0-based positions,
    /// or `-1` if the Nth occurrence is not found.
    fn str_find_instance<'a>(&'a self, target: &'a str, instance: i32) -> StrFindInstance<'a>;

    /// Performs SQL `LIKE` pattern matching using per-row patterns from another column.
    ///
    /// The `patterns` column must be a `STRING` column with the same number of
    /// rows. Each row is matched against its corresponding pattern. The
    /// `escape_char` escapes wildcards in the pattern.
    ///
    /// Returns a `BOOL8` column.
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
    ($name:ident -> $ret:ty, $body:expr) => {
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
simple_builder!(ToLower -> crate::column::UnboundColumn, |s: ToLower<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_lower(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::to_upper`].
pub struct ToUpper<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(ToUpper -> crate::column::UnboundColumn, |s: ToUpper<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_upper(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_contains`].
pub struct StrContains<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrContains -> crate::column::UnboundColumn, |s: StrContains<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target)?;
    let c = cudf_sys::strings::ffi::strings_contains(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_starts_with`].
pub struct StrStartsWith<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrStartsWith -> crate::column::UnboundColumn, |s: StrStartsWith<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target)?;
    let c = cudf_sys::strings::ffi::strings_starts_with(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_ends_with`].
pub struct StrEndsWith<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrEndsWith -> crate::column::UnboundColumn, |s: StrEndsWith<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target)?;
    let c = cudf_sys::strings::ffi::strings_ends_with(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_find`].
pub struct StrFind<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrFind -> crate::column::UnboundColumn, |s: StrFind<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.target)?;
    let c = cudf_sys::strings::ffi::strings_find(s.view.0, &ffi, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_replace`].
pub struct StrReplace<'a> {
    view: &'a ColumnView<'a>,
    target: &'a Scalar,
    replacement: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrReplace -> crate::column::UnboundColumn, |s: StrReplace<'_>| {
    let target_ffi = crate::scalar::scalar_to_ffi(s.target)?;
    let replacement_ffi = crate::scalar::scalar_to_ffi(s.replacement)?;
    let c = cudf_sys::strings::ffi::strings_replace(
        s.view.0, &target_ffi, &replacement_ffi, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_strip`].
pub struct StrStrip<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrStrip -> crate::column::UnboundColumn, |s: StrStrip<'_>| {
    let c = cudf_sys::strings::ffi::strings_strip(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_lstrip`].
pub struct StrLstrip<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrLstrip -> crate::column::UnboundColumn, |s: StrLstrip<'_>| {
    let c = cudf_sys::strings::ffi::strings_lstrip(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_rstrip`].
pub struct StrRstrip<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrRstrip -> crate::column::UnboundColumn, |s: StrRstrip<'_>| {
    let c = cudf_sys::strings::ffi::strings_rstrip(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::count_characters`].
pub struct CountCharacters<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(CountCharacters -> crate::column::UnboundColumn, |s: CountCharacters<'_>| {
    let c = cudf_sys::strings::ffi::strings_count_characters(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::count_bytes`].
pub struct CountBytes<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(CountBytes -> crate::column::UnboundColumn, |s: CountBytes<'_>| {
    let c = cudf_sys::strings::ffi::strings_count_bytes(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_from_integers`].
pub struct StrFromIntegers<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFromIntegers -> crate::column::UnboundColumn, |s: StrFromIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_integers(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_to_integers`].
pub struct StrToIntegers<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}
simple_builder!(StrToIntegers -> crate::column::UnboundColumn, |s: StrToIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_integers(s.view.0, s.output_type.repr, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_from_floats`].
pub struct StrFromFloats<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFromFloats -> crate::column::UnboundColumn, |s: StrFromFloats<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_floats(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_to_floats`].
pub struct StrToFloats<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}
simple_builder!(StrToFloats -> crate::column::UnboundColumn, |s: StrToFloats<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_floats(s.view.0, s.output_type.repr, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_pad`].
pub struct StrPad<'a> {
    view: &'a ColumnView<'a>,
    width: usize,
    side: SideType,
    fill_char: &'a str,
    stream: Stream,
}
simple_builder!(StrPad -> crate::column::UnboundColumn, |s: StrPad<'_>| {
    let c = cudf_sys::strings::ffi::strings_pad(
        s.view.0, crate::usize_to_i32(s.width), s.side.repr, s.fill_char, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_zfill`].
pub struct StrZfill<'a> {
    view: &'a ColumnView<'a>,
    width: usize,
    stream: Stream,
}
simple_builder!(StrZfill -> crate::column::UnboundColumn, |s: StrZfill<'_>| {
    let c = cudf_sys::strings::ffi::strings_zfill(s.view.0, crate::usize_to_i32(s.width), s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_zfill_by_widths`].
pub struct StrZfillByWidths<'a> {
    view: &'a ColumnView<'a>,
    widths: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrZfillByWidths -> crate::column::UnboundColumn, |s: StrZfillByWidths<'_>| {
    let c = cudf_sys::strings::ffi::strings_zfill_by_widths(s.view.0, s.widths.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_slice`].
pub struct StrSlice<'a> {
    view: &'a ColumnView<'a>,
    start: i32,
    stop: i32,
    step: i32,
    stream: Stream,
}
simple_builder!(StrSlice -> crate::column::UnboundColumn, |s: StrSlice<'_>| {
    let c = cudf_sys::strings::ffi::strings_slice(s.view.0, s.start, s.stop, s.step, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_repeat`].
pub struct StrRepeat<'a> {
    view: &'a ColumnView<'a>,
    times: usize,
    stream: Stream,
}
simple_builder!(StrRepeat -> crate::column::UnboundColumn, |s: StrRepeat<'_>| {
    let c = cudf_sys::strings::ffi::strings_repeat(s.view.0, crate::usize_to_i32(s.times), s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_split`].
pub struct StrSplit<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplit -> crate::table::UnboundTable, |s: StrSplit<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter)?;
    let t = cudf_sys::strings::ffi::strings_split_to_table(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_rsplit`].
pub struct StrRsplit<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplit -> crate::table::UnboundTable, |s: StrRsplit<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter)?;
    let t = cudf_sys::strings::ffi::strings_rsplit_to_table(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_split_part`].
pub struct StrSplitPart<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    index: i32,
    stream: Stream,
}
simple_builder!(StrSplitPart -> crate::column::UnboundColumn, |s: StrSplitPart<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter)?;
    let c = cudf_sys::strings::ffi::strings_split_part(s.view.0, &ffi, s.index, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_join`].
pub struct StrJoin<'a> {
    view: &'a ColumnView<'a>,
    separator: &'a Scalar,
    narep: &'a Scalar,
    stream: Stream,
}
simple_builder!(StrJoin -> crate::column::UnboundColumn, |s: StrJoin<'_>| {
    let sep_ffi = crate::scalar::scalar_to_ffi(s.separator)?;
    let na_ffi = crate::scalar::scalar_to_ffi(s.narep)?;
    let c = cudf_sys::strings::ffi::strings_join(s.view.0, &sep_ffi, &na_ffi, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_like`].
pub struct StrLike<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    escape_char: &'a str,
    stream: Stream,
}
simple_builder!(StrLike -> crate::column::UnboundColumn, |s: StrLike<'_>| {
    let c = cudf_sys::strings::ffi::strings_like(s.view.0, s.pattern, s.escape_char, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_contains_re`].
pub struct StrContainsRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrContainsRe -> crate::column::UnboundColumn, |s: StrContainsRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_contains_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_matches_re`].
pub struct StrMatchesRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrMatchesRe -> crate::column::UnboundColumn, |s: StrMatchesRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_matches_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_count_re`].
pub struct StrCountRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrCountRe -> crate::column::UnboundColumn, |s: StrCountRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_count_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_replace_re`].
pub struct StrReplaceRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    replacement: &'a str,
    stream: Stream,
}
simple_builder!(StrReplaceRe -> crate::column::UnboundColumn, |s: StrReplaceRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_re(s.view.0, s.pattern, s.replacement, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::swapcase`].
pub struct Swapcase<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Swapcase -> crate::column::UnboundColumn, |s: Swapcase<'_>| {
    let c = cudf_sys::strings::ffi::strings_swapcase(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_strip_chars`].
pub struct StrStripChars<'a> {
    view: &'a ColumnView<'a>,
    side: SideType,
    to_strip: &'a str,
    stream: Stream,
}
simple_builder!(StrStripChars -> crate::column::UnboundColumn, |s: StrStripChars<'_>| {
    let c = cudf_sys::strings::ffi::strings_strip_chars(s.view.0, s.side.repr, s.to_strip, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_replace_literal`].
pub struct StrReplaceLiteral<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    repl: &'a str,
    maxrepl: i32,
    stream: Stream,
}
simple_builder!(StrReplaceLiteral -> crate::column::UnboundColumn, |s: StrReplaceLiteral<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_literal(s.view.0, s.target, s.repl, s.maxrepl, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_find_str`].
pub struct StrFindStr<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    start: i32,
    stop: i32,
    stream: Stream,
}
simple_builder!(StrFindStr -> crate::column::UnboundColumn, |s: StrFindStr<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_str(s.view.0, s.target, s.start, s.stop, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_rfind`].
pub struct StrRfind<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    start: i32,
    stop: i32,
    stream: Stream,
}
simple_builder!(StrRfind -> crate::column::UnboundColumn, |s: StrRfind<'_>| {
    let c = cudf_sys::strings::ffi::strings_rfind(s.view.0, s.target, s.start, s.stop, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_contains_literal`].
pub struct StrContainsLiteral<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    stream: Stream,
}
simple_builder!(StrContainsLiteral -> crate::column::UnboundColumn, |s: StrContainsLiteral<'_>| {
    let c = cudf_sys::strings::ffi::strings_contains_str(s.view.0, s.target, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_starts_with_str`].
pub struct StrStartsWithStr<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    stream: Stream,
}
simple_builder!(StrStartsWithStr -> crate::column::UnboundColumn, |s: StrStartsWithStr<'_>| {
    let c = cudf_sys::strings::ffi::strings_starts_with_str(s.view.0, s.target, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_ends_with_str`].
pub struct StrEndsWithStr<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    stream: Stream,
}
simple_builder!(StrEndsWithStr -> crate::column::UnboundColumn, |s: StrEndsWithStr<'_>| {
    let c = cudf_sys::strings::ffi::strings_ends_with_str(s.view.0, s.target, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_reverse`].
pub struct StrReverse<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrReverse -> crate::column::UnboundColumn, |s: StrReverse<'_>| {
    let c = cudf_sys::strings::ffi::strings_reverse(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_extract`].
pub struct StrExtract<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrExtract -> crate::table::UnboundTable, |s: StrExtract<'_>| {
    let t = cudf_sys::strings::ffi::strings_extract(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_extract_all`].
pub struct StrExtractAll<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrExtractAll -> crate::column::UnboundColumn, |s: StrExtractAll<'_>| {
    let c = cudf_sys::strings::ffi::strings_extract_all_record(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_findall`].
pub struct StrFindall<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrFindall -> crate::column::UnboundColumn, |s: StrFindall<'_>| {
    let c = cudf_sys::strings::ffi::strings_findall(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_find_re`].
pub struct StrFindRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    stream: Stream,
}
simple_builder!(StrFindRe -> crate::column::UnboundColumn, |s: StrFindRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_re(s.view.0, s.pattern, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::capitalize`].
pub struct Capitalize<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Capitalize -> crate::column::UnboundColumn, |s: Capitalize<'_>| {
    let c = cudf_sys::strings::ffi::strings_capitalize(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::title`].
pub struct Title<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Title -> crate::column::UnboundColumn, |s: Title<'_>| {
    let c = cudf_sys::strings::ffi::strings_title(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::is_title`].
pub struct IsTitle<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(IsTitle -> crate::column::UnboundColumn, |s: IsTitle<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_title(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::wrap`].
pub struct Wrap<'a> {
    view: &'a ColumnView<'a>,
    width: i32,
    stream: Stream,
}
simple_builder!(Wrap -> crate::column::UnboundColumn, |s: Wrap<'_>| {
    let c = cudf_sys::strings::ffi::strings_wrap(s.view.0, s.width, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_to_timestamps`].
pub struct StrToTimestamps<'a> {
    view: &'a ColumnView<'a>,
    timestamp_type: TypeId,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrToTimestamps -> crate::column::UnboundColumn, |s: StrToTimestamps<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_timestamps(
        s.view.0, s.timestamp_type.repr, s.format, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_from_timestamps`].
pub struct StrFromTimestamps<'a> {
    view: &'a ColumnView<'a>,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrFromTimestamps -> crate::column::UnboundColumn, |s: StrFromTimestamps<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_timestamps(s.view.0, s.format, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_is_timestamp`].
pub struct StrIsTimestamp<'a> {
    view: &'a ColumnView<'a>,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrIsTimestamp -> crate::column::UnboundColumn, |s: StrIsTimestamp<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_timestamp(s.view.0, s.format, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_to_booleans`].
pub struct StrToBooleans<'a> {
    view: &'a ColumnView<'a>,
    true_string: &'a str,
    stream: Stream,
}
simple_builder!(StrToBooleans -> crate::column::UnboundColumn, |s: StrToBooleans<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_booleans(s.view.0, s.true_string, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_from_booleans`].
pub struct StrFromBooleans<'a> {
    view: &'a ColumnView<'a>,
    true_string: &'a str,
    false_string: &'a str,
    stream: Stream,
}
simple_builder!(StrFromBooleans -> crate::column::UnboundColumn, |s: StrFromBooleans<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_booleans(
        s.view.0, s.true_string, s.false_string, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_to_durations`].
pub struct StrToDurations<'a> {
    view: &'a ColumnView<'a>,
    duration_type: TypeId,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrToDurations -> crate::column::UnboundColumn, |s: StrToDurations<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_durations(
        s.view.0, s.duration_type.repr, s.format, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_from_durations`].
pub struct StrFromDurations<'a> {
    view: &'a ColumnView<'a>,
    format: &'a str,
    stream: Stream,
}
simple_builder!(StrFromDurations -> crate::column::UnboundColumn, |s: StrFromDurations<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_durations(s.view.0, s.format, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_to_fixed_point`].
pub struct StrToFixedPoint<'a> {
    view: &'a ColumnView<'a>,
    type_id: TypeId,
    scale: i32,
    stream: Stream,
}
simple_builder!(StrToFixedPoint -> crate::column::UnboundColumn, |s: StrToFixedPoint<'_>| {
    let c = cudf_sys::strings::ffi::strings_to_fixed_point(
        s.view.0, s.type_id.repr, s.scale, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_from_fixed_point`].
pub struct StrFromFixedPoint<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFromFixedPoint -> crate::column::UnboundColumn, |s: StrFromFixedPoint<'_>| {
    let c = cudf_sys::strings::ffi::strings_from_fixed_point(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_is_fixed_point`].
pub struct StrIsFixedPoint<'a> {
    view: &'a ColumnView<'a>,
    type_id: TypeId,
    scale: i32,
    stream: Stream,
}
simple_builder!(StrIsFixedPoint -> crate::column::UnboundColumn, |s: StrIsFixedPoint<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_fixed_point(
        s.view.0, s.type_id.repr, s.scale, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::url_encode`].
pub struct UrlEncode<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(UrlEncode -> crate::column::UnboundColumn, |s: UrlEncode<'_>| {
    let c = cudf_sys::strings::ffi::strings_url_encode(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::url_decode`].
pub struct UrlDecode<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(UrlDecode -> crate::column::UnboundColumn, |s: UrlDecode<'_>| {
    let c = cudf_sys::strings::ffi::strings_url_decode(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::ipv4_to_integers`].
pub struct Ipv4ToIntegers<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(Ipv4ToIntegers -> crate::column::UnboundColumn, |s: Ipv4ToIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_ipv4_to_integers(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::integers_to_ipv4`].
pub struct IntegersToIpv4<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(IntegersToIpv4 -> crate::column::UnboundColumn, |s: IntegersToIpv4<'_>| {
    let c = cudf_sys::strings::ffi::strings_integers_to_ipv4(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::is_ipv4`].
pub struct IsIpv4<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(IsIpv4 -> crate::column::UnboundColumn, |s: IsIpv4<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_ipv4(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_split_re`].
pub struct StrSplitRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplitRe -> crate::table::UnboundTable, |s: StrSplitRe<'_>| {
    let t = cudf_sys::strings::ffi::strings_split_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_rsplit_re`].
pub struct StrRsplitRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplitRe -> crate::table::UnboundTable, |s: StrRsplitRe<'_>| {
    let t = cudf_sys::strings::ffi::strings_rsplit_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_split_record_re`].
pub struct StrSplitRecordRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplitRecordRe -> crate::column::UnboundColumn, |s: StrSplitRecordRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_split_record_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_rsplit_record_re`].
pub struct StrRsplitRecordRe<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplitRecordRe -> crate::column::UnboundColumn, |s: StrRsplitRecordRe<'_>| {
    let c = cudf_sys::strings::ffi::strings_rsplit_record_re(s.view.0, s.pattern, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_partition`].
pub struct StrPartition<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a str,
    stream: Stream,
}
simple_builder!(StrPartition -> crate::table::UnboundTable, |s: StrPartition<'_>| {
    let t = cudf_sys::strings::ffi::strings_partition(s.view.0, s.delimiter, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_rpartition`].
pub struct StrRpartition<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a str,
    stream: Stream,
}
simple_builder!(StrRpartition -> crate::table::UnboundTable, |s: StrRpartition<'_>| {
    let t = cudf_sys::strings::ffi::strings_rpartition(s.view.0, s.delimiter, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_replace_with_backrefs`].
pub struct StrReplaceWithBackrefs<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    replacement: &'a str,
    stream: Stream,
}
simple_builder!(StrReplaceWithBackrefs -> crate::column::UnboundColumn, |s: StrReplaceWithBackrefs<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_with_backrefs(
        s.view.0, s.pattern, s.replacement, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_repeat_column`].
pub struct StrRepeatColumn<'a> {
    view: &'a ColumnView<'a>,
    repeat_times: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrRepeatColumn -> crate::column::UnboundColumn, |s: StrRepeatColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_repeat_column(s.view.0, s.repeat_times.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_contains_multiple`].
pub struct StrContainsMultiple<'a> {
    view: &'a ColumnView<'a>,
    targets: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrContainsMultiple -> crate::table::UnboundTable, |s: StrContainsMultiple<'_>| {
    let t = cudf_sys::strings::ffi::strings_contains_multiple(s.view.0, s.targets.0, s.stream.as_raw())?;
    Ok(crate::table::RawTable(t))
});

/// Builder for [`StringExt::str_find_multiple`].
pub struct StrFindMultiple<'a> {
    view: &'a ColumnView<'a>,
    targets: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrFindMultiple -> crate::column::UnboundColumn, |s: StrFindMultiple<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_multiple(s.view.0, s.targets.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::all_characters_of_type`].
pub struct AllCharactersOfType<'a> {
    view: &'a ColumnView<'a>,
    types: u32,
    verify_types: u32,
    stream: Stream,
}
simple_builder!(AllCharactersOfType -> crate::column::UnboundColumn, |s: AllCharactersOfType<'_>| {
    let c = cudf_sys::strings::ffi::strings_all_characters_of_type(
        s.view.0, s.types, s.verify_types, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::filter_characters_of_type`].
pub struct FilterCharactersOfType<'a> {
    view: &'a ColumnView<'a>,
    types_to_remove: u32,
    replacement: &'a str,
    types_to_keep: u32,
    stream: Stream,
}
simple_builder!(FilterCharactersOfType -> crate::column::UnboundColumn, |s: FilterCharactersOfType<'_>| {
    let c = cudf_sys::strings::ffi::strings_filter_characters_of_type(
        s.view.0, s.types_to_remove, s.replacement, s.types_to_keep, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_is_integer`].
pub struct StrIsInteger<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIsInteger -> crate::column::UnboundColumn, |s: StrIsInteger<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_integer(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_is_integer_with_type`].
pub struct StrIsIntegerWithType<'a> {
    view: &'a ColumnView<'a>,
    int_type: TypeId,
    stream: Stream,
}
simple_builder!(StrIsIntegerWithType -> crate::column::UnboundColumn, |s: StrIsIntegerWithType<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_integer_with_type(s.view.0, s.int_type.repr, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_is_float`].
pub struct StrIsFloat<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIsFloat -> crate::column::UnboundColumn, |s: StrIsFloat<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_float(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_hex_to_integers`].
pub struct StrHexToIntegers<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}
simple_builder!(StrHexToIntegers -> crate::column::UnboundColumn, |s: StrHexToIntegers<'_>| {
    let c = cudf_sys::strings::ffi::strings_hex_to_integers(s.view.0, s.output_type.repr, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_is_hex`].
pub struct StrIsHex<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIsHex -> crate::column::UnboundColumn, |s: StrIsHex<'_>| {
    let c = cudf_sys::strings::ffi::strings_is_hex(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_integers_to_hex`].
pub struct StrIntegersToHex<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrIntegersToHex -> crate::column::UnboundColumn, |s: StrIntegersToHex<'_>| {
    let c = cudf_sys::strings::ffi::strings_integers_to_hex(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_replace_slice`].
pub struct StrReplaceSlice<'a> {
    view: &'a ColumnView<'a>,
    repl: &'a str,
    start: i32,
    stop: i32,
    stream: Stream,
}
simple_builder!(StrReplaceSlice -> crate::column::UnboundColumn, |s: StrReplaceSlice<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_slice(
        s.view.0, s.repl, s.start, s.stop, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_replace_multiple`].
pub struct StrReplaceMultiple<'a> {
    view: &'a ColumnView<'a>,
    targets: &'a ColumnView<'a>,
    repls: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrReplaceMultiple -> crate::column::UnboundColumn, |s: StrReplaceMultiple<'_>| {
    let c = cudf_sys::strings::ffi::strings_replace_multiple(
        s.view.0, s.targets.0, s.repls.0, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_split_record`].
pub struct StrSplitRecord<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrSplitRecord -> crate::column::UnboundColumn, |s: StrSplitRecord<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter)?;
    let c = cudf_sys::strings::ffi::strings_split_record(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_rsplit_record`].
pub struct StrRsplitRecord<'a> {
    view: &'a ColumnView<'a>,
    delimiter: &'a Scalar,
    maxsplit: i32,
    stream: Stream,
}
simple_builder!(StrRsplitRecord -> crate::column::UnboundColumn, |s: StrRsplitRecord<'_>| {
    let ffi = crate::scalar::scalar_to_ffi(s.delimiter)?;
    let c = cudf_sys::strings::ffi::strings_rsplit_record(s.view.0, &ffi, s.maxsplit, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_join_list_elements`].
pub struct StrJoinListElements<'a> {
    view: &'a ColumnView<'a>,
    separator: &'a str,
    narep: &'a str,
    stream: Stream,
}
simple_builder!(StrJoinListElements -> crate::column::UnboundColumn, |s: StrJoinListElements<'_>| {
    let c = cudf_sys::strings::ffi::strings_join_list_elements(
        s.view.0, s.separator, s.narep, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_translate`].
pub struct StrTranslate<'a> {
    view: &'a ColumnView<'a>,
    from_chars: &'a [u32],
    to_chars: &'a [u32],
    stream: Stream,
}
simple_builder!(StrTranslate -> crate::column::UnboundColumn, |s: StrTranslate<'_>| {
    let c = cudf_sys::strings::ffi::strings_translate(
        s.view.0, s.from_chars, s.to_chars, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
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
simple_builder!(StrFilterCharacters -> crate::column::UnboundColumn, |s: StrFilterCharacters<'_>| {
    let c = cudf_sys::strings::ffi::strings_filter_characters(
        s.view.0, s.from_chars, s.to_chars, s.keep, s.replacement, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::code_points`].
pub struct CodePoints<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(CodePoints -> crate::column::UnboundColumn, |s: CodePoints<'_>| {
    let c = cudf_sys::strings::ffi::strings_code_points(s.view.0, s.stream.as_raw())?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_cast_to_integer`].
pub struct StrCastToInteger<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    big_endian: bool,
    stream: Stream,
}
simple_builder!(StrCastToInteger -> crate::column::UnboundColumn, |s: StrCastToInteger<'_>| {
    let c = cudf_sys::strings::ffi::strings_cast_to_integer(
        s.view.0, s.output_type.repr, s.big_endian, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_cast_from_integer`].
pub struct StrCastFromInteger<'a> {
    view: &'a ColumnView<'a>,
    big_endian: bool,
    stream: Stream,
}
simple_builder!(StrCastFromInteger -> crate::column::UnboundColumn, |s: StrCastFromInteger<'_>| {
    let c = cudf_sys::strings::ffi::strings_cast_from_integer(
        s.view.0, s.big_endian, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_slice_column`].
pub struct StrSliceColumn<'a> {
    view: &'a ColumnView<'a>,
    starts: &'a ColumnView<'a>,
    stops: &'a ColumnView<'a>,
    stream: Stream,
}
simple_builder!(StrSliceColumn -> crate::column::UnboundColumn, |s: StrSliceColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_slice_column(
        s.view.0, s.starts.0, s.stops.0, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_extract_single`].
pub struct StrExtractSingle<'a> {
    view: &'a ColumnView<'a>,
    pattern: &'a str,
    group_index: i32,
    stream: Stream,
}
simple_builder!(StrExtractSingle -> crate::column::UnboundColumn, |s: StrExtractSingle<'_>| {
    let c = cudf_sys::strings::ffi::strings_extract_single(
        s.view.0, s.pattern, s.group_index, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_join_list_elements_column`].
pub struct StrJoinListElementsColumn<'a> {
    view: &'a ColumnView<'a>,
    separators: &'a ColumnView<'a>,
    separator_narep: &'a str,
    string_narep: &'a str,
    stream: Stream,
}
simple_builder!(StrJoinListElementsColumn -> crate::column::UnboundColumn, |s: StrJoinListElementsColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_join_list_elements_column(
        s.view.0, s.separators.0, s.separator_narep, s.string_narep, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_join_strings`].
pub struct StrJoinStrings<'a> {
    view: &'a ColumnView<'a>,
    separator: &'a str,
    narep: &'a str,
    stream: Stream,
}
simple_builder!(StrJoinStrings -> crate::column::UnboundColumn, |s: StrJoinStrings<'_>| {
    let c = cudf_sys::strings::ffi::strings_join_strings(
        s.view.0, s.separator, s.narep, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_find_instance`].
pub struct StrFindInstance<'a> {
    view: &'a ColumnView<'a>,
    target: &'a str,
    instance: i32,
    stream: Stream,
}
simple_builder!(StrFindInstance -> crate::column::UnboundColumn, |s: StrFindInstance<'_>| {
    let c = cudf_sys::strings::ffi::strings_find_instance(
        s.view.0, s.target, s.instance, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`StringExt::str_like_column`].
pub struct StrLikeColumn<'a> {
    view: &'a ColumnView<'a>,
    patterns: &'a ColumnView<'a>,
    escape_char: &'a str,
    stream: Stream,
}
simple_builder!(StrLikeColumn -> crate::column::UnboundColumn, |s: StrLikeColumn<'_>| {
    let c = cudf_sys::strings::ffi::strings_like_column(
        s.view.0, s.patterns.0, s.escape_char, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`concatenate_strings_with_separator`].
///
/// Created by [`concatenate_strings_with_separator`]. Call `.call()` to
/// execute, or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ConcatenateStringsWithSeparator<'a> {
    tbl: &'a UnboundTable,
    separators: &'a ColumnView<'a>,
    separator_narep: &'a str,
    col_narep: &'a str,
    stream: Stream,
}
simple_builder!(ConcatenateStringsWithSeparator -> crate::column::UnboundColumn, |s: ConcatenateStringsWithSeparator<'_>| {
    let c = cudf_sys::strings::ffi::strings_concatenate_columns_sep_col(
        &s.tbl.0, s.separators.0, s.separator_narep, s.col_narep, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

/// Builder for [`get_json_object`].
///
/// Created by [`get_json_object`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct GetJsonObject<'a> {
    col: &'a ColumnView<'a>,
    json_path: &'a str,
    allow_single_quotes: bool,
    strip_quotes: bool,
    missing_fields_as_nulls: bool,
    stream: Stream,
}
simple_builder!(GetJsonObject -> crate::column::UnboundColumn, |s: GetJsonObject<'_>| {
    let c = cudf_sys::strings::ffi::get_json_object(
        s.col.0, s.json_path, s.allow_single_quotes, s.strip_quotes,
        s.missing_fields_as_nulls, s.stream.as_raw(),
    )?;
    Ok(crate::column::RawColumn(c))
});

// ---------------------------------------------------------------------------
// Trait implementation
// ---------------------------------------------------------------------------

#[path = "strings/ext_impl.rs"]
mod ext_impl;

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Concatenates string columns row-wise with per-row separators.
///
/// For each row, the strings from all columns in `tbl` are concatenated with
/// the separator from the corresponding row of `separators`. The
/// `separator_narep` is used when a separator is null. The `col_narep` is used
/// when a column value is null.
///
/// Returns a [`ConcatenateStringsWithSeparator`] builder. Call `.call()` to
/// execute, or chain `.stream(s)` to run on a specific CUDA stream.
///
/// # Errors
///
/// Returns an error if the table columns or separator column are not `STRING` type.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::stream::GpuOp;
/// use cudf::strings::concatenate_strings_with_separator;
/// use cudf::table::Table;
///
/// # let tbl = Table::from_columns(vec![
/// #     Column::from_strings(&["a", "b"]).call()?,
/// #     Column::from_strings(&["x", "y"]).call()?,
/// # ])?;
/// # let separators = Column::from_strings(&["-", "/"]).call()?;
/// let result = concatenate_strings_with_separator(
///     &tbl, &separators.view(), "-", "N/A",
/// ).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn concatenate_strings_with_separator<'a>(
    tbl: &'a UnboundTable,
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

/// Extracts values from JSON strings using a `JSONPath` expression.
///
/// Each row in the input `col` must be a valid JSON string. The `json_path`
/// expression (e.g. `"$.store.book[0].title"`) is applied to every row,
/// returning the matched values as strings.
///
/// Options:
/// - `allow_single_quotes` -- when `true`, accepts single-quoted JSON strings.
/// - `strip_quotes` -- when `true`, removes surrounding quotes from extracted
///   string values.
/// - `missing_fields_as_nulls` -- when `true`, missing fields produce null
///   rather than an error.
///
/// Returns a [`GetJsonObject`] builder. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
///
/// # Errors
///
/// Returns an error if the input column is not `STRING` type.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::stream::GpuOp;
/// use cudf::strings::get_json_object;
///
/// # let json_col = Column::from_strings(&[r#"{"name":"Ada"}"#]).call()?;
/// let values = get_json_object(
///     &json_col.view(), "$.name", false, true, true,
/// ).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
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

/// String character type bitmask constants for use with
/// [`StringExt::all_characters_of_type`] and
/// [`StringExt::filter_characters_of_type`].
///
/// Combine constants with bitwise OR to match multiple types (e.g.
/// `ALPHA | DIGIT` matches alphanumeric characters).
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

    fn make_string_col(values: &[&str]) -> crate::column::UnboundColumn {
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
        let strs: Vec<String> = result.to_vec_string().call().unwrap();
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
