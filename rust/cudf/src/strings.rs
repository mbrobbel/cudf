// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! String operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

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
