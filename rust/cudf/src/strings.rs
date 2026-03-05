// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! String operations on GPU columns.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::scalar::Scalar;

/// Converts each string to lower case.
pub fn to_lower(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_to_lower(col.0).expect("strings_to_lower failed"))
}

/// Converts each string to upper case.
pub fn to_upper(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_to_upper(col.0).expect("strings_to_upper failed"))
}

/// Returns a BOOL8 column indicating whether each string contains the target.
pub fn contains(col: &ColumnView, target: &Scalar) -> Column {
    Column(cudf_sys::ffi::strings_contains(col.0, &target.0).expect("strings_contains failed"))
}

/// Returns a BOOL8 column indicating whether each string starts with the target.
pub fn starts_with(col: &ColumnView, target: &Scalar) -> Column {
    Column(
        cudf_sys::ffi::strings_starts_with(col.0, &target.0)
            .expect("strings_starts_with failed"),
    )
}

/// Returns a BOOL8 column indicating whether each string ends with the target.
pub fn ends_with(col: &ColumnView, target: &Scalar) -> Column {
    Column(
        cudf_sys::ffi::strings_ends_with(col.0, &target.0).expect("strings_ends_with failed"),
    )
}

/// Returns an INT32 column with the position of the first occurrence of target in each string.
/// Returns -1 if not found.
pub fn find(col: &ColumnView, target: &Scalar) -> Column {
    Column(cudf_sys::ffi::strings_find(col.0, &target.0).expect("strings_find failed"))
}

/// Replaces all occurrences of `target` with `replacement` in each string.
pub fn replace(col: &ColumnView, target: &Scalar, replacement: &Scalar) -> Column {
    Column(
        cudf_sys::ffi::strings_replace(col.0, &target.0, &replacement.0)
            .expect("strings_replace failed"),
    )
}

/// Strips whitespace from both sides of each string.
pub fn strip(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_strip(col.0).expect("strings_strip failed"))
}

/// Strips whitespace from the left side of each string.
pub fn lstrip(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_lstrip(col.0).expect("strings_lstrip failed"))
}

/// Strips whitespace from the right side of each string.
pub fn rstrip(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_rstrip(col.0).expect("strings_rstrip failed"))
}

/// Returns an INT32 column with the character count of each string.
pub fn count_characters(col: &ColumnView) -> Column {
    Column(
        cudf_sys::ffi::strings_count_characters(col.0).expect("strings_count_characters failed"),
    )
}

/// Returns an INT32 column with the byte count of each string.
pub fn count_bytes(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_count_bytes(col.0).expect("strings_count_bytes failed"))
}

/// Converts an integer column to a string column.
pub fn from_integers(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_from_integers(col.0).expect("strings_from_integers failed"))
}

/// Converts a string column to an integer column of the specified type.
pub fn to_integers(col: &ColumnView, output_type: TypeId) -> Column {
    Column(
        cudf_sys::ffi::strings_to_integers(col.0, output_type.repr)
            .expect("strings_to_integers failed"),
    )
}

/// Converts a float column to a string column.
pub fn from_floats(col: &ColumnView) -> Column {
    Column(cudf_sys::ffi::strings_from_floats(col.0).expect("strings_from_floats failed"))
}

/// Converts a string column to a float column of the specified type.
pub fn to_floats(col: &ColumnView, output_type: TypeId) -> Column {
    Column(
        cudf_sys::ffi::strings_to_floats(col.0, output_type.repr)
            .expect("strings_to_floats failed"),
    )
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
        let result = to_lower(&col.view());
        assert_eq!(result.to_vec_string(), vec!["hello", "world", "foo"]);
    }

    #[test]
    fn to_upper_basic() {
        let col = make_string_col(&["hello", "World", "FOO"]);
        let result = to_upper(&col.view());
        assert_eq!(result.to_vec_string(), vec!["HELLO", "WORLD", "FOO"]);
    }

    #[test]
    fn contains_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("abc");
        let result = contains(&col.view(), &target);
        assert_eq!(result.to_vec_bool(), vec![true, false, true]);
    }

    #[test]
    fn starts_with_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("ab");
        let result = starts_with(&col.view(), &target);
        assert_eq!(result.to_vec_bool(), vec![true, false, true]);
    }

    #[test]
    fn ends_with_basic() {
        let col = make_string_col(&["abc", "def", "abcdef"]);
        let target = Scalar::from_string("ef");
        let result = ends_with(&col.view(), &target);
        assert_eq!(result.to_vec_bool(), vec![false, true, true]);
    }

    #[test]
    fn find_basic() {
        let col = make_string_col(&["hello world", "goodbye", "hello"]);
        let target = Scalar::from_string("lo");
        let result = find(&col.view(), &target);
        assert_eq!(result.to_vec_i32(), vec![3, -1, 3]);
    }

    #[test]
    fn replace_basic() {
        let col = make_string_col(&["hello", "world"]);
        let target = Scalar::from_string("o");
        let replacement = Scalar::from_string("0");
        let result = replace(&col.view(), &target, &replacement);
        assert_eq!(result.to_vec_string(), vec!["hell0", "w0rld"]);
    }

    #[test]
    fn strip_basic() {
        let col = make_string_col(&["  hello  ", " world", "foo "]);
        let result = strip(&col.view());
        assert_eq!(result.to_vec_string(), vec!["hello", "world", "foo"]);
    }

    #[test]
    fn lstrip_basic() {
        let col = make_string_col(&["  hello  ", " world"]);
        let result = lstrip(&col.view());
        assert_eq!(result.to_vec_string(), vec!["hello  ", "world"]);
    }

    #[test]
    fn rstrip_basic() {
        let col = make_string_col(&["  hello  ", "world "]);
        let result = rstrip(&col.view());
        assert_eq!(result.to_vec_string(), vec!["  hello", "world"]);
    }

    #[test]
    fn count_characters_basic() {
        let col = make_string_col(&["hello", "ab", ""]);
        let result = count_characters(&col.view());
        assert_eq!(result.to_vec_i32(), vec![5, 2, 0]);
    }

    #[test]
    fn count_bytes_basic() {
        let col = make_string_col(&["hello", "ab", ""]);
        let result = count_bytes(&col.view());
        assert_eq!(result.to_vec_i32(), vec![5, 2, 0]);
    }

    #[test]
    fn strings_from_integers_basic() {
        let col = Column::from_strings(&["1", "2", "3"]);
        let int_col = to_integers(&col.view(), TypeId::INT32);
        assert_eq!(int_col.to_vec_i32(), vec![1, 2, 3]);

        let back = from_integers(&int_col.view());
        assert_eq!(back.to_vec_string(), vec!["1", "2", "3"]);
    }

    #[test]
    fn strings_to_integers_basic() {
        let col = make_string_col(&["10", "20", "30"]);
        let result = to_integers(&col.view(), TypeId::INT32);
        assert_eq!(result.to_vec_i32(), vec![10, 20, 30]);
    }

    #[test]
    fn strings_from_floats_basic() {
        let col = Scalar::from_f64(1.5);
        let float_col = Column::from_scalar(&col, 2);
        let result = from_floats(&float_col.view());
        assert_eq!(result.len(), 2);
        // Float formatting may vary, just check it's non-empty strings
        let strs = result.to_vec_string();
        assert!(strs.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn strings_to_floats_basic() {
        let col = make_string_col(&["1.5", "2.5", "3.0"]);
        let result = to_floats(&col.view(), TypeId::FLOAT64);
        let vals = result.to_vec_f64();
        assert!((vals[0] - 1.5).abs() < 1e-9);
        assert!((vals[1] - 2.5).abs() < 1e-9);
        assert!((vals[2] - 3.0).abs() < 1e-9);
    }
}
