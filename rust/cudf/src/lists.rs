// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! List column operations.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Extension trait for list column operations.
pub trait ListExt {
    /// Returns an INT32 column with the number of elements in each list row.
    fn list_count_elements(&self) -> Result<Column>;
    /// Extract element at `index` from each list row.
    fn list_extract_element(&self, index: i32) -> Result<Column>;
    /// Sort elements within each list row.
    fn list_sort(&self, ascending: bool, nulls_last: bool) -> Result<Column>;
    /// Reverse elements within each list row.
    fn list_reverse(&self) -> Result<Column>;
    /// Returns a BOOL8 column indicating whether each list contains nulls.
    fn list_contains_nulls(&self) -> Result<Column>;
    /// Remove duplicate elements from each list.
    fn list_distinct(&self) -> Result<Column>;
    /// Concatenate nested list elements within each row (flatten one level).
    fn list_concatenate_elements(&self) -> Result<Column>;
    /// Check if each list row contains a scalar value (returns BOOL8 column).
    fn list_contains_scalar(&self, search_key: &Scalar) -> Result<Column>;
    /// Check if each list row contains the corresponding search_keys value.
    fn list_contains_column(&self, search_keys: &ColumnView<'_>) -> Result<Column>;
    /// Find position of scalar in each list row (-1 if not found).
    fn list_index_of_scalar(&self, search_key: &Scalar, find_first: bool) -> Result<Column>;
    /// Find position of each search_keys value in corresponding list row.
    fn list_index_of_column(&self, search_keys: &ColumnView<'_>, find_first: bool) -> Result<Column>;
    /// Gather elements from each list row using a gather map list column.
    fn list_segmented_gather(&self, gather_map: &ColumnView<'_>, nullify_oob: bool) -> Result<Column>;
    /// Format a list-of-strings column into formatted strings.
    fn list_format(&self, na_rep: &str) -> Result<Column>;
}

/// Generate sequences as list column from starts and sizes columns.
pub fn list_sequences(starts: &ColumnView<'_>, sizes: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::lists_sequences(starts.0, sizes.0, Stream::default_stream().as_raw())?;
    Ok(Column(c))
}

impl ListExt for ColumnView<'_> {
    fn list_count_elements(&self) -> Result<Column> {
        let c = cudf_sys::ffi::lists_count_elements(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_extract_element(&self, index: i32) -> Result<Column> {
        let c = cudf_sys::ffi::lists_extract_element(self.0, index, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_sort(&self, ascending: bool, nulls_last: bool) -> Result<Column> {
        let c = cudf_sys::ffi::lists_sort(self.0, ascending, nulls_last, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_reverse(&self) -> Result<Column> {
        let c = cudf_sys::ffi::lists_reverse(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_contains_nulls(&self) -> Result<Column> {
        let c = cudf_sys::ffi::lists_contains_nulls(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_distinct(&self) -> Result<Column> {
        let c = cudf_sys::ffi::lists_distinct(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_concatenate_elements(&self) -> Result<Column> {
        let c = cudf_sys::ffi::lists_concatenate_elements(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_contains_scalar(&self, search_key: &Scalar) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(search_key);
        let c = cudf_sys::ffi::lists_contains_scalar(self.0, &ffi, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_contains_column(&self, search_keys: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::lists_contains_column(self.0, search_keys.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_index_of_scalar(&self, search_key: &Scalar, find_first: bool) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(search_key);
        let c = cudf_sys::ffi::lists_index_of_scalar(self.0, &ffi, find_first, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_index_of_column(&self, search_keys: &ColumnView<'_>, find_first: bool) -> Result<Column> {
        let c = cudf_sys::ffi::lists_index_of_column(self.0, search_keys.0, find_first, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_segmented_gather(&self, gather_map: &ColumnView<'_>, nullify_oob: bool) -> Result<Column> {
        let c = cudf_sys::ffi::lists_segmented_gather(self.0, gather_map.0, nullify_oob, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn list_format(&self, na_rep: &str) -> Result<Column> {
        let c = cudf_sys::ffi::strings_format_list_column(self.0, na_rep, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
}

/// Check if two list columns have overlapping elements per row.
pub fn lists_have_overlap(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::lists_have_overlap(lhs.0, rhs.0, Stream::default_stream().as_raw())?;
    Ok(Column(c))
}

/// Intersect distinct elements of two list columns per row.
pub fn lists_intersect_distinct(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::lists_intersect_distinct(lhs.0, rhs.0, Stream::default_stream().as_raw())?;
    Ok(Column(c))
}

/// Union distinct elements of two list columns per row.
pub fn lists_union_distinct(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::lists_union_distinct(lhs.0, rhs.0, Stream::default_stream().as_raw())?;
    Ok(Column(c))
}

/// Difference distinct elements of two list columns per row.
pub fn lists_difference_distinct(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> Result<Column> {
    let c = cudf_sys::ffi::lists_difference_distinct(lhs.0, rhs.0, Stream::default_stream().as_raw())?;
    Ok(Column(c))
}
