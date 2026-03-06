// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! List column operations.

use crate::column::{Column, ColumnView};
use crate::error::Result;
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
}
