// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use cxx::UniquePtr;

use crate::column::ColumnView;

/// An owning GPU table (a set of columns of the same size).
///
/// Wraps a `cudf::table` via the CXX FFI layer. Dropping this value
/// frees the underlying GPU memory for all columns.
pub struct Table(pub(crate) UniquePtr<cudf_sys::ffi::Table>);

// GPU memory is globally accessible from any CPU thread.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Default for Table {
    /// Creates an empty table with zero columns and zero rows.
    fn default() -> Self {
        Self(cudf_sys::ffi::table_empty())
    }
}

impl Table {
    /// Returns the number of columns.
    pub fn columns_len(&self) -> usize {
        cudf_sys::ffi::table_num_columns(&self.0) as usize
    }

    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        cudf_sys::ffi::table_num_rows(&self.0) as usize
    }

    /// Returns `true` if the table has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns_len() == 0
    }

    /// Returns the total GPU allocation size in bytes.
    pub fn alloc_bytes(&self) -> usize {
        cudf_sys::ffi::table_alloc_size(&self.0)
    }

    /// Returns a view of the column at the given index, or `None` if out of bounds.
    pub fn column(&self, index: usize) -> Option<ColumnView<'_>> {
        if index >= self.columns_len() {
            return None;
        }
        let view = cudf_sys::ffi::table_get_column_view(&self.0, index as i32).ok()?;
        Some(ColumnView(view))
    }

    /// Returns an iterator over all columns as [`ColumnView`]s.
    pub fn columns(&self) -> Columns<'_> {
        Columns {
            table: self,
            index: 0,
            len: self.columns_len(),
        }
    }
}

/// An iterator over the columns of a [`Table`].
pub struct Columns<'a> {
    table: &'a Table,
    index: usize,
    len: usize,
}

impl<'a> Iterator for Columns<'a> {
    type Item = ColumnView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let col = self.table.column(self.index);
        if col.is_some() {
            self.index += 1;
        }
        col
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Columns<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_default() {
        let table = Table::default();
        assert_eq!(table.columns_len(), 0);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn empty_table_column_out_of_bounds() {
        let table = Table::default();
        assert!(table.column(0).is_none());
    }

    #[test]
    fn empty_table_columns_iterator() {
        let table = Table::default();
        let cols: Vec<_> = table.columns().collect();
        assert!(cols.is_empty());
        assert_eq!(table.columns().len(), 0);
    }
}
