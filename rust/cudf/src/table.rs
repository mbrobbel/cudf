// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use cxx::UniquePtr;

use crate::column::{Column, ColumnView};

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

/// A builder for constructing a [`Table`] from individual [`Column`]s.
pub struct TableBuilder(UniquePtr<cudf_sys::ffi::TableBuilder>);

impl Default for TableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TableBuilder {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self(cudf_sys::ffi::new_table_builder())
    }

    /// Adds a column to the builder, transferring ownership.
    pub fn push_column(&mut self, col: Column) {
        cudf_sys::ffi::table_builder_add_column(self.0.pin_mut(), col.0);
    }

    /// Consumes the builder and returns a [`Table`].
    pub fn build(mut self) -> Table {
        let tbl = cudf_sys::ffi::table_builder_build(self.0.pin_mut())
            .expect("table_builder_build should not fail");
        Table(tbl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;

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

    #[test]
    fn table_from_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 3);
        let c2 = Column::from_scalar(&Scalar::from_f64(2.5), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build();
        assert_eq!(table.columns_len(), 2);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn table_column_views() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build();

        let view = table.column(0).unwrap();
        assert_eq!(view.len(), 4);
        assert_eq!(view.type_id(), TypeId::INT32);
    }

    #[test]
    fn table_columns_iterator_with_data() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 2);
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 2);
        let c3 = Column::from_scalar(&Scalar::from_bool(true), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        builder.push_column(c3);
        let table = builder.build();

        let cols: Vec<_> = table.columns().collect();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].type_id(), TypeId::INT32);
        assert_eq!(cols[1].type_id(), TypeId::FLOAT64);
        assert_eq!(cols[2].type_id(), TypeId::BOOL8);
    }
}
