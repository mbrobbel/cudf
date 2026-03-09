// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Concatenation operations for GPU columns and tables.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::stream::Stream;
use crate::table::Table;

/// Builder for [`concatenate_columns`].
pub struct ConcatenateColumns<'a> {
    columns: &'a [&'a ColumnView<'a>],
    stream: Stream,
}

/// Concatenates multiple columns vertically into a single column.
///
/// All columns must have the same type.
pub fn concatenate_columns<'a>(columns: &'a [&'a ColumnView<'a>]) -> ConcatenateColumns<'a> {
    ConcatenateColumns {
        columns,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ConcatenateColumns<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut cat = cudf_sys::concatenate::ffi::new_column_concatenator();
        for col in self.columns {
            cudf_sys::concatenate::ffi::column_concatenator_add(cat.pin_mut(), col.0);
        }
        let c = cudf_sys::concatenate::ffi::column_concatenator_finish(
            cat.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`concatenate_tables`].
pub struct ConcatenateTables<'a> {
    tables: &'a [&'a Table],
    stream: Stream,
}

/// Concatenates multiple tables vertically into a single table.
///
/// All tables must have the same number of columns and matching types.
pub fn concatenate_tables<'a>(tables: &'a [&'a Table]) -> ConcatenateTables<'a> {
    ConcatenateTables {
        tables,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ConcatenateTables<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut cat = cudf_sys::concatenate::ffi::new_table_concatenator();
        for t in self.tables {
            cudf_sys::concatenate::ffi::table_concatenator_add(cat.pin_mut(), &t.0);
        }
        let t = cudf_sys::concatenate::ffi::table_concatenator_finish(
            cat.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn concat_two_columns() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let c2 = Col::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let result = concatenate_columns(&[&c1.view(), &c2.view()])
            .call()
            .unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.type_id(), TypeId::INT32);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn concat_two_tables() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let c2 = Col::from_scalar(&Scalar::from_i32(3), 2).call().unwrap();
        let mut b1 = TableBuilder::new();
        b1.push_column(c1);
        let t1 = b1.build().unwrap();
        let mut b2 = TableBuilder::new();
        b2.push_column(c2);
        let t2 = b2.build().unwrap();
        let result = concatenate_tables(&[&t1, &t2]).call().unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.columns_len(), 1);
    }

    #[test]
    fn concat_empty_column() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let c2 = Col::empty(TypeId::INT32);
        let result = concatenate_columns(&[&c1.view(), &c2.view()])
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 1, 1]);
    }
}
