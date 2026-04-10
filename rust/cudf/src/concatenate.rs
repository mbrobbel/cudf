// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Concatenation operations for GPU columns and tables.

use crate::column::ColumnView;
use crate::error::Result;
use crate::stream::Stream;
use crate::table::UnboundTable;

/// Builder for [`concatenate_columns`].
///
/// Created by [`concatenate_columns()`]. Call [`.call()`](crate::stream::GpuOp::call) to
/// execute. Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct ConcatenateColumns<'a> {
    columns: &'a [&'a ColumnView<'a>],
    stream: Stream,
}

/// Concatenates multiple columns vertically into a single column.
///
/// The output column contains all elements from each input column in order.
/// All input columns must have the same [`TypeId`](crate::data_type::TypeId);
/// null masks are concatenated accordingly.
///
/// # Arguments
///
/// * `columns` -- Slice of column views to concatenate. An empty slice produces
///   an error since the output type cannot be determined.
///
/// # Returns
///
/// A new [`Column`] whose length is the sum of all input column lengths.
///
/// # Errors
///
/// Returns an error if the columns have mismatched types or if the libcudf
/// call fails.
///
/// # Examples
///
/// ```no_run
/// use cudf::concatenate::concatenate_columns;
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::GpuOp;
///
/// let a = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
/// let b = Column::from_scalar(&Scalar::from_i32(2), 2).call()?;
/// let result = concatenate_columns(&[&a.view(), &b.view()]).call()?;
/// // result contains [1, 1, 1, 2, 2]
/// assert_eq!(result.len(), 5);
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn concatenate_columns<'a>(columns: &'a [&'a ColumnView<'a>]) -> ConcatenateColumns<'a> {
    ConcatenateColumns {
        columns,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ConcatenateColumns<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let Some(first) = self.columns.first() else {
            return Err(crate::error::Error::InvalidArgument(
                "at least one column is required for concatenation".into(),
            ));
        };
        if self
            .columns
            .iter()
            .skip(1)
            .any(|column| !crate::column_types_equivalent(first, column))
        {
            return Err(crate::error::Error::InvalidArgument(
                "all columns must have equivalent types".into(),
            ));
        }
        let mut cat = cudf_sys::concatenate::ffi::new_column_concatenator();
        for col in self.columns {
            cudf_sys::concatenate::ffi::column_concatenator_add(cat.pin_mut(), col.0);
        }
        let c = cudf_sys::concatenate::ffi::column_concatenator_finish(
            cat.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`concatenate_tables`].
///
/// Created by [`concatenate_tables()`]. Call [`.call()`](crate::stream::GpuOp::call) to
/// execute. Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct ConcatenateTables<'a> {
    tables: &'a [&'a UnboundTable],
    stream: Stream,
}

/// Concatenates multiple tables vertically into a single table.
///
/// Stacks the rows of each input table in order. All tables must have the same
/// number of columns and each corresponding column pair must have the same
/// [`TypeId`](crate::data_type::TypeId).
///
/// # Arguments
///
/// * `tables` -- Slice of tables to concatenate.
///
/// # Returns
///
/// A new [`Table`] whose row count is the sum of all input table row counts.
///
/// # Errors
///
/// Returns an error if the tables have mismatched schemas (column count or
/// types) or if the libcudf call fails.
///
/// # Examples
///
/// ```no_run
/// use cudf::concatenate::concatenate_tables;
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::GpuOp;
/// use cudf::table::TableBuilder;
///
/// let c1 = Column::from_scalar(&Scalar::from_i32(1), 2).call()?;
/// let mut b1 = TableBuilder::new();
/// b1.push_column(c1);
/// let t1 = b1.build()?;
///
/// let c2 = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
/// let mut b2 = TableBuilder::new();
/// b2.push_column(c2);
/// let t2 = b2.build()?;
///
/// let result = concatenate_tables(&[&t1, &t2]).call()?;
/// assert_eq!(result.len(), 4);
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn concatenate_tables<'a>(tables: &'a [&'a UnboundTable]) -> ConcatenateTables<'a> {
    ConcatenateTables {
        tables,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ConcatenateTables<'_> {
    type Output = crate::table::UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let Some(first) = self.tables.first() else {
            return Err(crate::error::Error::InvalidArgument(
                "at least one table is required for concatenation".into(),
            ));
        };
        if self.tables.iter().skip(1).any(|table| {
            table.columns_len() != first.columns_len()
                || !crate::tables_have_same_types(first, table)
        }) {
            return Err(crate::error::Error::InvalidArgument(
                "all tables must have matching column schemas".into(),
            ));
        }
        let mut cat = cudf_sys::concatenate::ffi::new_table_concatenator();
        for t in self.tables {
            cudf_sys::concatenate::ffi::table_concatenator_add(cat.pin_mut(), &t.0);
        }
        let t = cudf_sys::concatenate::ffi::table_concatenator_finish(
            cat.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(crate::table::RawTable(t))
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
        let c2 = Col::empty(TypeId::INT32).unwrap();
        let result = concatenate_columns(&[&c1.view(), &c2.view()])
            .call()
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn concat_columns_empty_errors() {
        assert!(concatenate_columns(&[]).call().is_err());
    }

    #[test]
    fn concat_columns_mismatched_types_error() {
        let ints = Col::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let floats = Col::from_scalar(&Scalar::from_f64(2.0), 2).call().unwrap();
        assert!(
            concatenate_columns(&[&ints.view(), &floats.view()])
                .call()
                .is_err()
        );
    }

    #[test]
    fn concat_tables_mismatched_schema_error() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let c2 = Col::from_scalar(&Scalar::from_f64(3.0), 2).call().unwrap();
        let mut b1 = TableBuilder::new();
        b1.push_column(c1);
        let t1 = b1.build().unwrap();

        let mut b2 = TableBuilder::new();
        b2.push_column(c2);
        let t2 = b2.build().unwrap();

        assert!(concatenate_tables(&[&t1, &t2]).call().is_err());
    }
}
