// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reshape operations on tables and columns.
//!
//! These operations change the shape of a table or column without modifying
//! the underlying data values.
//!
//! # Table methods
//!
//! - [`Table::interleave_columns`](crate::table::Table::interleave_columns) --
//!   merges all columns into a single column by interleaving rows.
//! - [`Table::tile`](crate::table::Table::tile) -- repeats the entire table
//!   vertically a specified number of times.
//!
//! # Free functions
//!
//! - [`one_hot_encode`] -- produces a `BOOL8` table from a column and a set
//!   of category values.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::column::Column;
//! use cudf::stream::GpuOp;
//! use cudf::table::TableBuilder;
//!
//! let c1 = Column::from_slice_i32(&[1, 2, 3]).call()?;
//! let c2 = Column::from_slice_i32(&[4, 5, 6]).call()?;
//! let mut b = TableBuilder::new();
//! b.push_column(c1);
//! b.push_column(c2);
//! let table = b.build()?;
//!
//! // Interleave: [1, 4, 2, 5, 3, 6]
//! let interleaved = table.interleave_columns().call()?;
//! assert_eq!(interleaved.len(), 6);
//!
//! // Tile: repeat twice => 6 rows
//! let tiled = table.tile(2).call()?;
//! assert_eq!(tiled.len(), 6);
//! # Ok::<(), cudf::error::Error>(())
//! ```

use crate::column::ColumnView;
use crate::error::Result;
use crate::stream::Stream;
use crate::table::Table;

/// Builder for [`one_hot_encode`]. See that function for details.
pub struct OneHotEncode<'a> {
    input: &'a ColumnView<'a>,
    categories: &'a ColumnView<'a>,
    stream: Stream,
}

/// One-hot encodes `input` against `categories`.
///
/// For each element in `categories`, the output table contains a `BOOL8`
/// column where row *i* is `true` if `input[i]` equals that category value,
/// and `false` otherwise. The output table has `categories.len()` columns
/// and `input.len()` rows.
///
/// `input` and `categories` must have the same data type.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::reshape::one_hot_encode;
/// use cudf::stream::GpuOp;
///
/// let input = Column::from_slice_i32(&[1, 2, 3, 1]).call()?;
/// let cats = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// let result = one_hot_encode(&input.view(), &cats.view()).call()?;
/// assert_eq!(result.columns_len(), 3); // one column per category
/// assert_eq!(result.len(), 4);         // same row count as input
/// # Ok::<(), cudf::error::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if the data types of `input` and `categories` are
/// incompatible, or if a GPU error occurs.
pub fn one_hot_encode<'a>(
    input: &'a ColumnView<'a>,
    categories: &'a ColumnView<'a>,
) -> OneHotEncode<'a> {
    OneHotEncode {
        input,
        categories,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for OneHotEncode<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::transform::ffi::one_hot_encode(
            self.input.0,
            self.categories.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn interleave_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]).call().unwrap();
        let c2 = Col::from_slice_i32(&[4, 5, 6]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        let result = table.interleave_columns().call().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.to_vec_i32().call().unwrap(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn tile_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let result = table.tile(2).call().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }
}
