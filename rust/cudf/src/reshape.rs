// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reshape operations on tables and columns.
//!
//! Available as methods on [`Table`](crate::Table):
//! `table.interleave_columns()`, `table.tile(...)`.
//!
//! Free functions:
//! - [`one_hot_encode`] — one-hot encode input against categories.

use crate::column::ColumnView;
use crate::error::Result;
use crate::stream::Stream;
use crate::table::Table;

/// One-hot encode `input` against `categories`, returning a table of BOOL8 columns
/// (one column per category).
pub fn one_hot_encode(input: &ColumnView<'_>, categories: &ColumnView<'_>) -> Result<Table> {
    one_hot_encode_on(input, categories, Stream::default_stream())
}

/// One-hot encode on a custom CUDA stream.
pub fn one_hot_encode_on(
    input: &ColumnView<'_>,
    categories: &ColumnView<'_>,
    stream: Stream,
) -> Result<Table> {
    let t = cudf_sys::ffi::one_hot_encode(input.0, categories.0, stream.as_raw())?;
    Ok(Table(t))
}

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::table::TableBuilder;

    #[test]
    fn interleave_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let c2 = Col::from_slice_i32(&[4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        let result = table.interleave_columns().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.to_vec_i32(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn tile_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let result = table.tile(2).unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }
}
