// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reshape operations on tables and columns.
//!
//! Available as methods on [`Table`]:
//! `table.interleave_columns()`, `table.tile(...)`.
//!
//! Free functions:
//! - [`one_hot_encode`] — one-hot encode input against categories.

use crate::column::ColumnView;
use crate::error::Result;
use crate::stream::Stream;
use crate::table::Table;

/// Builder for [`one_hot_encode`].
pub struct OneHotEncode<'a> {
    input: &'a ColumnView<'a>,
    categories: &'a ColumnView<'a>,
    stream: Stream,
}

/// One-hot encode `input` against `categories`, returning a table of BOOL8 columns
/// (one column per category).
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

impl OneHotEncode<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
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
    use crate::table::TableBuilder;

    #[test]
    fn interleave_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let c2 = Col::from_slice_i32(&[4, 5, 6]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        let result = table.interleave_columns().call().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.to_vec_i32(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn tile_basic() {
        let c1 = Col::from_slice_i32(&[1, 2, 3]);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let result = table.tile(2).call().unwrap();
        assert_eq!(result.len(), 6);
        assert_eq!(result.columns_len(), 1);
    }
}
