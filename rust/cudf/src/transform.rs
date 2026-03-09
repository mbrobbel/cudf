// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Transform operations on columns and tables.
//!
//! # Column transforms
//!
//! - [`ColumnView::nans_to_nulls`](crate::column::ColumnView::nans_to_nulls) --
//!   replaces NaN values with null in floating-point columns.
//!
//! # Table transforms
//!
//! - [`Table::encode`](crate::table::Table::encode) -- dictionary-encodes the
//!   table rows, returning an `INT32` column of indices into the sorted
//!   distinct rows.
//! - [`Table::encode_keys`](crate::table::Table::encode_keys) -- returns the
//!   sorted distinct row combinations as a [`Table`](crate::table::Table).
//!
//! `encode` and `encode_keys` are complementary: `encode` maps each input row
//! to an integer key, and `encode_keys` provides the look-up table for those
//! keys.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::column::Column;
//! use cudf::stream::GpuOp;
//! use cudf::table::TableBuilder;
//!
//! // NaN-to-null transform
//! let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]).call()?;
//! let clean = col.view().nans_to_nulls().call()?;
//! assert_eq!(clean.null_count(), 1);
//!
//! // Encode / encode_keys
//! let data = Column::from_slice_i32(&[3, 1, 2, 1, 3]).call()?;
//! let mut b = TableBuilder::new();
//! b.push_column(data);
//! let table = b.build()?;
//!
//! let indices = table.encode().call()?;    // INT32 column: [2, 0, 1, 0, 2]
//! let keys = table.encode_keys().call()?;  // 3-row table: [1, 2, 3]
//! # Ok::<(), cudf::error::Error>(())
//! ```

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn nans_to_nulls_basic() {
        let data = [1.0_f64, f64::NAN, 3.0, f64::NAN, 5.0];
        let col = Col::from_slice_f64(&data).call().unwrap();
        let result = col.view().nans_to_nulls().call().unwrap();
        assert_eq!(result.len(), 5);
        assert!(result.has_nulls());
        assert_eq!(result.null_count(), 2);
    }

    #[test]
    fn encode_basic() {
        let c1 = Col::from_slice_i32(&[3, 1, 2, 1, 3]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let indices = table.encode().call().unwrap();
        assert_eq!(indices.len(), 5);
        let vals = indices.to_vec_i32().call().unwrap();
        assert_eq!(vals, vec![2, 0, 1, 0, 2]);
    }

    #[test]
    fn encode_keys_basic() {
        let c1 = Col::from_slice_i32(&[3, 1, 2, 1, 3]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        let keys = table.encode_keys().call().unwrap();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys.columns_len(), 1);
    }
}
