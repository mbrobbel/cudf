// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Transform operations on columns and tables.
//!
//! - `column_view.nans_to_nulls()` converts NaN to null.
//! - `table.encode()` and `table.encode_keys()` for row encoding.

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
