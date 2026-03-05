// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Copying operations for GPU columns and tables.
//!
//! Available as methods: `table.gather(...)`, `table.empty_like()`,
//! `column_view.empty_like()`.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    #[test]
    fn gather_basic() {
        let col = Col::from_scalar(&Scalar::from_i32(10), 3);
        let indices = Col::from_scalar(&Scalar::from_i32(0), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.gather(&indices.view()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn empty_like_column_test() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 3);
        let empty = col.view().empty_like();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.type_id(), TypeId::INT32);
    }

    #[test]
    fn empty_like_table_test() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2);
        let c2 = Col::from_scalar(&Scalar::from_f64(2.5), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        let empty = table.empty_like();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.columns_len(), 2);
    }
}
