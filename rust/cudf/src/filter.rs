// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Filtering operations on GPU tables.
//!
//! Filtering is available as methods on [`Table`](crate::table::Table):
//! `table.filter(...)`, `table.drop_nulls()`.

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn filter_basic() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 4).call().unwrap();
        let mask = Column::from_scalar(&Scalar::from_bool(true), 4)
            .call()
            .unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.filter(&mask.view()).call().unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn drop_nulls_no_nulls() {
        let col = Column::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.drop_nulls().call().unwrap();
        assert_eq!(result.len(), 3);
    }
}
