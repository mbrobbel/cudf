// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Copying operations for GPU columns and tables.
//!
//! These operations copy, gather, scatter, or reshape column/table data on the
//! GPU. Available as methods on [`Table`](crate::table::Table) and
//! [`ColumnView`](crate::column::ColumnView):
//!
//! * [`Table::gather`](crate::table::Table::gather) -- select rows by index.
//! * [`Table::gather_checked`](crate::table::Table::gather_checked) -- gather
//!   with out-of-bounds policy.
//! * [`Table::scatter`](crate::table::Table::scatter) -- scatter rows into a
//!   target table at given positions.
//! * [`Table::scatter_scalars`](crate::table::Table::scatter_scalars) -- scatter
//!   scalar values to specified positions.
//! * [`Table::boolean_mask_scatter`](crate::table::Table::boolean_mask_scatter) --
//!   scatter using a boolean mask.
//! * [`Table::empty_like`](crate::table::Table::empty_like) -- empty table with
//!   same schema.
//! * [`Table::reverse`](crate::table::Table::reverse) -- reverse row order.
//! * [`ColumnView::empty_like`](crate::column::ColumnView::empty_like) -- empty
//!   column with same type.
//! * [`ColumnView::copy_range_into`](crate::column::ColumnView::copy_range_into) --
//!   copy a range of elements into another column.
//! * [`ColumnView::copy_if_else`](crate::column::ColumnView::copy_if_else) --
//!   select from two columns based on a mask.

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn gather_basic() {
        let col = Col::from_scalar(&Scalar::from_i32(10), 3).call().unwrap();
        let indices = Col::from_scalar(&Scalar::from_i32(0), 2).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.gather(&indices.view()).call().unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn empty_like_column_test() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
        let empty = col.view().empty_like();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.type_id(), TypeId::INT32);
    }

    #[test]
    fn empty_like_table_test() {
        let c1 = Col::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let c2 = Col::from_scalar(&Scalar::from_f64(2.5), 2).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        let empty = table.empty_like();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.columns_len(), 2);
    }
}
