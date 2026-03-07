// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Sorting operations on GPU tables.
//!
//! Sorting is available as methods on [`Table`](crate::Table):
//! `table.sort(...)`, `table.sorted_order(...)`, `table.is_sorted(...)`.

#[doc(alias = "null_order")]
pub use cudf_sys::ffi::NullOrder;
#[doc(alias = "order")]
pub use cudf_sys::ffi::Order;

#[doc(alias = "rank_method")]
/// Method for resolving ties in rank.
///
/// Mirrors `cudf::rank_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RankMethod {
    /// Stable sort order ranking (no ties).
    First = 0,
    /// Mean of first in the group.
    Average = 1,
    /// Min of first in the group.
    Min = 2,
    /// Max of first in the group.
    Max = 3,
    /// Rank always increases by 1 between groups.
    Dense = 4,
}

#[allow(clippy::as_conversions)]
impl From<RankMethod> for i32 {
    fn from(m: RankMethod) -> Self {
        m as Self
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column as Col;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::sorting::{NullOrder, Order};
    use crate::table::TableBuilder;

    #[test]
    fn sort_single_column_ascending() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let sorted = table.sort_ascending().call().unwrap();
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted.columns_len(), 1);
    }

    #[test]
    fn sort_descending() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let sorted = table
            .sort(&[Order::DESCENDING], &[NullOrder::AFTER])
            .call()
            .unwrap();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn sorted_order_basic() {
        let col = Col::from_scalar(&Scalar::from_i32(1), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let indices = table.sorted_order(&[], &[]).call().unwrap();
        assert_eq!(indices.len(), 3);
        assert_eq!(indices.type_id(), TypeId::INT32);
    }

    #[test]
    fn is_sorted_true_for_constant() {
        let col = Col::from_scalar(&Scalar::from_i32(7), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table
            .is_sorted(&[Order::ASCENDING], &[NullOrder::BEFORE])
            .call()
            .unwrap();
        assert!(result);
    }

    #[test]
    fn is_sorted_descending_constant() {
        let col = Col::from_scalar(&Scalar::from_i32(7), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table
            .is_sorted(&[Order::DESCENDING], &[NullOrder::AFTER])
            .call()
            .unwrap();
        assert!(result);
    }
}
