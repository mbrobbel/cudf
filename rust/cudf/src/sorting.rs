// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Sorting, ordering, and ranking operations on GPU tables and columns.
//!
//! # Table-level sorting
//!
//! These methods are available on [`Table`](crate::table::Table):
//!
//! - [`sort`](crate::table::Table::sort) -- sorts all rows by all columns
//!   using per-column [`Order`] and [`NullOrder`].
//! - [`sort_ascending`](crate::table::Table::sort_ascending) -- convenience
//!   shorthand: ascending order, nulls before.
//! - [`sorted_order`](crate::table::Table::sorted_order) -- returns an
//!   `INT32` column of row indices that would sort the table.
//! - [`is_sorted`](crate::table::Table::is_sorted) -- checks whether the
//!   table is already sorted.
//! - [`sort_by_key`](crate::table::Table::sort_by_key) / [`stable_sort_by_key`](crate::table::Table::stable_sort_by_key) --
//!   sort a values table using a separate keys table.
//!
//! # Column-level ranking
//!
//! - [`ColumnView::rank`](crate::column::ColumnView::rank) -- computes the
//!   rank of each element using a chosen [`RankMethod`].
//!
//! # Examples
//!
//! ```ignore
//! use cudf::column::Column;
//! use cudf::sorting::{Order, NullOrder};
//! use cudf::stream::GpuOp;
//! use cudf::table::TableBuilder;
//!
//! let col = Column::from_slice_i32(&[3, 1, 2]).call()?;
//! let mut b = TableBuilder::new();
//! b.push_column(col);
//! let table = b.build()?;
//!
//! // Sort ascending, nulls before
//! let sorted = table.sort(&[Order::ASCENDING], &[NullOrder::BEFORE]).call()?;
//!
//! // Get the sorted indices
//! let indices = table.sorted_order(&[Order::ASCENDING], &[NullOrder::BEFORE]).call()?;
//!
//! // Check if already sorted
//! let ok = table.is_sorted(&[Order::ASCENDING], &[NullOrder::BEFORE]).call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

/// Placement of null values in sorted output.
///
/// - `BEFORE` -- nulls appear before all non-null values.
/// - `AFTER` -- nulls appear after all non-null values.
///
/// Pass one `NullOrder` per column alongside [`Order`]. When the slice is
/// empty, all columns default to `AFTER`.
#[doc(alias = "null_order")]
pub use cudf_sys::ffi::NullOrder;

/// Sort direction for a column.
///
/// - `ASCENDING` -- smallest values first.
/// - `DESCENDING` -- largest values first.
///
/// Pass one `Order` per column to [`Table::sort`](crate::table::Table::sort),
/// [`Table::sorted_order`](crate::table::Table::sorted_order), and related
/// methods. When the slice is empty, all columns default to `ASCENDING`.
#[doc(alias = "order")]
pub use cudf_sys::ffi::Order;

#[doc(alias = "rank_method")]
/// Method for resolving ties when computing ranks.
///
/// Mirrors `cudf::rank_method`. Used by
/// [`ColumnView::rank`](crate::column::ColumnView::rank).
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::compaction::NullPolicy;
/// use cudf::sorting::{Order, NullOrder, RankMethod};
/// use cudf::stream::GpuOp;
///
/// let col = Column::from_slice_i32(&[30, 10, 20, 10]).call()?;
/// let view = col.view();
///
/// // Dense rank: [3, 1, 2, 1]
/// let ranks = view.rank(
///     RankMethod::Dense,
///     Order::ASCENDING,
///     NullPolicy::EXCLUDE,
///     NullOrder::AFTER,
///     false,
/// ).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RankMethod {
    /// Stable sort order ranking -- tied values receive distinct consecutive
    /// ranks based on their position in the input. No ties in the output.
    First = 0,
    /// Tied values receive the mean of the ranks they would span under
    /// [`First`](RankMethod::First). Output type is `FLOAT64`.
    Average = 1,
    /// Tied values all receive the minimum rank of the group.
    Min = 2,
    /// Tied values all receive the maximum rank of the group.
    Max = 3,
    /// Like [`Min`](RankMethod::Min), but ranks always increase by exactly 1
    /// between distinct values (no gaps).
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
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn sort_single_column_ascending() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 4).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let sorted = table.sort_ascending().call().unwrap();
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted.columns_len(), 1);
    }

    #[test]
    fn sort_descending() {
        let col = Col::from_scalar(&Scalar::from_i32(5), 3).call().unwrap();
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
        let col = Col::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let indices = table.sorted_order(&[], &[]).call().unwrap();
        assert_eq!(indices.len(), 3);
        assert_eq!(indices.type_id(), TypeId::INT32);
    }

    #[test]
    fn is_sorted_true_for_constant() {
        let col = Col::from_scalar(&Scalar::from_i32(7), 4).call().unwrap();
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
        let col = Col::from_scalar(&Scalar::from_i32(7), 4).call().unwrap();
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
