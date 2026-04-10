#![allow(clippy::wildcard_imports)]
use super::*;

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
    /// Sorts the table by all columns using the given sort directions.
    ///
    /// Each column is sorted according to its corresponding entry in
    /// `orders` (ascending or descending) and `nulls` (nulls placed first
    /// or last). Both slices must have exactly [`columns_len()`](Self::columns_len)
    /// elements, or be empty to use the defaults (ascending, nulls before).
    ///
    /// Returns a new [`Table`] with rows reordered. The original table is
    /// not modified.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::sorting::{Order, NullOrder};
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let col = Column::from_slice_i32(&[3, 1, 2]).call()?;
    /// let table = Table::from_columns(vec![col])?;
    /// let sorted = table.sort(
    ///     &[Order::ASCENDING],
    ///     &[NullOrder::BEFORE],
    /// ).call()?;
    /// assert_eq!(sorted.column(0)?.to_vec_i32().call()?, [1, 2, 3]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `orders` or `nulls` length does not match the
    /// number of columns (unless empty, which uses defaults).
    pub fn sort<'a>(&'a self, orders: &'a [Order], nulls: &'a [NullOrder]) -> Sort<'a, Raw> {
        Sort {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "sort")]
    /// Sorts the table in ascending order on all columns with nulls placed
    /// before non-null values.
    ///
    /// This is a convenience wrapper around [`sort`](Self::sort) with empty
    /// order/null-order slices, which default to ascending and nulls-before.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let col = Column::from_slice_i32(&[3, 1, 2]).call()?;
    /// let table = Table::from_columns(vec![col])?;
    /// let sorted = table.sort_ascending().call()?;
    /// assert_eq!(sorted.column(0)?.to_vec_i32().call()?, [1, 2, 3]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying sort operation fails.
    pub fn sort_ascending(&self) -> SortAscending<'_, Raw> {
        SortAscending {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the row indices that would sort this table.
    ///
    /// The result is an `INT32` [`Column`] of
    /// zero-based row indices. Gathering the table by these indices produces
    /// the same result as [`sort`](Table::sort).
    ///
    /// `orders` and `nulls` follow the same conventions as [`sort`](Table::sort):
    /// one element per column, or empty to use defaults.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let col = Column::from_slice_i32(&[30, 10, 20]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(col);
    /// let table = b.build()?;
    ///
    /// let indices = table.sorted_order(&[], &[]).call()?;
    /// assert_eq!(indices.len(), 3);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn sorted_order<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SortedOrder<'a, Raw> {
        SortedOrder {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Checks whether the table rows are sorted according to the given orders.
    ///
    /// Returns `true` if the rows are in the order specified by `orders` and
    /// `nulls`, `false` otherwise. Empty slices default to ascending order
    /// with nulls after.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::sorting::{Order, NullOrder};
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(col);
    /// let table = b.build()?;
    ///
    /// let sorted = table.is_sorted(
    ///     &[Order::ASCENDING],
    ///     &[NullOrder::BEFORE],
    /// ).call()?;
    /// assert!(sorted);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn is_sorted<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> IsSorted<'a, Raw> {
        IsSorted {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }
}
