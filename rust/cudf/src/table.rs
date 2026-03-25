// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU table types: owning [`Table`] and [`TableBuilder`].
//!
//! A [`Table`] is a collection of [`Column`]s that all share the same row
//! count, analogous to a data frame. It is the primary unit of data for
//! multi-column GPU operations such as joins, sorting, groupby, and
//! partitioning.
//!
//! Tables are constructed using [`TableBuilder`], which collects columns one
//! at a time and validates that they share the same length when
//! [`build`](TableBuilder::build) is called.
//!
//! # Builder pattern
//!
//! All GPU operations on `Table` return lightweight builder structs that
//! implement [`GpuOp`](crate::stream::GpuOp). Call [`.call()`](crate::stream::GpuOp::call)
//! to execute the operation, or chain [`.stream()`](crate::stream::GpuOp::stream)
//! first to run on a non-default CUDA stream.
//!
//! ```ignore
//! use cudf::column::Column;
//! use cudf::stream::GpuOp;
//! use cudf::table::TableBuilder;
//!
//! let col = Column::from_slice_i32(&[10, 20, 30]).call()?;
//! let mut tb = TableBuilder::new();
//! tb.push_column(col);
//! let table = tb.build()?;
//! assert_eq!(table.len(), 3);
//! assert_eq!(table.columns_len(), 1);
//! # Ok::<(), cudf::error::Error>(())
//! ```

use cxx::UniquePtr;

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::groupby::AggregationKind;
use crate::scalar::Scalar;
use crate::sorting::{NullOrder, Order};
use crate::stream::Stream;
use crate::{i32_to_usize, usize_to_i32};

#[doc(alias = "table")]
/// An owning GPU table consisting of zero or more [`Column`]s of equal row
/// count.
///
/// `Table` owns the GPU memory of every column it contains. Dropping the
/// table frees all underlying device allocations.
///
/// Use [`TableBuilder`] or [`Table::from_columns`] to construct a table,
/// and [`Table::column`] or [`Table::columns`] to access individual columns
/// as non-owning [`ColumnView`]s.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::GpuOp;
/// use cudf::table::Table;
///
/// let c1 = Column::from_scalar(&Scalar::from_i32(42), 5).call()?;
/// let c2 = Column::from_scalar(&Scalar::from_f64(3.14), 5).call()?;
/// let table = Table::from_columns(vec![c1, c2])?;
///
/// assert_eq!(table.len(), 5);
/// assert_eq!(table.columns_len(), 2);
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct Table(pub(crate) UniquePtr<cudf_sys::ffi::Table>);

impl Default for Table {
    /// Creates an empty table with zero columns and zero rows.
    ///
    /// Equivalent to `TableBuilder::new().build().unwrap()`.
    fn default() -> Self {
        Self(cudf_sys::ffi::table_empty())
    }
}

// ---------------------------------------------------------------------------
// Builder structs for Table operations
// ---------------------------------------------------------------------------

/// Builder for [`Table::sort`].
///
/// Created by [`Table::sort`]. Call [`.call()`](crate::stream::GpuOp::call)
/// to execute, or chain [`.stream()`](crate::stream::GpuOp::stream) to run
/// on a non-default CUDA stream.
pub struct Sort<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for Sort<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::sort_table(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::sort_ascending`]. See that method for details.
pub struct SortAscending<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for SortAscending<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        self.table.sort(&[], &[]).stream(self.stream).call()
    }
}

/// Builder for [`Table::sorted_order`]. See that method for details.
pub struct SortedOrder<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for SortedOrder<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let col = cudf_sys::sorting::ffi::sorted_order(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`Table::is_sorted`]. See that method for details.
pub struct IsSorted<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for IsSorted<'_> {
    type Output = bool;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        cudf_sys::sorting::ffi::is_sorted_table(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

/// Builder for [`Table::inner_join`]. See that method for details.
pub struct InnerJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for InnerJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::inner_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_join`]. See that method for details.
pub struct LeftJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for LeftJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::full_join`]. See that method for details.
pub struct FullJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for FullJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::full_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_semi_join`]. See that method for details.
pub struct LeftSemiJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for LeftSemiJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_semi_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_anti_join`]. See that method for details.
pub struct LeftAntiJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for LeftAntiJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_anti_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::inner_join_indices`]. See that method for details.
#[doc(alias = "inner_join")]
pub struct InnerJoinIndices<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for InnerJoinIndices<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::inner_join_indices(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_join_indices`]. See that method for details.
#[doc(alias = "left_join")]
pub struct LeftJoinIndices<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for LeftJoinIndices<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_join_indices(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::full_join_indices`]. See that method for details.
#[doc(alias = "full_join")]
pub struct FullJoinIndices<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for FullJoinIndices<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::full_join_indices(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_semi_join_indices`]. See that method for details.
#[doc(alias = "left_semi_join")]
pub struct LeftSemiJoinIndices<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for LeftSemiJoinIndices<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_semi_join_indices(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_anti_join_indices`]. See that method for details.
#[doc(alias = "left_anti_join")]
pub struct LeftAntiJoinIndices<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for LeftAntiJoinIndices<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_anti_join_indices(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::compute_column`]. See that method for details.
pub struct ComputeColumn<'a> {
    table: &'a Table,
    tree: &'a crate::ast::ExpressionTree,
    root: crate::ast::ExprRef,
    stream: Stream,
}

impl crate::stream::GpuOp for ComputeColumn<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ast::ffi::ast_compute_column(
            &self.table.0,
            self.tree.raw(),
            self.root.index(),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::conditional_inner_join`]. See that method for details.
pub struct ConditionalInnerJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    tree: &'a crate::ast::ExpressionTree,
    predicate: crate::ast::ExprRef,
    stream: Stream,
}

impl crate::stream::GpuOp for ConditionalInnerJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_inner_join(
            &self.table.0,
            &self.right.0,
            self.tree.raw(),
            self.predicate.index(),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::conditional_left_join`]. See that method for details.
pub struct ConditionalLeftJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    tree: &'a crate::ast::ExpressionTree,
    predicate: crate::ast::ExprRef,
    stream: Stream,
}

impl crate::stream::GpuOp for ConditionalLeftJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_left_join(
            &self.table.0,
            &self.right.0,
            self.tree.raw(),
            self.predicate.index(),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::conditional_full_join`]. See that method for details.
pub struct ConditionalFullJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    tree: &'a crate::ast::ExpressionTree,
    predicate: crate::ast::ExprRef,
    stream: Stream,
}

impl crate::stream::GpuOp for ConditionalFullJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_full_join(
            &self.table.0,
            &self.right.0,
            self.tree.raw(),
            self.predicate.index(),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::conditional_left_semi_join`]. See that method for details.
pub struct ConditionalLeftSemiJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    tree: &'a crate::ast::ExpressionTree,
    predicate: crate::ast::ExprRef,
    stream: Stream,
}

impl crate::stream::GpuOp for ConditionalLeftSemiJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_left_semi_join(
            &self.table.0,
            &self.right.0,
            self.tree.raw(),
            self.predicate.index(),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::conditional_left_anti_join`]. See that method for details.
pub struct ConditionalLeftAntiJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    tree: &'a crate::ast::ExpressionTree,
    predicate: crate::ast::ExprRef,
    stream: Stream,
}

impl crate::stream::GpuOp for ConditionalLeftAntiJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_left_anti_join(
            &self.table.0,
            &self.right.0,
            self.tree.raw(),
            self.predicate.index(),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::groupby`]. See that method for details.
pub struct Groupby<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_column: i32,
    agg: AggregationKind,
    stream: Stream,
}

impl crate::stream::GpuOp for Groupby<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let result = cudf_sys::groupby::ffi::groupby_single(
            &self.table.0,
            self.key_columns,
            self.value_column,
            self.agg.repr,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::groupby_multi`]. See that method for details.
pub struct GroupbyMulti<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    aggs: &'a [AggregationKind],
    stream: Stream,
}

impl crate::stream::GpuOp for GroupbyMulti<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        if self.value_columns.len() != self.aggs.len() {
            return Err(crate::error::Error::OutOfBounds {
                index: self.aggs.len(),
                len: self.value_columns.len(),
            });
        }
        let agg_kinds = cudf_sys::agg_kinds_as_i32(self.aggs);
        let result = cudf_sys::groupby::ffi::groupby_multi(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            agg_kinds,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::filter`]. See that method for details.
pub struct Filter<'a> {
    table: &'a Table,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Filter<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::apply_boolean_mask(
            &self.table.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::drop_nulls`]. See that method for details.
pub struct DropNulls<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for DropNulls<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nulls_all(&self.table.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::gather`]. See that method for details.
pub struct Gather<'a> {
    table: &'a Table,
    indices: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Gather<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::gather_table(
            &self.table.0,
            self.indices.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::murmur3`]. See that method for details.
pub struct Murmur3<'a> {
    table: &'a Table,
    seed: u32,
    stream: Stream,
}

impl crate::stream::GpuOp for Murmur3<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            Column(cudf_sys::hashing::ffi::hash_murmur3(
                &self.table.0,
                self.seed,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::xxhash64`]. See that method for details.
pub struct Xxhash64<'a> {
    table: &'a Table,
    seed: u64,
    stream: Stream,
}

impl crate::stream::GpuOp for Xxhash64<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            Column(cudf_sys::hashing::ffi::hash_xxhash64(
                &self.table.0,
                self.seed,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::md5`]. See that method for details.
pub struct Md5<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Md5<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            Column(cudf_sys::hashing::ffi::hash_md5(
                &self.table.0,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::sha256`]. See that method for details.
pub struct Sha256<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Sha256<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            Column(cudf_sys::hashing::ffi::hash_sha256(
                &self.table.0,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::merge`]. See that method for details.
pub struct Merge<'a> {
    table: &'a Table,
    right: &'a Table,
    key_columns: &'a [i32],
    orders: &'a [Order],
    null_orders: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for Merge<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.null_orders);
        let tbl = cudf_sys::merge::ffi::merge_tables(
            &self.table.0,
            &self.right.0,
            self.key_columns,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::hash_partition`]. See that method for details.
pub struct HashPartition<'a> {
    table: &'a Table,
    columns: &'a [i32],
    num_partitions: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for HashPartition<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::partitioning::ffi::hash_partition_table(
            &self.table.0,
            self.columns,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::hash_partition_offsets`]. See that method for details.
pub struct HashPartitionOffsets<'a> {
    table: &'a Table,
    columns: &'a [i32],
    num_partitions: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for HashPartitionOffsets<'_> {
    type Output = Vec<usize>;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let offsets = cudf_sys::partitioning::ffi::hash_partition_offsets(
            &self.table.0,
            self.columns,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::round_robin`]. See that method for details.
pub struct RoundRobin<'a> {
    table: &'a Table,
    num_partitions: usize,
    start: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for RoundRobin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::partitioning::ffi::round_robin_partition_table(
            &self.table.0,
            usize_to_i32(self.num_partitions),
            usize_to_i32(self.start),
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::round_robin_offsets`]. See that method for details.
pub struct RoundRobinOffsets<'a> {
    table: &'a Table,
    num_partitions: usize,
    start: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for RoundRobinOffsets<'_> {
    type Output = Vec<usize>;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let offsets = cudf_sys::partitioning::ffi::round_robin_partition_offsets(
            &self.table.0,
            usize_to_i32(self.num_partitions),
            usize_to_i32(self.start),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::interleave_columns`]. See that method for details.
pub struct InterleaveColumns<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for InterleaveColumns<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::reshape::ffi::interleave_columns(&self.table.0, self.stream.as_raw())?;
        Ok(Column(col))
    }
}

/// Builder for [`Table::tile`]. See that method for details.
pub struct Tile<'a> {
    table: &'a Table,
    count: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Tile<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::reshape::ffi::tile_table(
            &self.table.0,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::repeat`]. See that method for details.
pub struct Repeat<'a> {
    table: &'a Table,
    count: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Repeat<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::filling::ffi::repeat_table(
            &self.table.0,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::encode`]. See that method for details.
pub struct Encode<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Encode<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::transform::ffi::encode_table(&self.table.0, self.stream.as_raw())?;
        Ok(Column(col))
    }
}

/// Builder for [`Table::encode_keys`]. See that method for details.
pub struct EncodeKeys<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for EncodeKeys<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::transform::ffi::encode_keys(&self.table.0, self.stream.as_raw())?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::drop_nans`]. See that method for details.
pub struct DropNans<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for DropNans<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t =
            cudf_sys::compaction::ffi::drop_nans(&self.table.0, self.keys, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::drop_nulls_with_threshold`]. See that method for details.
pub struct DropNullsWithThreshold<'a> {
    table: &'a Table,
    keys: &'a [i32],
    threshold: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for DropNullsWithThreshold<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nulls_with_threshold(
            &self.table.0,
            self.keys,
            usize_to_i32(self.threshold),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::unique`]. See that method for details.
pub struct Unique<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for Unique<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::unique_table(
            &self.table.0,
            self.keys,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::distinct`]. See that method for details.
pub struct Distinct<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for Distinct<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::distinct_table(
            &self.table.0,
            self.keys,
            0,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::stable_distinct`]. See that method for details.
pub struct StableDistinct<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for StableDistinct<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::stable_distinct_table(
            &self.table.0,
            self.keys,
            0,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::scatter`]. See that method for details.
pub struct Scatter<'a> {
    table: &'a Table,
    source: &'a Table,
    map: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Scatter<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::scatter_table(
            &self.source.0,
            self.map.0,
            &self.table.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::reverse`]. See that method for details.
pub struct Reverse<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Reverse<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::reverse_table(&self.table.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::slice`]. See that method for details.
pub struct Slice<'a> {
    table: &'a Table,
    begin: usize,
    end: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Slice<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::slice_table(
            &self.table.0,
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::sample`]. See that method for details.
pub struct Sample<'a> {
    table: &'a Table,
    n: usize,
    with_replacement: bool,
    seed: i64,
    stream: Stream,
}

impl crate::stream::GpuOp for Sample<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::sample_table(
            &self.table.0,
            usize_to_i32(self.n),
            self.with_replacement,
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::transpose`]. See that method for details.
pub struct Transpose<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Transpose<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::reshape::ffi::transpose_table(&self.table.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::lower_bound`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an `INT32`
/// [`Column`] of insertion indices.
pub struct LowerBound<'a> {
    table: &'a Table,
    needles: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for LowerBound<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::search::ffi::lower_bound(
            &self.table.0,
            &self.needles.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::upper_bound`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an `INT32`
/// [`Column`] of insertion indices.
pub struct UpperBound<'a> {
    table: &'a Table,
    needles: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for UpperBound<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::search::ffi::upper_bound(
            &self.table.0,
            &self.needles.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::explode`]. See that method for details.
pub struct Explode<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Explode<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::explode_position`]. See that method for details.
pub struct ExplodePosition<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for ExplodePosition<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_position_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::explode_outer`]. See that method for details.
pub struct ExplodeOuter<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for ExplodeOuter<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_outer_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::explode_outer_position`]. See that method for details.
pub struct ExplodeOuterPosition<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for ExplodeOuterPosition<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_outer_position_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::gather_checked`]. See that method for details.
pub struct GatherChecked<'a> {
    table: &'a Table,
    indices: &'a ColumnView<'a>,
    nullify_oob: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for GatherChecked<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::gather_table_checked(
            &self.table.0,
            self.indices.0,
            self.nullify_oob,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

#[doc(alias = "negative_index_policy")]
/// Builder for [`Table::gather_with_policy`]. See that method for details.
pub struct GatherWithPolicy<'a> {
    table: &'a Table,
    indices: &'a ColumnView<'a>,
    nullify_oob: bool,
    allow_negative: bool,
    stream: Stream,
}

impl GatherWithPolicy<'_> {
    /// If `true`, out-of-bounds indices produce null rows in the output.
    /// If `false` (the default), behavior is undefined for out-of-bounds indices.
    pub fn nullify_oob(mut self, nullify: bool) -> Self {
        self.nullify_oob = nullify;
        self
    }

    /// If `true` (the default), negative indices wrap around (`i + n`).
    /// If `false`, negative indices are undefined behavior.
    pub fn allow_negative(mut self, allow: bool) -> Self {
        self.allow_negative = allow;
        self
    }
}

impl crate::stream::GpuOp for GatherWithPolicy<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::gather_table_with_policy(
            &self.table.0,
            self.indices.0,
            self.nullify_oob,
            self.allow_negative,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::cross_join`]. See that method for details.
pub struct CrossJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for CrossJoin<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t =
            cudf_sys::join::ffi::cross_join(&self.table.0, &self.right.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::partition_by_map`]. See that method for details.
pub struct PartitionByMap<'a> {
    table: &'a Table,
    partition_map: &'a ColumnView<'a>,
    num_partitions: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for PartitionByMap<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::partitioning::ffi::partition_by_map(
            &self.table.0,
            self.partition_map.0,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::partition_by_map_offsets`]. See that method for details.
pub struct PartitionByMapOffsets<'a> {
    table: &'a Table,
    partition_map: &'a ColumnView<'a>,
    num_partitions: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for PartitionByMapOffsets<'_> {
    type Output = Vec<usize>;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let offsets = cudf_sys::partitioning::ffi::partition_by_map_offsets(
            &self.table.0,
            self.partition_map.0,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::groupby_scan`]. See that method for details.
pub struct GroupbyScan<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    aggs: &'a [AggregationKind],
    stream: Stream,
}

impl crate::stream::GpuOp for GroupbyScan<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        if self.value_columns.len() != self.aggs.len() {
            return Err(crate::error::Error::OutOfBounds {
                index: self.aggs.len(),
                len: self.value_columns.len(),
            });
        }
        let agg_kinds = cudf_sys::agg_kinds_as_i32(self.aggs);
        let result = cudf_sys::groupby::ffi::groupby_scan(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            agg_kinds,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::groupby_shift`]. See that method for details.
pub struct GroupbyShift<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    offsets: &'a [i32],
    fill_values: &'a [Scalar],
    stream: Stream,
}

impl crate::stream::GpuOp for GroupbyShift<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.fill_values {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let result = cudf_sys::groupby::ffi::groupby_shift(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            self.offsets,
            list.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::groupby_replace_nulls`]. See that method for details.
pub struct GroupbyReplaceNulls<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    policies: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for GroupbyReplaceNulls<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let result = cudf_sys::groupby::ffi::groupby_replace_nulls(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            self.policies,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::sha1`]. See that method for details.
pub struct Sha1<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Sha1<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha1(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::sha224`]. See that method for details.
pub struct Sha224<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Sha224<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha224(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::sha384`]. See that method for details.
pub struct Sha384<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Sha384<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha384(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::sha512`]. See that method for details.
pub struct Sha512<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Sha512<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha512(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::murmurhash3_x64_128`]. See that method for details.
pub struct MurmurHash3X64_128<'a> {
    table: &'a Table,
    seed: u64,
    stream: Stream,
}

impl crate::stream::GpuOp for MurmurHash3X64_128<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::hashing::ffi::hash_murmurhash3_x64_128(
            &self.table.0,
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::xxhash_32`]. See that method for details.
pub struct XxHash32<'a> {
    table: &'a Table,
    seed: u32,
    stream: Stream,
}

impl crate::stream::GpuOp for XxHash32<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::hashing::ffi::hash_xxhash_32(&self.table.0, self.seed, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::row_bit_count`]. See that method for details.
pub struct RowBitCount<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for RowBitCount<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::transform::ffi::row_bit_count(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::segmented_row_bit_count`]. See that method for details.
pub struct SegmentedRowBitCount<'a> {
    table: &'a Table,
    segment_length: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedRowBitCount<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::transform::ffi::segmented_row_bit_count(
            &self.table.0,
            self.segment_length,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::byte_cast`]. See that method for details.
pub struct ByteCast<'a> {
    col: &'a ColumnView<'a>,
    flip_endian: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ByteCast<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::reshape::ffi::byte_cast_column(
            self.col.0,
            self.flip_endian,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::stable_sorted_order`]. See that method for details.
pub struct StableSortedOrder<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for StableSortedOrder<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::stable_sorted_order(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::stable_sort`]. See that method for details.
pub struct StableSort<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for StableSort<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_sort_table(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::sort_by_key`]. See that method for details.
pub struct SortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for SortByKey<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::sort_by_key(
            &self.table.0,
            &self.keys.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::stable_sort_by_key`]. See that method for details.
pub struct StableSortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for StableSortByKey<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_sort_by_key(
            &self.table.0,
            &self.keys.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::segmented_sorted_order`]. See that method for details.
pub struct SegmentedSortedOrder<'a> {
    table: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedSortedOrder<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::segmented_sorted_order(
            &self.table.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::stable_segmented_sorted_order`]. See that method for details.
pub struct StableSegmentedSortedOrder<'a> {
    table: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for StableSegmentedSortedOrder<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::stable_segmented_sorted_order(
            &self.table.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::segmented_sort_by_key`]. See that method for details.
pub struct SegmentedSortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedSortByKey<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::segmented_sort_by_key(
            &self.table.0,
            &self.keys.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::stable_segmented_sort_by_key`]. See that method for details.
pub struct StableSegmentedSortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for StableSegmentedSortByKey<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_segmented_sort_by_key(
            &self.table.0,
            &self.keys.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::quantiles`]. See that method for details.
pub struct Quantiles<'a> {
    table: &'a Table,
    q: &'a [f64],
    interp: crate::quantile::Interpolation,
    is_sorted: bool,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl crate::stream::GpuOp for Quantiles<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::quantile::ffi::quantiles_table(
            &self.table.0,
            self.q,
            self.interp.repr,
            self.is_sorted,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::boolean_mask_scatter`]. See that method for details.
pub struct BooleanMaskScatter<'a> {
    table: &'a Table,
    source: &'a Table,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for BooleanMaskScatter<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::boolean_mask_scatter_table(
            &self.source.0,
            &self.table.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::scatter_scalars`]. See that method for details.
pub struct ScatterScalars<'a> {
    table: &'a Table,
    scalars: &'a [Scalar],
    map: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ScatterScalars<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.scalars {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::copying::ffi::scatter_scalars(
            list.pin_mut(),
            self.map.0,
            &self.table.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::boolean_mask_scatter_scalars`]. See that method for details.
pub struct BooleanMaskScatterScalars<'a> {
    table: &'a Table,
    scalars: &'a [Scalar],
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for BooleanMaskScatterScalars<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.scalars {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::copying::ffi::boolean_mask_scatter_scalars(
            list.pin_mut(),
            &self.table.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::repeat_by_column`]. See that method for details.
pub struct RepeatByColumn<'a> {
    table: &'a Table,
    counts: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for RepeatByColumn<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::filling::ffi::repeat_table_column(
            &self.table.0,
            self.counts.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::distinct_count`]. See that method for details.
pub struct DistinctCount<'a> {
    table: &'a Table,
    nulls_equal: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for DistinctCount<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            let ne = i32::from(!self.nulls_equal); // EQUAL=0, UNEQUAL=1
            i32_to_usize(cudf_sys::compaction::ffi::distinct_count_table(
                &self.table.0,
                ne,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::bitmask_and_to_bools`]. See that method for details.
pub struct BitmaskAndToBools<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for BitmaskAndToBools<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::bitmask_and_to_bools(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::bitmask_or_to_bools`]. See that method for details.
pub struct BitmaskOrToBools<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for BitmaskOrToBools<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::bitmask_or_to_bools(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::unique_count`]. See that method for details.
pub struct UniqueCount<'a> {
    table: &'a Table,
    nulls_equal: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for UniqueCount<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            let ne = i32::from(!self.nulls_equal);
            i32_to_usize(cudf_sys::compaction::ffi::unique_count_table(
                &self.table.0,
                ne,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::drop_nans_with_threshold`]. See that method for details.
pub struct DropNansWithThreshold<'a> {
    table: &'a Table,
    keys: &'a [i32],
    threshold: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for DropNansWithThreshold<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nans_with_threshold(
            &self.table.0,
            self.keys,
            usize_to_i32(self.threshold),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::approx_distinct_count`]. See that method for details.
pub struct ApproxDistinctCount<'a> {
    table: &'a Table,
    precision: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for ApproxDistinctCount<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            cudf_sys::compaction::ffi::approx_distinct_count(
                &self.table.0,
                self.precision,
                self.stream.as_raw(),
            )
        })
    }
}

/// Builder for [`Table::from_dlpack`]. See that method for details.
pub struct FromDlpack {
    managed_tensor_ptr: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for FromDlpack {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::io::ffi::from_dlpack(self.managed_tensor_ptr, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::to_dlpack`]. See that method for details.
pub struct ToDlpack<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for ToDlpack<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok(cudf_sys::io::ffi::to_dlpack(
            &self.table.0,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::concatenate_columns`]. See that method for details.
pub struct ConcatenateTableColumns<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for ConcatenateTableColumns<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::concatenate::ffi::concatenate_columns(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::concatenate`]. See that method for details.
pub struct ConcatenateWith<'a> {
    table: &'a Table,
    other: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for ConcatenateWith<'_> {
    type Output = Table;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::concatenate::ffi::concatenate_tables(
            &self.table.0,
            &self.other.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

// ---------------------------------------------------------------------------
// impl Table
// ---------------------------------------------------------------------------

impl Table {
    /// Creates a table from a vector of columns.
    ///
    /// All columns must have the same row count. This is a convenience
    /// constructor that builds a [`TableBuilder`] internally. Ownership of
    /// each [`Column`] is transferred into the new table.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let a = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let b = Column::from_slice_i32(&[4, 5, 6]).call()?;
    /// let table = Table::from_columns(vec![a, b])?;
    /// assert_eq!(table.columns_len(), 2);
    /// assert_eq!(table.len(), 3);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have mismatched row counts.
    pub fn from_columns(columns: Vec<Column>) -> Result<Self> {
        let mut builder = TableBuilder::new();
        for col in columns {
            builder.push_column(col);
        }
        builder.build()
    }

    #[doc(alias = "num_columns")]
    /// Returns the number of columns in this table.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::table::Table;
    ///
    /// let table = Table::default();
    /// assert_eq!(table.columns_len(), 0);
    /// ```
    pub fn columns_len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::table_num_columns(&self.0))
    }

    #[doc(alias = "num_rows")]
    /// Returns the number of rows in this table.
    ///
    /// All columns in a table share the same row count.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let table = Table::from_columns(vec![col])?;
    /// assert_eq!(table.len(), 3);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::table_num_rows(&self.0))
    }

    /// Returns `true` if the table has zero columns.
    ///
    /// Note that a table can have zero columns but still report zero rows
    /// (from [`Default::default`]). A table with columns that each have zero
    /// rows is *not* empty by this definition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::table::Table;
    ///
    /// assert!(Table::default().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.columns_len() == 0
    }

    /// Returns the total GPU memory allocation size in bytes for all columns
    /// in this table, including data buffers, null masks, and child columns.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let table = Table::from_columns(vec![col])?;
    /// assert!(table.alloc_bytes() > 0);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn alloc_bytes(&self) -> usize {
        cudf_sys::ffi::table_alloc_size(&self.0)
    }

    /// Returns an immutable view of the column at `index` (zero-based).
    ///
    /// The returned [`ColumnView`] borrows from this table and cannot
    /// outlive it.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let col = Column::from_slice_i32(&[10, 20]).call()?;
    /// let table = Table::from_columns(vec![col])?;
    /// let view = table.column(0)?;
    /// assert_eq!(view.len(), 2);
    /// assert_eq!(view.type_id(), TypeId::INT32);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfBounds`](crate::error::Error::OutOfBounds) if
    /// `index >= self.columns_len()`.
    pub fn column(&self, index: usize) -> Result<ColumnView<'_>> {
        if index >= self.columns_len() {
            return Err(crate::error::Error::OutOfBounds {
                index,
                len: self.columns_len(),
            });
        }
        let view = cudf_sys::ffi::table_get_column_view(&self.0, usize_to_i32(index))?;
        Ok(ColumnView(view))
    }

    /// Returns an iterator over all columns as [`ColumnView`]s.
    ///
    /// The iterator yields views in column-index order (0, 1, 2, ...) and
    /// implements [`ExactSizeIterator`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let a = Column::from_slice_i32(&[1, 2]).call()?;
    /// let b = Column::from_slice_i32(&[3, 4]).call()?;
    /// let table = Table::from_columns(vec![a, b])?;
    /// assert_eq!(table.columns().len(), 2);
    /// for col in table.columns() {
    ///     assert_eq!(col.len(), 2);
    /// }
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn columns(&self) -> Columns<'_> {
        Columns {
            table: self,
            index: 0,
            len: self.columns_len(),
        }
    }

    // -- Sorting --

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
    /// ```ignore
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
    pub fn sort<'a>(&'a self, orders: &'a [Order], nulls: &'a [NullOrder]) -> Sort<'a> {
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
    /// ```ignore
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
    pub fn sort_ascending(&self) -> SortAscending<'_> {
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
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::sorting::{Order, NullOrder};
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
    ) -> SortedOrder<'a> {
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
    /// ```ignore
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
    pub fn is_sorted<'a>(&'a self, orders: &'a [Order], nulls: &'a [NullOrder]) -> IsSorted<'a> {
        IsSorted {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    // -- Joins --

    /// Performs an inner join with `right` on the specified key columns.
    ///
    /// Matches rows where `self[left_on[i]] == right[right_on[i]]` for all
    /// key pairs. The output table contains all columns from `self` followed
    /// by all columns from `right`, with only the matching rows included.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is a
    /// zero-based column index. The key columns at matching positions must
    /// have compatible types.
    ///
    /// When duplicate keys exist, the output contains the Cartesian product
    /// of matching rows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// // Left: id=[1,2,3], value=[10,20,30]
    /// let id_l = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let val = Column::from_slice_i32(&[10, 20, 30]).call()?;
    /// let mut lb = TableBuilder::new();
    /// lb.push_column(id_l);
    /// lb.push_column(val);
    /// let left = lb.build()?;
    ///
    /// // Right: id=[2,3,4], label=[200,300,400]
    /// let id_r = Column::from_slice_i32(&[2, 3, 4]).call()?;
    /// let label = Column::from_slice_i32(&[200, 300, 400]).call()?;
    /// let mut rb = TableBuilder::new();
    /// rb.push_column(id_r);
    /// rb.push_column(label);
    /// let right = rb.build()?;
    ///
    /// let joined = left.inner_join(&right, &[0], &[0]).call()?;
    /// assert_eq!(joined.len(), 2);          // rows with id=2 and id=3
    /// assert_eq!(joined.columns_len(), 4);  // left(2) + right(2)
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn inner_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> InnerJoin<'a> {
        InnerJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a left join with `right` on the specified key columns.
    ///
    /// All rows from `self` (the left table) are preserved. For each left
    /// row, matching right rows are appended; where no match exists the
    /// right columns are filled with nulls.
    ///
    /// The output table contains all columns from `self` followed by all
    /// columns from `right`. The row count is at least `self.len()`.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is a
    /// zero-based column index with compatible types at matching positions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let left = {
    ///     let k = Column::from_slice_i32(&[1, 2, 3]).call()?;
    ///     let v = Column::from_slice_i32(&[10, 20, 30]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    /// let right = {
    ///     let k = Column::from_slice_i32(&[2, 3, 4]).call()?;
    ///     let v = Column::from_slice_i32(&[200, 300, 400]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    ///
    /// let joined = left.left_join(&right, &[0], &[0]).call()?;
    /// assert_eq!(joined.len(), 3);          // all 3 left rows kept
    /// assert_eq!(joined.columns_len(), 4);  // left(2) + right(2)
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn left_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftJoin<'a> {
        LeftJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a full outer join with `right` on the specified key columns.
    ///
    /// All rows from both tables are preserved. Where a row from one side
    /// has no match in the other, the missing side's columns are filled with
    /// nulls.
    ///
    /// The output table contains all columns from `self` followed by all
    /// columns from `right`.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is a
    /// zero-based column index with compatible types at matching positions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let left = {
    ///     let k = Column::from_slice_i32(&[1, 2]).call()?;
    ///     let v = Column::from_slice_i32(&[10, 20]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    /// let right = {
    ///     let k = Column::from_slice_i32(&[2, 3]).call()?;
    ///     let v = Column::from_slice_i32(&[200, 300]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    ///
    /// let joined = left.full_join(&right, &[0], &[0]).call()?;
    /// assert_eq!(joined.len(), 3);          // id=1 + id=2 + id=3
    /// assert_eq!(joined.columns_len(), 4);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn full_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> FullJoin<'a> {
        FullJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a left semi join with `right` on the specified key columns.
    ///
    /// Returns only the columns from `self` (the left table) for rows that
    /// have at least one matching row in `right`. Unlike an inner join, no
    /// columns from `right` appear in the output and duplicate matches do
    /// not produce extra rows.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is a
    /// zero-based column index with compatible types at matching positions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let left = {
    ///     let k = Column::from_slice_i32(&[1, 2, 3]).call()?;
    ///     let v = Column::from_slice_i32(&[10, 20, 30]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    /// let right = {
    ///     let k = Column::from_slice_i32(&[2, 3, 4]).call()?;
    ///     let v = Column::from_slice_i32(&[200, 300, 400]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    ///
    /// let result = left.left_semi_join(&right, &[0], &[0]).call()?;
    /// assert_eq!(result.len(), 2);          // rows with id=2 and id=3
    /// assert_eq!(result.columns_len(), 2);  // left columns only
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn left_semi_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftSemiJoin<'a> {
        LeftSemiJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a left anti join with `right` on the specified key columns.
    ///
    /// Returns only the columns from `self` (the left table) for rows that
    /// do **not** have any matching row in `right`. This is the complement
    /// of [`left_semi_join`](Table::left_semi_join).
    ///
    /// `left_on` and `right_on` must have the same length. Each element is a
    /// zero-based column index with compatible types at matching positions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let left = {
    ///     let k = Column::from_slice_i32(&[1, 2, 3]).call()?;
    ///     let v = Column::from_slice_i32(&[10, 20, 30]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    /// let right = {
    ///     let k = Column::from_slice_i32(&[2, 3, 4]).call()?;
    ///     let v = Column::from_slice_i32(&[200, 300, 400]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.push_column(v);
    ///     b.build()?
    /// };
    ///
    /// let result = left.left_anti_join(&right, &[0], &[0]).call()?;
    /// assert_eq!(result.len(), 1);          // only row with id=1
    /// assert_eq!(result.columns_len(), 2);  // left columns only
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn left_anti_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftAntiJoin<'a> {
        LeftAntiJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    // -- Index-only join variants --

    /// Returns the raw gather map indices for an inner join without
    /// materializing the gathered output rows.
    ///
    /// The result is a [`Table`] with two `INT32` columns: the first
    /// contains left-table row indices and the second contains
    /// right-table row indices. Callers can use
    /// [`Table::gather`](Table::gather) or
    /// [`Table::gather_with_policy`](Table::gather_with_policy) to
    /// materialize only the columns they need.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is
    /// a zero-based column index with compatible types at matching
    /// positions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let left = {
    ///     let k = Column::from_slice_i32(&[1, 2, 3]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.build()?
    /// };
    /// let right = {
    ///     let k = Column::from_slice_i32(&[2, 3, 4]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(k);
    ///     b.build()?
    /// };
    ///
    /// let indices = left.inner_join_indices(&right, &[0], &[0]).call()?;
    /// assert_eq!(indices.columns_len(), 2); // left indices + right indices
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    #[doc(alias = "inner_join")]
    pub fn inner_join_indices<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> InnerJoinIndices<'a> {
        InnerJoinIndices {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the raw gather map indices for a left join without
    /// materializing the gathered output rows.
    ///
    /// The result is a [`Table`] with two `INT32` columns: the first
    /// contains left-table row indices and the second contains
    /// right-table row indices. Right indices are `-1` for left rows
    /// with no match.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is
    /// a zero-based column index with compatible types at matching
    /// positions.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    #[doc(alias = "left_join")]
    pub fn left_join_indices<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftJoinIndices<'a> {
        LeftJoinIndices {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the raw gather map indices for a full outer join without
    /// materializing the gathered output rows.
    ///
    /// The result is a [`Table`] with two `INT32` columns: the first
    /// contains left-table row indices and the second contains
    /// right-table row indices. Indices are `-1` for rows with no
    /// match on that side.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is
    /// a zero-based column index with compatible types at matching
    /// positions.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    #[doc(alias = "full_join")]
    pub fn full_join_indices<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> FullJoinIndices<'a> {
        FullJoinIndices {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the raw gather map indices for a left semi join without
    /// materializing the gathered output rows.
    ///
    /// The result is a [`Table`] with a single `INT32` column containing
    /// left-table row indices for rows that have at least one match in
    /// `right`.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is
    /// a zero-based column index with compatible types at matching
    /// positions.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    #[doc(alias = "left_semi_join")]
    pub fn left_semi_join_indices<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftSemiJoinIndices<'a> {
        LeftSemiJoinIndices {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the raw gather map indices for a left anti join without
    /// materializing the gathered output rows.
    ///
    /// The result is a [`Table`] with a single `INT32` column containing
    /// left-table row indices for rows that have **no** match in `right`.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is
    /// a zero-based column index with compatible types at matching
    /// positions.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    #[doc(alias = "left_anti_join")]
    pub fn left_anti_join_indices<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftAntiJoinIndices<'a> {
        LeftAntiJoinIndices {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    // -- AST / Computed columns --

    /// Computes a new column by evaluating an AST expression tree on this
    /// table.
    ///
    /// The expression tree may reference columns of this table by index
    /// using [`ExpressionTree::col`](crate::ast::ExpressionTree::col). The
    /// `root` argument specifies which node in the tree is the root of the
    /// expression to evaluate.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::ast::ExpressionTree;
    /// use cudf::stream::GpuOp;
    ///
    /// let mut tree = ExpressionTree::new();
    /// let a = tree.col(0);
    /// let b = tree.col(1);
    /// let sum = tree.add(a, b);
    ///
    /// let result = table.compute_column(&tree, sum).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds, if the
    /// expression is invalid, or if a GPU error occurs.
    pub fn compute_column<'a>(
        &'a self,
        tree: &'a crate::ast::ExpressionTree,
        root: crate::ast::ExprRef,
    ) -> ComputeColumn<'a> {
        ComputeColumn {
            table: self,
            tree,
            root,
            stream: Stream::default_stream(),
        }
    }

    // -- Conditional joins --

    /// Performs a conditional inner join with `right` using an AST
    /// predicate.
    ///
    /// Unlike equality-based joins, conditional joins evaluate an arbitrary
    /// boolean expression per row-pair. Only row-pairs where the predicate
    /// evaluates to `true` appear in the output.
    ///
    /// The expression tree should use
    /// [`col_in`](crate::ast::ExpressionTree::col_in) with
    /// [`TableSide::Left`](crate::ast::TableSide::Left) and
    /// [`TableSide::Right`](crate::ast::TableSide::Right) to reference
    /// columns from the two tables.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::ast::{ExpressionTree, TableSide};
    /// use cudf::stream::GpuOp;
    ///
    /// let mut tree = ExpressionTree::new();
    /// let lc = tree.col_in(0, TableSide::Left);
    /// let rc = tree.col_in(0, TableSide::Right);
    /// let pred = tree.eq(lc, rc);
    ///
    /// let result = left.conditional_inner_join(&right, &tree, pred).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or a GPU error occurs.
    pub fn conditional_inner_join<'a>(
        &'a self,
        right: &'a Table,
        tree: &'a crate::ast::ExpressionTree,
        predicate: crate::ast::ExprRef,
    ) -> ConditionalInnerJoin<'a> {
        ConditionalInnerJoin {
            table: self,
            right,
            tree,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a conditional left join with `right` using an AST predicate.
    ///
    /// All rows from `self` (the left table) are preserved. For each left
    /// row, matching right rows (where the predicate is `true`) are
    /// appended; where no match exists the right columns are filled with
    /// nulls.
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or a GPU error occurs.
    pub fn conditional_left_join<'a>(
        &'a self,
        right: &'a Table,
        tree: &'a crate::ast::ExpressionTree,
        predicate: crate::ast::ExprRef,
    ) -> ConditionalLeftJoin<'a> {
        ConditionalLeftJoin {
            table: self,
            right,
            tree,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a conditional full outer join with `right` using an AST
    /// predicate.
    ///
    /// All rows from both tables are preserved. Where a row from one side
    /// has no match (predicate is `false` for all pairings), the other
    /// side's columns are filled with nulls.
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or a GPU error occurs.
    pub fn conditional_full_join<'a>(
        &'a self,
        right: &'a Table,
        tree: &'a crate::ast::ExpressionTree,
        predicate: crate::ast::ExprRef,
    ) -> ConditionalFullJoin<'a> {
        ConditionalFullJoin {
            table: self,
            right,
            tree,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a conditional left semi join with `right` using an AST
    /// predicate.
    ///
    /// Returns rows from `self` (the left table) that have at least one
    /// matching row in `right` where the predicate is `true`. The output
    /// contains only the left table's columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or a GPU error occurs.
    pub fn conditional_left_semi_join<'a>(
        &'a self,
        right: &'a Table,
        tree: &'a crate::ast::ExpressionTree,
        predicate: crate::ast::ExprRef,
    ) -> ConditionalLeftSemiJoin<'a> {
        ConditionalLeftSemiJoin {
            table: self,
            right,
            tree,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a conditional left anti join with `right` using an AST
    /// predicate.
    ///
    /// Returns rows from `self` (the left table) that have NO matching row
    /// in `right` where the predicate is `true`. The output contains only
    /// the left table's columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or a GPU error occurs.
    pub fn conditional_left_anti_join<'a>(
        &'a self,
        right: &'a Table,
        tree: &'a crate::ast::ExpressionTree,
        predicate: crate::ast::ExprRef,
    ) -> ConditionalLeftAntiJoin<'a> {
        ConditionalLeftAntiJoin {
            table: self,
            right,
            tree,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    // -- GroupBy --

    /// Groups the table by key columns and applies a single aggregation.
    ///
    /// Partitions the table into groups defined by the columns at indices
    /// `key_columns`, then applies `agg` to the column at `value_column`
    /// within each group.
    ///
    /// The output table contains the unique key column values followed by a
    /// single aggregated result column. The output row count equals the
    /// number of distinct key combinations. Output row order is **not**
    /// guaranteed.
    ///
    /// - `key_columns` -- zero-based column indices that define the groups.
    /// - `value_column` -- zero-based column index to aggregate.
    /// - `agg` -- the [`AggregationKind`] to apply (e.g. `SUM`, `MIN`,
    ///   `MAX`, `MEAN`, `COUNT`).
    ///
    /// The output type of the aggregated column depends on `agg`. For
    /// example, `SUM` on `INT32` produces `INT64`, and `MEAN` always
    /// produces `FLOAT64`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::groupby::AggregationKind;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let keys = Column::from_slice_i32(&[1, 1, 2, 2]).call()?;
    /// let vals = Column::from_slice_i32(&[10, 20, 30, 40]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(keys);
    /// b.push_column(vals);
    /// let table = b.build()?;
    ///
    /// let result = table.groupby(&[0], 1, AggregationKind::SUM).call()?;
    /// assert_eq!(result.columns_len(), 2); // key + aggregated value
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn groupby<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_column: i32,
        agg: AggregationKind,
    ) -> Groupby<'a> {
        Groupby {
            table: self,
            key_columns,
            value_column,
            agg,
            stream: Stream::default_stream(),
        }
    }

    /// Groups the table by key columns and applies multiple aggregations.
    ///
    /// Like [`groupby`](Table::groupby), but applies a different (or the
    /// same) aggregation to each of several value columns in a single pass.
    ///
    /// `value_columns` and `aggs` must have the same length. Each
    /// `aggs[i]` is applied to the column at `value_columns[i]`. A value
    /// column may appear more than once to compute multiple aggregations on
    /// the same column.
    ///
    /// The output table contains the key columns followed by one result
    /// column per aggregation, in the order given.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::groupby::AggregationKind;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let keys = Column::from_slice_i32(&[1, 1, 2, 2]).call()?;
    /// let vals = Column::from_slice_i32(&[10, 20, 30, 40]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(keys);
    /// b.push_column(vals);
    /// let table = b.build()?;
    ///
    /// // SUM and MIN of the same value column
    /// let result = table.groupby_multi(
    ///     &[0],
    ///     &[1, 1],
    ///     &[AggregationKind::SUM, AggregationKind::MIN],
    /// ).call()?;
    /// assert_eq!(result.columns_len(), 3); // key + sum + min
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `value_columns` and `aggs` differ in length,
    /// column indices are out of bounds, or a GPU error occurs.
    pub fn groupby_multi<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        aggs: &'a [AggregationKind],
    ) -> GroupbyMulti<'a> {
        GroupbyMulti {
            table: self,
            key_columns,
            value_columns,
            aggs,
            stream: Stream::default_stream(),
        }
    }

    // -- Filter --

    /// Filters the table by a boolean mask column.
    ///
    /// Returns a new table containing only the rows where `mask` is `true`.
    /// Null entries in `mask` are treated as `false` (those rows are excluded).
    ///
    /// # Arguments
    ///
    /// * `mask` -- A `BOOL8` column with the same number of rows as this table.
    ///
    /// # Returns
    ///
    /// A new [`Table`] with the filtered rows.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` is not a boolean column or has a mismatched
    /// length.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::stream::GpuOp;
    ///
    /// let mask = Column::from_slice_bool(&[true, false, true]).call()?;
    /// let result = table.filter(&mask.view()).call()?;
    /// // Only rows 0 and 2 are retained
    /// ```
    pub fn filter<'a>(&'a self, mask: &'a ColumnView<'a>) -> Filter<'a> {
        Filter {
            table: self,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Drops rows where **all** columns are null.
    ///
    /// A row is removed only if every column in that row contains a null value.
    /// If any column in a row is non-null, the row is kept.
    ///
    /// # Returns
    ///
    /// A new [`Table`] with null-only rows removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn drop_nulls(&self) -> DropNulls<'_> {
        DropNulls {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying --

    /// Gathers (selects) rows by index.
    ///
    /// Creates a new table by selecting rows from `self` at the positions
    /// specified in `indices`. Negative indices are not supported; use
    /// [`gather_checked`](Table::gather_checked) for out-of-bounds handling.
    ///
    /// # Arguments
    ///
    /// * `indices` -- An `INT32` column of row indices to select. Each value
    ///   must be in `[0, self.len())`. Duplicate indices are allowed (rows
    ///   can be repeated).
    ///
    /// # Returns
    ///
    /// A new [`Table`] whose *i*-th row is `self[indices[i]]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails. Behavior is undefined for
    /// out-of-bounds indices; use [`gather_checked`](Table::gather_checked)
    /// for safe handling.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::stream::GpuOp;
    ///
    /// let idx = Column::from_slice_i32(&[2, 0]).call()?;
    /// let result = table.gather(&idx.view()).call()?;
    /// // result has 2 rows: row 2 and row 0 of the original table
    /// ```
    pub fn gather<'a>(&'a self, indices: &'a ColumnView<'a>) -> Gather<'a> {
        Gather {
            table: self,
            indices,
            stream: Stream::default_stream(),
        }
    }

    /// Creates an empty table with the same column types but zero rows.
    ///
    /// The returned table has the same number of columns and same
    /// [`TypeId`](crate::data_type::TypeId) per column as `self`, but
    /// contains no data.
    pub fn empty_like(&self) -> Table {
        Table(cudf_sys::copying::ffi::empty_like_table(&self.0))
    }

    // -- Hashing --

    /// Computes a `MurmurHash3` 32-bit hash of each row.
    ///
    /// Returns a `UINT32` column with one hash value per row, computed across
    /// all columns. Two rows with the same values in all columns produce the
    /// same hash (given the same `seed`). The hash is deterministic and
    /// suitable for partitioning and hash joins.
    ///
    /// # Arguments
    ///
    /// * `seed` -- Initial seed for the hash function. Use `0` for no seeding.
    ///
    /// # Returns
    ///
    /// A [`Column`] of type `UINT32` with `self.len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::stream::GpuOp;
    ///
    /// let hashes = table.murmur3(0).call()?;
    /// assert_eq!(hashes.len(), table.len());
    /// ```
    pub fn murmur3(&self, seed: u32) -> Murmur3<'_> {
        Murmur3 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// Computes an `XXHash64` hash of each row.
    ///
    /// Returns a `UINT64` column with one hash value per row, computed across
    /// all columns. Deterministic given the same `seed`.
    ///
    /// # Arguments
    ///
    /// * `seed` -- Initial seed for the hash function. Use `0` for no seeding.
    ///
    /// # Returns
    ///
    /// A [`Column`] of type `UINT64`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn xxhash64(&self, seed: u64) -> Xxhash64<'_> {
        Xxhash64 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the MD5 cryptographic hash of each row.
    ///
    /// Returns a `STRING` column where each element is a 32-character
    /// lowercase hex-encoded MD5 digest computed across all columns of the
    /// corresponding row.
    ///
    /// # Returns
    ///
    /// A [`Column`] of type `STRING`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn md5(&self) -> Md5<'_> {
        Md5 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the SHA-256 cryptographic hash of each row.
    ///
    /// Returns a `STRING` column where each element is a 64-character
    /// lowercase hex-encoded SHA-256 digest.
    ///
    /// # Returns
    ///
    /// A [`Column`] of type `STRING`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn sha256(&self) -> Sha256<'_> {
        Sha256 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes a SHA-1 hash of each row.
    ///
    /// Returns a `STRING` [`Column`] containing the 40-character lowercase
    /// hexadecimal digest for each row.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn sha1(&self) -> Sha1<'_> {
        Sha1 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes a `MurmurHash3` 128-bit (x64) hash of each row.
    ///
    /// `seed` initializes the hash state. Returns a [`Table`] with two
    /// `UINT64` columns representing the high and low 64 bits of the
    /// 128-bit hash, respectively.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn murmurhash3_x64_128(&self, seed: u64) -> MurmurHash3X64_128<'_> {
        MurmurHash3X64_128 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// Computes an `XXHash` 32-bit hash of each row.
    ///
    /// `seed` initializes the hash state. Returns an `INT32` [`Column`]
    /// with one hash value per row.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn xxhash_32(&self, seed: u32) -> XxHash32<'_> {
        XxHash32 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// Computes a SHA-224 hash of each row.
    ///
    /// Returns a `STRING` [`Column`] containing the 56-character lowercase
    /// hexadecimal digest for each row.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn sha224(&self) -> Sha224<'_> {
        Sha224 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes a SHA-384 hash of each row.
    ///
    /// Returns a `STRING` [`Column`] containing the 96-character lowercase
    /// hexadecimal digest for each row.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn sha384(&self) -> Sha384<'_> {
        Sha384 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes a SHA-512 hash of each row.
    ///
    /// Returns a `STRING` [`Column`] containing the 128-character lowercase
    /// hexadecimal digest for each row.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn sha512(&self) -> Sha512<'_> {
        Sha512 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the approximate number of bits needed to store each row.
    ///
    /// Returns an `INT32` [`Column`] where each element is the total bit
    /// count for that row across all columns (data + null masks).
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn row_bit_count(&self) -> RowBitCount<'_> {
        RowBitCount {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes per-segment cumulative row bit counts.
    ///
    /// Divides the table into segments of `segment_length` rows and returns
    /// an `INT32` [`Column`] with the cumulative bit count within each
    /// segment.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn segmented_row_bit_count(&self, segment_length: i32) -> SegmentedRowBitCount<'_> {
        SegmentedRowBitCount {
            table: self,
            segment_length,
            stream: Stream::default_stream(),
        }
    }

    /// Converts each element of a fixed-width column to a list of bytes.
    ///
    /// This is an associated function (not a method on `self`). Pass a
    /// [`ColumnView`] of a fixed-width numeric type. If `flip_endian` is
    /// `true`, the byte order of each element is reversed.
    ///
    /// Returns a `LIST<UINT8>` [`Column`].
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a fixed-width type or a GPU
    /// error occurs.
    pub fn byte_cast<'a>(col: &'a ColumnView<'a>, flip_endian: bool) -> ByteCast<'a> {
        ByteCast {
            col,
            flip_endian,
            stream: Stream::default_stream(),
        }
    }

    // -- Merge --

    /// Merges two pre-sorted tables into a single sorted table.
    ///
    /// Both `self` and `right` must already be sorted on the columns at
    /// `key_columns` according to `orders` and `null_orders`. The merge is
    /// a stable O(n+m) operation. Both tables must have the same number of
    /// columns with matching types.
    ///
    /// The output has `self.len() + right.len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables are not properly sorted, column types
    /// differ, or a GPU error occurs.
    pub fn merge<'a>(
        &'a self,
        right: &'a Table,
        key_columns: &'a [i32],
        orders: &'a [Order],
        null_orders: &'a [NullOrder],
    ) -> Merge<'a> {
        Merge {
            table: self,
            right,
            key_columns,
            orders,
            null_orders,
            stream: Stream::default_stream(),
        }
    }

    // -- Partitioning --

    /// Hash-partitions the table into `num_partitions` groups.
    ///
    /// Rows are assigned to partitions by hashing the columns at the
    /// zero-based indices in `columns`. The output is a single [`Table`]
    /// with rows reordered so that each partition's rows are contiguous.
    /// Use [`hash_partition_offsets`](Self::hash_partition_offsets) to
    /// obtain the boundary offsets between partitions.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn hash_partition<'a>(
        &'a self,
        columns: &'a [i32],
        num_partitions: usize,
    ) -> HashPartition<'a> {
        HashPartition {
            table: self,
            columns,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Returns partition boundary offsets for
    /// [`hash_partition`](Self::hash_partition).
    ///
    /// The returned `Vec<usize>` has `num_partitions` elements. Element `i`
    /// is the starting row index of partition `i` in the reordered table.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn hash_partition_offsets<'a>(
        &'a self,
        columns: &'a [i32],
        num_partitions: usize,
    ) -> HashPartitionOffsets<'a> {
        HashPartitionOffsets {
            table: self,
            columns,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Distributes rows across `num_partitions` in round-robin order.
    ///
    /// Row assignment begins at partition `start` and cycles through all
    /// partitions. The output table has rows reordered so each partition's
    /// rows are contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn round_robin(&self, num_partitions: usize, start: usize) -> RoundRobin<'_> {
        RoundRobin {
            table: self,
            num_partitions,
            start,
            stream: Stream::default_stream(),
        }
    }

    /// Returns partition boundary offsets for
    /// [`round_robin`](Self::round_robin).
    ///
    /// The returned `Vec<usize>` has `num_partitions` elements. Element `i`
    /// is the starting row index of partition `i` in the reordered table.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn round_robin_offsets(
        &self,
        num_partitions: usize,
        start: usize,
    ) -> RoundRobinOffsets<'_> {
        RoundRobinOffsets {
            table: self,
            num_partitions,
            start,
            stream: Stream::default_stream(),
        }
    }

    // -- Reshape --

    /// Interleaves the table's columns into a single column, row by row.
    ///
    /// All columns in the table must have the same data type and the same
    /// length. The output column has length `self.len() * self.columns_len()`
    /// and contains elements in the order:
    /// `col0[0], col1[0], ..., colN[0], col0[1], col1[1], ...`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let c1 = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let c2 = Column::from_slice_i32(&[4, 5, 6]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(c1);
    /// b.push_column(c2);
    /// let table = b.build()?;
    ///
    /// let result = table.interleave_columns().call()?;
    /// assert_eq!(result.len(), 6);
    /// // Elements: [1, 4, 2, 5, 3, 6]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different types or lengths, or
    /// if a GPU error occurs.
    pub fn interleave_columns(&self) -> InterleaveColumns<'_> {
        InterleaveColumns {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Tiles (repeats) the table vertically `count` times.
    ///
    /// The output table has the same columns as `self` but with
    /// `self.len() * count` rows. The rows of the original table are
    /// repeated `count` consecutive times.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(col);
    /// let table = b.build()?;
    ///
    /// let tiled = table.tile(3).call()?;
    /// assert_eq!(tiled.len(), 9);         // 3 rows * 3
    /// assert_eq!(tiled.columns_len(), 1);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn tile(&self, count: usize) -> Tile<'_> {
        Tile {
            table: self,
            count,
            stream: Stream::default_stream(),
        }
    }

    // -- Fill (table-level) --

    /// Repeats each row of the table `count` times consecutively.
    ///
    /// The output has `self.len() * count` rows. Unlike
    /// [`tile`](Self::tile), which appends copies of the entire table,
    /// `repeat` duplicates each row in place.
    ///
    /// See also [`repeat_by_column`](Self::repeat_by_column) for per-row
    /// repeat counts.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn repeat(&self, count: usize) -> Repeat<'_> {
        Repeat {
            table: self,
            count,
            stream: Stream::default_stream(),
        }
    }

    // -- Transform --

    /// Dictionary-encodes the table rows as integer indices.
    ///
    /// Each unique row combination is assigned a consecutive integer key
    /// starting from 0, with keys ordered by the sorted distinct rows.
    /// The output is an `INT32` [`Column`] of the
    /// same length as the table, where each element is the key for that row.
    ///
    /// Use [`encode_keys`](Table::encode_keys) to obtain the look-up table
    /// that maps each integer key back to the original row values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let col = Column::from_slice_i32(&[3, 1, 2, 1, 3]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(col);
    /// let table = b.build()?;
    ///
    /// let indices = table.encode().call()?;
    /// // Sorted distinct values: [1, 2, 3] => keys [0, 1, 2]
    /// // Input mapping: [3->2, 1->0, 2->1, 1->0, 3->2]
    /// assert_eq!(indices.to_vec_i32().call()?, vec![2, 0, 1, 0, 2]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn encode(&self) -> Encode<'_> {
        Encode {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the sorted distinct row combinations from encoding.
    ///
    /// The output is a [`Table`] with the same column schema as `self` but
    /// containing only the unique rows, sorted in ascending order. Row *i*
    /// of the output corresponds to key *i* in the result of
    /// [`encode`](Table::encode).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let col = Column::from_slice_i32(&[3, 1, 2, 1, 3]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(col);
    /// let table = b.build()?;
    ///
    /// let keys = table.encode_keys().call()?;
    /// assert_eq!(keys.len(), 3);          // 3 distinct values
    /// assert_eq!(keys.columns_len(), 1);  // same schema as input
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn encode_keys(&self) -> EncodeKeys<'_> {
        EncodeKeys {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Stream compaction --

    /// Drops rows where any of the specified key columns contain NaN.
    ///
    /// `keys` is a slice of zero-based column indices identifying which
    /// columns to examine. A row is dropped if **any** of the key columns
    /// in that row is NaN. Non-floating-point key columns are ignored for
    /// the NaN check.
    ///
    /// See also [`drop_nans_with_threshold`](Self::drop_nans_with_threshold)
    /// for threshold-based dropping.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn drop_nans<'a>(&'a self, keys: &'a [i32]) -> DropNans<'a> {
        DropNans {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    /// Drops rows where the number of non-null key values is below
    /// `threshold`.
    ///
    /// `keys` is a slice of zero-based column indices to examine. A row is
    /// kept only if at least `threshold` of those key columns are non-null
    /// in that row.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn drop_nulls_with_threshold<'a>(
        &'a self,
        keys: &'a [i32],
        threshold: usize,
    ) -> DropNullsWithThreshold<'a> {
        DropNullsWithThreshold {
            table: self,
            keys,
            threshold,
            stream: Stream::default_stream(),
        }
    }

    /// Removes consecutive duplicate rows based on key columns.
    ///
    /// `keys` is a slice of zero-based column indices that define equality.
    /// Only **consecutive** duplicate rows (as determined by key column
    /// values) are collapsed -- the first of each run is kept. This is
    /// analogous to the Unix `uniq` command.
    ///
    /// For global deduplication regardless of row order, use
    /// [`distinct`](Self::distinct).
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn unique<'a>(&'a self, keys: &'a [i32]) -> Unique<'a> {
        Unique {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    /// Returns globally distinct rows based on key columns.
    ///
    /// `keys` is a slice of zero-based column indices that define equality.
    /// All duplicate rows (anywhere in the table, not just consecutive) are
    /// removed. The output order is not guaranteed.
    ///
    /// For order-preserving deduplication, use
    /// [`stable_distinct`](Self::stable_distinct).
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn distinct<'a>(&'a self, keys: &'a [i32]) -> Distinct<'a> {
        Distinct {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    /// Returns globally distinct rows, preserving the original input order.
    ///
    /// Like [`distinct`](Self::distinct), but the first occurrence of each
    /// key combination retains its position relative to other first
    /// occurrences.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn stable_distinct<'a>(&'a self, keys: &'a [i32]) -> StableDistinct<'a> {
        StableDistinct {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying extras --

    /// Scatters rows from `source` into this table at the positions given
    /// by `scatter_map`.
    ///
    /// `scatter_map` is an `INT32` column of the same length as `source`.
    /// For each row `i` in `source`, the row is written to position
    /// `scatter_map[i]` in a copy of `self`. Rows in `self` that are not
    /// targeted by any scatter index retain their original values.
    ///
    /// `source` and `self` must have the same number of columns with
    /// matching types.
    ///
    /// # Errors
    ///
    /// Returns an error if scatter map indices are out of bounds, column
    /// types differ, or a GPU error occurs.
    pub fn scatter<'a>(
        &'a self,
        source: &'a Table,
        scatter_map: &'a ColumnView<'a>,
    ) -> Scatter<'a> {
        Scatter {
            table: self,
            source,
            map: scatter_map,
            stream: Stream::default_stream(),
        }
    }

    /// Reverses the row order of the table.
    ///
    /// Returns a new [`Table`] where the last row becomes the first, and
    /// so on. The column count and types are unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn reverse(&self) -> Reverse<'_> {
        Reverse {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Extracts a contiguous range of rows `[begin, end)` as a new table.
    ///
    /// `begin` is inclusive and `end` is exclusive (zero-based). The output
    /// has `end - begin` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds or a GPU error
    /// occurs.
    pub fn slice(&self, begin: usize, end: usize) -> Slice<'_> {
        Slice {
            table: self,
            begin,
            end,
            stream: Stream::default_stream(),
        }
    }

    /// Randomly samples `n` rows from the table.
    ///
    /// If `with_replacement` is `true`, the same row may appear more than
    /// once. `seed` controls the random number generator for reproducible
    /// results.
    ///
    /// # Errors
    ///
    /// Returns an error if `n` exceeds `self.len()` when
    /// `with_replacement` is `false`, or if a GPU error occurs.
    pub fn sample(&self, n: usize, with_replacement: bool, seed: i64) -> Sample<'_> {
        Sample {
            table: self,
            n,
            with_replacement,
            seed,
            stream: Stream::default_stream(),
        }
    }

    // -- Transpose --

    /// Transposes the table so that rows become columns and columns become
    /// rows.
    ///
    /// The output table has `self.columns_len()` rows and `self.len()`
    /// columns. All source columns must have the same data type.
    ///
    /// # Errors
    ///
    /// Returns an error if columns have different types or a GPU error
    /// occurs.
    pub fn transpose(&self) -> Transpose<'_> {
        Transpose {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Search --

    /// Finds the lower-bound insertion point for each row in `needles`
    /// within this **sorted** table.
    ///
    /// For each needle row, the returned index is the first position where
    /// the needle could be inserted while preserving the sort order (like
    /// C++ `std::lower_bound`).
    ///
    /// # Parameters
    ///
    /// - `needles` -- a [`Table`] with the same number of columns and
    ///   compatible types as `self`. Each row is looked up independently.
    /// - `orders` -- one [`Order`] per column, specifying whether the column
    ///   is sorted `ASCENDING` or `DESCENDING`. Must match `self`'s actual
    ///   sort order.
    /// - `nulls` -- one [`NullOrder`] per column, specifying whether nulls
    ///   sort `BEFORE` or `AFTER` non-null values.
    ///
    /// # Returns
    ///
    /// An `INT32` [`Column`] with one entry per needle row, containing the
    /// insertion index.
    ///
    /// Returns a [`LowerBound`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables have different column counts or
    /// incompatible types.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::sorting::{NullOrder, Order};
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let hay = {
    ///     let c = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(c);
    ///     b.build()?
    /// };
    /// let needles = {
    ///     let c = Column::from_slice_i32(&[15, 30, 55]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(c);
    ///     b.build()?
    /// };
    /// let idx = hay.lower_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
    ///     .call()?;
    /// assert_eq!(idx.to_vec_i32().call()?, [1, 2, 5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn lower_bound<'a>(
        &'a self,
        needles: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> LowerBound<'a> {
        LowerBound {
            table: self,
            needles,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Finds the upper-bound insertion point for each row in `needles`
    /// within this **sorted** table.
    ///
    /// For each needle row, the returned index is one past the last
    /// position of an equal element -- i.e., the first position where
    /// the needle could be inserted *after* all existing equal rows (like
    /// C++ `std::upper_bound`).
    ///
    /// # Parameters
    ///
    /// - `needles` -- a [`Table`] with the same column layout as `self`.
    /// - `orders` -- one [`Order`] per column (see [`lower_bound`](Self::lower_bound)).
    /// - `nulls` -- one [`NullOrder`] per column.
    ///
    /// # Returns
    ///
    /// An `INT32` [`Column`] with one entry per
    /// needle row.
    ///
    /// Returns an [`UpperBound`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables have different column counts or
    /// incompatible types.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::sorting::{NullOrder, Order};
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let hay = {
    ///     let c = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(c);
    ///     b.build()?
    /// };
    /// let needles = {
    ///     let c = Column::from_slice_i32(&[15, 30, 55]).call()?;
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(c);
    ///     b.build()?
    /// };
    /// let idx = hay.upper_bound(&needles, &[Order::ASCENDING], &[NullOrder::BEFORE])
    ///     .call()?;
    /// // 15 -> 1 (before 20), 30 -> 3 (after 30), 55 -> 5 (past end)
    /// assert_eq!(idx.to_vec_i32().call()?, [1, 3, 5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn upper_bound<'a>(
        &'a self,
        needles: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> UpperBound<'a> {
        UpperBound {
            table: self,
            needles,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    // -- Sorting (new) --

    /// Returns a column of row indices that would sort this table, with
    /// stable ordering (equal elements preserve their original relative
    /// order).
    ///
    /// Semantics are identical to [`sorted_order`](Self::sorted_order)
    /// except for stability.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn stable_sorted_order<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSortedOrder<'a> {
        StableSortedOrder {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Sorts the table with stable ordering (equal elements preserve their
    /// original relative order).
    ///
    /// Semantics are identical to [`sort`](Self::sort) except for stability.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn stable_sort<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSort<'a> {
        StableSort {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Sorts this (values) table by the sort order of a separate `keys`
    /// table.
    ///
    /// `keys` must have the same number of rows as `self`. The sort
    /// permutation is determined by sorting `keys` according to `orders`
    /// and `nulls`, then that same permutation is applied to `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if row counts differ or a GPU error occurs.
    pub fn sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SortByKey<'a> {
        SortByKey {
            table: self,
            keys,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable sort this (values) table by a separate `keys` table.
    ///
    /// Like [`sort_by_key`](Self::sort_by_key) but with stable ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if row counts differ or a GPU error occurs.
    pub fn stable_sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSortByKey<'a> {
        StableSortByKey {
            table: self,
            keys,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Returns sorted row indices within each segment of the table.
    ///
    /// `segment_offsets` is an `INT32` column of starting row indices that
    /// define contiguous segments. Sorting is performed independently
    /// within each segment according to `orders` and `nulls`.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn segmented_sorted_order<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SegmentedSortedOrder<'a> {
        SegmentedSortedOrder {
            table: self,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable version of [`segmented_sorted_order`](Self::segmented_sorted_order),
    /// preserving the relative order of equal elements within each segment.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn stable_segmented_sorted_order<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSegmentedSortedOrder<'a> {
        StableSegmentedSortedOrder {
            table: self,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Sorts this (values) table by a separate `keys` table within each
    /// segment defined by `segment_offsets`.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn segmented_sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SegmentedSortByKey<'a> {
        SegmentedSortByKey {
            table: self,
            keys,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable version of [`segmented_sort_by_key`](Self::segmented_sort_by_key),
    /// preserving the relative order of equal elements within each segment.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn stable_segmented_sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSegmentedSortByKey<'a> {
        StableSegmentedSortByKey {
            table: self,
            keys,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Selects rows at the given quantile positions from the table.
    ///
    /// For each quantile value in `q`, selects the row at that position in
    /// the sorted table. If `is_sorted` is `true`, the table is assumed to
    /// already be sorted; otherwise it is sorted internally.
    ///
    /// # Arguments
    ///
    /// * `q` -- Quantile values in `[0.0, 1.0]`. Each value selects a row.
    /// * `interp` -- [`Interpolation`](crate::quantile::Interpolation) method
    ///   when the quantile falls between two rows.
    /// * `is_sorted` -- If `true`, assumes the table is already sorted.
    /// * `orders` -- Sort order per column.
    /// * `nulls` -- Null placement per column.
    ///
    /// # Returns
    ///
    /// A new [`Table`] with one row per quantile value.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn quantiles<'a>(
        &'a self,
        q: &'a [f64],
        interp: crate::quantile::Interpolation,
        is_sorted: bool,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> Quantiles<'a> {
        Quantiles {
            table: self,
            q,
            interp,
            is_sorted,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying (new) --

    /// Scatters rows from `source` into this table at positions where `mask`
    /// is `true`.
    ///
    /// Row *i* of `source` replaces the *i*-th `true` position in `mask`.
    /// `source` must have as many rows as the number of `true` values in
    /// `mask`. Both tables must have matching column schemas.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn boolean_mask_scatter<'a>(
        &'a self,
        source: &'a Table,
        mask: &'a ColumnView<'a>,
    ) -> BooleanMaskScatter<'a> {
        BooleanMaskScatter {
            table: self,
            source,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Scatters scalar values (one per column) to specified row positions.
    ///
    /// For each index in `scatter_map`, the corresponding row in the output
    /// receives the scalar values. `scalars` must have one element per column.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn scatter_scalars<'a>(
        &'a self,
        scalars: &'a [Scalar],
        scatter_map: &'a ColumnView<'a>,
    ) -> ScatterScalars<'a> {
        ScatterScalars {
            table: self,
            scalars,
            map: scatter_map,
            stream: Stream::default_stream(),
        }
    }

    /// Scatters scalar values (one per column) into rows where `mask` is
    /// `true`.
    ///
    /// Every row where `mask` is `true` receives the scalar values from
    /// `scalars`. `scalars` must have one element per column, with types
    /// matching the table columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn boolean_mask_scatter_scalars<'a>(
        &'a self,
        scalars: &'a [Scalar],
        mask: &'a ColumnView<'a>,
    ) -> BooleanMaskScatterScalars<'a> {
        BooleanMaskScatterScalars {
            table: self,
            scalars,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Repeats table rows using per-row counts from a column.
    ///
    /// Each row *i* is repeated `counts[i]` times. `counts` must be an
    /// integer column with the same number of rows as `self`. A count of 0
    /// removes that row from the output.
    ///
    /// See also [`repeat`](Table::repeat) for uniform repetition.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn repeat_by_column<'a>(&'a self, counts: &'a ColumnView<'a>) -> RepeatByColumn<'a> {
        RepeatByColumn {
            table: self,
            counts,
            stream: Stream::default_stream(),
        }
    }

    // -- Explode --

    /// Explodes a list column, expanding each list element into its own row.
    ///
    /// `column_idx` is the zero-based index of a `LIST`-type column. Each
    /// list element becomes a separate row, with all other columns
    /// duplicated accordingly. Empty lists produce no output rows.
    ///
    /// See also [`explode_outer`](Self::explode_outer) to keep empty/null
    /// list rows as null rows.
    ///
    /// # Errors
    ///
    /// Returns an error if `column_idx` is out of bounds, the column is
    /// not a list type, or a GPU error occurs.
    pub fn explode(&self, column_idx: usize) -> Explode<'_> {
        Explode {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Explodes a list column like [`explode`](Self::explode), and adds an
    /// extra `INT32` column at the front containing the zero-based position
    /// of each element within its original list.
    ///
    /// # Errors
    ///
    /// Returns an error if `column_idx` is out of bounds, the column is
    /// not a list type, or a GPU error occurs.
    pub fn explode_position(&self, column_idx: usize) -> ExplodePosition<'_> {
        ExplodePosition {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Explodes a list column like [`explode`](Self::explode), but preserves
    /// rows with null or empty lists by emitting a single null row for each.
    ///
    /// # Errors
    ///
    /// Returns an error if `column_idx` is out of bounds, the column is
    /// not a list type, or a GPU error occurs.
    pub fn explode_outer(&self, column_idx: usize) -> ExplodeOuter<'_> {
        ExplodeOuter {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Combines [`explode_outer`](Self::explode_outer) and
    /// [`explode_position`](Self::explode_position): keeps null/empty list
    /// rows and adds an element-position column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column_idx` is out of bounds, the column is
    /// not a list type, or a GPU error occurs.
    pub fn explode_outer_position(&self, column_idx: usize) -> ExplodeOuterPosition<'_> {
        ExplodeOuterPosition {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Gathers rows by index with configurable out-of-bounds handling.
    ///
    /// Like [`gather`](Table::gather), but allows control over what happens
    /// when `indices` contains out-of-bounds values.
    ///
    /// # Arguments
    ///
    /// * `indices` -- An `INT32` column of row indices to select.
    /// * `nullify_oob` -- If `true`, out-of-bounds indices produce null rows
    ///   in the output. If `false`, behavior is undefined for out-of-bounds
    ///   indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn gather_checked<'a>(
        &'a self,
        indices: &'a ColumnView<'a>,
        nullify_oob: bool,
    ) -> GatherChecked<'a> {
        GatherChecked {
            table: self,
            indices,
            nullify_oob,
            stream: Stream::default_stream(),
        }
    }

    /// Gathers rows by index with configurable out-of-bounds and negative-index
    /// handling.
    ///
    /// This is a more flexible version of [`gather_checked`](Table::gather_checked)
    /// that additionally controls whether negative indices are allowed (wrap
    /// around) or treated as undefined behavior.
    ///
    /// Use the builder methods [`.nullify_oob()`](GatherWithPolicy::nullify_oob)
    /// and [`.allow_negative()`](GatherWithPolicy::allow_negative) to configure
    /// the policies before calling [`.call()`](crate::stream::GpuOp::call).
    ///
    /// # Arguments
    ///
    /// * `indices` -- An `INT32` column of row indices to select.
    ///
    /// # Defaults
    ///
    /// * `nullify_oob` -- `false` (`DONT_CHECK`)
    /// * `allow_negative` -- `true` (`ALLOWED`, negative indices wrap around)
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    #[doc(alias = "negative_index_policy")]
    pub fn gather_with_policy<'a>(&'a self, indices: &'a ColumnView<'a>) -> GatherWithPolicy<'a> {
        GatherWithPolicy {
            table: self,
            indices,
            nullify_oob: false,
            allow_negative: true,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the cross join (Cartesian product) of this table with
    /// `right`.
    ///
    /// The output table has `self.len() * right.len()` rows and
    /// `self.columns_len() + right.columns_len()` columns (all left
    /// columns followed by all right columns).
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn cross_join<'a>(&'a self, right: &'a Table) -> CrossJoin<'a> {
        CrossJoin {
            table: self,
            right,
            stream: Stream::default_stream(),
        }
    }

    /// Partitions the table using a `partition_map` column that assigns each
    /// row to a partition.
    ///
    /// `partition_map` is an integer column with values in
    /// `[0, num_partitions)`. The output table has rows reordered so that
    /// each partition's rows are contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn partition_by_map<'a>(
        &'a self,
        partition_map: &'a ColumnView<'a>,
        num_partitions: usize,
    ) -> PartitionByMap<'a> {
        PartitionByMap {
            table: self,
            partition_map,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Returns partition boundary offsets for
    /// [`partition_by_map`](Self::partition_by_map).
    ///
    /// The returned `Vec<usize>` has `num_partitions` elements. Element `i`
    /// is the starting row index of partition `i` in the reordered table.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn partition_by_map_offsets<'a>(
        &'a self,
        partition_map: &'a ColumnView<'a>,
        num_partitions: usize,
    ) -> PartitionByMapOffsets<'a> {
        PartitionByMapOffsets {
            table: self,
            partition_map,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct rows in this table.
    ///
    /// If `nulls_equal` is `true`, all null rows are considered equal and
    /// count as a single distinct value. If `false`, each null row is
    /// treated as unique.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn distinct_count(&self, nulls_equal: bool) -> DistinctCount<'_> {
        DistinctCount {
            table: self,
            nulls_equal,
            stream: Stream::default_stream(),
        }
    }

    // -- Bitmask combining --

    /// Computes the bitwise AND of all column null masks.
    ///
    /// Returns a `BOOL8` [`Column`] where `true` means the row is valid
    /// (non-null) in **all** columns.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn bitmask_and_to_bools(&self) -> BitmaskAndToBools<'_> {
        BitmaskAndToBools {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the bitwise OR of all column null masks.
    ///
    /// Returns a `BOOL8` [`Column`] where `true` means the row is valid
    /// (non-null) in **at least one** column.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn bitmask_or_to_bools(&self) -> BitmaskOrToBools<'_> {
        BitmaskOrToBools {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Groupby scan/shift/replace_nulls --

    /// Performs cumulative (scan) aggregation within groups.
    ///
    /// Groups are defined by `key_columns`. For each group, a running
    /// aggregation is computed over the `value_columns` using the
    /// corresponding `aggs`. The output table has the same number of rows
    /// as the input, with key columns followed by the scan result columns.
    ///
    /// `value_columns` and `aggs` must have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if `value_columns` and `aggs` differ in length, or
    /// a GPU error occurs.
    pub fn groupby_scan<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        aggs: &'a [AggregationKind],
    ) -> GroupbyScan<'a> {
        GroupbyScan {
            table: self,
            key_columns,
            value_columns,
            aggs,
            stream: Stream::default_stream(),
        }
    }

    /// Shifts values within groups by specified offsets, filling vacated
    /// positions with scalars.
    ///
    /// Groups are defined by `key_columns`. For each `value_columns[i]`,
    /// values are shifted by `offsets[i]` positions within each group.
    /// Positive offsets shift forward (creating fill values at the start);
    /// negative offsets shift backward. `fill_values[i]` provides the
    /// fill scalar for the vacated positions in `value_columns[i]`.
    ///
    /// All four slices (`value_columns`, `offsets`, `fill_values`) must
    /// have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn groupby_shift<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        offsets: &'a [i32],
        fill_values: &'a [Scalar],
    ) -> GroupbyShift<'a> {
        GroupbyShift {
            table: self,
            key_columns,
            value_columns,
            offsets,
            fill_values,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces null values within groups using forward or backward fill.
    ///
    /// Groups are defined by `key_columns`. For each column index in
    /// `value_columns`, the corresponding entry in `policies` determines
    /// the fill direction: `0` fills forward (PRECEDING -- propagates the
    /// last non-null value), `1` fills backward (FOLLOWING -- propagates
    /// the next non-null value).
    ///
    /// The output table contains the key columns followed by the
    /// null-replaced value columns.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let keys = Column::from_slice_i32(&[1, 1, 1, 2, 2]).call()?;
    /// let vals = Column::from_slice_i32(&[10, 0, 30, 0, 50]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(keys);
    /// b.push_column(vals);
    /// let tbl = b.build()?;
    ///
    /// // Forward-fill nulls within each group
    /// let result = tbl.groupby_replace_nulls(&[0], &[1], &[0]).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn groupby_replace_nulls<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        policies: &'a [i32],
    ) -> GroupbyReplaceNulls<'a> {
        GroupbyReplaceNulls {
            table: self,
            key_columns,
            value_columns,
            policies,
            stream: Stream::default_stream(),
        }
    }

    // -- Unique count --

    /// Counts the number of consecutive unique rows in the table.
    ///
    /// Like the Unix `uniq` command -- only counts transitions between
    /// adjacent distinct rows. If the table is sorted, this equals the
    /// total number of distinct rows.
    ///
    /// # Arguments
    ///
    /// * `nulls_equal` -- If `true`, consecutive null rows count as equal.
    ///
    /// # Returns
    ///
    /// The count as `usize`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn unique_count(&self, nulls_equal: bool) -> UniqueCount<'_> {
        UniqueCount {
            table: self,
            nulls_equal,
            stream: Stream::default_stream(),
        }
    }

    // -- Drop NaNs with threshold --

    /// Drops rows with NaN values, keeping only rows where at least
    /// `threshold` of the specified key columns contain non-NaN values.
    ///
    /// # Arguments
    ///
    /// * `keys` -- Zero-based column indices to examine for NaN.
    /// * `threshold` -- Minimum number of non-NaN key values required to
    ///   keep the row.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds or a GPU error
    /// occurs.
    pub fn drop_nans_with_threshold<'a>(
        &'a self,
        keys: &'a [i32],
        threshold: usize,
    ) -> DropNansWithThreshold<'a> {
        DropNansWithThreshold {
            table: self,
            keys,
            threshold,
            stream: Stream::default_stream(),
        }
    }

    // -- Approximate distinct count --

    /// Estimates the approximate number of distinct rows using the
    /// `HyperLogLog++` algorithm.
    ///
    /// `precision` controls accuracy versus memory usage and must be in
    /// the range 4..=18 (a typical default is 12). The standard error is
    /// approximately `1.04 / sqrt(2^precision)`, so higher precision
    /// gives a more accurate estimate at the cost of more memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 1, 2]).call()?;
    /// let mut b = TableBuilder::new();
    /// b.push_column(col);
    /// let tbl = b.build()?;
    ///
    /// let approx = tbl.approx_distinct_count(12).call()?;
    /// assert!(approx >= 2); // at least close to 3 distinct values
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    pub fn approx_distinct_count(&self, precision: i32) -> ApproxDistinctCount<'_> {
        ApproxDistinctCount {
            table: self,
            precision,
            stream: Stream::default_stream(),
        }
    }

    /// Returns `true` if any column in this table has a nested data type
    /// ([`LIST`](crate::data_type::TypeId::LIST) or
    /// [`STRUCT`](crate::data_type::TypeId::STRUCT)).
    ///
    /// This is a metadata check and does not launch GPU work.
    pub fn has_nested_columns(&self) -> bool {
        cudf_sys::ffi::table_has_nested_columns(&self.0)
    }

    /// Returns `true` if any nested (child) column within this table
    /// contains null values.
    ///
    /// Only inspects children of `LIST` and `STRUCT` columns; top-level
    /// nulls are not considered. This is a metadata check and does not
    /// launch GPU work.
    pub fn has_nested_nulls(&self) -> bool {
        cudf_sys::ffi::table_has_nested_nulls(&self.0)
    }

    /// Returns `true` if any nested (child) column within this table has
    /// a null mask allocated (i.e., is nullable), regardless of whether it
    /// actually contains null values.
    ///
    /// This is a metadata check and does not launch GPU work.
    pub fn has_nested_nullable_columns(&self) -> bool {
        cudf_sys::ffi::table_has_nested_nullable_columns(&self.0)
    }

    /// Creates a table from a `DLPack` `DLManagedTensor` pointer.
    ///
    /// `managed_tensor_ptr` is a raw pointer (passed as `usize`) to a
    /// valid `DLManagedTensor`. The tensor's data must reside on a CUDA
    /// device. Ownership of the tensor is transferred to libcudf, which
    /// will call the tensor's deleter when it is no longer needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor format is unsupported or if a GPU
    /// error occurs.
    pub fn from_dlpack(managed_tensor_ptr: usize) -> FromDlpack {
        FromDlpack {
            managed_tensor_ptr,
            stream: Stream::default_stream(),
        }
    }

    /// Converts this table into a `DLPack` `DLManagedTensor` pointer.
    ///
    /// All columns must have the same numeric data type and a null count
    /// of zero. The returned `usize` is a raw pointer to a newly
    /// allocated `DLManagedTensor`; the caller is responsible for
    /// eventually calling the tensor's `deleter` function to free it.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different types, contain
    /// nulls, or if a GPU error occurs.
    pub fn to_dlpack(&self) -> ToDlpack<'_> {
        ToDlpack {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Concatenates all columns in this table into a single [`Column`],
    /// stacking them end-to-end in column-index order.
    ///
    /// All columns must have the same data type. The resulting column has
    /// `self.len() * self.columns_len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different data types or if a
    /// GPU error occurs.
    pub fn concatenate_columns(&self) -> ConcatenateTableColumns<'_> {
        ConcatenateTableColumns {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Concatenates two tables vertically (row-wise append).
    ///
    /// `other` is appended below `self`. Both tables must have the same
    /// number of columns, and corresponding columns must have matching
    /// data types. The resulting table has `self.len() + other.len()`
    /// rows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::TableBuilder;
    ///
    /// let c1 = Column::from_slice_i32(&[1, 2]).call()?;
    /// let c2 = Column::from_slice_i32(&[3, 4]).call()?;
    /// let t1 = {
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(c1);
    ///     b.build()?
    /// };
    /// let t2 = {
    ///     let mut b = TableBuilder::new();
    ///     b.push_column(c2);
    ///     b.build()?
    /// };
    ///
    /// let combined = t1.concatenate(&t2).call()?;
    /// assert_eq!(combined.len(), 4);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tables have different column counts or
    /// mismatched column types, or if a GPU error occurs.
    pub fn concatenate<'a>(&'a self, other: &'a Table) -> ConcatenateWith<'a> {
        ConcatenateWith {
            table: self,
            other,
            stream: Stream::default_stream(),
        }
    }
}

/// An iterator over the columns of a [`Table`], yielding
/// [`ColumnView`]s in index order.
///
/// Created by [`Table::columns`]. Implements [`ExactSizeIterator`], so
/// [`ExactSizeIterator::len`] returns the number of remaining columns.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::stream::GpuOp;
/// use cudf::table::TableBuilder;
///
/// let c = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// let mut b = TableBuilder::new();
/// b.push_column(c);
/// let tbl = b.build()?;
///
/// for col_view in tbl.columns() {
///     assert_eq!(col_view.len(), 3);
/// }
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct Columns<'a> {
    table: &'a Table,
    index: usize,
    len: usize,
}

impl<'a> Iterator for Columns<'a> {
    type Item = ColumnView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let col = self.table.column(self.index).ok();
        if col.is_some() {
            self.index += 1;
        }
        col
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Columns<'_> {}

/// A builder for constructing a [`Table`] from individual [`Column`]s.
///
/// Columns are added one at a time via [`push_column`](Self::push_column),
/// which transfers ownership of each [`Column`]'s GPU memory to the
/// builder. When all columns have been added, call
/// [`build`](Self::build) to produce the final [`Table`].
///
/// All columns must have the same row count; `build` will return an
/// error if they do not.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::stream::GpuOp;
/// use cudf::table::TableBuilder;
///
/// let c1 = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// let c2 = Column::from_slice_f64(&[1.0, 2.0, 3.0]).call()?;
/// let mut builder = TableBuilder::new();
/// builder.push_column(c1);
/// builder.push_column(c2);
/// let table = builder.build()?;
///
/// assert_eq!(table.len(), 3);
/// assert_eq!(table.columns_len(), 2);
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct TableBuilder(UniquePtr<cudf_sys::ffi::TableBuilder>);

impl Default for TableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TableBuilder {
    /// Creates a new, empty `TableBuilder` with no columns.
    pub fn new() -> Self {
        Self(cudf_sys::ffi::new_table_builder())
    }

    /// Appends `col` to this builder, transferring ownership of the
    /// column's GPU memory. The column must have the same row count as
    /// any columns previously added (enforced by [`build`](Self::build)).
    pub fn push_column(&mut self, col: Column) {
        cudf_sys::ffi::table_builder_add_column(self.0.pin_mut(), col.0);
    }

    /// Consumes the builder and returns a [`Table`] owning all added
    /// columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have mismatched row counts or if
    /// a GPU error occurs.
    pub fn build(mut self) -> crate::Result<Table> {
        let tbl = cudf_sys::ffi::table_builder_build(self.0.pin_mut())?;
        Ok(Table(tbl))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;

    #[test]
    fn empty_table_default() {
        let table = Table::default();
        assert_eq!(table.columns_len(), 0);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn empty_table_column_out_of_bounds() {
        let table = Table::default();
        assert!(table.column(0).is_err());
    }

    #[test]
    fn empty_table_columns_iterator() {
        let table = Table::default();
        let cols: Vec<_> = table.columns().collect();
        assert!(cols.is_empty());
        assert_eq!(table.columns().len(), 0);
    }

    #[test]
    fn table_from_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 3).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_f64(2.5), 3)
            .call()
            .unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        assert_eq!(table.columns_len(), 2);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn table_column_views() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 4)
            .call()
            .unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let view = table.column(0).unwrap();
        assert_eq!(view.len(), 4);
        assert_eq!(view.type_id(), TypeId::INT32);
    }

    #[test]
    fn table_columns_iterator_with_data() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 2)
            .call()
            .unwrap();
        let c3 = Column::from_scalar(&Scalar::from_bool(true), 2)
            .call()
            .unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        builder.push_column(c3);
        let table = builder.build().unwrap();

        let cols: Vec<_> = table.columns().collect();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].type_id(), TypeId::INT32);
        assert_eq!(cols[1].type_id(), TypeId::FLOAT64);
        assert_eq!(cols[2].type_id(), TypeId::BOOL8);
    }

    #[test]
    fn table_alloc_bytes() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 100)
            .call()
            .unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        assert!(table.alloc_bytes() > 0);
    }

    #[test]
    fn table_builder_default() {
        let builder = TableBuilder::default();
        let table = builder.build().unwrap();
        assert_eq!(table.columns_len(), 0);
    }

    #[test]
    fn gather_with_policy_default() {
        let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();

        let idx = Column::from_slice_i32(&[2, 0]).call().unwrap();
        let result = table.gather_with_policy(&idx.view()).call().unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn gather_with_policy_nullify_oob() {
        let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();

        let idx = Column::from_slice_i32(&[0, 100]).call().unwrap();
        let result = table
            .gather_with_policy(&idx.view())
            .nullify_oob(true)
            .call()
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn gather_with_policy_allow_negative() {
        let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();

        // -1 should wrap to index 2 (30)
        let idx = Column::from_slice_i32(&[-1]).call().unwrap();
        let result = table
            .gather_with_policy(&idx.view())
            .allow_negative(true)
            .call()
            .unwrap();
        assert_eq!(result.len(), 1);
    }
}
