// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU table types: owning [`Table`] and [`TableBuilder`].
//!
//! A [`Table`] is a collection of [`Column`]s that all share the same row
//! count, analogous to a data frame. It is the primary unit of data for
//! multi-column GPU operations such as joins, sorting, groupby, and
//! partitioning.
//!
//! The preferred safe path is to allocate tables from an explicit
//! [`rmm::gpu_context::GpuContext`] via [`Table::from_columns_in`]. The
//! [`UnboundTable`] type remains available as a legacy/raw-owner escape hatch
//! for the older ambient-allocation model.
//!
//! Tables can also be constructed using [`TableBuilder`], which collects
//! columns one at a time and validates that they share the same length when
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
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::stream::GpuOp;
//! use cudf::stream::GpuOpExt;
//! use cudf::table::Table;
//! use rmm::device::current_device;
//! use rmm::gpu_context::GpuContext;
//!
//! let ctx = GpuContext::<()>::new(current_device())?;
//! let alloc = ctx.default_device_allocator();
//! let exec = ctx.default_stream();
//!
//! let keys = Column::from_slice_i32_in(&alloc, &[3, 1, 2])?;
//! let vals = Column::from_slice_i32_in(&alloc, &[30, 10, 20])?;
//! let table = Table::from_columns_in(&alloc, vec![keys, vals])?;
//! let sorted = table.sort_ascending().in_alloc(&alloc).call_on(&exec)?;
//!
//! assert_eq!(sorted.column(0)?.to_vec_i32().call()?, vec![1, 2, 3]);
//! # Ok::<(), cudf::error::Error>(())
//! ```

use cxx::UniquePtr;
use rmm::gpu_context::{Allocator, ContextBound};

use crate::column::{ColumnView, RawColumn, UnboundColumn};
use crate::error::Result;
use crate::groupby::AggregationKind;
use crate::scalar::Scalar;
use crate::sorting::{NullOrder, Order};
use crate::stream::Stream;
use crate::{i32_to_usize, usize_to_i32};

/// A table bound to an explicit allocator/context lifetime.
pub type Table<'ctx> = OwnedTable<'ctx>;

/// Backward-compatible alias for the bound safe-table surface.
pub type OwnedTable<'ctx> = BoundTable<'ctx, ()>;

/// A legacy unbound owning table.
///
/// Prefer [`Table<'_>`] for the safe context-bound surface. `UnboundTable`
/// is kept for compatibility with the older ambient-allocation model and for
/// bridging APIs that still materialize raw-owner values.
pub type UnboundTable = RawTable<UniquePtr<cudf_sys::ffi::Table>>;

#[doc(hidden)]
pub type BoundTable<'ctx, Brand> =
    RawTable<ContextBound<'ctx, Brand, UniquePtr<cudf_sys::ffi::Table>>>;

#[doc(hidden)]
pub trait TableOwner {
    fn as_unique_ptr(&self) -> &UniquePtr<cudf_sys::ffi::Table>;
}

impl TableOwner for UniquePtr<cudf_sys::ffi::Table> {
    fn as_unique_ptr(&self) -> &UniquePtr<cudf_sys::ffi::Table> {
        self
    }
}

impl<Brand> TableOwner for ContextBound<'_, Brand, UniquePtr<cudf_sys::ffi::Table>> {
    fn as_unique_ptr(&self) -> &UniquePtr<cudf_sys::ffi::Table> {
        self
    }
}

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
pub struct RawTable<Raw = UniquePtr<cudf_sys::ffi::Table>>(pub(crate) Raw);

impl Default for UnboundTable {
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
pub struct Sort<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sort<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        validate_sort_args(self.table, self.orders, self.nulls)?;
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::sort_table(
            self.table.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::sort_ascending`]. See that method for details.
pub struct SortAscending<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for SortAscending<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        self.table.sort(&[], &[]).stream(self.stream).call()
    }
}

/// Builder for [`Table::sorted_order`]. See that method for details.
pub struct SortedOrder<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for SortedOrder<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        validate_sort_args(self.table, self.orders, self.nulls)?;
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let col = cudf_sys::sorting::ffi::sorted_order(
            self.table.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(col))
    }
}

/// Builder for [`Table::is_sorted`]. See that method for details.
pub struct IsSorted<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for IsSorted<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = bool;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        validate_sort_args(self.table, self.orders, self.nulls)?;
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        cudf_sys::sorting::ffi::is_sorted_table(
            self.table.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

fn validate_sort_args<Raw: TableOwner>(
    table: &RawTable<Raw>,
    orders: &[Order],
    nulls: &[NullOrder],
) -> Result<()> {
    let columns_len = table.columns_len();
    if !orders.is_empty() && orders.len() != columns_len {
        return Err(crate::error::Error::InvalidArgument(format!(
            "orders length {} does not match table column count {}",
            orders.len(),
            columns_len
        )));
    }
    if !nulls.is_empty() && nulls.len() != columns_len {
        return Err(crate::error::Error::InvalidArgument(format!(
            "nulls length {} does not match table column count {}",
            nulls.len(),
            columns_len
        )));
    }
    Ok(())
}

/// Builder for [`Table::inner_join`]. See that method for details.
pub struct InnerJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for InnerJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::inner_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::left_join`]. See that method for details.
pub struct LeftJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LeftJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::full_join`]. See that method for details.
pub struct FullJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for FullJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::full_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::left_semi_join`]. See that method for details.
pub struct LeftSemiJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LeftSemiJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_semi_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::left_anti_join`]. See that method for details.
pub struct LeftAntiJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LeftAntiJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_anti_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::inner_join_indices`]. See that method for details.
#[doc(alias = "inner_join")]
pub struct InnerJoinIndices<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for InnerJoinIndices<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::inner_join_indices(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::left_join_indices`]. See that method for details.
#[doc(alias = "left_join")]
pub struct LeftJoinIndices<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LeftJoinIndices<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_join_indices(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::full_join_indices`]. See that method for details.
#[doc(alias = "full_join")]
pub struct FullJoinIndices<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for FullJoinIndices<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::full_join_indices(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::left_semi_join_indices`]. See that method for details.
#[doc(alias = "left_semi_join")]
pub struct LeftSemiJoinIndices<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LeftSemiJoinIndices<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_semi_join_indices(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::left_anti_join_indices`]. See that method for details.
#[doc(alias = "left_anti_join")]
pub struct LeftAntiJoinIndices<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LeftAntiJoinIndices<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::left_anti_join_indices(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::compute_column`]. See that method for details.
pub struct ComputeColumn<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    root: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ComputeColumn<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ast::ffi::ast_compute_column(
            self.table.0.as_unique_ptr(),
            self.root.tree.raw(),
            self.root.index,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::filter_with_ast`]. See that method for details.
pub struct FilterWithAst<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    predicate_table: &'a RawTable<Raw>,
    filter_table: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for FilterWithAst<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::ast_filter(
            self.predicate_table.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.filter_table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::conditional_inner_join`]. See that method for details.
pub struct ConditionalInnerJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalInnerJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_inner_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::conditional_inner_join_size`]. See that method for details.
pub struct ConditionalInnerJoinSize<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalInnerJoinSize<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        cudf_sys::ast::ffi::conditional_inner_join_size(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

/// Builder for [`Table::conditional_left_join`]. See that method for details.
pub struct ConditionalLeftJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalLeftJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_left_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::conditional_left_join_size`]. See that method for details.
pub struct ConditionalLeftJoinSize<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalLeftJoinSize<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        cudf_sys::ast::ffi::conditional_left_join_size(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

/// Builder for [`Table::conditional_full_join`]. See that method for details.
pub struct ConditionalFullJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalFullJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_full_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::conditional_left_semi_join`]. See that method for details.
pub struct ConditionalLeftSemiJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalLeftSemiJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_left_semi_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::conditional_left_semi_join_size`]. See that method for details.
pub struct ConditionalLeftSemiJoinSize<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalLeftSemiJoinSize<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        cudf_sys::ast::ffi::conditional_left_semi_join_size(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

/// Builder for [`Table::conditional_left_anti_join`]. See that method for details.
pub struct ConditionalLeftAntiJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalLeftAntiJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::ast::ffi::conditional_left_anti_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::conditional_left_anti_join_size`]. See that method for details.
pub struct ConditionalLeftAntiJoinSize<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    predicate: crate::ast::RootExpr<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConditionalLeftAntiJoinSize<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        cudf_sys::ast::ffi::conditional_left_anti_join_size(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.predicate.tree.raw(),
            self.predicate.index,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

/// Builder for [`Table::groupby`]. See that method for details.
pub struct Groupby<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    key_columns: &'a [i32],
    value_column: i32,
    agg: AggregationKind,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Groupby<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let result = cudf_sys::groupby::ffi::groupby_single(
            self.table.0.as_unique_ptr(),
            self.key_columns,
            self.value_column,
            self.agg.repr,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(result))
    }
}

/// Builder for [`Table::groupby_multi`]. See that method for details.
pub struct GroupbyMulti<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    aggs: &'a [AggregationKind],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for GroupbyMulti<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

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
            self.table.0.as_unique_ptr(),
            self.key_columns,
            self.value_columns,
            agg_kinds,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(result))
    }
}

/// Builder for [`Table::filter`]. See that method for details.
pub struct Filter<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Filter<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::apply_boolean_mask(
            self.table.0.as_unique_ptr(),
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::drop_nulls`]. See that method for details.
pub struct DropNulls<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for DropNulls<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nulls_all(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::gather`]. See that method for details.
pub struct Gather<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    indices: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Gather<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::gather_table(
            self.table.0.as_unique_ptr(),
            self.indices.0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::murmur3`]. See that method for details.
pub struct Murmur3<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    seed: u32,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Murmur3<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            RawColumn(cudf_sys::hashing::ffi::hash_murmur3(
                self.table.0.as_unique_ptr(),
                self.seed,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::xxhash64`]. See that method for details.
pub struct Xxhash64<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    seed: u64,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Xxhash64<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            RawColumn(cudf_sys::hashing::ffi::hash_xxhash64(
                self.table.0.as_unique_ptr(),
                self.seed,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::md5`]. See that method for details.
pub struct Md5<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Md5<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            RawColumn(cudf_sys::hashing::ffi::hash_md5(
                self.table.0.as_unique_ptr(),
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::sha256`]. See that method for details.
pub struct Sha256<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sha256<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            RawColumn(cudf_sys::hashing::ffi::hash_sha256(
                self.table.0.as_unique_ptr(),
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::merge`]. See that method for details.
pub struct Merge<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    key_columns: &'a [i32],
    orders: &'a [Order],
    null_orders: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Merge<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.null_orders);
        let tbl = cudf_sys::merge::ffi::merge_tables(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.key_columns,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(tbl))
    }
}

/// Builder for [`Table::hash_partition`]. See that method for details.
pub struct HashPartition<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    columns: &'a [i32],
    num_partitions: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for HashPartition<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::partitioning::ffi::hash_partition_table(
            self.table.0.as_unique_ptr(),
            self.columns,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(tbl))
    }
}

/// Builder for [`Table::hash_partition_offsets`]. See that method for details.
pub struct HashPartitionOffsets<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    columns: &'a [i32],
    num_partitions: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for HashPartitionOffsets<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = Vec<usize>;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let offsets = cudf_sys::partitioning::ffi::hash_partition_offsets(
            self.table.0.as_unique_ptr(),
            self.columns,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::round_robin`]. See that method for details.
pub struct RoundRobin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    num_partitions: usize,
    start: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for RoundRobin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::partitioning::ffi::round_robin_partition_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.num_partitions),
            usize_to_i32(self.start),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(tbl))
    }
}

/// Builder for [`Table::round_robin_offsets`]. See that method for details.
pub struct RoundRobinOffsets<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    num_partitions: usize,
    start: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for RoundRobinOffsets<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = Vec<usize>;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let offsets = cudf_sys::partitioning::ffi::round_robin_partition_offsets(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.num_partitions),
            usize_to_i32(self.start),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::interleave_columns`]. See that method for details.
pub struct InterleaveColumns<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for InterleaveColumns<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::reshape::ffi::interleave_columns(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(col))
    }
}

/// Builder for [`Table::tile`]. See that method for details.
pub struct Tile<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    count: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Tile<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::reshape::ffi::tile_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(tbl))
    }
}

/// Builder for [`Table::repeat`]. See that method for details.
pub struct Repeat<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    count: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Repeat<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::filling::ffi::repeat_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::encode`]. See that method for details.
pub struct Encode<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Encode<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::transform::ffi::encode_table(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(col))
    }
}

/// Builder for [`Table::encode_keys`]. See that method for details.
pub struct EncodeKeys<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for EncodeKeys<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let tbl = cudf_sys::transform::ffi::encode_keys(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(tbl))
    }
}

/// Builder for [`Table::drop_nans`]. See that method for details.
pub struct DropNans<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for DropNans<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nans(
            self.table.0.as_unique_ptr(),
            self.keys,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::drop_nulls_with_threshold`]. See that method for details.
pub struct DropNullsWithThreshold<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a [i32],
    threshold: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for DropNullsWithThreshold<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nulls_with_threshold(
            self.table.0.as_unique_ptr(),
            self.keys,
            usize_to_i32(self.threshold),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::unique`]. See that method for details.
pub struct Unique<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Unique<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::unique_table(
            self.table.0.as_unique_ptr(),
            self.keys,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::distinct`]. See that method for details.
pub struct Distinct<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Distinct<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::distinct_table(
            self.table.0.as_unique_ptr(),
            self.keys,
            0,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::stable_distinct`]. See that method for details.
pub struct StableDistinct<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for StableDistinct<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::stable_distinct_table(
            self.table.0.as_unique_ptr(),
            self.keys,
            0,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::scatter`]. See that method for details.
pub struct Scatter<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    source: &'a RawTable<Raw>,
    map: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Scatter<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::scatter_table(
            self.source.0.as_unique_ptr(),
            self.map.0,
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::reverse`]. See that method for details.
pub struct Reverse<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Reverse<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::reverse_table(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::slice`]. See that method for details.
pub struct Slice<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    begin: usize,
    end: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Slice<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::slice_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::sample`]. See that method for details.
pub struct Sample<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    n: usize,
    with_replacement: bool,
    seed: i64,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sample<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::sample_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.n),
            self.with_replacement,
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::transpose`]. See that method for details.
pub struct Transpose<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Transpose<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::reshape::ffi::transpose_table(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::lower_bound`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an `INT32`
/// [`Column`] of insertion indices.
pub struct LowerBound<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    needles: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for LowerBound<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::search::ffi::lower_bound(
            self.table.0.as_unique_ptr(),
            self.needles.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::upper_bound`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an `INT32`
/// [`Column`] of insertion indices.
pub struct UpperBound<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    needles: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for UpperBound<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::search::ffi::upper_bound(
            self.table.0.as_unique_ptr(),
            self.needles.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::explode`]. See that method for details.
pub struct Explode<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    column_idx: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Explode<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::explode_position`]. See that method for details.
pub struct ExplodePosition<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    column_idx: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ExplodePosition<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_position_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::explode_outer`]. See that method for details.
pub struct ExplodeOuter<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    column_idx: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ExplodeOuter<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_outer_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::explode_outer_position`]. See that method for details.
pub struct ExplodeOuterPosition<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    column_idx: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ExplodeOuterPosition<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::lists::ffi::explode_outer_position_table(
            self.table.0.as_unique_ptr(),
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::gather_checked`]. See that method for details.
pub struct GatherChecked<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    indices: &'a ColumnView<'a>,
    nullify_oob: bool,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for GatherChecked<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::gather_table_checked(
            self.table.0.as_unique_ptr(),
            self.indices.0,
            self.nullify_oob,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

#[doc(alias = "negative_index_policy")]
/// Builder for [`Table::gather_with_policy`]. See that method for details.
pub struct GatherWithPolicy<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    indices: &'a ColumnView<'a>,
    nullify_oob: bool,
    allow_negative: bool,
    stream: Stream,
}

impl<Raw> GatherWithPolicy<'_, Raw> {
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

impl<Raw> crate::stream::GpuOp for GatherWithPolicy<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::gather_table_with_policy(
            self.table.0.as_unique_ptr(),
            self.indices.0,
            self.nullify_oob,
            self.allow_negative,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::cross_join`]. See that method for details.
pub struct CrossJoin<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    right: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for CrossJoin<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::cross_join(
            self.table.0.as_unique_ptr(),
            self.right.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::partition_by_map`]. See that method for details.
pub struct PartitionByMap<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    partition_map: &'a ColumnView<'a>,
    num_partitions: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for PartitionByMap<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::partitioning::ffi::partition_by_map(
            self.table.0.as_unique_ptr(),
            self.partition_map.0,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::partition_by_map_offsets`]. See that method for details.
pub struct PartitionByMapOffsets<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    partition_map: &'a ColumnView<'a>,
    num_partitions: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for PartitionByMapOffsets<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = Vec<usize>;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let offsets = cudf_sys::partitioning::ffi::partition_by_map_offsets(
            self.table.0.as_unique_ptr(),
            self.partition_map.0,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::groupby_scan`]. See that method for details.
pub struct GroupbyScan<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    aggs: &'a [AggregationKind],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for GroupbyScan<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

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
            self.table.0.as_unique_ptr(),
            self.key_columns,
            self.value_columns,
            agg_kinds,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(result))
    }
}

/// Builder for [`Table::groupby_shift`]. See that method for details.
pub struct GroupbyShift<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    offsets: &'a [i32],
    fill_values: &'a [Scalar],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for GroupbyShift<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.fill_values {
            let ffi = crate::scalar::scalar_to_ffi(s)?;
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let result = cudf_sys::groupby::ffi::groupby_shift(
            self.table.0.as_unique_ptr(),
            self.key_columns,
            self.value_columns,
            self.offsets,
            list.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(result))
    }
}

/// Builder for [`Table::groupby_replace_nulls`]. See that method for details.
pub struct GroupbyReplaceNulls<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    policies: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for GroupbyReplaceNulls<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let result = cudf_sys::groupby::ffi::groupby_replace_nulls(
            self.table.0.as_unique_ptr(),
            self.key_columns,
            self.value_columns,
            self.policies,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(result))
    }
}

/// Builder for [`Table::sha1`]. See that method for details.
pub struct Sha1<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sha1<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::hashing::ffi::hash_sha1(self.table.0.as_unique_ptr(), self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::sha224`]. See that method for details.
pub struct Sha224<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sha224<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha224(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::sha384`]. See that method for details.
pub struct Sha384<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sha384<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha384(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::sha512`]. See that method for details.
pub struct Sha512<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Sha512<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_sha512(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::murmurhash3_x64_128`]. See that method for details.
pub struct MurmurHash3X64_128<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    seed: u64,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for MurmurHash3X64_128<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::hashing::ffi::hash_murmurhash3_x64_128(
            self.table.0.as_unique_ptr(),
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::xxhash_32`]. See that method for details.
pub struct XxHash32<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    seed: u32,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for XxHash32<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::hashing::ffi::hash_xxhash_32(
            self.table.0.as_unique_ptr(),
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::row_bit_count`]. See that method for details.
pub struct RowBitCount<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for RowBitCount<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::transform::ffi::row_bit_count(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::segmented_row_bit_count`]. See that method for details.
pub struct SegmentedRowBitCount<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    segment_length: i32,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for SegmentedRowBitCount<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::transform::ffi::segmented_row_bit_count(
            self.table.0.as_unique_ptr(),
            self.segment_length,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::byte_cast`]. See that method for details.
pub struct ByteCast<'a> {
    col: &'a ColumnView<'a>,
    flip_endian: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ByteCast<'_> {
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::stable_sorted_order`]. See that method for details.
pub struct StableSortedOrder<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for StableSortedOrder<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::stable_sorted_order(
            self.table.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::stable_sort`]. See that method for details.
pub struct StableSort<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for StableSort<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_sort_table(
            self.table.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::sort_by_key`]. See that method for details.
pub struct SortByKey<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for SortByKey<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::sort_by_key(
            self.table.0.as_unique_ptr(),
            self.keys.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::stable_sort_by_key`]. See that method for details.
pub struct StableSortByKey<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a RawTable<Raw>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for StableSortByKey<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_sort_by_key(
            self.table.0.as_unique_ptr(),
            self.keys.0.as_unique_ptr(),
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::segmented_sorted_order`]. See that method for details.
pub struct SegmentedSortedOrder<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for SegmentedSortedOrder<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::segmented_sorted_order(
            self.table.0.as_unique_ptr(),
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::stable_segmented_sorted_order`]. See that method for details.
pub struct StableSegmentedSortedOrder<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for StableSegmentedSortedOrder<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::stable_segmented_sorted_order(
            self.table.0.as_unique_ptr(),
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::segmented_sort_by_key`]. See that method for details.
pub struct SegmentedSortByKey<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a RawTable<Raw>,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for SegmentedSortByKey<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::segmented_sort_by_key(
            self.table.0.as_unique_ptr(),
            self.keys.0.as_unique_ptr(),
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::stable_segmented_sort_by_key`]. See that method for details.
pub struct StableSegmentedSortByKey<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a RawTable<Raw>,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for StableSegmentedSortByKey<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_segmented_sort_by_key(
            self.table.0.as_unique_ptr(),
            self.keys.0.as_unique_ptr(),
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::quantiles`]. See that method for details.
pub struct Quantiles<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    q: &'a [f64],
    interp: crate::quantile::Interpolation,
    is_sorted: bool,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Quantiles<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::quantile::ffi::quantiles_table(
            self.table.0.as_unique_ptr(),
            self.q,
            self.interp.repr,
            self.is_sorted,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::boolean_mask_scatter`]. See that method for details.
pub struct BooleanMaskScatter<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    source: &'a RawTable<Raw>,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for BooleanMaskScatter<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::copying::ffi::boolean_mask_scatter_table(
            self.source.0.as_unique_ptr(),
            self.table.0.as_unique_ptr(),
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::scatter_scalars`]. See that method for details.
pub struct ScatterScalars<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    scalars: &'a [Scalar],
    map: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ScatterScalars<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.scalars {
            let ffi = crate::scalar::scalar_to_ffi(s)?;
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::copying::ffi::scatter_scalars(
            list.pin_mut(),
            self.map.0,
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::boolean_mask_scatter_scalars`]. See that method for details.
pub struct BooleanMaskScatterScalars<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    scalars: &'a [Scalar],
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for BooleanMaskScatterScalars<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.scalars {
            let ffi = crate::scalar::scalar_to_ffi(s)?;
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::copying::ffi::boolean_mask_scatter_scalars(
            list.pin_mut(),
            self.table.0.as_unique_ptr(),
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::repeat_by_column`]. See that method for details.
pub struct RepeatByColumn<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    counts: &'a ColumnView<'a>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for RepeatByColumn<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::filling::ffi::repeat_table_column(
            self.table.0.as_unique_ptr(),
            self.counts.0,
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::distinct_count`]. See that method for details.
pub struct DistinctCount<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    nulls_equal: bool,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for DistinctCount<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            let ne = i32::from(!self.nulls_equal); // EQUAL=0, UNEQUAL=1
            i32_to_usize(cudf_sys::compaction::ffi::distinct_count_table(
                self.table.0.as_unique_ptr(),
                ne,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::bitmask_and_to_bools`]. See that method for details.
pub struct BitmaskAndToBools<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for BitmaskAndToBools<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::bitmask_and_to_bools(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::bitmask_or_to_bools`]. See that method for details.
pub struct BitmaskOrToBools<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for BitmaskOrToBools<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::ffi::bitmask_or_to_bools(self.table.0.as_unique_ptr(), self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::unique_count`]. See that method for details.
pub struct UniqueCount<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    nulls_equal: bool,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for UniqueCount<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            let ne = i32::from(!self.nulls_equal);
            i32_to_usize(cudf_sys::compaction::ffi::unique_count_table(
                self.table.0.as_unique_ptr(),
                ne,
                self.stream.as_raw(),
            ))
        })
    }
}

/// Builder for [`Table::drop_nans_with_threshold`]. See that method for details.
pub struct DropNansWithThreshold<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    keys: &'a [i32],
    threshold: usize,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for DropNansWithThreshold<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::compaction::ffi::drop_nans_with_threshold(
            self.table.0.as_unique_ptr(),
            self.keys,
            usize_to_i32(self.threshold),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::approx_distinct_count`]. See that method for details.
pub struct ApproxDistinctCount<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    precision: i32,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ApproxDistinctCount<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok({
            cudf_sys::compaction::ffi::approx_distinct_count(
                self.table.0.as_unique_ptr(),
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
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::io::ffi::from_dlpack(self.managed_tensor_ptr, self.stream.as_raw())?;
        Ok(RawTable(t))
    }
}

/// Builder for [`Table::to_dlpack`]. See that method for details.
pub struct ToDlpack<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ToDlpack<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        Ok(cudf_sys::io::ffi::to_dlpack(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::concatenate_columns`]. See that method for details.
pub struct ConcatenateTableColumns<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConcatenateTableColumns<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::concatenate::ffi::concatenate_columns(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Table::concatenate`]. See that method for details.
pub struct ConcatenateWith<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    other: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ConcatenateWith<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::concatenate::ffi::concatenate_tables(
            self.table.0.as_unique_ptr(),
            self.other.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(RawTable(t))
    }
}

#[path = "table/join_impl.rs"]
mod join_impl;
#[path = "table/sort_impl.rs"]
mod sort_impl;
#[path = "table/table_impl.rs"]
mod table_impl;

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
pub struct Columns<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    index: usize,
    len: usize,
}

impl<'a, Raw> Iterator for Columns<'a, Raw>
where
    Raw: TableOwner,
{
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

impl<Raw> ExactSizeIterator for Columns<'_, Raw> where Raw: TableOwner {}

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
    pub fn push_column<Raw>(&mut self, col: RawColumn<Raw>)
    where
        Raw: crate::column::IntoColumnRaw,
    {
        cudf_sys::ffi::table_builder_add_column(self.0.pin_mut(), col.0.into_unique_ptr());
    }

    /// Appends an allocator-bound column, preserving its allocator lifetime in
    /// the final table built from this builder.
    pub fn push_owned_column<Brand>(&mut self, col: crate::column::BoundColumn<'_, Brand>) {
        self.push_column(col);
    }

    /// Consumes the builder and returns a [`Table`] owning all added
    /// columns.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have mismatched row counts or if
    /// a GPU error occurs.
    pub fn build(mut self) -> crate::Result<UnboundTable> {
        let tbl = cudf_sys::ffi::table_builder_build(self.0.pin_mut())?;
        Ok(RawTable(tbl))
    }

    /// Builds a table in `alloc`, preserving allocator provenance.
    pub fn build_in<'ctx, Brand>(
        mut self,
        alloc: &Allocator<'ctx, Brand>,
    ) -> crate::Result<BoundTable<'ctx, Brand>> {
        alloc.with_current(|| {
            let tbl = cudf_sys::ffi::table_builder_build(self.0.pin_mut())?;
            Ok(RawTable(alloc.bind(tbl)))
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
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
    fn table_builder_rejects_mismatched_row_counts() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let c2 = Column::from_scalar(&Scalar::from_i32(2), 3).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        assert!(builder.build().is_err());
    }

    #[test]
    fn sort_rejects_mismatched_order_lengths() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 2).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        assert!(
            table
                .sort(&[Order::ASCENDING, Order::DESCENDING], &[NullOrder::BEFORE])
                .call()
                .is_err()
        );
        assert!(
            table
                .sorted_order(&[Order::ASCENDING], &[NullOrder::BEFORE, NullOrder::AFTER])
                .call()
                .is_err()
        );
        assert!(
            table
                .is_sorted(&[Order::ASCENDING, Order::DESCENDING], &[NullOrder::BEFORE])
                .call()
                .is_err()
        );
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
