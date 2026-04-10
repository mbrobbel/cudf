#![allow(clippy::wildcard_imports)]
use super::*;

// ---------------------------------------------------------------------------
// impl UnboundTable
// ---------------------------------------------------------------------------

impl UnboundTable {
    /// Creates a table from a vector of columns.
    ///
    /// All columns must have the same row count. This is a convenience
    /// constructor that builds a [`TableBuilder`] internally. Ownership of
    /// each [`Column`] is transferred into the new table.
    ///
    /// # Examples
    ///
    /// ```no_run
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
    pub fn from_columns(columns: Vec<UnboundColumn>) -> Result<Self> {
        let mut builder = TableBuilder::new();
        for col in columns {
            builder.push_column(col);
        }
        builder.build()
    }

    /// Binds this unbound table to `alloc`'s context lifetime.
    ///
    /// This is a transitional bridge from the legacy unbound owner surface to
    /// the explicit allocator-bound safe path.
    pub fn into_owned_in<'ctx>(
        self,
        alloc: &rmm::gpu_context::Allocator<'ctx>,
    ) -> OwnedTable<'ctx> {
        RawTable(alloc.bind(self.0))
    }

    /// Creates a table in `alloc` from allocator-bound columns.
    pub fn from_columns_in<'ctx, Brand>(
        alloc: &rmm::gpu_context::Allocator<'ctx, Brand>,
        columns: Vec<crate::column::BoundColumn<'ctx, Brand>>,
    ) -> Result<crate::table::BoundTable<'ctx, Brand>> {
        let mut builder = TableBuilder::new();
        for col in columns {
            builder.push_owned_column(col);
        }
        builder.build_in(alloc)
    }
}

impl RawTable<rmm::gpu_context::ContextBound<'_, (), UniquePtr<cudf_sys::ffi::Table>>> {
    /// Creates an unbound table from already-owned columns.
    pub fn from_columns(columns: Vec<UnboundColumn>) -> Result<UnboundTable> {
        UnboundTable::from_columns(columns)
    }

    /// Creates a context-bound table from allocator-bound columns.
    pub fn from_columns_in<'ctx, Brand>(
        alloc: &rmm::gpu_context::Allocator<'ctx, Brand>,
        columns: Vec<crate::column::BoundColumn<'ctx, Brand>>,
    ) -> Result<crate::table::BoundTable<'ctx, Brand>> {
        UnboundTable::from_columns_in(alloc, columns)
    }

    #[allow(clippy::should_implement_trait)]
    /// Creates an empty unbound table.
    pub fn default() -> UnboundTable {
        UnboundTable::default()
    }
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
    #[doc(alias = "num_columns")]
    /// Returns the number of columns in this table.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::table::Table;
    ///
    /// let table = Table::default();
    /// assert_eq!(table.columns_len(), 0);
    /// ```
    pub fn columns_len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::table_num_columns(self.0.as_unique_ptr()))
    }

    #[doc(alias = "num_rows")]
    /// Returns the number of rows in this table.
    ///
    /// All columns in a table share the same row count.
    ///
    /// # Examples
    ///
    /// ```no_run
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
        i32_to_usize(cudf_sys::ffi::table_num_rows(self.0.as_unique_ptr()))
    }

    /// Returns `true` if the table has zero columns.
    ///
    /// Note that a table can have zero columns but still report zero rows
    /// (from [`Default::default`]). A table with columns that each have zero
    /// rows is *not* empty by this definition.
    ///
    /// # Examples
    ///
    /// ```no_run
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
    /// ```no_run
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
        cudf_sys::ffi::table_alloc_size(self.0.as_unique_ptr())
    }

    /// Returns an immutable view of the column at `index` (zero-based).
    ///
    /// The returned [`ColumnView`] borrows from this table and cannot
    /// outlive it.
    ///
    /// # Examples
    ///
    /// ```no_run
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
        let view =
            cudf_sys::ffi::table_get_column_view(self.0.as_unique_ptr(), usize_to_i32(index))?;
        Ok(ColumnView(view))
    }

    /// Returns an iterator over all columns as [`ColumnView`]s.
    ///
    /// The iterator yields views in column-index order (0, 1, 2, ...) and
    /// implements [`ExactSizeIterator`].
    ///
    /// # Examples
    ///
    /// ```no_run
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
    pub fn columns(&self) -> Columns<'_, Raw> {
        Columns {
            table: self,
            index: 0,
            len: self.columns_len(),
        }
    }
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
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
    /// ```no_run
    /// use cudf::ast::ExpressionTree;
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let table = Table::from_columns(vec![
    ///     Column::from_slice_i32(&[1, 2, 3]).call()?,
    ///     Column::from_slice_i32(&[10, 20, 30]).call()?,
    /// ])?;
    /// let mut tree = ExpressionTree::new();
    /// let a = tree.col(0);
    /// let b = tree.col(1);
    /// let sum = tree.add(a, b);
    ///
    /// let result = table.compute_column(tree.root(sum)?).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds, if the
    /// expression is invalid, or if a GPU error occurs.
    pub fn compute_column<'a>(&'a self, root: crate::ast::RootExpr<'a>) -> ComputeColumn<'a, Raw> {
        ComputeColumn {
            table: self,
            root,
            stream: Stream::default_stream(),
        }
    }
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
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
    /// ```no_run
    /// use cudf::ast::{ExpressionTree, TableSide};
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let left = Table::from_columns(vec![
    ///     Column::from_slice_i32(&[1, 2, 3]).call()?,
    ///     Column::from_slice_i32(&[10, 20, 30]).call()?,
    /// ])?;
    /// let right = Table::from_columns(vec![
    ///     Column::from_slice_i32(&[2, 3, 4]).call()?,
    ///     Column::from_slice_i32(&[200, 300, 400]).call()?,
    /// ])?;
    /// let mut tree = ExpressionTree::new();
    /// let lc = tree.col_in(0, TableSide::Left);
    /// let rc = tree.col_in(0, TableSide::Right);
    /// let pred = tree.eq(lc, rc);
    ///
    /// let result = left.conditional_inner_join(&right, tree.root(pred)?).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or a GPU error occurs.
    pub fn conditional_inner_join<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalInnerJoin<'a, Raw> {
        ConditionalInnerJoin {
            table: self,
            right,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the exact output row count for a conditional inner join.
    pub fn conditional_inner_join_size<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalInnerJoinSize<'a, Raw> {
        ConditionalInnerJoinSize {
            table: self,
            right,
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
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalLeftJoin<'a, Raw> {
        ConditionalLeftJoin {
            table: self,
            right,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the exact output row count for a conditional left join.
    pub fn conditional_left_join_size<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalLeftJoinSize<'a, Raw> {
        ConditionalLeftJoinSize {
            table: self,
            right,
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
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalFullJoin<'a, Raw> {
        ConditionalFullJoin {
            table: self,
            right,
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
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalLeftSemiJoin<'a, Raw> {
        ConditionalLeftSemiJoin {
            table: self,
            right,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the exact output row count for a conditional left semi join.
    pub fn conditional_left_semi_join_size<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalLeftSemiJoinSize<'a, Raw> {
        ConditionalLeftSemiJoinSize {
            table: self,
            right,
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
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalLeftAntiJoin<'a, Raw> {
        ConditionalLeftAntiJoin {
            table: self,
            right,
            predicate,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the exact output row count for a conditional left anti join.
    pub fn conditional_left_anti_join_size<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> ConditionalLeftAntiJoinSize<'a, Raw> {
        ConditionalLeftAntiJoinSize {
            table: self,
            right,
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
    ) -> Groupby<'a, Raw> {
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
    ) -> GroupbyMulti<'a, Raw> {
        GroupbyMulti {
            table: self,
            key_columns,
            value_columns,
            aggs,
            stream: Stream::default_stream(),
        }
    }
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
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
    pub fn filter<'a>(&'a self, mask: &'a ColumnView<'a>) -> Filter<'a, Raw> {
        Filter {
            table: self,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Filters `self` using an AST predicate evaluated against `predicate_table`.
    ///
    /// This wraps libcudf's AST-based table filter. In the common case where
    /// the predicate is evaluated against the same table being filtered, pass
    /// `self` as `predicate_table`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::ast::ExpressionTree;
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    /// use cudf::table::Table;
    ///
    /// let table = Table::from_columns(vec![
    ///     Column::from_slice_i32(&[1, 2, 3]).call()?,
    ///     Column::from_slice_i32(&[10, 20, 30]).call()?,
    /// ])?;
    ///
    /// let mut tree = ExpressionTree::new();
    /// let col = tree.col(0);
    /// let threshold = tree.lit_i32(1);
    /// let pred = tree.gt(col, threshold);
    ///
    /// let result = table.filter_with_ast(&table, tree.root(pred)?).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the predicate is invalid or if it does not produce
    /// a boolean result.
    pub fn filter_with_ast<'a>(
        &'a self,
        predicate_table: &'a RawTable<Raw>,
        predicate: crate::ast::RootExpr<'a>,
    ) -> FilterWithAst<'a, Raw> {
        FilterWithAst {
            predicate_table,
            filter_table: self,
            predicate,
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
    pub fn drop_nulls(&self) -> DropNulls<'_, Raw> {
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
    pub fn gather<'a>(&'a self, indices: &'a ColumnView<'a>) -> Gather<'a, Raw> {
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
    pub fn empty_like(&self) -> UnboundTable {
        RawTable(cudf_sys::copying::ffi::empty_like_table(
            self.0.as_unique_ptr(),
        ))
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
    pub fn murmur3(&self, seed: u32) -> Murmur3<'_, Raw> {
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
    pub fn xxhash64(&self, seed: u64) -> Xxhash64<'_, Raw> {
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
    pub fn md5(&self) -> Md5<'_, Raw> {
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
    pub fn sha256(&self) -> Sha256<'_, Raw> {
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
    pub fn sha1(&self) -> Sha1<'_, Raw> {
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
    pub fn murmurhash3_x64_128(&self, seed: u64) -> MurmurHash3X64_128<'_, Raw> {
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
    pub fn xxhash_32(&self, seed: u32) -> XxHash32<'_, Raw> {
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
    pub fn sha224(&self) -> Sha224<'_, Raw> {
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
    pub fn sha384(&self) -> Sha384<'_, Raw> {
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
    pub fn sha512(&self) -> Sha512<'_, Raw> {
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
    pub fn row_bit_count(&self) -> RowBitCount<'_, Raw> {
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
    pub fn segmented_row_bit_count(&self, segment_length: i32) -> SegmentedRowBitCount<'_, Raw> {
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
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
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
        right: &'a RawTable<Raw>,
        key_columns: &'a [i32],
        orders: &'a [Order],
        null_orders: &'a [NullOrder],
    ) -> Merge<'a, Raw> {
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
    ) -> HashPartition<'a, Raw> {
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
    ) -> HashPartitionOffsets<'a, Raw> {
        HashPartitionOffsets {
            table: self,
            columns,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
    /// Distributes rows across `num_partitions` in round-robin order.
    ///
    /// Row assignment begins at partition `start` and cycles through all
    /// partitions. The output table has rows reordered so each partition's
    /// rows are contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn round_robin(&self, num_partitions: usize, start: usize) -> RoundRobin<'_, Raw> {
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
    ) -> RoundRobinOffsets<'_, Raw> {
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
    pub fn interleave_columns(&self) -> InterleaveColumns<'_, Raw> {
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
    pub fn tile(&self, count: usize) -> Tile<'_, Raw> {
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
    pub fn repeat(&self, count: usize) -> Repeat<'_, Raw> {
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
    pub fn encode(&self) -> Encode<'_, Raw> {
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
    pub fn encode_keys(&self) -> EncodeKeys<'_, Raw> {
        EncodeKeys {
            table: self,
            stream: Stream::default_stream(),
        }
    }
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
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
    pub fn drop_nans<'a>(&'a self, keys: &'a [i32]) -> DropNans<'a, Raw> {
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
    ) -> DropNullsWithThreshold<'a, Raw> {
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
    pub fn unique<'a>(&'a self, keys: &'a [i32]) -> Unique<'a, Raw> {
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
    pub fn distinct<'a>(&'a self, keys: &'a [i32]) -> Distinct<'a, Raw> {
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
    pub fn stable_distinct<'a>(&'a self, keys: &'a [i32]) -> StableDistinct<'a, Raw> {
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
        source: &'a RawTable<Raw>,
        scatter_map: &'a ColumnView<'a>,
    ) -> Scatter<'a, Raw> {
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
    pub fn reverse(&self) -> Reverse<'_, Raw> {
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
    pub fn slice(&self, begin: usize, end: usize) -> Slice<'_, Raw> {
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
    pub fn sample(&self, n: usize, with_replacement: bool, seed: i64) -> Sample<'_, Raw> {
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
    pub fn transpose(&self) -> Transpose<'_, Raw> {
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
        needles: &'a RawTable<Raw>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> LowerBound<'a, Raw> {
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
        needles: &'a RawTable<Raw>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> UpperBound<'a, Raw> {
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
    ) -> StableSortedOrder<'a, Raw> {
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
    ) -> StableSort<'a, Raw> {
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
        keys: &'a RawTable<Raw>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SortByKey<'a, Raw> {
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
        keys: &'a RawTable<Raw>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSortByKey<'a, Raw> {
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
    ) -> SegmentedSortedOrder<'a, Raw> {
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
    ) -> StableSegmentedSortedOrder<'a, Raw> {
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
        keys: &'a RawTable<Raw>,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SegmentedSortByKey<'a, Raw> {
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
        keys: &'a RawTable<Raw>,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSegmentedSortByKey<'a, Raw> {
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
    ) -> Quantiles<'a, Raw> {
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
        source: &'a RawTable<Raw>,
        mask: &'a ColumnView<'a>,
    ) -> BooleanMaskScatter<'a, Raw> {
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
    ) -> ScatterScalars<'a, Raw> {
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
    ) -> BooleanMaskScatterScalars<'a, Raw> {
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
    pub fn repeat_by_column<'a>(&'a self, counts: &'a ColumnView<'a>) -> RepeatByColumn<'a, Raw> {
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
    pub fn explode(&self, column_idx: usize) -> Explode<'_, Raw> {
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
    pub fn explode_position(&self, column_idx: usize) -> ExplodePosition<'_, Raw> {
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
    pub fn explode_outer(&self, column_idx: usize) -> ExplodeOuter<'_, Raw> {
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
    pub fn explode_outer_position(&self, column_idx: usize) -> ExplodeOuterPosition<'_, Raw> {
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
    ) -> GatherChecked<'a, Raw> {
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
    pub fn gather_with_policy<'a>(
        &'a self,
        indices: &'a ColumnView<'a>,
    ) -> GatherWithPolicy<'a, Raw> {
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
    pub fn cross_join<'a>(&'a self, right: &'a RawTable<Raw>) -> CrossJoin<'a, Raw> {
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
    ) -> PartitionByMap<'a, Raw> {
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
    ) -> PartitionByMapOffsets<'a, Raw> {
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
    pub fn distinct_count(&self, nulls_equal: bool) -> DistinctCount<'_, Raw> {
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
    pub fn bitmask_and_to_bools(&self) -> BitmaskAndToBools<'_, Raw> {
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
    pub fn bitmask_or_to_bools(&self) -> BitmaskOrToBools<'_, Raw> {
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
    ) -> GroupbyScan<'a, Raw> {
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
    ) -> GroupbyShift<'a, Raw> {
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
    ) -> GroupbyReplaceNulls<'a, Raw> {
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
    pub fn unique_count(&self, nulls_equal: bool) -> UniqueCount<'_, Raw> {
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
    ) -> DropNansWithThreshold<'a, Raw> {
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
    pub fn approx_distinct_count(&self, precision: i32) -> ApproxDistinctCount<'_, Raw> {
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
        cudf_sys::ffi::table_has_nested_columns(self.0.as_unique_ptr())
    }

    /// Returns `true` if any nested (child) column within this table
    /// contains null values.
    ///
    /// Only inspects children of `LIST` and `STRUCT` columns; top-level
    /// nulls are not considered. This is a metadata check and does not
    /// launch GPU work.
    pub fn has_nested_nulls(&self) -> bool {
        cudf_sys::ffi::table_has_nested_nulls(self.0.as_unique_ptr())
    }

    /// Returns `true` if any nested (child) column within this table has
    /// a null mask allocated (i.e., is nullable), regardless of whether it
    /// actually contains null values.
    ///
    /// This is a metadata check and does not launch GPU work.
    pub fn has_nested_nullable_columns(&self) -> bool {
        cudf_sys::ffi::table_has_nested_nullable_columns(self.0.as_unique_ptr())
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
    pub fn to_dlpack(&self) -> ToDlpack<'_, Raw> {
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
    pub fn concatenate_columns(&self) -> ConcatenateTableColumns<'_, Raw> {
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
    pub fn concatenate<'a>(&'a self, other: &'a RawTable<Raw>) -> ConcatenateWith<'a, Raw> {
        ConcatenateWith {
            table: self,
            other,
            stream: Stream::default_stream(),
        }
    }
}
