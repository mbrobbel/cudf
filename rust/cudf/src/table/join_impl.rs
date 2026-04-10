#![allow(clippy::wildcard_imports)]
use super::*;

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
    /// Performs an inner join with `right` on the specified key columns.
    ///
    /// Matches rows where `self[left_on[i]] == right[right_on[i]]` for all
    /// key pairs. The output table contains all columns from `self` followed
    /// by all columns from `right`.
    ///
    /// `left_on` and `right_on` must have the same length. Each element is a
    /// zero-based column index with compatible types at matching positions.
    ///
    /// # Examples
    ///
    /// ```no_run
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
    /// assert_eq!(joined.len(), 2);
    /// assert_eq!(joined.columns_len(), 4);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn inner_join<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> InnerJoin<'a, Raw> {
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
    /// ```no_run
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
    /// assert_eq!(joined.len(), 3);
    /// assert_eq!(joined.columns_len(), 4);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn left_join<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftJoin<'a, Raw> {
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
    /// ```no_run
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
    /// assert_eq!(joined.len(), 3);
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
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> FullJoin<'a, Raw> {
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
    /// ```no_run
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
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result.columns_len(), 2);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn left_semi_join<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftSemiJoin<'a, Raw> {
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
    /// ```no_run
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
    /// assert_eq!(result.len(), 1);
    /// assert_eq!(result.columns_len(), 2);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn left_anti_join<'a>(
        &'a self,
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftAntiJoin<'a, Raw> {
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
    /// ```no_run
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
    /// assert_eq!(indices.columns_len(), 2);
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
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> InnerJoinIndices<'a, Raw> {
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
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftJoinIndices<'a, Raw> {
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
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> FullJoinIndices<'a, Raw> {
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
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftSemiJoinIndices<'a, Raw> {
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
        right: &'a RawTable<Raw>,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftAntiJoinIndices<'a, Raw> {
        LeftAntiJoinIndices {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }
}
