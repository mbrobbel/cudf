// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Join operations on GPU tables.
//!
//! This module provides SQL-style join operations that combine two tables
//! based on equality of key columns. Each join variant determines which rows
//! appear in the output:
//!
//! | Join type | Output rows |
//! |---|---|
//! | [`inner_join`](crate::table::Table::inner_join) | Only rows with matching keys in both tables |
//! | [`left_join`](crate::table::Table::left_join) | All left rows; right columns nulled where no match |
//! | [`full_join`](crate::table::Table::full_join) | All rows from both sides; nulls where no match |
//! | [`left_semi_join`](crate::table::Table::left_semi_join) | Left rows that have a match in right (left columns only) |
//! | [`left_anti_join`](crate::table::Table::left_anti_join) | Left rows that have no match in right (left columns only) |
//!
//! All join methods take `left_on` and `right_on` slices of zero-based column
//! indices that identify the key columns. The key columns at corresponding
//! positions must have compatible types.
//!
//! For inner, left, and full joins the output table contains all columns from
//! the left table followed by all columns from the right table. For semi and
//! anti joins the output contains only the left table's columns.
//!
//! Additionally, the [`MarkJoin`] type provides a stateful hash join that
//! builds a hash table once from a "build" table and can be probed multiple
//! times with different "probe" tables for semi and anti joins.
//!
//! Joins are available as methods on [`Table`]:
//!
//! ```ignore
//! use cudf::stream::GpuOp;
//!
//! // Inner join on column 0
//! let result = left.inner_join(&right, &[0], &[0]).call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

use cxx::UniquePtr;
use rmm::gpu_context::ContextBound;

use crate::error::Result;
use crate::stream::Stream;
use crate::table::UnboundTable;

#[doc(hidden)]
pub trait UniquePtrOwner<T: cxx::memory::UniquePtrTarget> {
    fn as_unique_ptr(&self) -> &UniquePtr<T>;
}

impl<T: cxx::memory::UniquePtrTarget> UniquePtrOwner<T> for UniquePtr<T> {
    fn as_unique_ptr(&self) -> &UniquePtr<T> {
        self
    }
}

impl<Brand, T: cxx::memory::UniquePtrTarget> UniquePtrOwner<T>
    for ContextBound<'_, Brand, UniquePtr<T>>
{
    fn as_unique_ptr(&self) -> &UniquePtr<T> {
        self
    }
}

#[doc(alias = "mark_join")]
/// A stateful mark-based hash join that builds a hash table once and supports
/// repeated semi-join and anti-join probes.
///
/// `MarkJoin` is constructed from a build table and a set of key column
/// indices. The hash table is built during construction. Subsequent calls to
/// [`semi_join`](MarkJoin::semi_join) and [`anti_join`](MarkJoin::anti_join)
/// probe this hash table with different probe tables, returning the matching
/// (or non-matching) rows from the **build** table.
///
/// This is useful when the same build table is joined against many probe
/// tables, as the hash table construction cost is paid only once.
///
/// # Examples
///
/// ```ignore
/// use cudf::join::MarkJoin;
/// use cudf::stream::GpuOp;
///
/// let build = /* ... */;
/// let joiner = MarkJoin::new(&build, &[0], true).call()?;
///
/// let result1 = joiner.semi_join(&build, &probe1, &[0]).call()?;
/// let result2 = joiner.anti_join(&build, &probe2, &[0]).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct MarkJoin<Raw = UniquePtr<cudf_sys::join::ffi::MarkJoin>>(pub(crate) Raw);

/// Builder for [`MarkJoin::new`]. See that method for details.
pub struct MarkJoinNew<'a> {
    build: &'a UnboundTable,
    keys: &'a [i32],
    compare_nulls_equal: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for MarkJoinNew<'_> {
    type Output = MarkJoin;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let inner = cudf_sys::join::ffi::mark_join_new(
            &self.build.0,
            self.keys,
            self.compare_nulls_equal,
            self.stream.as_raw(),
        )?;
        Ok(MarkJoin(inner))
    }
}

/// Builder for [`MarkJoin::semi_join`]. See that method for details.
pub struct MarkJoinSemi<'a, Raw> {
    joiner: &'a MarkJoin<Raw>,
    build: &'a UnboundTable,
    probe: &'a UnboundTable,
    probe_keys: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for MarkJoinSemi<'_, Raw>
where
    Raw: UniquePtrOwner<cudf_sys::join::ffi::MarkJoin>,
{
    type Output = crate::table::UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::mark_join_semi(
            self.joiner.0.as_unique_ptr(),
            &self.build.0,
            &self.probe.0,
            self.probe_keys,
            self.stream.as_raw(),
        )?;
        Ok(crate::table::RawTable(t))
    }
}

/// Builder for [`MarkJoin::anti_join`]. See that method for details.
pub struct MarkJoinAnti<'a, Raw> {
    joiner: &'a MarkJoin<Raw>,
    build: &'a UnboundTable,
    probe: &'a UnboundTable,
    probe_keys: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for MarkJoinAnti<'_, Raw>
where
    Raw: UniquePtrOwner<cudf_sys::join::ffi::MarkJoin>,
{
    type Output = crate::table::UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let t = cudf_sys::join::ffi::mark_join_anti(
            self.joiner.0.as_unique_ptr(),
            &self.build.0,
            &self.probe.0,
            self.probe_keys,
            self.stream.as_raw(),
        )?;
        Ok(crate::table::RawTable(t))
    }
}

impl MarkJoin {
    /// Creates a new `MarkJoin` by building a hash table from the specified
    /// key columns of the build table.
    ///
    /// # Arguments
    ///
    /// * `build` -- The table whose key columns form the hash table.
    /// * `keys` -- Zero-based column indices identifying the key columns in
    ///   `build`.
    /// * `compare_nulls_equal` -- If `true`, null key values are considered
    ///   equal when matching. If `false`, nulls never match.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column indices are out of bounds or a GPU
    /// error occurs.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<'a>(
        build: &'a UnboundTable,
        keys: &'a [i32],
        compare_nulls_equal: bool,
    ) -> MarkJoinNew<'a> {
        MarkJoinNew {
            build,
            keys,
            compare_nulls_equal,
            stream: Stream::default_stream(),
        }
    }
}

impl<Raw> MarkJoin<Raw>
where
    Raw: UniquePtrOwner<cudf_sys::join::ffi::MarkJoin>,
{
    /// Performs a semi-join probe: returns rows from `build` whose key columns
    /// have at least one match in the `probe` table.
    ///
    /// The output table has the same columns as `build` and contains only the
    /// matching rows.
    ///
    /// # Arguments
    ///
    /// * `build` -- The original build table (must be the same table used
    ///   during construction).
    /// * `probe` -- The table to probe against the build hash table.
    /// * `probe_keys` -- Zero-based column indices identifying the key columns
    ///   in `probe`. Must have the same length as the `keys` used during
    ///   construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn semi_join<'a>(
        &'a self,
        build: &'a UnboundTable,
        probe: &'a UnboundTable,
        probe_keys: &'a [i32],
    ) -> MarkJoinSemi<'a, Raw> {
        MarkJoinSemi {
            joiner: self,
            build,
            probe,
            probe_keys,
            stream: Stream::default_stream(),
        }
    }

    /// Performs an anti-join probe: returns rows from `build` whose key columns
    /// have NO matches in the `probe` table.
    ///
    /// The output table has the same columns as `build` and contains only the
    /// non-matching rows.
    ///
    /// # Arguments
    ///
    /// * `build` -- The original build table (must be the same table used
    ///   during construction).
    /// * `probe` -- The table to probe against the build hash table.
    /// * `probe_keys` -- Zero-based column indices identifying the key columns
    ///   in `probe`. Must have the same length as the `keys` used during
    ///   construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the key column types are incompatible or a GPU
    /// error occurs.
    pub fn anti_join<'a>(
        &'a self,
        build: &'a UnboundTable,
        probe: &'a UnboundTable,
        probe_keys: &'a [i32],
    ) -> MarkJoinAnti<'a, Raw> {
        MarkJoinAnti {
            joiner: self,
            build,
            probe,
            probe_keys,
            stream: Stream::default_stream(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MarkJoin;
    use crate::column::Column;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    /// Helper to build a two-column table from i32 slices.
    fn make_two_col_table(keys: &[i32], vals: &[i32]) -> crate::table::UnboundTable {
        let key_col = Column::from_slice_i32(keys).call().unwrap();
        let val_col = Column::from_slice_i32(vals).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(key_col);
        builder.push_column(val_col);
        builder.build().unwrap()
    }

    #[test]
    fn inner_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn left_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left.left_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn full_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left.full_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn left_semi_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left
            .left_semi_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn left_anti_join_basic() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let result = left
            .left_anti_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn inner_join_empty_result() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[4, 5, 6], &[400, 500, 600]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn inner_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn inner_join_duplicate_keys() {
        let left = make_two_col_table(&[1, 1, 2], &[10, 11, 20]);
        let right = make_two_col_table(&[1, 1, 3], &[100, 101, 300]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn left_join_no_matches() {
        let left = make_two_col_table(&[1, 2], &[10, 20]);
        let right = make_two_col_table(&[3, 4], &[300, 400]);
        let result = left.left_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn full_join_no_overlap() {
        let left = make_two_col_table(&[1, 2], &[10, 20]);
        let right = make_two_col_table(&[3, 4], &[300, 400]);
        let result = left.full_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn semi_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);
        let result = left
            .left_semi_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn anti_join_no_matches() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[4, 5, 6], &[400, 500, 600]);
        let result = left
            .left_anti_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn anti_join_all_match() {
        let left = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let right = make_two_col_table(&[1, 2, 3], &[100, 200, 300]);
        let result = left
            .left_anti_join(&right, &[0i32], &[0i32])
            .call()
            .unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn inner_join_with_f64_values() {
        let key_left = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
        let val_left = Column::from_slice_f64(&[1.1, 2.2, 3.3]).call().unwrap();
        let key_right = Column::from_slice_i32(&[2, 3, 4]).call().unwrap();
        let val_right = Column::from_slice_f64(&[20.0, 30.0, 40.0]).call().unwrap();

        let mut lb = TableBuilder::new();
        lb.push_column(key_left);
        lb.push_column(val_left);
        let left = lb.build().unwrap();

        let mut rb = TableBuilder::new();
        rb.push_column(key_right);
        rb.push_column(val_right);
        let right = rb.build().unwrap();

        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 4);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn join_single_row_tables() {
        let left = make_two_col_table(&[1], &[10]);
        let right = make_two_col_table(&[1], &[100]);
        let result = left.inner_join(&right, &[0i32], &[0i32]).call().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.columns_len(), 4);
    }

    // -- MarkJoin tests --

    #[test]
    fn mark_join_semi_basic() {
        let build = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let joiner = MarkJoin::new(&build, &[0i32], true).call().unwrap();

        let probe = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let result = joiner.semi_join(&build, &probe, &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn mark_join_anti_basic() {
        let build = make_two_col_table(&[2, 3, 4], &[200, 300, 400]);
        let joiner = MarkJoin::new(&build, &[0i32], true).call().unwrap();

        let probe = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let result = joiner.anti_join(&build, &probe, &[0i32]).call().unwrap();
        assert_eq!(result.columns_len(), 2);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn mark_join_reuse_build() {
        let build = make_two_col_table(&[1, 2], &[10, 20]);
        let joiner = MarkJoin::new(&build, &[0i32], true).call().unwrap();

        let probe1 = make_two_col_table(&[1, 3, 5], &[100, 300, 500]);
        let probe2 = make_two_col_table(&[2, 4, 6], &[200, 400, 600]);

        let r1 = joiner.semi_join(&build, &probe1, &[0i32]).call().unwrap();
        let r2 = joiner.semi_join(&build, &probe2, &[0i32]).call().unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn mark_join_no_matches() {
        let build = make_two_col_table(&[10, 20], &[100, 200]);
        let joiner = MarkJoin::new(&build, &[0i32], true).call().unwrap();

        let probe = make_two_col_table(&[1, 2, 3], &[10, 20, 30]);
        let semi = joiner.semi_join(&build, &probe, &[0i32]).call().unwrap();
        let anti = joiner.anti_join(&build, &probe, &[0i32]).call().unwrap();
        assert_eq!(semi.len(), 0);
        assert_eq!(anti.len(), 2);
    }
}
