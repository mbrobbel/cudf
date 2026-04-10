// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! List column operations.
//!
//! List columns contain variable-length arrays of elements per row. The
//! [`ListExt`] trait provides per-row operations such as element extraction,
//! sorting, searching, deduplication, and gathering.
//!
//! Free functions in this module provide cross-column list operations:
//! [`list_sequences`], [`lists_concatenate_rows`], [`lists_have_overlap`],
//! [`lists_intersect_distinct`], [`lists_union_distinct`], and
//! [`lists_difference_distinct`].
//!
//! # Examples
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::lists::ListExt;
//! use cudf::stream::GpuOp;
//!
//! let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
//! let values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
//! let list_col = Column::from_lists(2, offsets, values).call()?;
//!
//! // Count elements in each list row
//! let counts = list_col.view().list_count_elements().call()?;
//!
//! // Extract the first element from each list
//! let firsts = list_col.view().list_extract_element(0).call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

use crate::column::ColumnView;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

#[path = "lists/ext_impl.rs"]
mod ext_impl;

/// Extension trait for list column operations.
///
/// Provides per-row operations on columns whose type is `LIST`. Each row in a
/// list column contains a variable-length sequence of elements of the same child
/// type.
///
/// This trait is implemented for [`ColumnView`] and is sealed -- it cannot be
/// implemented outside this crate.
///
/// All methods return builder structs that implement [`GpuOp`](crate::stream::GpuOp).
/// Call `.call()` to execute the operation, or chain `.stream(s)` first to run
/// on a non-default CUDA stream.
///
/// # Errors
///
/// Methods return an error if the input column is not a `LIST` column.
pub trait ListExt: private::Sealed {
    /// Returns the number of elements in each list row.
    ///
    /// Returns an `INT32` column where each value is the length of the
    /// corresponding list. Null list rows produce null counts.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(3, offsets, values).call()?;
    /// let counts = list_col.view().list_count_elements().call()?;
    /// // For [[1,2,3], [4,5], []] => [3, 2, 0]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_count_elements(&self) -> ListCountElements<'_>;

    /// Extracts the element at `index` from each list row.
    ///
    /// Negative indices count from the end of the list (`-1` is the last
    /// element). If `index` is out of bounds for a given row, the result for
    /// that row is null.
    ///
    /// Returns a column whose type matches the list's child type.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// // Extract the first element from each list
    /// let firsts = list_col.view().list_extract_element(0).call()?;
    ///
    /// // Extract the last element from each list
    /// let lasts = list_col.view().list_extract_element(-1).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_extract_element(&self, index: i32) -> ListExtractElement<'_>;

    /// Sorts the elements within each list row independently.
    ///
    /// When `ascending` is `true`, elements are sorted in ascending order;
    /// otherwise in descending order. When `nulls_last` is `true`, null
    /// elements are placed at the end of each sorted list; otherwise at the
    /// beginning.
    ///
    /// Returns a new `LIST` column with sorted lists. The sort is **not**
    /// guaranteed to be stable; use [`list_stable_sort`](ListExt::list_stable_sort)
    /// if stability is required.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[3, 1, 2, 5, 4]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let sorted = list_col.view().list_sort(true, true).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_sort(&self, ascending: bool, nulls_last: bool) -> ListSort<'_>;

    /// Reverses the element order within each list row.
    ///
    /// Returns a new `LIST` column where each list's elements appear in
    /// reverse order.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let reversed = list_col.view().list_reverse().call()?;
    /// // [[1,2,3], [4,5]] => [[3,2,1], [5,4]]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_reverse(&self) -> ListReverse<'_>;

    /// Tests whether each list row contains at least one null element.
    ///
    /// Returns a `BOOL8` column: `true` if the list contains any null, `false`
    /// otherwise. Null list rows produce null results.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 2]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2]).call()?;
    /// # let list_col = Column::from_lists(1, offsets, values).call()?;
    /// let has_nulls = list_col.view().list_contains_nulls().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_contains_nulls(&self) -> ListContainsNulls<'_>;

    /// Removes duplicate elements from each list row.
    ///
    /// Returns a new `LIST` column where each list contains only distinct
    /// values. Nulls are treated as equal (at most one null is kept).
    /// The order of the remaining elements is not guaranteed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 4, 6]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 1, 3, 4, 4]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let unique = list_col.view().list_distinct().call()?;
    /// // [[1,2,1,3], [4,4]] => [[1,2,3], [4]]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_distinct(&self) -> ListDistinct<'_>;

    /// Flattens one level of nesting within each list row.
    ///
    /// For a column of type `LIST(LIST(T))`, this concatenates the inner lists
    /// within each row to produce a column of type `LIST(T)`.
    ///
    /// Returns a new `LIST` column with one fewer level of nesting.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let inner_offsets = Column::from_slice_i32(&[0, 2, 3, 5]).call()?;
    /// # let inner_values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// # let inner_lists = Column::from_lists(3, inner_offsets, inner_values).call()?;
    /// # let outer_offsets = Column::from_slice_i32(&[0, 2, 3]).call()?;
    /// # let nested_list_col = Column::from_lists(2, outer_offsets, inner_lists).call()?;
    /// // [[[1,2],[3]], [[4,5]]] => [[1,2,3], [4,5]]
    /// let flat = nested_list_col.view().list_concatenate_elements().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_concatenate_elements(&self) -> ListConcatenateElements<'_>;

    /// Tests whether each list row contains a given scalar value.
    ///
    /// The `search_key` scalar must have the same type as the list's child
    /// elements. Returns a `BOOL8` column: `true` if the list contains the
    /// value, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 42, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let key = Scalar::from_i32(42);
    /// let found = list_col.view().list_contains_scalar(&key).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_contains_scalar<'a>(&'a self, search_key: &'a Scalar) -> ListContainsScalar<'a>;

    /// Tests whether each list row contains the corresponding search key.
    ///
    /// The `search_keys` column must have the same type as the list's child
    /// elements and the same number of rows. Row `i` is checked for
    /// `search_keys[i]`.
    ///
    /// Returns a `BOOL8` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// # let keys = Column::from_slice_i32(&[2, 4]).call()?;
    /// let found = list_col.view().list_contains_column(&keys.view()).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_contains_column<'a>(
        &'a self,
        search_keys: &'a ColumnView<'a>,
    ) -> ListContainsColumn<'a>;

    /// Finds the position of a scalar value within each list row.
    ///
    /// Returns an `INT32` column. If the value is found, the position
    /// (0-based) is returned; if not found, `-1` is returned. When
    /// `find_first` is `true`, the first occurrence is returned; when `false`,
    /// the last occurrence is returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 4, 6]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 42, 2, 42, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let key = Scalar::from_i32(42);
    /// let positions = list_col.view().list_index_of_scalar(&key, true).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_index_of_scalar<'a>(
        &'a self,
        search_key: &'a Scalar,
        find_first: bool,
    ) -> ListIndexOfScalar<'a>;

    /// Finds the position of each per-row search key within the corresponding list.
    ///
    /// The `search_keys` column must have the same type as the list's child
    /// elements and the same number of rows. Returns an `INT32` column with
    /// positions (`-1` if not found). When `find_first` is `true`, the first
    /// occurrence is returned; when `false`, the last.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// # let keys = Column::from_slice_i32(&[2, 5]).call()?;
    /// let positions = list_col.view().list_index_of_column(&keys.view(), true).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_index_of_column<'a>(
        &'a self,
        search_keys: &'a ColumnView<'a>,
        find_first: bool,
    ) -> ListIndexOfColumn<'a>;

    /// Gathers elements from each list row at the positions specified by a gather map.
    ///
    /// The `gather_map` is a `LIST(INT32)` column with the same number of rows.
    /// For each row, the integers in the gather map select elements from the
    /// corresponding source list. Negative indices count from the end.
    ///
    /// When `nullify_oob` is `true`, out-of-bounds indices produce null
    /// elements; when `false`, the behavior for out-of-bounds indices is
    /// undefined.
    ///
    /// Returns a new `LIST` column with the gathered elements.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// # let gather_offsets = Column::from_slice_i32(&[0, 2, 4]).call()?;
    /// # let gather_values = Column::from_slice_i32(&[0, 2, 1, 0]).call()?;
    /// # let gather_map = Column::from_lists(2, gather_offsets, gather_values).call()?;
    /// let gathered = list_col.view()
    ///     .list_segmented_gather(&gather_map.view(), true)
    ///     .call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_segmented_gather<'a>(
        &'a self,
        gather_map: &'a ColumnView<'a>,
        nullify_oob: bool,
    ) -> ListSegmentedGather<'a>;

    /// Formats a list-of-strings column into a single string per row.
    ///
    /// Each list of strings is formatted into a bracketed, comma-separated
    /// representation. The `na_rep` string is used in place of null elements.
    ///
    /// Returns a `STRING` column.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 2, 4]).call()?;
    /// # let values = Column::from_strings(&["a", "b", "c", "d"]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let formatted = list_col.view().list_format("NULL").call()?;
    /// // [["a","b"], ["c",null]] => ["[a, b]", "[c, NULL]"]
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_format<'a>(&'a self, na_rep: &'a str) -> ListFormat<'a>;

    /// Extracts an element from each list using per-row indices from a column.
    ///
    /// The `indices` column must be an `INT32` column with the same number of
    /// rows. Negative indices count from the end of the list. Out-of-bounds
    /// indices produce null results.
    ///
    /// Returns a column whose type matches the list's child type.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// # let indices = Column::from_slice_i32(&[0, -1]).call()?;
    /// let elements = list_col.view()
    ///     .list_extract_element_column(&indices.view())
    ///     .call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_extract_element_column<'a>(
        &'a self,
        indices: &'a ColumnView<'a>,
    ) -> ListExtractElementColumn<'a>;

    /// Sorts the elements within each list row with guaranteed stability.
    ///
    /// Behaves identically to [`list_sort`](ListExt::list_sort) but guarantees
    /// that equal elements retain their original relative order.
    ///
    /// When `ascending` is `true`, elements are sorted in ascending order.
    /// When `nulls_last` is `true`, null elements are placed at the end.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[3, 1, 2, 5, 4]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// let sorted = list_col.view().list_stable_sort(true, true).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_stable_sort(&self, ascending: bool, nulls_last: bool) -> ListStableSort<'_>;

    /// Filters elements within each list row using a boolean mask.
    ///
    /// The `boolean_mask` is a `LIST(BOOL8)` column with the same number of
    /// rows. For each row, only elements whose corresponding mask value is
    /// `true` are kept. The mask lists must have the same lengths as the source
    /// lists.
    ///
    /// Returns a new `LIST` column with filtered elements.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use cudf::column::Column;
    /// use cudf::lists::ListExt;
    /// use cudf::stream::GpuOp;
    ///
    /// # let offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let values = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    /// # let list_col = Column::from_lists(2, offsets, values).call()?;
    /// # let mask_offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
    /// # let mask_values = Column::from_slice_bool(&[true, false, true, true, false]).call()?;
    /// # let mask = Column::from_lists(2, mask_offsets, mask_values).call()?;
    /// let filtered = list_col.view()
    ///     .list_apply_boolean_mask(&mask.view())
    ///     .call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn list_apply_boolean_mask<'a>(
        &'a self,
        boolean_mask: &'a ColumnView<'a>,
    ) -> ListApplyBooleanMask<'a>;
}

// ---------------------------------------------------------------------------
// Builder structs — trait methods
// ---------------------------------------------------------------------------

/// Builder for [`ListExt::list_count_elements`].
///
/// Created by [`ListExt::list_count_elements`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListCountElements<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListCountElements<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_count_elements(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_extract_element`].
///
/// Created by [`ListExt::list_extract_element`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListExtractElement<'a> {
    view: &'a ColumnView<'a>,
    index: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for ListExtractElement<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_extract_element(
            self.view.0,
            self.index,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_sort`].
///
/// Created by [`ListExt::list_sort`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListSort<'a> {
    view: &'a ColumnView<'a>,
    ascending: bool,
    nulls_last: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ListSort<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_sort(
            self.view.0,
            self.ascending,
            self.nulls_last,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_reverse`].
///
/// Created by [`ListExt::list_reverse`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListReverse<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListReverse<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_reverse(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_contains_nulls`].
///
/// Created by [`ListExt::list_contains_nulls`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListContainsNulls<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListContainsNulls<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_contains_nulls(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_distinct`].
///
/// Created by [`ListExt::list_distinct`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListDistinct<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListDistinct<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_distinct(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_concatenate_elements`].
///
/// Created by [`ListExt::list_concatenate_elements`]. Call `.call()` to
/// execute, or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListConcatenateElements<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListConcatenateElements<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::lists::ffi::lists_concatenate_elements(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_contains_scalar`].
///
/// Created by [`ListExt::list_contains_scalar`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListContainsScalar<'a> {
    view: &'a ColumnView<'a>,
    search_key: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for ListContainsScalar<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.search_key)?;
        let c =
            cudf_sys::lists::ffi::lists_contains_scalar(self.view.0, &ffi, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_contains_column`].
///
/// Created by [`ListExt::list_contains_column`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListContainsColumn<'a> {
    view: &'a ColumnView<'a>,
    search_keys: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListContainsColumn<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_contains_column(
            self.view.0,
            self.search_keys.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_index_of_scalar`].
///
/// Created by [`ListExt::list_index_of_scalar`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListIndexOfScalar<'a> {
    view: &'a ColumnView<'a>,
    search_key: &'a Scalar,
    find_first: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ListIndexOfScalar<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.search_key)?;
        let c = cudf_sys::lists::ffi::lists_index_of_scalar(
            self.view.0,
            &ffi,
            self.find_first,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_index_of_column`].
///
/// Created by [`ListExt::list_index_of_column`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListIndexOfColumn<'a> {
    view: &'a ColumnView<'a>,
    search_keys: &'a ColumnView<'a>,
    find_first: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ListIndexOfColumn<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_index_of_column(
            self.view.0,
            self.search_keys.0,
            self.find_first,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_segmented_gather`].
///
/// Created by [`ListExt::list_segmented_gather`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListSegmentedGather<'a> {
    view: &'a ColumnView<'a>,
    gather_map: &'a ColumnView<'a>,
    nullify_oob: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ListSegmentedGather<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_segmented_gather(
            self.view.0,
            self.gather_map.0,
            self.nullify_oob,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_format`].
///
/// Created by [`ListExt::list_format`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListFormat<'a> {
    view: &'a ColumnView<'a>,
    na_rep: &'a str,
    stream: Stream,
}

impl crate::stream::GpuOp for ListFormat<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::strings::ffi::strings_format_list_column(
            self.view.0,
            self.na_rep,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_extract_element_column`].
///
/// Created by [`ListExt::list_extract_element_column`]. Call `.call()` to
/// execute, or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListExtractElementColumn<'a> {
    view: &'a ColumnView<'a>,
    indices: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListExtractElementColumn<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_extract_element_column(
            self.view.0,
            self.indices.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_stable_sort`].
///
/// Created by [`ListExt::list_stable_sort`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListStableSort<'a> {
    view: &'a ColumnView<'a>,
    ascending: bool,
    nulls_last: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ListStableSort<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_stable_sort(
            self.view.0,
            self.ascending,
            self.nulls_last,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`ListExt::list_apply_boolean_mask`].
///
/// Created by [`ListExt::list_apply_boolean_mask`]. Call `.call()` to execute,
/// or chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListApplyBooleanMask<'a> {
    view: &'a ColumnView<'a>,
    boolean_mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ListApplyBooleanMask<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_apply_boolean_mask(
            self.view.0,
            self.boolean_mask.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

// ---------------------------------------------------------------------------
// Builder structs — free functions
// ---------------------------------------------------------------------------

/// Builder for [`list_sequences`].
///
/// Created by [`list_sequences`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct ListSequences<'a> {
    starts: &'a ColumnView<'a>,
    sizes: &'a ColumnView<'a>,
    stream: Stream,
}

/// Generates integer sequences as a list column from `starts` and `sizes` columns.
///
/// For each row, generates a sequence starting at `starts[i]` with `sizes[i]`
/// elements and a step of `1`. Both columns must be integer types with the
/// same number of rows.
///
/// Returns a `LIST` column where each list contains the generated sequence.
///
/// # Errors
///
/// Returns an error if the column types are incompatible or sizes are negative.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::list_sequences;
/// use cudf::stream::GpuOp;
///
/// let starts = Column::from_slice_i32(&[0, 10]).call()?;
/// let sizes = Column::from_slice_i32(&[3, 2]).call()?;
/// let seqs = list_sequences(&starts.view(), &sizes.view()).call()?;
/// // [[0, 1, 2], [10, 11]]
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn list_sequences<'a>(
    starts: &'a ColumnView<'a>,
    sizes: &'a ColumnView<'a>,
) -> ListSequences<'a> {
    ListSequences {
        starts,
        sizes,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListSequences<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_sequences(
            self.starts.0,
            self.sizes.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`list_sequences_with_step`].
///
/// Created by [`list_sequences_with_step`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct ListSequencesWithStep<'a> {
    starts: &'a ColumnView<'a>,
    steps: &'a ColumnView<'a>,
    sizes: &'a ColumnView<'a>,
    stream: Stream,
}

/// Generates integer sequences with a custom step from `starts`, `steps`, and `sizes` columns.
///
/// For each row, generates a sequence starting at `starts[i]`, incrementing by
/// `steps[i]`, with `sizes[i]` elements. All three columns must be integer
/// types with the same number of rows.
///
/// Returns a `LIST` column where each list contains the generated sequence.
///
/// # Errors
///
/// Returns an error if the column types are incompatible.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::list_sequences_with_step;
/// use cudf::stream::GpuOp;
///
/// let starts = Column::from_slice_i32(&[0, 100]).call()?;
/// let steps = Column::from_slice_i32(&[2, -10]).call()?;
/// let sizes = Column::from_slice_i32(&[3, 4]).call()?;
/// let seqs = list_sequences_with_step(
///     &starts.view(), &steps.view(), &sizes.view(),
/// ).call()?;
/// // [[0, 2, 4], [100, 90, 80, 70]]
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn list_sequences_with_step<'a>(
    starts: &'a ColumnView<'a>,
    steps: &'a ColumnView<'a>,
    sizes: &'a ColumnView<'a>,
) -> ListSequencesWithStep<'a> {
    ListSequencesWithStep {
        starts,
        steps,
        sizes,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListSequencesWithStep<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_sequences_with_step(
            self.starts.0,
            self.steps.0,
            self.sizes.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`lists_concatenate_rows`].
///
/// Created by [`lists_concatenate_rows`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct ListsConcatenateRows<'a> {
    tbl: &'a crate::table::UnboundTable,
    stream: Stream,
}

/// Concatenates list columns row-wise from a [`Table`](crate::table::Table) into a single list column.
///
/// For each row, the lists from all columns in the table are concatenated into
/// one list. All columns in the table must be `LIST` columns with the same
/// child type.
///
/// Returns a single `LIST` column with the concatenated lists.
///
/// # Errors
///
/// Returns an error if the table columns have incompatible types.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::lists_concatenate_rows;
/// use cudf::stream::GpuOp;
/// use cudf::table::Table;
///
/// # let offsets_a = Column::from_slice_i32(&[0, 2, 3]).call()?;
/// # let values_a = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// # let lists_a = Column::from_lists(2, offsets_a, values_a).call()?;
/// # let offsets_b = Column::from_slice_i32(&[0, 1, 3]).call()?;
/// # let values_b = Column::from_slice_i32(&[10, 20, 30]).call()?;
/// # let lists_b = Column::from_lists(2, offsets_b, values_b).call()?;
/// # let tbl = Table::from_columns(vec![lists_a, lists_b])?;
/// let combined = lists_concatenate_rows(&tbl).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn lists_concatenate_rows(tbl: &crate::table::UnboundTable) -> ListsConcatenateRows<'_> {
    ListsConcatenateRows {
        tbl,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListsConcatenateRows<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_concatenate_rows(&self.tbl.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`lists_have_overlap`].
///
/// Created by [`lists_have_overlap`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct ListsHaveOverlap<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Tests whether two list columns share any common elements per row.
///
/// For each row, returns `true` if any element in `lhs[i]` is also present in
/// `rhs[i]`. Both columns must be `LIST` columns with the same child type and
/// the same number of rows.
///
/// Returns a `BOOL8` column.
///
/// # Errors
///
/// Returns an error if the column types are incompatible.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::lists_have_overlap;
/// use cudf::stream::GpuOp;
///
/// # let lhs_offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
/// # let lhs_values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
/// # let lhs = Column::from_lists(2, lhs_offsets, lhs_values).call()?;
/// # let rhs_offsets = Column::from_slice_i32(&[0, 2, 4]).call()?;
/// # let rhs_values = Column::from_slice_i32(&[3, 9, 4, 8]).call()?;
/// # let rhs = Column::from_lists(2, rhs_offsets, rhs_values).call()?;
/// let overlap = lists_have_overlap(&lhs.view(), &rhs.view()).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn lists_have_overlap<'a>(
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
) -> ListsHaveOverlap<'a> {
    ListsHaveOverlap {
        lhs,
        rhs,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListsHaveOverlap<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::lists::ffi::lists_have_overlap(self.lhs.0, self.rhs.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`lists_intersect_distinct`].
///
/// Created by [`lists_intersect_distinct`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct ListsIntersectDistinct<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Computes the distinct intersection of two list columns per row.
///
/// For each row, returns a list containing only the distinct elements that
/// appear in both `lhs[i]` and `rhs[i]`. Both columns must be `LIST` columns
/// with the same child type and the same number of rows.
///
/// Returns a `LIST` column.
///
/// # Errors
///
/// Returns an error if the column types are incompatible.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::lists_intersect_distinct;
/// use cudf::stream::GpuOp;
///
/// # let lhs_offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
/// # let lhs_values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
/// # let lhs = Column::from_lists(2, lhs_offsets, lhs_values).call()?;
/// # let rhs_offsets = Column::from_slice_i32(&[0, 2, 4]).call()?;
/// # let rhs_values = Column::from_slice_i32(&[3, 9, 4, 8]).call()?;
/// # let rhs = Column::from_lists(2, rhs_offsets, rhs_values).call()?;
/// let common = lists_intersect_distinct(&lhs.view(), &rhs.view()).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn lists_intersect_distinct<'a>(
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
) -> ListsIntersectDistinct<'a> {
    ListsIntersectDistinct {
        lhs,
        rhs,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListsIntersectDistinct<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_intersect_distinct(
            self.lhs.0,
            self.rhs.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`lists_union_distinct`].
///
/// Created by [`lists_union_distinct`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct ListsUnionDistinct<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Computes the distinct union of two list columns per row.
///
/// For each row, returns a list containing the distinct elements from either
/// `lhs[i]` or `rhs[i]` (or both). Both columns must be `LIST` columns with
/// the same child type and the same number of rows.
///
/// Returns a `LIST` column.
///
/// # Errors
///
/// Returns an error if the column types are incompatible.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::lists_union_distinct;
/// use cudf::stream::GpuOp;
///
/// # let lhs_offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
/// # let lhs_values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
/// # let lhs = Column::from_lists(2, lhs_offsets, lhs_values).call()?;
/// # let rhs_offsets = Column::from_slice_i32(&[0, 2, 4]).call()?;
/// # let rhs_values = Column::from_slice_i32(&[3, 9, 4, 8]).call()?;
/// # let rhs = Column::from_lists(2, rhs_offsets, rhs_values).call()?;
/// let merged = lists_union_distinct(&lhs.view(), &rhs.view()).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn lists_union_distinct<'a>(
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
) -> ListsUnionDistinct<'a> {
    ListsUnionDistinct {
        lhs,
        rhs,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListsUnionDistinct<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_union_distinct(
            self.lhs.0,
            self.rhs.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`lists_difference_distinct`].
///
/// Created by [`lists_difference_distinct`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct ListsDifferenceDistinct<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Computes the distinct set difference of two list columns per row.
///
/// For each row, returns a list containing the distinct elements that appear in
/// `lhs[i]` but not in `rhs[i]`. Both columns must be `LIST` columns with the
/// same child type and the same number of rows.
///
/// Returns a `LIST` column.
///
/// # Errors
///
/// Returns an error if the column types are incompatible.
///
/// # Examples
///
/// ```no_run
/// # use cudf::column::Column;
/// use cudf::lists::lists_difference_distinct;
/// use cudf::stream::GpuOp;
///
/// # let lhs_offsets = Column::from_slice_i32(&[0, 3, 5]).call()?;
/// # let lhs_values = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
/// # let lhs = Column::from_lists(2, lhs_offsets, lhs_values).call()?;
/// # let rhs_offsets = Column::from_slice_i32(&[0, 2, 4]).call()?;
/// # let rhs_values = Column::from_slice_i32(&[3, 9, 4, 8]).call()?;
/// # let rhs = Column::from_lists(2, rhs_offsets, rhs_values).call()?;
/// let diff = lists_difference_distinct(&lhs.view(), &rhs.view()).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub fn lists_difference_distinct<'a>(
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
) -> ListsDifferenceDistinct<'a> {
    ListsDifferenceDistinct {
        lhs,
        rhs,
        stream: Stream::default_stream(),
    }
}

impl crate::stream::GpuOp for ListsDifferenceDistinct<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::lists::ffi::lists_difference_distinct(
            self.lhs.0,
            self.rhs.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}
