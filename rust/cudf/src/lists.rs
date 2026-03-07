// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! List column operations.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

/// Extension trait for list column operations.
pub trait ListExt: private::Sealed {
    /// Returns an INT32 column with the number of elements in each list row.
    fn list_count_elements(&self) -> ListCountElements<'_>;
    /// Extract element at `index` from each list row.
    fn list_extract_element(&self, index: i32) -> ListExtractElement<'_>;
    /// Sort elements within each list row.
    fn list_sort(&self, ascending: bool, nulls_last: bool) -> ListSort<'_>;
    /// Reverse elements within each list row.
    fn list_reverse(&self) -> ListReverse<'_>;
    /// Returns a BOOL8 column indicating whether each list contains nulls.
    fn list_contains_nulls(&self) -> ListContainsNulls<'_>;
    /// Remove duplicate elements from each list.
    fn list_distinct(&self) -> ListDistinct<'_>;
    /// Concatenate nested list elements within each row (flatten one level).
    fn list_concatenate_elements(&self) -> ListConcatenateElements<'_>;
    /// Check if each list row contains a scalar value (returns BOOL8 column).
    fn list_contains_scalar<'a>(&'a self, search_key: &'a Scalar) -> ListContainsScalar<'a>;
    /// Check if each list row contains the corresponding `search_keys` value.
    fn list_contains_column<'a>(
        &'a self,
        search_keys: &'a ColumnView<'a>,
    ) -> ListContainsColumn<'a>;
    /// Find position of scalar in each list row (-1 if not found).
    fn list_index_of_scalar<'a>(
        &'a self,
        search_key: &'a Scalar,
        find_first: bool,
    ) -> ListIndexOfScalar<'a>;
    /// Find position of each `search_keys` value in corresponding list row.
    fn list_index_of_column<'a>(
        &'a self,
        search_keys: &'a ColumnView<'a>,
        find_first: bool,
    ) -> ListIndexOfColumn<'a>;
    /// Gather elements from each list row using a gather map list column.
    fn list_segmented_gather<'a>(
        &'a self,
        gather_map: &'a ColumnView<'a>,
        nullify_oob: bool,
    ) -> ListSegmentedGather<'a>;
    /// Format a list-of-strings column into formatted strings.
    fn list_format<'a>(&'a self, na_rep: &'a str) -> ListFormat<'a>;
    /// Extract element from each list using per-row column indices.
    fn list_extract_element_column<'a>(
        &'a self,
        indices: &'a ColumnView<'a>,
    ) -> ListExtractElementColumn<'a>;
    /// Stable sort elements within each list row.
    fn list_stable_sort(&self, ascending: bool, nulls_last: bool) -> ListStableSort<'_>;
    /// Filter elements within each list row using a boolean mask list column.
    fn list_apply_boolean_mask<'a>(
        &'a self,
        boolean_mask: &'a ColumnView<'a>,
    ) -> ListApplyBooleanMask<'a>;
}

// ---------------------------------------------------------------------------
// Builder structs — trait methods
// ---------------------------------------------------------------------------

/// Builder for [`ListExt::list_count_elements`].
pub struct ListCountElements<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListCountElements<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_count_elements(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_extract_element`].
pub struct ListExtractElement<'a> {
    view: &'a ColumnView<'a>,
    index: i32,
    stream: Stream,
}

impl ListExtractElement<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_extract_element(
            self.view.0,
            self.index,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_sort`].
pub struct ListSort<'a> {
    view: &'a ColumnView<'a>,
    ascending: bool,
    nulls_last: bool,
    stream: Stream,
}

impl ListSort<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_sort(
            self.view.0,
            self.ascending,
            self.nulls_last,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_reverse`].
pub struct ListReverse<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListReverse<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_reverse(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_contains_nulls`].
pub struct ListContainsNulls<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListContainsNulls<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_contains_nulls(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_distinct`].
pub struct ListDistinct<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListDistinct<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_distinct(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_concatenate_elements`].
pub struct ListConcatenateElements<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListConcatenateElements<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::lists::ffi::lists_concatenate_elements(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_contains_scalar`].
pub struct ListContainsScalar<'a> {
    view: &'a ColumnView<'a>,
    search_key: &'a Scalar,
    stream: Stream,
}

impl ListContainsScalar<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(self.search_key);
        let c =
            cudf_sys::lists::ffi::lists_contains_scalar(self.view.0, &ffi, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_contains_column`].
pub struct ListContainsColumn<'a> {
    view: &'a ColumnView<'a>,
    search_keys: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListContainsColumn<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_contains_column(
            self.view.0,
            self.search_keys.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_index_of_scalar`].
pub struct ListIndexOfScalar<'a> {
    view: &'a ColumnView<'a>,
    search_key: &'a Scalar,
    find_first: bool,
    stream: Stream,
}

impl ListIndexOfScalar<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(self.search_key);
        let c = cudf_sys::lists::ffi::lists_index_of_scalar(
            self.view.0,
            &ffi,
            self.find_first,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_index_of_column`].
pub struct ListIndexOfColumn<'a> {
    view: &'a ColumnView<'a>,
    search_keys: &'a ColumnView<'a>,
    find_first: bool,
    stream: Stream,
}

impl ListIndexOfColumn<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_index_of_column(
            self.view.0,
            self.search_keys.0,
            self.find_first,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_segmented_gather`].
pub struct ListSegmentedGather<'a> {
    view: &'a ColumnView<'a>,
    gather_map: &'a ColumnView<'a>,
    nullify_oob: bool,
    stream: Stream,
}

impl ListSegmentedGather<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_segmented_gather(
            self.view.0,
            self.gather_map.0,
            self.nullify_oob,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_format`].
pub struct ListFormat<'a> {
    view: &'a ColumnView<'a>,
    na_rep: &'a str,
    stream: Stream,
}

impl ListFormat<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::strings::ffi::strings_format_list_column(
            self.view.0,
            self.na_rep,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_extract_element_column`].
pub struct ListExtractElementColumn<'a> {
    view: &'a ColumnView<'a>,
    indices: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListExtractElementColumn<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_extract_element_column(
            self.view.0,
            self.indices.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_stable_sort`].
pub struct ListStableSort<'a> {
    view: &'a ColumnView<'a>,
    ascending: bool,
    nulls_last: bool,
    stream: Stream,
}

impl ListStableSort<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_stable_sort(
            self.view.0,
            self.ascending,
            self.nulls_last,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ListExt::list_apply_boolean_mask`].
pub struct ListApplyBooleanMask<'a> {
    view: &'a ColumnView<'a>,
    boolean_mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl ListApplyBooleanMask<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_apply_boolean_mask(
            self.view.0,
            self.boolean_mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

// ---------------------------------------------------------------------------
// Builder structs — free functions
// ---------------------------------------------------------------------------

/// Builder for [`list_sequences`].
pub struct ListSequences<'a> {
    starts: &'a ColumnView<'a>,
    sizes: &'a ColumnView<'a>,
    stream: Stream,
}

/// Generate sequences as list column from starts and sizes columns.
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

impl ListSequences<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_sequences(
            self.starts.0,
            self.sizes.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`list_sequences_with_step`].
pub struct ListSequencesWithStep<'a> {
    starts: &'a ColumnView<'a>,
    steps: &'a ColumnView<'a>,
    sizes: &'a ColumnView<'a>,
    stream: Stream,
}

/// Generate sequences with custom step from starts, steps, and sizes columns.
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

impl ListSequencesWithStep<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_sequences_with_step(
            self.starts.0,
            self.steps.0,
            self.sizes.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`lists_concatenate_rows`].
pub struct ListsConcatenateRows<'a> {
    tbl: &'a crate::table::Table,
    stream: Stream,
}

/// Row-wise concatenation of list columns from a table into a single list column.
pub fn lists_concatenate_rows(tbl: &crate::table::Table) -> ListsConcatenateRows<'_> {
    ListsConcatenateRows {
        tbl,
        stream: Stream::default_stream(),
    }
}

impl ListsConcatenateRows<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_concatenate_rows(&self.tbl.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`lists_have_overlap`].
pub struct ListsHaveOverlap<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Check if two list columns have overlapping elements per row.
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

impl ListsHaveOverlap<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::lists::ffi::lists_have_overlap(self.lhs.0, self.rhs.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`lists_intersect_distinct`].
pub struct ListsIntersectDistinct<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Intersect distinct elements of two list columns per row.
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

impl ListsIntersectDistinct<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_intersect_distinct(
            self.lhs.0,
            self.rhs.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`lists_union_distinct`].
pub struct ListsUnionDistinct<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Union distinct elements of two list columns per row.
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

impl ListsUnionDistinct<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_union_distinct(
            self.lhs.0,
            self.rhs.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`lists_difference_distinct`].
pub struct ListsDifferenceDistinct<'a> {
    lhs: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

/// Difference distinct elements of two list columns per row.
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

impl ListsDifferenceDistinct<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::lists::ffi::lists_difference_distinct(
            self.lhs.0,
            self.rhs.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

impl private::Sealed for ColumnView<'_> {}

impl ListExt for ColumnView<'_> {
    fn list_count_elements(&self) -> ListCountElements<'_> {
        ListCountElements {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn list_extract_element(&self, index: i32) -> ListExtractElement<'_> {
        ListExtractElement {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }
    fn list_sort(&self, ascending: bool, nulls_last: bool) -> ListSort<'_> {
        ListSort {
            view: self,
            ascending,
            nulls_last,
            stream: Stream::default_stream(),
        }
    }
    fn list_reverse(&self) -> ListReverse<'_> {
        ListReverse {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn list_contains_nulls(&self) -> ListContainsNulls<'_> {
        ListContainsNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn list_distinct(&self) -> ListDistinct<'_> {
        ListDistinct {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn list_concatenate_elements(&self) -> ListConcatenateElements<'_> {
        ListConcatenateElements {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn list_contains_scalar<'a>(&'a self, search_key: &'a Scalar) -> ListContainsScalar<'a> {
        ListContainsScalar {
            view: self,
            search_key,
            stream: Stream::default_stream(),
        }
    }
    fn list_contains_column<'a>(
        &'a self,
        search_keys: &'a ColumnView<'a>,
    ) -> ListContainsColumn<'a> {
        ListContainsColumn {
            view: self,
            search_keys,
            stream: Stream::default_stream(),
        }
    }
    fn list_index_of_scalar<'a>(
        &'a self,
        search_key: &'a Scalar,
        find_first: bool,
    ) -> ListIndexOfScalar<'a> {
        ListIndexOfScalar {
            view: self,
            search_key,
            find_first,
            stream: Stream::default_stream(),
        }
    }
    fn list_index_of_column<'a>(
        &'a self,
        search_keys: &'a ColumnView<'a>,
        find_first: bool,
    ) -> ListIndexOfColumn<'a> {
        ListIndexOfColumn {
            view: self,
            search_keys,
            find_first,
            stream: Stream::default_stream(),
        }
    }
    fn list_segmented_gather<'a>(
        &'a self,
        gather_map: &'a ColumnView<'a>,
        nullify_oob: bool,
    ) -> ListSegmentedGather<'a> {
        ListSegmentedGather {
            view: self,
            gather_map,
            nullify_oob,
            stream: Stream::default_stream(),
        }
    }
    fn list_format<'a>(&'a self, na_rep: &'a str) -> ListFormat<'a> {
        ListFormat {
            view: self,
            na_rep,
            stream: Stream::default_stream(),
        }
    }
    fn list_extract_element_column<'a>(
        &'a self,
        indices: &'a ColumnView<'a>,
    ) -> ListExtractElementColumn<'a> {
        ListExtractElementColumn {
            view: self,
            indices,
            stream: Stream::default_stream(),
        }
    }
    fn list_stable_sort(&self, ascending: bool, nulls_last: bool) -> ListStableSort<'_> {
        ListStableSort {
            view: self,
            ascending,
            nulls_last,
            stream: Stream::default_stream(),
        }
    }
    fn list_apply_boolean_mask<'a>(
        &'a self,
        boolean_mask: &'a ColumnView<'a>,
    ) -> ListApplyBooleanMask<'a> {
        ListApplyBooleanMask {
            view: self,
            boolean_mask,
            stream: Stream::default_stream(),
        }
    }
}
