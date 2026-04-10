#![allow(clippy::wildcard_imports)]
use super::*;

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
