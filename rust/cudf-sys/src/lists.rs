// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        type Scalar = crate::ffi::Scalar;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Lists operations --

        /// Count elements in each list row.
        fn lists_count_elements(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Extract element at given index from each list row.
        fn lists_extract_element(
            col: &column_view,
            index: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Sort elements within each list row.
        fn lists_sort(
            col: &column_view,
            ascending: bool,
            nulls_last: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Reverse elements within each list row.
        fn lists_reverse(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Check if each list contains nulls.
        fn lists_contains_nulls(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Remove duplicates from each list.
        fn lists_distinct(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Concatenate nested list elements within each row.
        fn lists_concatenate_elements(
            col: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Generate sequences as lists from starts and sizes columns.
        fn lists_sequences(
            starts: &column_view,
            sizes: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- List set operations --

        /// Check if two list columns have overlapping elements per row.
        fn lists_have_overlap(
            lhs: &column_view,
            rhs: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Intersect distinct elements of two list columns per row.
        fn lists_intersect_distinct(
            lhs: &column_view,
            rhs: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Union distinct elements of two list columns per row.
        fn lists_union_distinct(
            lhs: &column_view,
            rhs: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Difference distinct elements of two list columns per row.
        fn lists_difference_distinct(
            lhs: &column_view,
            rhs: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Lists contains / index_of / segmented_gather --

        /// Check if each list row contains a scalar value (returns BOOL8).
        fn lists_contains_scalar(
            col: &column_view,
            search_key: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Check if each list row contains the corresponding search_keys value.
        fn lists_contains_column(
            col: &column_view,
            search_keys: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Find position of scalar in each list row. find_first=true for FIND_FIRST.
        fn lists_index_of_scalar(
            col: &column_view,
            search_key: &Scalar,
            find_first: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Find position of each search_keys value in corresponding list row.
        fn lists_index_of_column(
            col: &column_view,
            search_keys: &column_view,
            find_first: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Gather elements from each list row using a gather map list column.
        fn lists_segmented_gather(
            col: &column_view,
            gather_map: &column_view,
            nullify_oob: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Lists: apply_boolean_mask --

        /// Filter elements within each list row using a boolean mask list column.
        fn lists_apply_boolean_mask(
            col: &column_view,
            boolean_mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Lists: extract (column), stable sort, concatenate_rows --

        /// Extract element from each list using per-row column indices.
        fn lists_extract_element_column(
            col: &column_view,
            indices: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Stable sort elements within each list row.
        fn lists_stable_sort(
            col: &column_view,
            ascending: bool,
            nulls_last: bool,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Row-wise concatenation of list columns from a table into a single list column.
        fn lists_concatenate_rows(tbl: &Table, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Lists: sequences with step --

        /// Generate sequences with custom step from starts, steps, and sizes columns.
        fn lists_sequences_with_step(
            starts: &column_view,
            steps: &column_view,
            sizes: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Explode --

        /// Explode a list column in a table.
        fn explode_table(tbl: &Table, column_idx: i32, stream: usize) -> Result<UniquePtr<Table>>;
        /// Explode a list column with position column.
        fn explode_position_table(
            tbl: &Table,
            column_idx: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
        /// Explode a list column, keeping null/empty list rows.
        fn explode_outer_table(
            tbl: &Table,
            column_idx: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Explode: outer_position --

        /// Explode outer with position column.
        fn explode_outer_position_table(
            tbl: &Table,
            column_idx: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
