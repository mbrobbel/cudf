// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Table = crate::ffi::Table;
        type Scalar = crate::ffi::Scalar;
        type ScalarList = crate::ffi::ScalarList;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Gather --

        /// Gathers rows from a table using index column.
        fn gather_table(
            tbl: &Table,
            indices: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Gather rows with out-of-bounds policy (nullify_oob=true -> NULLIFY).
        fn gather_table_checked(
            tbl: &Table,
            gather_map: &column_view,
            nullify_oob: bool,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Deep copy --

        /// Deep-copies a column_view into an owning Column.
        fn copy_column(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Empty / Allocate --

        /// Creates an empty column with the same type as input.
        fn empty_like_column(col: &column_view) -> UniquePtr<Column>;

        /// Creates an empty table with the same schema as input.
        fn empty_like_table(tbl: &Table) -> UniquePtr<Table>;

        /// Create uninitialized column of same type and size.
        fn allocate_like_column(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Scatter --

        /// Scatters source table rows into target table at given indices.
        fn scatter_table(
            source: &Table,
            scatter_map: &column_view,
            target: &Table,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Scatter scalar values to specified indices in a target table.
        fn scatter_scalars(
            sources: Pin<&mut ScalarList>,
            indices: &column_view,
            target: &Table,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Scatter rows from source table into target using boolean mask.
        fn boolean_mask_scatter_table(
            source: &Table,
            target: &Table,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        /// Scatter scalar values into target where boolean mask is true.
        fn boolean_mask_scatter_scalars(
            sources: Pin<&mut ScalarList>,
            target: &Table,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Reverse --

        /// Reverses the elements of a column.
        fn reverse_column(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Reverses the rows of a table.
        fn reverse_table(tbl: &Table, stream: usize) -> Result<UniquePtr<Table>>;

        // -- Shift --

        /// Shifts column values by offset, filling with the given scalar.
        fn shift_column(
            col: &column_view,
            offset: i32,
            fill_value: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Get element --

        /// Returns a single element from a column as a scalar.
        fn get_element(col: &column_view, index: i32, stream: usize) -> Result<UniquePtr<Scalar>>;

        // -- Copy-if-else --

        /// Selects elements from lhs or rhs based on a boolean mask.
        fn copy_if_else_columns(
            lhs: &column_view,
            rhs: &column_view,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Selects elements from a scalar or column based on a boolean mask.
        fn copy_if_else_scalar_column(
            lhs: &Scalar,
            rhs: &column_view,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Selects elements from a column or scalar based on a boolean mask.
        fn copy_if_else_column_scalar(
            lhs: &column_view,
            rhs: &Scalar,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Copy-if-else with two scalars and a boolean mask.
        fn copy_if_else_scalars(
            lhs: &Scalar,
            rhs: &Scalar,
            mask: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Slice --

        /// Extracts a slice [begin, end) of a column as a new owned column.
        fn slice_column(
            col: &column_view,
            begin: i32,
            end: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Extracts a slice [begin, end) of a table as a new owned table.
        fn slice_table(
            tbl: &Table,
            begin: i32,
            end: i32,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Sample --

        /// Randomly samples rows from a table.
        fn sample_table(
            tbl: &Table,
            n: i32,
            with_replacement: bool,
            seed: i64,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;

        // -- Copy range --

        /// Copy range of elements from source into target (out-of-place).
        fn copy_range(
            source: &column_view,
            target: &column_view,
            source_begin: i32,
            source_end: i32,
            target_begin: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Null inspection --

        /// Check if column has non-empty null rows (LIST/STRING).
        fn has_nonempty_nulls(col: &column_view, stream: usize) -> Result<bool>;

        /// Check if a column may have non-empty data in null rows (no stream needed).
        fn may_have_nonempty_nulls(col: &column_view) -> bool;

        /// Purge non-empty null row contents.
        fn purge_nonempty_nulls(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- In-place operations --

        /// Fill a range [begin, end) of a column with a scalar value in-place.
        fn fill_in_place(
            col: Pin<&mut Column>,
            begin: i32,
            end: i32,
            value: &Scalar,
            stream: usize,
        ) -> Result<()>;

        /// Copy a range from source into dest in-place.
        fn copy_range_in_place(
            dest: Pin<&mut Column>,
            source: &column_view,
            source_begin: i32,
            source_end: i32,
            dest_begin: i32,
            stream: usize,
        ) -> Result<()>;

        // -- Gather with policy --

        /// Gather rows with both out-of-bounds and negative-index policies.
        ///
        /// `nullify_oob=true` means NULLIFY, `false` means DONT_CHECK.
        /// `allow_negative=true` means negative indices wrap around (ALLOWED),
        /// `false` means they are undefined behavior (NOT_ALLOWED).
        fn gather_table_with_policy(
            tbl: &Table,
            gather_map: &column_view,
            nullify_oob: bool,
            allow_negative: bool,
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
