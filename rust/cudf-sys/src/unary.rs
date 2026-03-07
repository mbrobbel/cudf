// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Unary operations --

        /// Casts a column to a different type.
        fn unary_cast(
            col: &column_view,
            target_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates a null value.
        fn unary_is_null(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates a valid value.
        fn unary_is_valid(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates NaN.
        fn unary_is_nan(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Negates every element of the column.
        fn unary_negate(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns the absolute value of every element.
        fn unary_abs(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Generic unary operation --

        /// Applies a unary operation to a column.
        fn unary_operation(col: &column_view, op: i32, stream: usize) -> Result<UniquePtr<Column>>;

        /// Returns a BOOL8 column where true indicates a non-NaN value.
        fn unary_is_not_nan(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;

        // -- Round --

        /// Rounds column values to the given number of decimal places.
        fn round_column(
            col: &column_view,
            decimal_places: i32,
            method: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
