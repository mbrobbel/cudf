// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Scalar = crate::ffi::Scalar;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Replace operations --

        /// Replaces null values with corresponding values from replacement column.
        fn replace_nulls_column(
            col: &column_view,
            replacement: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces null values with a scalar.
        fn replace_nulls_scalar(
            col: &column_view,
            replacement: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces NaN values with corresponding values from replacement column.
        fn replace_nans_column(
            col: &column_view,
            replacement: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Replaces NaN values with a scalar.
        fn replace_nans_scalar(
            col: &column_view,
            replacement: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Clamps column values to [lo, hi] range.
        fn clamp_column(
            col: &column_view,
            lo: &Scalar,
            hi: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Finds and replaces all matching values in a column.
        fn find_and_replace_all(
            col: &column_view,
            values_to_replace: &column_view,
            replacement_values: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Replace (new) --

        /// Replace nulls using preceding/following policy. policy: 0=PRECEDING, 1=FOLLOWING.
        fn replace_nulls_policy(
            col: &column_view,
            policy: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Clamp with separate replacement values for lo and hi.
        fn clamp_column_with_replace(
            col: &column_view,
            lo: &Scalar,
            lo_replace: &Scalar,
            hi: &Scalar,
            hi_replace: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Normalize NaNs and zeros (convert -NaN to NaN, -0.0 to 0.0).
        fn normalize_nans_and_zeros(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
    }
}
