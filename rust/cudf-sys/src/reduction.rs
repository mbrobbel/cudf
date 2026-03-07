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

        // -- Reduction --

        /// Computes the sum of all elements.
        fn reduce_sum(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Computes the minimum value.
        fn reduce_min(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Computes the maximum value.
        fn reduce_max(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Computes the product of all elements.
        fn reduce_product(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Returns true if any element is non-zero.
        fn reduce_any(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Returns true if all elements are non-zero.
        fn reduce_all(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        // -- Additional reductions --

        /// Computes the mean of all elements.
        fn reduce_mean(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Computes the standard deviation.
        fn reduce_std(
            col: &column_view,
            output_type_id: i32,
            ddof: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Computes the variance.
        fn reduce_var(
            col: &column_view,
            output_type_id: i32,
            ddof: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Computes the median.
        fn reduce_median(
            col: &column_view,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Counts the number of unique elements.
        fn reduce_nunique(
            col: &column_view,
            null_policy: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Prefix scan (cumulative operation).
        fn scan_column(
            col: &column_view,
            agg_kind: i32,
            scan_type: i32,
            null_policy: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Returns the minimum value from a column (via minmax).
        fn minmax_min(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        /// Returns the maximum value from a column (via minmax).
        fn minmax_max(col: &column_view, stream: usize) -> Result<UniquePtr<Scalar>>;

        // -- Generic reduce --

        /// Reduce a column with any aggregation kind (ddof used for STD/VAR).
        fn reduce_generic(
            col: &column_view,
            agg_kind: i32,
            ddof: i32,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Reduces a column with an initial value (supports SUM, PRODUCT, MIN, MAX, ANY, ALL).
        fn reduce_with_init(
            col: &column_view,
            agg_kind: i32,
            ddof: i32,
            output_type_id: i32,
            init: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;

        /// Segmented reduce: reduces each segment of a column defined by offsets.
        fn segmented_reduce(
            col: &column_view,
            offsets: &column_view,
            agg_kind: i32,
            ddof: i32,
            output_type_id: i32,
            null_handling: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
