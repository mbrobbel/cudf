// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Column = crate::ffi::Column;
        type Scalar = crate::ffi::Scalar;
        type BinaryOperator = crate::ffi::BinaryOperator;
        #[namespace = "cudf"]
        type column_view = crate::ffi::column_view;

        // -- Binary operations --

        /// Binary operation between two column_views.
        fn binary_operation_columns(
            lhs: &column_view,
            rhs: &column_view,
            op: BinaryOperator,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Binary operation between a column_view and a Scalar.
        fn binary_operation_column_scalar(
            lhs: &column_view,
            rhs: &Scalar,
            op: BinaryOperator,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        /// Binary operation between a Scalar and a column_view.
        fn binary_operation_scalar_column(
            lhs: &Scalar,
            rhs: &column_view,
            op: BinaryOperator,
            output_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;

        // -- Binary: fixed_point_scale / is_supported_operation --

        /// Compute the output scale for a fixed-point binary operation.
        fn binary_operation_fixed_point_scale(op: i32, left_scale: i32, right_scale: i32) -> i32;

        /// Check if a binary operation is supported for the given types.
        fn is_supported_binaryop(
            out_type_id: i32,
            lhs_type_id: i32,
            rhs_type_id: i32,
            op: i32,
        ) -> bool;
    }
}
