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

        // -- Label bins --

        /// Assigns bin labels to column values.
        fn label_bins_column(
            col: &column_view,
            left_edges: &column_view,
            left_inclusive: i32,
            right_edges: &column_view,
            right_inclusive: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
    }
}
