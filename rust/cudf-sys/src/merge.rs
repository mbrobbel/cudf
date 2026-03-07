// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "cudf_sys")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cudf-sys/lib.hpp");

        // Shared types from the core bridge.
        type Table = crate::ffi::Table;

        // -- Merge --

        /// Merges two sorted tables maintaining sort order.
        fn merge_tables(
            left: &Table,
            right: &Table,
            key_indices: &[i32],
            column_orders: &[i32],
            null_orders: &[i32],
            stream: usize,
        ) -> Result<UniquePtr<Table>>;
    }
}
