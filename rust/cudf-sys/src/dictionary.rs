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

        // -- Dictionary operations --

        /// Dictionary-encode a column (returns DICTIONARY32 column).
        fn dictionary_encode(
            col: &column_view,
            indices_type_id: i32,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Decode a dictionary column back to its value type.
        fn dictionary_decode(col: &column_view, stream: usize) -> Result<UniquePtr<Column>>;
        /// Add new keys to a dictionary column.
        fn dictionary_add_keys(
            col: &column_view,
            new_keys: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Remove keys from a dictionary column.
        fn dictionary_remove_keys(
            col: &column_view,
            keys_to_remove: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Replace the keyset of a dictionary column.
        fn dictionary_set_keys(
            col: &column_view,
            keys: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Remove keys not referenced by any indices.
        fn dictionary_remove_unused_keys(
            col: &column_view,
            stream: usize,
        ) -> Result<UniquePtr<Column>>;
        /// Look up a scalar key in a dictionary, returning its index.
        fn dictionary_get_index(
            col: &column_view,
            key: &Scalar,
            stream: usize,
        ) -> Result<UniquePtr<Scalar>>;
    }
}
