// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Dictionary column encode, decode, and key management operations.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Extension trait for dictionary column operations.
pub trait DictionaryExt {
    /// Decode a dictionary column back to its value type.
    fn dict_decode(&self) -> Result<Column>;
    /// Add new keys to a dictionary column.
    fn dict_add_keys(&self, new_keys: &ColumnView<'_>) -> Result<Column>;
    /// Remove keys from a dictionary column.
    fn dict_remove_keys(&self, keys_to_remove: &ColumnView<'_>) -> Result<Column>;
    /// Replace the keyset of a dictionary column.
    fn dict_set_keys(&self, keys: &ColumnView<'_>) -> Result<Column>;
    /// Remove keys not referenced by any indices.
    fn dict_remove_unused_keys(&self) -> Result<Column>;
    /// Look up a scalar key in a dictionary, returning its index scalar.
    fn dict_get_index(&self, key: &Scalar) -> Result<Scalar>;
}

/// Dictionary-encode a column. Returns a DICTIONARY32 column.
///
/// The `indices_type` controls the integer type used for indices
/// (default `TypeId::INT32`).
pub fn dictionary_encode(col: &ColumnView<'_>, indices_type: TypeId) -> Result<Column> {
    let c = cudf_sys::ffi::dictionary_encode(
        col.0,
        indices_type.repr,
        Stream::default_stream().as_raw(),
    )?;
    Ok(Column(c))
}

impl DictionaryExt for ColumnView<'_> {
    fn dict_decode(&self) -> Result<Column> {
        let c = cudf_sys::ffi::dictionary_decode(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn dict_add_keys(&self, new_keys: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::dictionary_add_keys(
            self.0,
            new_keys.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }
    fn dict_remove_keys(&self, keys_to_remove: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::dictionary_remove_keys(
            self.0,
            keys_to_remove.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }
    fn dict_set_keys(&self, keys: &ColumnView<'_>) -> Result<Column> {
        let c =
            cudf_sys::ffi::dictionary_set_keys(self.0, keys.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }
    fn dict_remove_unused_keys(&self) -> Result<Column> {
        let c = cudf_sys::ffi::dictionary_remove_unused_keys(
            self.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }
    fn dict_get_index(&self, key: &Scalar) -> Result<Scalar> {
        let ffi = crate::scalar::scalar_to_ffi(key);
        let s =
            cudf_sys::ffi::dictionary_get_index(self.0, &ffi, Stream::default_stream().as_raw())?;
        Ok(crate::scalar::scalar_from_ffi(&s))
    }
}
