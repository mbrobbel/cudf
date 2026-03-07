// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Dictionary column encode, decode, and key management operations.

use crate::column::{Column, ColumnView};
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

/// Extension trait for dictionary column operations.
pub trait DictionaryExt: private::Sealed {
    /// Decode a dictionary column back to its value type.
    fn dict_decode(&self) -> DictDecode<'_>;
    /// Add new keys to a dictionary column.
    fn dict_add_keys<'a>(&'a self, new_keys: &'a ColumnView<'a>) -> DictAddKeys<'a>;
    /// Remove keys from a dictionary column.
    fn dict_remove_keys<'a>(&'a self, keys_to_remove: &'a ColumnView<'a>) -> DictRemoveKeys<'a>;
    /// Replace the keyset of a dictionary column.
    fn dict_set_keys<'a>(&'a self, keys: &'a ColumnView<'a>) -> DictSetKeys<'a>;
    /// Remove keys not referenced by any indices.
    fn dict_remove_unused_keys(&self) -> DictRemoveUnusedKeys<'_>;
    /// Look up a scalar key in a dictionary, returning its index scalar.
    fn dict_get_index<'a>(&'a self, key: &'a Scalar) -> DictGetIndex<'a>;
}

// ---------------------------------------------------------------------------
// Builder structs — trait methods
// ---------------------------------------------------------------------------

/// Builder for [`DictionaryExt::dict_decode`].
pub struct DictDecode<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl DictDecode<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::dictionary::ffi::dictionary_decode(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`DictionaryExt::dict_add_keys`].
pub struct DictAddKeys<'a> {
    view: &'a ColumnView<'a>,
    new_keys: &'a ColumnView<'a>,
    stream: Stream,
}

impl DictAddKeys<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::dictionary::ffi::dictionary_add_keys(
            self.view.0,
            self.new_keys.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DictionaryExt::dict_remove_keys`].
pub struct DictRemoveKeys<'a> {
    view: &'a ColumnView<'a>,
    keys_to_remove: &'a ColumnView<'a>,
    stream: Stream,
}

impl DictRemoveKeys<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::dictionary::ffi::dictionary_remove_keys(
            self.view.0,
            self.keys_to_remove.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DictionaryExt::dict_set_keys`].
pub struct DictSetKeys<'a> {
    view: &'a ColumnView<'a>,
    keys: &'a ColumnView<'a>,
    stream: Stream,
}

impl DictSetKeys<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::dictionary::ffi::dictionary_set_keys(
            self.view.0,
            self.keys.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DictionaryExt::dict_remove_unused_keys`].
pub struct DictRemoveUnusedKeys<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl DictRemoveUnusedKeys<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::dictionary::ffi::dictionary_remove_unused_keys(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`DictionaryExt::dict_get_index`].
pub struct DictGetIndex<'a> {
    view: &'a ColumnView<'a>,
    key: &'a Scalar,
    stream: Stream,
}

impl DictGetIndex<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Scalar> {
        let ffi = crate::scalar::scalar_to_ffi(self.key);
        let s = cudf_sys::dictionary::ffi::dictionary_get_index(
            self.view.0,
            &ffi,
            self.stream.as_raw(),
        )?;
        Ok(crate::scalar::scalar_from_ffi(&s))
    }
}

// ---------------------------------------------------------------------------
// Builder struct — free function
// ---------------------------------------------------------------------------

/// Builder for [`dictionary_encode`].
pub struct DictionaryEncode<'a> {
    col: &'a ColumnView<'a>,
    indices_type: TypeId,
    stream: Stream,
}

/// Dictionary-encode a column. Returns a DICTIONARY32 column.
///
/// The `indices_type` controls the integer type used for indices
/// (default `TypeId::INT32`).
pub fn dictionary_encode<'a>(
    col: &'a ColumnView<'a>,
    indices_type: TypeId,
) -> DictionaryEncode<'a> {
    DictionaryEncode {
        col,
        indices_type,
        stream: Stream::default_stream(),
    }
}

impl DictionaryEncode<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::dictionary::ffi::dictionary_encode(
            self.col.0,
            self.indices_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

impl private::Sealed for ColumnView<'_> {}

impl DictionaryExt for ColumnView<'_> {
    fn dict_decode(&self) -> DictDecode<'_> {
        DictDecode {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn dict_add_keys<'a>(&'a self, new_keys: &'a ColumnView<'a>) -> DictAddKeys<'a> {
        DictAddKeys {
            view: self,
            new_keys,
            stream: Stream::default_stream(),
        }
    }
    fn dict_remove_keys<'a>(&'a self, keys_to_remove: &'a ColumnView<'a>) -> DictRemoveKeys<'a> {
        DictRemoveKeys {
            view: self,
            keys_to_remove,
            stream: Stream::default_stream(),
        }
    }
    fn dict_set_keys<'a>(&'a self, keys: &'a ColumnView<'a>) -> DictSetKeys<'a> {
        DictSetKeys {
            view: self,
            keys,
            stream: Stream::default_stream(),
        }
    }
    fn dict_remove_unused_keys(&self) -> DictRemoveUnusedKeys<'_> {
        DictRemoveUnusedKeys {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn dict_get_index<'a>(&'a self, key: &'a Scalar) -> DictGetIndex<'a> {
        DictGetIndex {
            view: self,
            key,
            stream: Stream::default_stream(),
        }
    }
}
