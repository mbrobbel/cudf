// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Dictionary column encode, decode, and key management operations.
//!
//! Dictionary-encoded columns store data as a pair of components: a sorted set
//! of unique **keys** and an integer **indices** column that maps each row to
//! its corresponding key. This encoding is space-efficient when a column has
//! many repeated values.
//!
//! The [`DictionaryExt`] trait provides methods for decoding, adding/removing
//! keys, and looking up indices. The free function [`dictionary_encode`]
//! converts a plain column into dictionary form.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::column::Column;
//! use cudf::data_type::TypeId;
//! use cudf::dictionary::{DictionaryExt, dictionary_encode};
//! use cudf::stream::GpuOp;
//!
//! // Encode a string column as a dictionary
//! let col = Column::from_strings(&["a", "b", "a", "c"]).call()?;
//! let dict = dictionary_encode(&col.view(), TypeId::INT32).call()?;
//!
//! // Decode back to a plain string column
//! let plain = dict.view().dict_decode().call()?;
//! # Ok::<(), cudf::error::Error>(())
//! ```

use crate::column::ColumnView;
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

/// Extension trait for dictionary column operations.
///
/// Provides methods for manipulating dictionary-encoded columns, including
/// decoding, key management (add, remove, set, prune), and index lookup.
///
/// This trait is implemented for [`ColumnView`] and is sealed -- it cannot be
/// implemented outside this crate.
///
/// All methods return builder structs that implement [`GpuOp`](crate::stream::GpuOp).
/// Call `.call()` to execute the operation, or chain `.stream(s)` first to run
/// on a non-default CUDA stream.
pub trait DictionaryExt: private::Sealed {
    /// Decodes a dictionary column back to its underlying value type.
    ///
    /// Expands the dictionary encoding by replacing each index with its
    /// corresponding key value. The returned [`Column`] has the same type as
    /// the dictionary's keys column.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not dictionary-encoded.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::dictionary::{DictionaryExt, dictionary_encode};
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let dict = dictionary_encode(&col.view(), TypeId::INT32).call()?;
    /// let plain = dict.view().dict_decode().call()?;
    /// assert_eq!(plain.type_id(), TypeId::STRING);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dict_decode(&self) -> DictDecode<'_>;

    /// Adds new keys to a dictionary column's key set.
    ///
    /// The `new_keys` column must have the same type as the existing keys.
    /// Existing indices remain unchanged; the new keys are merged into the
    /// sorted key set. Duplicate keys are ignored.
    ///
    /// Returns a new dictionary [`Column`] with the expanded key set.
    ///
    /// # Errors
    ///
    /// Returns an error if `new_keys` has a different type than the existing keys,
    /// or if the column is not dictionary-encoded.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::dictionary::DictionaryExt;
    /// use cudf::stream::GpuOp;
    ///
    /// let new_keys = Column::from_strings(&["d", "e"]).call()?;
    /// let expanded = dict.view().dict_add_keys(&new_keys.view()).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dict_add_keys<'a>(&'a self, new_keys: &'a ColumnView<'a>) -> DictAddKeys<'a>;

    /// Removes the specified keys from a dictionary column.
    ///
    /// Any row whose index pointed to a removed key becomes null in the output.
    /// The `keys_to_remove` column must have the same type as the dictionary's
    /// keys. Keys not present in the dictionary are silently ignored.
    ///
    /// Returns a new dictionary [`Column`] with the reduced key set.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not dictionary-encoded.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::dictionary::DictionaryExt;
    /// use cudf::stream::GpuOp;
    ///
    /// let to_remove = Column::from_strings(&["a"]).call()?;
    /// let pruned = dict.view().dict_remove_keys(&to_remove.view()).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dict_remove_keys<'a>(&'a self, keys_to_remove: &'a ColumnView<'a>) -> DictRemoveKeys<'a>;

    /// Replaces the entire key set of a dictionary column.
    ///
    /// The `keys` column provides the new set of keys. Indices are recomputed
    /// so that each row maps to the matching key in the new set. Rows whose
    /// original key is not present in `keys` become null.
    ///
    /// Returns a new dictionary [`Column`] with the replaced key set and
    /// updated indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not dictionary-encoded.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::dictionary::DictionaryExt;
    /// use cudf::stream::GpuOp;
    ///
    /// let new_keys = Column::from_strings(&["x", "y", "z"]).call()?;
    /// let reindexed = dict.view().dict_set_keys(&new_keys.view()).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dict_set_keys<'a>(&'a self, keys: &'a ColumnView<'a>) -> DictSetKeys<'a>;

    /// Removes keys that are not referenced by any index in the column.
    ///
    /// This is a compaction operation: unused keys are dropped and indices are
    /// recomputed to remain contiguous. No row values change.
    ///
    /// Returns a new dictionary [`Column`] with only the keys that are
    /// actually used.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not dictionary-encoded.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::dictionary::DictionaryExt;
    /// use cudf::stream::GpuOp;
    ///
    /// let compacted = dict.view().dict_remove_unused_keys().call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dict_remove_unused_keys(&self) -> DictRemoveUnusedKeys<'_>;

    /// Looks up a single key in the dictionary and returns its index as a [`Scalar`].
    ///
    /// The `key` scalar must have the same type as the dictionary's keys. If
    /// the key is found, the returned scalar contains its integer index value.
    /// If the key is not found, the returned scalar is null.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not dictionary-encoded.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::dictionary::DictionaryExt;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let key = Scalar::from_string("b");
    /// let index = dict.view().dict_get_index(&key).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    fn dict_get_index<'a>(&'a self, key: &'a Scalar) -> DictGetIndex<'a>;
}

// ---------------------------------------------------------------------------
// Builder structs — trait methods
// ---------------------------------------------------------------------------

/// Builder for [`DictionaryExt::dict_decode`].
///
/// Created by [`DictionaryExt::dict_decode`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DictDecode<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DictDecode<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::dictionary::ffi::dictionary_decode(self.view.0, self.stream.as_raw())?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DictionaryExt::dict_add_keys`].
///
/// Created by [`DictionaryExt::dict_add_keys`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DictAddKeys<'a> {
    view: &'a ColumnView<'a>,
    new_keys: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DictAddKeys<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::dictionary::ffi::dictionary_add_keys(
            self.view.0,
            self.new_keys.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DictionaryExt::dict_remove_keys`].
///
/// Created by [`DictionaryExt::dict_remove_keys`]. Call `.call()` to execute,
/// or chain `.stream(s)` to run on a specific CUDA stream.
pub struct DictRemoveKeys<'a> {
    view: &'a ColumnView<'a>,
    keys_to_remove: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DictRemoveKeys<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::dictionary::ffi::dictionary_remove_keys(
            self.view.0,
            self.keys_to_remove.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DictionaryExt::dict_set_keys`].
///
/// Created by [`DictionaryExt::dict_set_keys`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DictSetKeys<'a> {
    view: &'a ColumnView<'a>,
    keys: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DictSetKeys<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::dictionary::ffi::dictionary_set_keys(
            self.view.0,
            self.keys.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DictionaryExt::dict_remove_unused_keys`].
///
/// Created by [`DictionaryExt::dict_remove_unused_keys`]. Call `.call()` to
/// execute, or chain `.stream(s)` to run on a specific CUDA stream.
pub struct DictRemoveUnusedKeys<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for DictRemoveUnusedKeys<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::dictionary::ffi::dictionary_remove_unused_keys(
            self.view.0,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
    }
}

/// Builder for [`DictionaryExt::dict_get_index`].
///
/// Created by [`DictionaryExt::dict_get_index`]. Call `.call()` to execute, or
/// chain `.stream(s)` to run on a specific CUDA stream.
pub struct DictGetIndex<'a> {
    view: &'a ColumnView<'a>,
    key: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for DictGetIndex<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.key)?;
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
///
/// Created by [`dictionary_encode`]. Call `.call()` to execute, or chain
/// `.stream(s)` to run on a specific CUDA stream.
pub struct DictionaryEncode<'a> {
    col: &'a ColumnView<'a>,
    indices_type: TypeId,
    stream: Stream,
}

/// Dictionary-encodes a plain column, returning a `DICTIONARY32` column.
///
/// The unique values become the dictionary keys and each row is replaced by
/// an integer index into the key set. The `indices_type` parameter controls
/// the integer type used for the indices column (typically [`TypeId::INT32`]).
///
/// # Errors
///
/// Returns an error if the column type cannot be dictionary-encoded.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::data_type::TypeId;
/// use cudf::dictionary::dictionary_encode;
/// use cudf::stream::GpuOp;
///
/// let col = Column::from_strings(&["a", "b", "a", "c"]).call()?;
/// let dict = dictionary_encode(&col.view(), TypeId::INT32).call()?;
/// assert_eq!(dict.type_id(), TypeId::DICTIONARY32);
/// # Ok::<(), cudf::error::Error>(())
/// ```
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

impl crate::stream::GpuOp for DictionaryEncode<'_> {
    type Output = crate::column::UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::dictionary::ffi::dictionary_encode(
            self.col.0,
            self.indices_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::column::RawColumn(c))
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
