// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use crate::data_type::TypeId;

/// A non-owning, immutable view of a GPU column.
///
/// The lifetime parameter ties this view to the owning [`Table`](crate::Table).
pub struct ColumnView<'a>(pub(crate) &'a cudf_sys::ffi::column_view);

impl ColumnView<'_> {
    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        cudf_sys::ffi::column_view_size(self.0) as usize
    }

    /// Returns `true` if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the count of null elements.
    pub fn null_count(&self) -> usize {
        cudf_sys::ffi::column_view_null_count(self.0) as usize
    }

    /// Returns `true` if the view contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_view_has_nulls(self.0)
    }

    /// Returns the offset of the view into the underlying data.
    pub fn offset(&self) -> usize {
        cudf_sys::ffi::column_view_offset(self.0) as usize
    }

    /// Returns the type identifier of the column.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_view_type_id(self.0);
        // Safety: type_id values from C++ are valid TypeId discriminants
        unsafe { std::mem::transmute::<i32, TypeId>(id) }
    }
}
