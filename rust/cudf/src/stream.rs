// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA stream handle for GPU operations.

#[doc(alias = "cuda_stream_view")]
/// A CUDA stream handle.
///
/// `Stream` is a lightweight `Copy` type wrapping a `cudaStream_t`.
/// Use `Stream::default_stream()` to get the cudf default stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Stream(usize);

impl Stream {
    /// Returns the default CUDA stream used by cudf.
    pub fn default_stream() -> Self {
        Self(cudf_sys::ffi::get_default_stream())
    }

    /// Creates a `Stream` from a raw `cudaStream_t` handle.
    ///
    /// The caller must ensure the stream handle remains valid for the
    /// duration of any operation using this `Stream`.
    pub fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    /// Returns the raw stream handle as a `usize`.
    pub fn as_raw(self) -> usize {
        self.0
    }
}
