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

impl Default for Stream {
    fn default() -> Self {
        Self(cudf_sys::ffi::get_default_stream())
    }
}

impl Stream {
    /// Returns the default CUDA stream used by cudf.
    pub fn default_stream() -> Self {
        Self::default()
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

/// Trait for GPU operation builders that support stream selection and execution.
///
/// All builders returned by cudf operations implement this trait. Use
/// `.stream()` to override the CUDA stream (defaults to the cudf default
/// stream), then `.call()` to execute.
pub trait GpuOp: Sized {
    /// The result type produced by [`call`](GpuOp::call).
    type Output;

    /// Sets the CUDA stream for this operation.
    fn stream(self, stream: Stream) -> Self;

    /// Executes the operation.
    fn call(self) -> crate::error::Result<Self::Output>;
}
