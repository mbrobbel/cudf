// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA stream handle for GPU operations.

#[doc(alias = "cuda_stream_view")]
/// A CUDA stream handle.
///
/// `Stream` is a lightweight `Copy` type wrapping a raw `cudaStream_t` pointer
/// (represented as `usize`). Streams provide ordering guarantees for GPU
/// operations: operations submitted to the same stream execute in order, while
/// operations on different streams may execute concurrently.
///
/// Use [`Stream::default_stream()`] to obtain the cudf default stream, which
/// is used by all operations unless overridden via the
/// [`.stream()`](GpuOp::stream) builder method.
///
/// # Examples
///
/// ```ignore
/// use cudf::stream::Stream;
///
/// // Use the default stream (most common case)
/// let stream = Stream::default_stream();
///
/// // Wrap an existing CUDA stream handle
/// let custom = Stream::from_raw(raw_cuda_stream);
/// ```
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
    ///
    /// This is the stream that all cudf operations use unless a custom stream
    /// is set via [`.stream()`](GpuOp::stream). In a per-thread default
    /// stream build of cudf, each thread gets its own default stream; in the
    /// legacy default stream build, all threads share the same stream.
    pub fn default_stream() -> Self {
        Self::default()
    }

    /// Creates a `Stream` from a raw `cudaStream_t` handle.
    ///
    /// The `raw` value is the `cudaStream_t` pointer cast to `usize`.
    ///
    /// # Safety note
    ///
    /// While this method is not marked `unsafe` (the unsafety is deferred to
    /// the FFI call site in cudf-sys), the caller must ensure:
    /// - The stream handle remains valid for the duration of any operation
    ///   using this `Stream`.
    /// - The stream was created by the CUDA runtime on the current device.
    pub fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    /// Returns the raw `cudaStream_t` handle as a `usize`.
    ///
    /// This is useful when interacting with other CUDA libraries that accept
    /// raw stream handles.
    pub fn as_raw(self) -> usize {
        self.0
    }

    /// Creates a `Stream` from an owning [`rmm::stream::Stream`].
    ///
    /// The returned cudf `Stream` borrows the raw handle. The caller must
    /// ensure the owning `rmm::stream::Stream` outlives any operations
    /// submitted to this handle.
    pub fn from_rmm(stream: &rmm::stream::Stream) -> Self {
        Self(stream.as_raw())
    }
}

impl From<&rmm::stream::Stream> for Stream {
    fn from(stream: &rmm::stream::Stream) -> Self {
        Self(stream.as_raw())
    }
}

/// Trait for GPU operation builders that support stream selection and execution.
///
/// All builders returned by cudf operations implement this trait. The typical
/// usage pattern is:
///
/// 1. Call a method that returns a builder (e.g. `column.view().fill(...)`).
/// 2. Optionally call `.stream()` to override the CUDA stream.
/// 3. Call `.call()` to execute the operation and get the result.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::{GpuOp, Stream};
///
/// let col = Column::from_scalar(&Scalar::from_i32(0), 10)
///     .stream(Stream::default_stream())
///     .call()?;
/// ```
pub trait GpuOp: Sized {
    /// The result type produced by [`call`](GpuOp::call).
    type Output;

    /// Sets the CUDA stream for this operation.
    ///
    /// If not called, the operation uses the cudf default stream
    /// ([`Stream::default_stream()`]).
    fn stream(self, stream: Stream) -> Self;

    /// Executes the GPU operation and returns the result.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](crate::error::Error) if the underlying libcudf
    /// call fails.
    fn call(self) -> crate::error::Result<Self::Output>;
}
