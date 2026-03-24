// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Untyped GPU memory buffers.
//!
//! [`DeviceBuffer`] is the primary type for managing GPU memory allocations.
//! It wraps `rmm::device_buffer` and provides RAII-based lifetime management
//! — the GPU memory is freed when the buffer is dropped.

use std::fmt;

use cxx::UniquePtr;

/// An untyped GPU memory buffer.
///
/// Wraps `rmm::device_buffer`, providing an owning handle to a contiguous
/// region of GPU memory. The buffer contents are uninitialized after
/// allocation. Memory is freed automatically when the buffer is dropped.
///
/// # Examples
///
/// ```ignore
/// use rmm::buffer::DeviceBuffer;
///
/// let buf = DeviceBuffer::new(1024);
/// assert_eq!(buf.size(), 1024);
/// assert!(!buf.is_empty());
///
/// let ptr = buf.as_ptr();
/// // pass `ptr` to GPU kernels...
/// ```
#[doc(alias = "rmm::device_buffer")]
pub struct DeviceBuffer(UniquePtr<rmm_sys::ffi::DeviceBuffer>);

impl DeviceBuffer {
    /// Allocates a new device buffer of `size` uninitialized bytes.
    ///
    /// Uses the default CUDA stream (stream `0`) for the allocation.
    ///
    /// # Parameters
    ///
    /// - `size`: Number of bytes to allocate.
    pub fn new(size: usize) -> Self {
        Self(rmm_sys::ffi::device_buffer_new(size, 0))
    }

    /// Allocates a new device buffer of `size` uninitialized bytes on the
    /// given stream.
    ///
    /// # Parameters
    ///
    /// - `size`: Number of bytes to allocate.
    /// - `stream`: Raw `cudaStream_t` handle as `usize`.
    pub fn with_stream(size: usize, stream: usize) -> Self {
        Self(rmm_sys::ffi::device_buffer_new(size, stream))
    }

    /// Returns the size in bytes of the buffer.
    ///
    /// This is the logical size, which may be less than [`capacity`](Self::capacity).
    pub fn size(&self) -> usize {
        rmm_sys::ffi::device_buffer_size(&self.0)
    }

    /// Returns the capacity in bytes of the underlying allocation.
    ///
    /// The invariant `size() <= capacity()` always holds.
    pub fn capacity(&self) -> usize {
        rmm_sys::ffi::device_buffer_capacity(&self.0)
    }

    /// Returns `true` if the buffer is empty (`size == 0`).
    pub fn is_empty(&self) -> bool {
        rmm_sys::ffi::device_buffer_is_empty(&self.0)
    }

    /// Resizes the buffer to `new_size` bytes using the default stream.
    ///
    /// If `new_size <= capacity`, only the logical size is updated.
    /// If `new_size > capacity`, a new allocation is made and existing
    /// contents are copied. New bytes are uninitialized.
    ///
    /// # Parameters
    ///
    /// - `new_size`: The new size in bytes.
    pub fn resize(&mut self, new_size: usize) {
        rmm_sys::ffi::device_buffer_resize(self.0.pin_mut(), new_size, 0);
    }

    /// Resizes the buffer to `new_size` bytes on the given stream.
    ///
    /// # Parameters
    ///
    /// - `new_size`: The new size in bytes.
    /// - `stream`: Raw `cudaStream_t` handle as `usize`.
    pub fn resize_on_stream(&mut self, new_size: usize, stream: usize) {
        rmm_sys::ffi::device_buffer_resize(self.0.pin_mut(), new_size, stream);
    }

    /// Returns the raw device pointer as `usize`.
    ///
    /// The returned value is an opaque handle representing a GPU memory
    /// address. It should only be passed to other FFI functions or GPU
    /// kernels that expect a device pointer.
    pub fn as_ptr(&self) -> usize {
        rmm_sys::ffi::device_buffer_data(&self.0)
    }
}

impl fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("size", &self.size())
            .field("capacity", &self.capacity())
            .field("ptr", &format_args!("{:#x}", self.as_ptr()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer() {
        let buf = DeviceBuffer::new(1024);
        assert_eq!(buf.size(), 1024);
        assert!(buf.capacity() >= 1024);
        assert!(!buf.is_empty());
    }

    #[test]
    fn empty_buffer() {
        let buf = DeviceBuffer::new(0);
        assert_eq!(buf.size(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn resize_buffer() {
        let mut buf = DeviceBuffer::new(256);
        assert_eq!(buf.size(), 256);

        buf.resize(512);
        assert_eq!(buf.size(), 512);

        buf.resize(128);
        assert_eq!(buf.size(), 128);
    }

    #[test]
    fn buffer_has_valid_ptr() {
        let buf = DeviceBuffer::new(1024);
        // Non-empty buffers should have a non-null pointer.
        assert_ne!(buf.as_ptr(), 0);
    }

    #[test]
    fn buffer_debug() {
        let buf = DeviceBuffer::new(256);
        let debug = format!("{buf:?}");
        assert!(debug.contains("DeviceBuffer"));
        assert!(debug.contains("size: 256"));
    }

    #[test]
    fn with_stream_default() {
        let buf = DeviceBuffer::with_stream(512, 0);
        assert_eq!(buf.size(), 512);
    }
}
