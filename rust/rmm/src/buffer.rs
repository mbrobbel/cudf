// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Untyped GPU memory buffers.
//!
//! [`DeviceBuffer`] is the primary type for managing GPU memory allocations.
//! It wraps `rmm::device_buffer` and provides RAII-based lifetime management
//! — the GPU memory is freed when the buffer is dropped.

use std::fmt;

use crate::error::Result;
use crate::gpu_context::{Allocator, ContextBound};
use cxx::UniquePtr;

/// An untyped GPU memory buffer.
///
/// Wraps `rmm::device_buffer`, providing an owning handle to a contiguous
/// region of GPU memory. The buffer contents are uninitialized after
/// allocation. Memory is freed automatically when the buffer is dropped.
///
/// # Examples
///
/// ```no_run
/// use rmm::buffer::DeviceBuffer;
///
/// let buf = DeviceBuffer::new(1024)?;
/// assert_eq!(buf.size(), 1024);
/// assert!(!buf.is_empty());
///
/// let ptr = buf.as_ptr();
/// // pass `ptr` to GPU kernels...
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[doc(alias = "rmm::device_buffer")]
pub struct DeviceBuffer<'ctx, Brand = ()> {
    raw: ContextBound<'ctx, Brand, UniquePtr<rmm_sys::ffi::DeviceBuffer>>,
}

impl DeviceBuffer<'static, ()> {
    /// Allocates a new device buffer of `size` uninitialized bytes.
    ///
    /// Uses the default CUDA stream (stream `0`) for the allocation.
    ///
    /// # Parameters
    ///
    /// - `size`: Number of bytes to allocate.
    pub fn new(size: usize) -> Result<Self> {
        Ok(Self::from_ffi_unbound(rmm_sys::ffi::device_buffer_new(
            size, 0,
        )?))
    }

    /// Allocates a new device buffer of `size` uninitialized bytes on the
    /// given stream.
    ///
    /// # Parameters
    ///
    /// - `size`: Number of bytes to allocate.
    /// - `stream`: Raw `cudaStream_t` handle as `usize`.
    pub fn with_stream(size: usize, stream: usize) -> Result<Self> {
        Ok(Self::from_ffi_unbound(rmm_sys::ffi::device_buffer_new(
            size, stream,
        )?))
    }
}

impl<'ctx, Brand> DeviceBuffer<'ctx, Brand> {
    fn from_ffi_unbound(raw: UniquePtr<rmm_sys::ffi::DeviceBuffer>) -> Self {
        Self {
            raw: ContextBound::unbound(raw),
        }
    }

    fn from_ffi_bound(
        raw: UniquePtr<rmm_sys::ffi::DeviceBuffer>,
        alloc: Allocator<'ctx, Brand>,
    ) -> Self {
        Self {
            raw: ContextBound::bound(raw, alloc),
        }
    }

    /// Allocates a new device buffer of `size` bytes in `alloc`.
    ///
    /// The returned buffer is tied to `alloc`'s context lifetime, so it
    /// cannot be used with allocators or execution handles from another
    /// context in safe Rust.
    ///
    /// ```no_run
    /// use rmm::buffer::DeviceBuffer;
    /// use rmm::gpu_context::GpuContext;
    ///
    /// let ctx = GpuContext::<()>::current()?;
    /// let alloc = ctx.default_device_allocator();
    /// let buf = DeviceBuffer::new_in(&alloc, 1024)?;
    /// assert_eq!(buf.size(), 1024);
    /// # Ok::<(), rmm::error::Error>(())
    /// ```
    ///
    /// ```compile_fail
    /// use rmm::buffer::DeviceBuffer;
    /// use rmm::gpu_context::{DeviceAllocator, GpuContext};
    ///
    /// fn require_same_context<'ctx>(
    ///     _: DeviceBuffer<'ctx>,
    ///     _: DeviceAllocator<'ctx>,
    /// ) {}
    ///
    /// let ctx_a = GpuContext::<()>::current().unwrap();
    /// let ctx_b = GpuContext::<()>::current().unwrap();
    /// let alloc_a = ctx_a.default_device_allocator();
    /// let alloc_b = ctx_b.default_device_allocator();
    /// let buf = DeviceBuffer::new_in(&alloc_a, 64).unwrap();
    ///
    /// require_same_context(buf, alloc_b);
    /// ```
    pub fn new_in(alloc: &Allocator<'ctx, Brand>, size: usize) -> Result<Self> {
        alloc.with_current(|| {
            Ok(Self::from_ffi_bound(
                rmm_sys::ffi::device_buffer_new(size, 0)?,
                *alloc,
            ))
        })?
    }

    /// Allocates a new device buffer of `size` bytes in `alloc` on `stream`.
    pub fn with_stream_in(
        alloc: &Allocator<'ctx, Brand>,
        size: usize,
        stream: usize,
    ) -> Result<Self> {
        alloc.with_current(|| {
            Ok(Self::from_ffi_bound(
                rmm_sys::ffi::device_buffer_new(size, stream)?,
                *alloc,
            ))
        })?
    }

    /// Returns the size in bytes of the buffer.
    ///
    /// This is the logical size, which may be less than [`capacity`](Self::capacity).
    pub fn size(&self) -> usize {
        rmm_sys::ffi::device_buffer_size(&self.raw)
    }

    /// Returns the capacity in bytes of the underlying allocation.
    ///
    /// The invariant `size() <= capacity()` always holds.
    pub fn capacity(&self) -> usize {
        rmm_sys::ffi::device_buffer_capacity(&self.raw)
    }

    /// Returns `true` if the buffer is empty (`size == 0`).
    pub fn is_empty(&self) -> bool {
        rmm_sys::ffi::device_buffer_is_empty(&self.raw)
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
    pub fn resize(&mut self, new_size: usize) -> Result<()> {
        rmm_sys::ffi::device_buffer_resize(self.raw.pin_mut(), new_size, 0)?;
        Ok(())
    }

    /// Resizes the buffer to `new_size` bytes on the given stream.
    ///
    /// # Parameters
    ///
    /// - `new_size`: The new size in bytes.
    /// - `stream`: Raw `cudaStream_t` handle as `usize`.
    pub fn resize_on_stream(&mut self, new_size: usize, stream: usize) -> Result<()> {
        rmm_sys::ffi::device_buffer_resize(self.raw.pin_mut(), new_size, stream)?;
        Ok(())
    }

    /// Returns the raw device pointer as `usize`.
    ///
    /// The returned value is an opaque handle representing a GPU memory
    /// address. It should only be passed to other FFI functions or GPU
    /// kernels that expect a device pointer.
    pub fn as_ptr(&self) -> usize {
        rmm_sys::ffi::device_buffer_data(&self.raw)
    }
}

impl<Brand> fmt::Debug for DeviceBuffer<'_, Brand> {
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
    use crate::gpu_context::GpuContext;

    #[test]
    fn new_buffer() {
        let _test_lock = crate::test_lock();
        let buf = DeviceBuffer::new(1024).unwrap();
        assert_eq!(buf.size(), 1024);
        assert!(buf.capacity() >= 1024);
        assert!(!buf.is_empty());
    }

    #[test]
    fn empty_buffer() {
        let _test_lock = crate::test_lock();
        let buf = DeviceBuffer::new(0).unwrap();
        assert_eq!(buf.size(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn resize_buffer() {
        let _test_lock = crate::test_lock();
        let mut buf = DeviceBuffer::new(256).unwrap();
        assert_eq!(buf.size(), 256);

        buf.resize(512).unwrap();
        assert_eq!(buf.size(), 512);

        buf.resize(128).unwrap();
        assert_eq!(buf.size(), 128);
    }

    #[test]
    fn buffer_has_valid_ptr() {
        let _test_lock = crate::test_lock();
        let buf = DeviceBuffer::new(1024).unwrap();
        // Non-empty buffers should have a non-null pointer.
        assert_ne!(buf.as_ptr(), 0);
    }

    #[test]
    fn buffer_debug() {
        let _test_lock = crate::test_lock();
        let buf = DeviceBuffer::new(256).unwrap();
        let debug = format!("{buf:?}");
        assert!(debug.contains("DeviceBuffer"));
        assert!(debug.contains("size: 256"));
    }

    #[test]
    fn with_stream_default() {
        let _test_lock = crate::test_lock();
        let buf = DeviceBuffer::with_stream(512, 0).unwrap();
        assert_eq!(buf.size(), 512);
    }

    #[test]
    fn new_in_binds_buffer_to_context() {
        let _test_lock = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.default_device_allocator();
        let buf = DeviceBuffer::new_in(&alloc, 1024).unwrap();
        assert_eq!(buf.size(), 1024);
        assert!(!buf.is_empty());
    }
}
