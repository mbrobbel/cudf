// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU memory resource allocators.
//!
//! Memory resources control how GPU memory is allocated and freed. They are
//! the central abstraction in RMM, allowing applications to swap allocation
//! strategies without changing application code.
//!
//! # Available resources
//!
//! | Type | Allocation strategy |
//! |---|---|
//! | [`CudaMemoryResource`] | `cudaMalloc`/`cudaFree` (default, simple) |
//! | [`PoolMemoryResource`] | Pre-allocated pool with sub-allocation (best for production) |
//! | [`CudaAsyncMemoryResource`] | `cudaMallocAsync`/`cudaFreeAsync` (CUDA 11.2+) |
//! | [`ManagedMemoryResource`] | `cudaMallocManaged` (UVM, pages migrate automatically) |
//!
//! # Per-device default
//!
//! Every CUDA device has a default memory resource that is used by all RMM
//! allocations (including [`DeviceBuffer`](crate::buffer::DeviceBuffer)) unless
//! an explicit resource is provided. Use [`set_current_device_resource`] and
//! [`reset_current_device_resource`] to control it.
//!
//! # Examples
//!
//! ```no_run
//! use rmm::memory_resource::{PoolMemoryResource, set_current_device_resource};
//!
//! // Create a pool that pre-allocates 1 GiB, growing up to 4 GiB.
//! let pool = PoolMemoryResource::with_limits(1 << 30, 4 << 30)?;
//! let _guard = set_current_device_resource(pool.as_ref());
//!
//! // All subsequent RMM allocations use the pool.
//! let buf = rmm::buffer::DeviceBuffer::new(4096)?;
//! # Ok::<(), rmm::error::Error>(())
//! ```

use std::marker::PhantomData;

use cxx::UniquePtr;

use crate::error::Result;
use crate::gpu_context::{ContextMarker, GpuContext};

// ---------------------------------------------------------------------------
// Concrete memory resources
// ---------------------------------------------------------------------------

/// The default `cudaMalloc`/`cudaFree`-based allocator.
///
/// Every allocation calls `cudaMalloc` and every deallocation calls
/// `cudaFree`. This is the simplest strategy but has high per-allocation
/// overhead, making it unsuitable for workloads with many small allocations.
///
/// This is the built-in default memory resource — if no resource is
/// explicitly configured, all RMM allocations use this strategy.
#[doc(alias = "rmm::mr::cuda_memory_resource")]
pub struct CudaMemoryResource(UniquePtr<rmm_sys::ffi::CudaMemoryResource>);

impl CudaMemoryResource {
    /// Creates a new `CudaMemoryResource`.
    pub fn new() -> Result<Self> {
        Ok(Self(rmm_sys::ffi::cuda_memory_resource_new()?))
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    ///
    /// The returned ref borrows `self` — the `CudaMemoryResource` must
    /// outlive all allocations made through the ref.
    pub fn as_ref(&self) -> MemoryResourceRef<'_> {
        MemoryResourceRef {
            raw: rmm_sys::ffi::memory_resource_ref_from_cuda(&self.0),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl std::fmt::Debug for CudaMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMemoryResource").finish()
    }
}

/// A stream-ordered pooled sub-allocator backed by `cudaMallocAsync`.
///
/// Uses CUDA's built-in asynchronous memory allocator, available on
/// CUDA 11.2+ with a compatible driver. Provides stream-ordered allocation
/// without requiring a user-space pool.
#[doc(alias = "rmm::mr::cuda_async_memory_resource")]
pub struct CudaAsyncMemoryResource(UniquePtr<rmm_sys::ffi::CudaAsyncMemoryResource>);

impl CudaAsyncMemoryResource {
    /// Creates a new `CudaAsyncMemoryResource` with default pool settings.
    pub fn new() -> Result<Self> {
        Ok(Self(rmm_sys::ffi::cuda_async_memory_resource_new()?))
    }

    /// Creates a new `CudaAsyncMemoryResource` with explicit pool parameters.
    ///
    /// * `initial_pool_size` — initial pool size in bytes (primed by
    ///   allocating and immediately freeing this amount).
    /// * `release_threshold` — when the pool exceeds this size, unused memory
    ///   is released at the next synchronization event.
    pub fn with_limits(initial_pool_size: usize, release_threshold: usize) -> Result<Self> {
        Ok(Self(rmm_sys::ffi::cuda_async_memory_resource_with_size(
            initial_pool_size,
            release_threshold,
        )?))
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    pub fn as_ref(&self) -> MemoryResourceRef<'_> {
        MemoryResourceRef {
            raw: rmm_sys::ffi::memory_resource_ref_from_async(&self.0),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl std::fmt::Debug for CudaAsyncMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAsyncMemoryResource").finish()
    }
}

/// CUDA Unified Virtual Memory (UVM) allocator.
///
/// Allocates managed memory via `cudaMallocManaged`. Pages automatically
/// migrate between host and device as needed, which is convenient for
/// prototyping but may have lower peak throughput than explicit device
/// allocations.
#[doc(alias = "rmm::mr::managed_memory_resource")]
pub struct ManagedMemoryResource(UniquePtr<rmm_sys::ffi::ManagedMemoryResource>);

impl ManagedMemoryResource {
    /// Creates a new `ManagedMemoryResource`.
    pub fn new() -> Result<Self> {
        Ok(Self(rmm_sys::ffi::managed_memory_resource_new()?))
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    pub fn as_ref(&self) -> MemoryResourceRef<'_> {
        MemoryResourceRef {
            raw: rmm_sys::ffi::memory_resource_ref_from_managed(&self.0),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl std::fmt::Debug for ManagedMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedMemoryResource").finish()
    }
}

/// A coalescing best-fit pool sub-allocator backed by `cudaMalloc`.
///
/// This is the most commonly used memory resource in production. It
/// pre-allocates a large chunk of device memory and sub-allocates from it,
/// dramatically reducing the overhead of frequent small allocations.
///
/// The pool grows automatically up to `maximum_size`. Freed blocks are
/// coalesced and reused.
#[doc(alias = "rmm::mr::pool_memory_resource")]
pub struct PoolMemoryResource(UniquePtr<rmm_sys::ffi::PoolMemoryResource>);

impl PoolMemoryResource {
    /// Creates a new `PoolMemoryResource` with default settings.
    ///
    /// The initial pool size is 50% of free device memory. The maximum
    /// pool size is the total device memory.
    pub fn new() -> Result<Self> {
        Ok(Self(rmm_sys::ffi::pool_memory_resource_new()?))
    }

    /// Creates a new `PoolMemoryResource` with explicit size limits.
    ///
    /// * `initial_size` — minimum initial pool size in bytes (must be
    ///   aligned to 256 bytes).
    /// * `maximum_size` — maximum pool size in bytes (must be aligned
    ///   to 256 bytes).
    pub fn with_limits(initial_size: usize, maximum_size: usize) -> Result<Self> {
        Ok(Self(rmm_sys::ffi::pool_memory_resource_with_size(
            initial_size,
            maximum_size,
        )?))
    }

    /// Returns the current total pool size in bytes, including both
    /// allocated and free regions.
    pub fn pool_size(&self) -> usize {
        rmm_sys::ffi::pool_memory_resource_pool_size(&self.0)
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    pub fn as_ref(&self) -> MemoryResourceRef<'_> {
        MemoryResourceRef {
            raw: rmm_sys::ffi::memory_resource_ref_from_pool(&self.0),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl std::fmt::Debug for PoolMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolMemoryResource")
            .field("pool_size", &self.pool_size())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Type-erased reference
// ---------------------------------------------------------------------------

/// A non-owning, type-erased handle to any memory resource.
///
/// Obtained from the `as_ref()` method on any concrete memory resource. The
/// referenced memory resource must outlive this handle, and that requirement
/// is enforced by the borrow carried in `'mr`.
#[doc(alias = "rmm::mr::device_memory_resource")]
pub struct MemoryResourceRef<'mr> {
    raw: UniquePtr<rmm_sys::ffi::MemoryResourceRef>,
    _lifetime: PhantomData<&'mr ()>,
    _not_send_sync: PhantomData<std::rc::Rc<()>>,
}

impl std::fmt::Debug for MemoryResourceRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryResourceRef").finish()
    }
}

impl Clone for MemoryResourceRef<'_> {
    fn clone(&self) -> Self {
        Self {
            raw: rmm_sys::ffi::memory_resource_ref_clone(&self.raw),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-device resource management
// ---------------------------------------------------------------------------

/// Sets the memory resource for the current CUDA device.
///
/// The returned guard keeps `mr` borrowed for the duration of the override, so
/// the installed allocator cannot be dropped while it remains active.
///
/// # Examples
///
/// ```no_run
/// use rmm::memory_resource::{PoolMemoryResource, set_current_device_resource};
///
/// let pool = PoolMemoryResource::new()?;
/// let _guard = set_current_device_resource(pool.as_ref());
/// // ... all allocations now go through `pool` ...
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[doc(alias = "rmm::mr::set_current_device_resource")]
pub fn set_current_device_resource(mr: MemoryResourceRef<'_>) -> ScopedCurrentDeviceResource<'_> {
    ScopedCurrentDeviceResource {
        previous: rmm_sys::ffi::set_current_device_resource(&mr.raw),
        _current: mr,
        _not_send: std::marker::PhantomData,
    }
}

/// Resets the current device's memory resource to the built-in default
/// (`CudaMemoryResource`).
#[doc(alias = "rmm::mr::reset_current_device_resource_ref")]
pub fn reset_current_device_resource() {
    rmm_sys::ffi::reset_current_device_resource();
}

/// RAII guard for temporarily overriding the current device memory resource.
///
/// The guard borrows the resource that was installed, preventing the caller
/// from dropping that allocator while it remains active on the current thread.
pub struct ScopedCurrentDeviceResource<'a> {
    previous: UniquePtr<rmm_sys::ffi::MemoryResourceRef>,
    _current: MemoryResourceRef<'a>,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for ScopedCurrentDeviceResource<'_> {
    fn drop(&mut self) {
        let _ = rmm_sys::ffi::set_current_device_resource(&self.previous);
    }
}

/// A memory resource that allocates page-locked (pinned) host memory.
///
/// Pinned memory is registered with the CUDA driver, enabling DMA transfers
/// at full `PCIe` bandwidth. Use this for host-side buffers that will be
/// transferred to/from the GPU via `cudaMemcpyAsync`.
pub struct PinnedHostMemoryResource(UniquePtr<rmm_sys::ffi::PinnedHostMemoryResource>);

impl PinnedHostMemoryResource {
    /// Creates a new pinned host memory resource.
    pub fn new() -> Result<Self> {
        Ok(Self(rmm_sys::ffi::pinned_host_memory_resource_new()?))
    }

    /// Allocates `size` bytes of pinned host memory. Returns raw pointer as `usize`.
    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        rmm_sys::ffi::pinned_host_allocate(self.0.pin_mut(), size).map_err(Into::into)
    }

    /// Deallocates pinned host memory.
    pub fn deallocate(&mut self, ptr: usize, size: usize) {
        rmm_sys::ffi::pinned_host_deallocate(self.0.pin_mut(), ptr, size);
    }
}

impl std::fmt::Debug for PinnedHostMemoryResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostMemoryResource").finish()
    }
}

#[derive(Debug)]
enum PinnedBufferOwner<'ctx, Brand> {
    Unbound(PinnedHostMemoryResource),
    Context {
        context: &'ctx GpuContext<Brand>,
        resource_id: usize,
    },
}

/// RAII wrapper for a page-locked host allocation.
///
/// This type owns a pinned host allocation and frees it on drop. It provides
/// a safer lifetime model than raw `usize` pointers, while still exposing the
/// raw address for low-level CUDA interop.
pub struct PinnedHostBuffer<'ctx, Brand = ()> {
    owner: PinnedBufferOwner<'ctx, Brand>,
    ptr: usize,
    size: usize,
    _context_marker: ContextMarker<'ctx>,
    _brand: PhantomData<fn() -> Brand>,
}

impl PinnedHostBuffer<'static, ()> {
    /// Allocates a pinned host buffer of `size` bytes.
    pub fn new(size: usize) -> Result<Self> {
        let mut mr = PinnedHostMemoryResource::new()?;
        let ptr = mr.allocate(size)?;
        Ok(Self {
            owner: PinnedBufferOwner::Unbound(mr),
            ptr,
            size,
            _context_marker: PhantomData,
            _brand: PhantomData,
        })
    }

    /// Creates a pinned host buffer initialized with `src`.
    pub fn from_slice(src: &[u8]) -> Result<Self> {
        let mut buffer = Self::new(src.len())?;
        buffer.write(src)?;
        Ok(buffer)
    }
}

impl<'ctx, Brand> PinnedHostBuffer<'ctx, Brand> {
    pub(crate) fn from_context(
        context: &'ctx GpuContext<Brand>,
        resource_id: usize,
        ptr: usize,
        size: usize,
    ) -> Self {
        Self {
            owner: PinnedBufferOwner::Context {
                context,
                resource_id,
            },
            ptr,
            size,
            _context_marker: PhantomData,
            _brand: PhantomData,
        }
    }

    /// Returns the raw host pointer as `usize`.
    pub fn as_ptr(&self) -> usize {
        self.ptr
    }

    /// Returns the raw mutable host pointer as `usize`.
    pub fn as_mut_ptr(&mut self) -> usize {
        self.ptr
    }

    /// Returns the allocation size in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copies `src` into this pinned host allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`](crate::error::Error::InvalidArgument)
    /// if `src.len() > self.len()`.
    pub fn write(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.size {
            return Err(crate::error::Error::InvalidArgument(format!(
                "source slice length {} exceeds pinned buffer length {}",
                src.len(),
                self.size
            )));
        }
        rmm_sys::ffi::pinned_host_copy_from_slice(self.ptr, src);
        Ok(())
    }

    /// Copies bytes from this pinned allocation into `dst`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`](crate::error::Error::InvalidArgument)
    /// if `dst.len() > self.len()`.
    pub fn read(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() > self.size {
            return Err(crate::error::Error::InvalidArgument(format!(
                "destination slice length {} exceeds pinned buffer length {}",
                dst.len(),
                self.size
            )));
        }
        rmm_sys::ffi::pinned_host_copy_to_slice(self.ptr, dst);
        Ok(())
    }

    /// Copies the entire pinned allocation into a fresh `Vec<u8>`.
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        let mut dst = vec![0; self.size];
        self.read(&mut dst)?;
        Ok(dst)
    }

    /// Copies bytes from a device pointer into this pinned allocation.
    pub fn copy_from_device(&mut self, src_ptr: usize, stream: usize) -> Result<()> {
        copy_device_to_pinned_host(self, src_ptr, stream)
    }

    /// Copies bytes from this pinned allocation to a device pointer.
    pub fn copy_to_device(&self, dst_ptr: usize, stream: usize) -> Result<()> {
        copy_pinned_host_to_device(dst_ptr, self, stream)
    }
}

impl<Brand> std::fmt::Debug for PinnedHostBuffer<'_, Brand> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("ptr", &format_args!("{:#x}", self.ptr))
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<Brand> Drop for PinnedHostBuffer<'_, Brand> {
    fn drop(&mut self) {
        if self.size != 0 {
            match &mut self.owner {
                PinnedBufferOwner::Unbound(mr) => mr.deallocate(self.ptr, self.size),
                PinnedBufferOwner::Context {
                    context,
                    resource_id,
                } => context.deallocate_pinned_bytes(*resource_id, self.ptr, self.size),
            }
        }
    }
}

/// Copies bytes from device to host and blocks until the transfer completes.
///
/// Copies `dst.len()` bytes from the device pointer `src_ptr` to the host
/// slice `dst`. For maximum throughput, `dst` should be in pinned memory.
pub fn memcpy_d2h(dst: &mut [u8], src_ptr: usize, stream: usize) -> Result<()> {
    rmm_sys::ffi::cuda_memcpy_d2h(dst, src_ptr, stream)?;
    stream_synchronize(stream)
}

/// Copies bytes from host to device and blocks until the transfer completes.
///
/// Copies `src.len()` bytes from the host slice `src` to the device pointer
/// `dst_ptr`. For maximum throughput, `src` should be in pinned memory.
pub fn memcpy_h2d(dst_ptr: usize, src: &[u8], stream: usize) -> Result<()> {
    rmm_sys::ffi::cuda_memcpy_h2d(dst_ptr, src, stream)?;
    stream_synchronize(stream)
}

/// Synchronizes a CUDA stream, blocking until all operations complete.
pub fn stream_synchronize(stream: usize) -> Result<()> {
    rmm_sys::ffi::cuda_stream_synchronize_raw(stream)?;
    Ok(())
}

/// Copies bytes from device to host and blocks until the transfer completes.
pub fn copy_device_to_host(dst: &mut [u8], src_ptr: usize, stream: usize) -> Result<()> {
    memcpy_d2h(dst, src_ptr, stream)
}

/// Copies bytes from host to device and blocks until the transfer completes.
pub fn copy_host_to_device(dst_ptr: usize, src: &[u8], stream: usize) -> Result<()> {
    memcpy_h2d(dst_ptr, src, stream)
}

/// Copies bytes from device to a pinned host allocation and blocks until complete.
pub fn copy_device_to_pinned_host<Brand>(
    dst: &mut PinnedHostBuffer<'_, Brand>,
    src_ptr: usize,
    stream: usize,
) -> Result<()> {
    rmm_sys::ffi::cuda_memcpy_d2h_raw(dst.as_mut_ptr(), src_ptr, dst.len(), stream)?;
    stream_synchronize(stream)
}

/// Copies bytes from a pinned host allocation to device and blocks until complete.
pub fn copy_pinned_host_to_device<Brand>(
    dst_ptr: usize,
    src: &PinnedHostBuffer<'_, Brand>,
    stream: usize,
) -> Result<()> {
    rmm_sys::ffi::cuda_memcpy_h2d_raw(dst_ptr, src.as_ptr(), src.len(), stream)?;
    stream_synchronize(stream)
}

#[cfg(test)]
mod tests {
    use super::{PinnedHostBuffer, PoolMemoryResource};

    #[test]
    fn pool_memory_resource_rejects_misaligned_sizes() {
        let _test_lock = crate::test_lock();
        let alignment = rmm_sys::ffi::cuda_allocation_alignment();
        let result = PoolMemoryResource::with_limits(alignment + 1, alignment * 2);
        assert!(result.is_err());
    }

    #[test]
    fn pinned_host_buffer_roundtrips_bytes() {
        let _test_lock = crate::test_lock();
        let buf = PinnedHostBuffer::from_slice(&[1, 2, 3, 4]).unwrap();
        assert_eq!(buf.to_vec().unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn pinned_host_buffer_rejects_oversized_write() {
        let _test_lock = crate::test_lock();
        let mut buf = PinnedHostBuffer::new(4).unwrap();
        assert!(buf.write(&[1, 2, 3, 4, 5]).is_err());
    }

    #[test]
    fn pinned_host_buffer_rejects_oversized_read() {
        let _test_lock = crate::test_lock();
        let buf = PinnedHostBuffer::new(4).unwrap();
        let mut dst = [0_u8; 5];
        assert!(buf.read(&mut dst).is_err());
    }
}
