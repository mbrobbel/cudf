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
//! [`get_current_device_resource`] to control it.
//!
//! # Examples
//!
//! ```ignore
//! use rmm::memory_resource::{PoolMemoryResource, set_current_device_resource};
//!
//! // Create a pool that pre-allocates 1 GiB, growing up to 4 GiB.
//! let mut pool = PoolMemoryResource::with_limits(1 << 30, 4 << 30)?;
//! set_current_device_resource(&mut pool);
//!
//! // All subsequent RMM allocations use the pool.
//! let buf = rmm::buffer::DeviceBuffer::new(4096);
//!
//! // Reset to the built-in default when done.
//! rmm::memory_resource::reset_current_device_resource();
//! # Ok::<(), rmm::error::Error>(())
//! ```

use cxx::UniquePtr;

use crate::error::Result;

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
        Ok(Self(rmm_sys::ffi::cuda_memory_resource_new()))
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    ///
    /// The returned ref borrows `self` — the `CudaMemoryResource` must
    /// outlive all allocations made through the ref.
    pub fn as_ref(&mut self) -> MemoryResourceRef {
        MemoryResourceRef(rmm_sys::ffi::memory_resource_ref_from_cuda(
            self.0.pin_mut(),
        ))
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
        Ok(Self(rmm_sys::ffi::cuda_async_memory_resource_new()))
    }

    /// Creates a new `CudaAsyncMemoryResource` with explicit pool parameters.
    ///
    /// * `initial_pool_size` — initial pool size in bytes (primed by
    ///   allocating and immediately freeing this amount).
    /// * `release_threshold` — when the pool exceeds this size, unused memory
    ///   is released at the next synchronization event.
    pub fn with_limits(
        initial_pool_size: usize,
        release_threshold: usize,
    ) -> Result<Self> {
        Ok(Self(
            rmm_sys::ffi::cuda_async_memory_resource_with_size(
                initial_pool_size,
                release_threshold,
            ),
        ))
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    pub fn as_ref(&mut self) -> MemoryResourceRef {
        MemoryResourceRef(rmm_sys::ffi::memory_resource_ref_from_async(
            self.0.pin_mut(),
        ))
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
        Ok(Self(rmm_sys::ffi::managed_memory_resource_new()))
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    pub fn as_ref(&mut self) -> MemoryResourceRef {
        MemoryResourceRef(rmm_sys::ffi::memory_resource_ref_from_managed(
            self.0.pin_mut(),
        ))
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
        Ok(Self(rmm_sys::ffi::pool_memory_resource_new()))
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
        )))
    }

    /// Returns the current total pool size in bytes, including both
    /// allocated and free regions.
    pub fn pool_size(&self) -> usize {
        rmm_sys::ffi::pool_memory_resource_pool_size(&self.0)
    }

    /// Returns a type-erased [`MemoryResourceRef`] pointing to this resource.
    pub fn as_ref(&mut self) -> MemoryResourceRef {
        MemoryResourceRef(rmm_sys::ffi::memory_resource_ref_from_pool(
            self.0.pin_mut(),
        ))
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
/// Obtained from the `as_ref()` method on any concrete memory resource, or
/// from [`get_current_device_resource`]. The referenced memory resource must
/// outlive this handle.
#[doc(alias = "rmm::mr::device_memory_resource")]
pub struct MemoryResourceRef(UniquePtr<rmm_sys::ffi::MemoryResourceRef>);

impl std::fmt::Debug for MemoryResourceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryResourceRef").finish()
    }
}

// ---------------------------------------------------------------------------
// Per-device resource management
// ---------------------------------------------------------------------------

/// Returns a [`MemoryResourceRef`] to the current device's memory resource.
///
/// If no resource has been explicitly set via [`set_current_device_resource`],
/// this returns a reference to the built-in default (`CudaMemoryResource`).
#[doc(alias = "rmm::mr::get_current_device_resource")]
pub fn get_current_device_resource() -> MemoryResourceRef {
    MemoryResourceRef(rmm_sys::ffi::get_current_device_resource())
}

/// Sets the memory resource for the current CUDA device.
///
/// Returns a [`MemoryResourceRef`] to the *previous* resource.
///
/// # Lifetime requirement
///
/// The concrete memory resource behind `mr` must outlive all allocations
/// made through it. Dropping the resource while allocations are still live
/// is undefined behavior.
///
/// # Examples
///
/// ```ignore
/// use rmm::memory_resource::{PoolMemoryResource, set_current_device_resource};
///
/// let mut pool = PoolMemoryResource::new()?;
/// let _prev = set_current_device_resource(&pool.as_ref());
/// // ... all allocations now go through `pool` ...
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[doc(alias = "rmm::mr::set_current_device_resource")]
pub fn set_current_device_resource(mr: &MemoryResourceRef) -> MemoryResourceRef {
    MemoryResourceRef(rmm_sys::ffi::set_current_device_resource(&mr.0))
}

/// Resets the current device's memory resource to the built-in default
/// (`CudaMemoryResource`).
#[doc(alias = "rmm::mr::reset_current_device_resource_ref")]
pub fn reset_current_device_resource() {
    rmm_sys::ffi::reset_current_device_resource();
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
        Ok(Self(rmm_sys::ffi::pinned_host_memory_resource_new()))
    }

    /// Allocates `size` bytes of pinned host memory. Returns raw pointer as `usize`.
    pub fn allocate(&mut self, size: usize) -> usize {
        rmm_sys::ffi::pinned_host_allocate(self.0.pin_mut(), size)
    }

    /// Deallocates pinned host memory.
    pub fn deallocate(&mut self, ptr: usize, size: usize) {
        rmm_sys::ffi::pinned_host_deallocate(self.0.pin_mut(), ptr, size);
    }
}

/// Asynchronous device-to-host memory copy.
///
/// Copies `dst.len()` bytes from the device pointer `src_ptr` to the host
/// slice `dst`. For maximum throughput, `dst` should be in pinned memory.
pub fn memcpy_d2h(dst: &mut [u8], src_ptr: usize, stream: usize) {
    rmm_sys::ffi::cuda_memcpy_d2h(dst, src_ptr, stream);
}

/// Asynchronous host-to-device memory copy.
///
/// Copies `src.len()` bytes from the host slice `src` to the device pointer
/// `dst_ptr`. For maximum throughput, `src` should be in pinned memory.
pub fn memcpy_h2d(dst_ptr: usize, src: &[u8], stream: usize) {
    rmm_sys::ffi::cuda_memcpy_h2d(dst_ptr, src, stream);
}

/// Synchronizes a CUDA stream, blocking until all operations complete.
pub fn stream_synchronize(stream: usize) {
    rmm_sys::ffi::cuda_stream_synchronize_raw(stream);
}

/// Batched device-to-host copy via `cudaMemcpyBatchAsync` (CUDA 12.8+).
///
/// Fires all copies in a single driver call. All three slices must have the
/// same length. Each element `i` copies `sizes[i]` bytes from device pointer
/// `src_ptrs[i]` to host pointer `dst_ptrs[i]`.
pub fn memcpy_batch_d2h(dst_ptrs: &[usize], src_ptrs: &[usize], sizes: &[usize], stream: usize) {
    rmm_sys::ffi::cuda_memcpy_batch_d2h(dst_ptrs, src_ptrs, sizes, stream);
}

/// Batched host-to-device copy via `cudaMemcpyBatchAsync` (CUDA 12.8+).
///
/// Fires all copies in a single driver call. All three slices must have the
/// same length. Each element `i` copies `sizes[i]` bytes from host pointer
/// `src_ptrs[i]` to device pointer `dst_ptrs[i]`.
pub fn memcpy_batch_h2d(dst_ptrs: &[usize], src_ptrs: &[usize], sizes: &[usize], stream: usize) {
    rmm_sys::ffi::cuda_memcpy_batch_h2d(dst_ptrs, src_ptrs, sizes, stream);
}

