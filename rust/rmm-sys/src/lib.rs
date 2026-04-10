// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Low-level CXX FFI bindings for RMM (RAPIDS Memory Manager).
//!
//! This crate provides device queries, GPU memory allocation
//! ([`DeviceBuffer`](ffi::DeviceBuffer)), CUDA stream management
//! ([`CudaStream`](ffi::CudaStream), [`CudaStreamPool`](ffi::CudaStreamPool)),
//! memory resources ([`CudaMemoryResource`](ffi::CudaMemoryResource),
//! [`PoolMemoryResource`](ffi::PoolMemoryResource),
//! [`CudaAsyncMemoryResource`](ffi::CudaAsyncMemoryResource),
//! [`ManagedMemoryResource`](ffi::ManagedMemoryResource)),
//! per-device resource management, scoped device switching
//! ([`ScopedDevice`](ffi::ScopedDevice)), prefetch, and alignment utilities.

#![deny(clippy::undocumented_unsafe_blocks)]
// CXX-generated shared enum variants and repr fields cannot carry doc comments.
#![allow(missing_docs)]

#[cxx::bridge(namespace = "rmm_sys")]
pub mod ffi {
    /// A CUDA device identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct DeviceId {
        value: i32,
    }

    unsafe extern "C++" {
        include!("lib.hpp");

        // ---- Device queries ----

        /// Returns the number of CUDA devices available.
        fn get_num_cuda_devices() -> i32;

        /// Returns the current CUDA device.
        fn get_current_device() -> DeviceId;

        /// Returns the available (free) device memory in bytes.
        fn available_device_memory() -> usize;

        /// Returns the total device memory in bytes.
        fn total_device_memory() -> usize;

        /// Returns the specified percentage of free device memory in bytes.
        fn percent_of_free_device_memory(percent: i32) -> usize;

        // ---- DeviceBuffer ----

        /// Opaque wrapper around `rmm::device_buffer`.
        ///
        /// Represents an untyped, uninitialized GPU memory allocation managed
        /// by RMM. The buffer is automatically deallocated when dropped.
        ///
        /// See: [`rmm::device_buffer`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1device__buffer.html)
        type DeviceBuffer;

        /// Allocates a new device buffer of `size` uninitialized bytes.
        ///
        /// The `stream` parameter is the raw `cudaStream_t` cast to `usize`.
        /// Pass `0` for the default CUDA stream.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if allocation fails.
        fn device_buffer_new(size: usize, stream: usize) -> Result<UniquePtr<DeviceBuffer>>;

        /// Returns the size in bytes of the device buffer.
        fn device_buffer_size(buf: &DeviceBuffer) -> usize;

        /// Returns the capacity in bytes of the underlying allocation.
        ///
        /// The invariant `size <= capacity` always holds.
        fn device_buffer_capacity(buf: &DeviceBuffer) -> usize;

        /// Resizes the device buffer to `new_size` bytes.
        ///
        /// If `new_size <= capacity`, only the logical size is updated.
        /// If `new_size > capacity`, a new allocation is made and existing
        /// contents are copied. New bytes are uninitialized.
        ///
        /// The `stream` parameter is the raw `cudaStream_t` cast to `usize`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if reallocation fails.
        fn device_buffer_resize(
            buf: Pin<&mut DeviceBuffer>,
            new_size: usize,
            stream: usize,
        ) -> Result<()>;

        /// Increases the capacity to at least `new_capacity` bytes without
        /// changing the logical size.
        ///
        /// If `new_capacity <= capacity`, this is a no-op. Otherwise a new
        /// allocation is made and existing contents are copied.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if allocation fails.
        fn device_buffer_reserve(
            buf: Pin<&mut DeviceBuffer>,
            new_capacity: usize,
            stream: usize,
        ) -> Result<()>;

        /// Releases any excess capacity so that `capacity == size`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if reallocation fails.
        fn device_buffer_shrink_to_fit(buf: Pin<&mut DeviceBuffer>, stream: usize) -> Result<()>;

        /// Returns the raw device pointer as `usize`.
        ///
        /// The returned value is an opaque handle; it should only be passed
        /// to other FFI functions that expect a device pointer.
        fn device_buffer_data(buf: &DeviceBuffer) -> usize;

        /// Returns `true` if the buffer is empty (`size == 0`).
        fn device_buffer_is_empty(buf: &DeviceBuffer) -> bool;

        // ---- CudaStream ----

        /// Opaque wrapper around `rmm::cuda_stream`.
        ///
        /// Owns a CUDA stream and destroys it on drop. This is a move-only
        /// type on the C++ side.
        ///
        /// See: [`rmm::cuda_stream`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1cuda__stream.html)
        type CudaStream;

        /// Creates a new CUDA stream.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if stream creation fails.
        fn cuda_stream_new() -> Result<UniquePtr<CudaStream>>;

        /// Returns the raw default `cudaStream_t` handle as `usize`.
        fn cuda_stream_default_view() -> usize;

        /// Returns the raw `cudaStream_t` handle as `usize`.
        fn cuda_stream_view(stream: &CudaStream) -> usize;

        /// Synchronizes the stream, blocking until all preceding work completes.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if synchronization fails.
        fn cuda_stream_synchronize(stream: &CudaStream) -> Result<()>;

        /// Returns `true` if the stream handle is valid (non-null).
        fn cuda_stream_is_valid(stream: &CudaStream) -> bool;

        // ---- CudaEvent ----

        /// Opaque wrapper around `cudaEvent_t`.
        type CudaEvent;

        /// Creates a new CUDA event with timing disabled.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if event creation fails.
        fn cuda_event_new() -> Result<UniquePtr<CudaEvent>>;

        /// Records the event on `stream`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if the record fails.
        fn cuda_event_record(event: Pin<&mut CudaEvent>, stream: usize) -> Result<()>;

        /// Synchronizes the event, blocking until the record completes.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if synchronization fails.
        fn cuda_event_synchronize(event: &CudaEvent) -> Result<()>;

        /// Returns `true` if the event has completed.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if querying the event fails.
        fn cuda_event_query(event: &CudaEvent) -> Result<bool>;

        /// Enqueues a wait for `event` on `stream`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if the wait cannot be enqueued.
        fn cuda_stream_wait_event_raw(stream: usize, event: &CudaEvent) -> Result<()>;

        // ---- CudaStreamPool ----

        /// Opaque wrapper around `rmm::cuda_stream_pool`.
        ///
        /// Manages a pool of CUDA streams for concurrent work. Streams are
        /// handed out in round-robin fashion.
        ///
        /// See: [`rmm::cuda_stream_pool`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1cuda__stream__pool.html)
        type CudaStreamPool;

        /// Creates a new stream pool with `pool_size` streams.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if `pool_size` is zero or stream creation fails.
        fn cuda_stream_pool_new(pool_size: usize) -> Result<UniquePtr<CudaStreamPool>>;

        /// Returns a stream handle (`cudaStream_t` as `usize`) from the pool.
        ///
        /// Successive calls cycle through the pool in round-robin order.
        fn cuda_stream_pool_get_stream(pool: &CudaStreamPool) -> usize;

        /// Returns the number of streams in the pool.
        fn cuda_stream_pool_get_pool_size(pool: &CudaStreamPool) -> usize;

        // ---- Memory Resources ----

        /// Opaque wrapper around `rmm::mr::cuda_memory_resource`.
        ///
        /// Uses `cudaMalloc`/`cudaFree` for allocation and deallocation.
        /// This is the default memory resource if none is explicitly configured.
        ///
        /// See: [`rmm::mr::cuda_memory_resource`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1mr_1_1cuda__memory__resource.html)
        type CudaMemoryResource;

        /// Creates a new `CudaMemoryResource`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if construction fails.
        fn cuda_memory_resource_new() -> Result<UniquePtr<CudaMemoryResource>>;

        /// Opaque wrapper around `rmm::mr::cuda_async_memory_resource`.
        ///
        /// Uses `cudaMallocAsync`/`cudaFreeAsync` for allocation and deallocation.
        /// Requires CUDA 11.2+ runtime with a compatible driver.
        ///
        /// See: [`rmm::mr::cuda_async_memory_resource`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1mr_1_1cuda__async__memory__resource.html)
        type CudaAsyncMemoryResource;

        /// Creates a new `CudaAsyncMemoryResource` with default pool settings.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if `cudaMallocAsync` is not supported.
        fn cuda_async_memory_resource_new() -> Result<UniquePtr<CudaAsyncMemoryResource>>;

        /// Creates a new `CudaAsyncMemoryResource` with explicit pool parameters.
        ///
        /// # Parameters
        ///
        /// - `initial_pool_size`: Initial pool size in bytes (the pool is
        ///   primed by allocating and immediately deallocating this amount).
        /// - `release_threshold`: When the pool exceeds this size, unused
        ///   memory is released at the next synchronization event.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if `cudaMallocAsync` is not supported.
        fn cuda_async_memory_resource_with_size(
            initial_pool_size: usize,
            release_threshold: usize,
        ) -> Result<UniquePtr<CudaAsyncMemoryResource>>;

        /// Opaque wrapper around `rmm::mr::managed_memory_resource`.
        ///
        /// Uses `cudaMallocManaged`/`cudaFree` for allocation and deallocation
        /// (CUDA Unified Virtual Memory).
        ///
        /// See: [`rmm::mr::managed_memory_resource`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1mr_1_1managed__memory__resource.html)
        type ManagedMemoryResource;

        /// Creates a new `ManagedMemoryResource`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if construction fails.
        fn managed_memory_resource_new() -> Result<UniquePtr<ManagedMemoryResource>>;

        /// Opaque wrapper around `rmm::mr::pool_memory_resource<cuda_memory_resource>`.
        ///
        /// A coalescing best-fit suballocator backed by `cudaMalloc`. This is
        /// the most commonly used memory resource in production because it
        /// dramatically reduces the overhead of frequent small allocations.
        ///
        /// See: [`rmm::mr::pool_memory_resource`](https://docs.rapids.ai/api/rmm/stable/classrmm_1_1mr_1_1pool__memory__resource.html)
        type PoolMemoryResource;

        /// Creates a new `PoolMemoryResource` with a default initial pool
        /// size of 50% of free device memory.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if the initial pool allocation fails.
        fn pool_memory_resource_new() -> Result<UniquePtr<PoolMemoryResource>>;

        /// Creates a new `PoolMemoryResource` with explicit size parameters.
        ///
        /// Both sizes must be aligned to 256 bytes
        /// ([`cuda_allocation_alignment`]).
        ///
        /// # Parameters
        ///
        /// - `initial_size`: Minimum initial pool size in bytes.
        /// - `maximum_size`: Maximum pool size in bytes.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if the sizes are misaligned or if the
        /// initial allocation fails.
        fn pool_memory_resource_with_size(
            initial_size: usize,
            maximum_size: usize,
        ) -> Result<UniquePtr<PoolMemoryResource>>;

        /// Returns the current total pool size in bytes, including both
        /// allocated and free regions.
        fn pool_memory_resource_pool_size(mr: &PoolMemoryResource) -> usize;

        // ---- MemoryResourceRef (type-erased) ----

        /// Non-owning, type-erased handle to any `rmm::mr::device_memory_resource`.
        ///
        /// Obtained from one of the `memory_resource_ref_from_*` functions or
        /// from [`get_current_device_resource`]. The referenced memory resource
        /// must outlive this handle.
        type MemoryResourceRef;

        /// Creates a [`MemoryResourceRef`] pointing to a [`CudaMemoryResource`].
        fn memory_resource_ref_from_cuda(mr: &CudaMemoryResource) -> UniquePtr<MemoryResourceRef>;

        /// Creates a [`MemoryResourceRef`] pointing to a [`PoolMemoryResource`].
        fn memory_resource_ref_from_pool(mr: &PoolMemoryResource) -> UniquePtr<MemoryResourceRef>;

        /// Creates a [`MemoryResourceRef`] pointing to a [`CudaAsyncMemoryResource`].
        fn memory_resource_ref_from_async(
            mr: &CudaAsyncMemoryResource,
        ) -> UniquePtr<MemoryResourceRef>;

        /// Creates a [`MemoryResourceRef`] pointing to a [`ManagedMemoryResource`].
        fn memory_resource_ref_from_managed(
            mr: &ManagedMemoryResource,
        ) -> UniquePtr<MemoryResourceRef>;

        /// Clones a non-owning [`MemoryResourceRef`].
        fn memory_resource_ref_clone(mr: &MemoryResourceRef) -> UniquePtr<MemoryResourceRef>;

        // ---- Per-device resource management ----

        /// Returns a [`MemoryResourceRef`] to the current device's memory resource.
        ///
        /// If no resource has been explicitly set, this returns a reference to
        /// the default `cuda_memory_resource`.
        fn get_current_device_resource() -> UniquePtr<MemoryResourceRef>;

        /// Sets the memory resource for the current device.
        ///
        /// Returns a [`MemoryResourceRef`] to the *previous* resource.
        ///
        /// # Safety (lifetime)
        ///
        /// The concrete memory resource referenced by `mr` must outlive all
        /// allocations made through it. It is the caller's responsibility to
        /// ensure this.
        fn set_current_device_resource(mr: &MemoryResourceRef) -> UniquePtr<MemoryResourceRef>;

        /// Resets the current device's memory resource to the built-in default
        /// (`cuda_memory_resource`).
        fn reset_current_device_resource();

        // ---- Pinned host memory ----

        type PinnedHostMemoryResource;

        fn pinned_host_memory_resource_new() -> Result<UniquePtr<PinnedHostMemoryResource>>;

        fn pinned_host_allocate(
            mr: Pin<&mut PinnedHostMemoryResource>,
            size: usize,
        ) -> Result<usize>;

        fn pinned_host_deallocate(mr: Pin<&mut PinnedHostMemoryResource>, ptr: usize, size: usize);

        fn pinned_host_copy_to_slice(ptr: usize, dst: &mut [u8]);

        fn pinned_host_copy_from_slice(ptr: usize, src: &[u8]);

        // ---- Async memcpy ----

        /// D2H: copies `dst.len()` bytes from device pointer `src_ptr` to `dst`.
        fn cuda_memcpy_d2h(dst: &mut [u8], src_ptr: usize, stream: usize) -> Result<()>;

        /// H2D: copies `src.len()` bytes from `src` to device pointer `dst_ptr`.
        fn cuda_memcpy_h2d(dst_ptr: usize, src: &[u8], stream: usize) -> Result<()>;

        /// D2H: copies `size` bytes between raw pinned/host pointers.
        fn cuda_memcpy_d2h_raw(
            dst_ptr: usize,
            src_ptr: usize,
            size: usize,
            stream: usize,
        ) -> Result<()>;

        /// H2D: copies `size` bytes between raw pinned/host pointers.
        fn cuda_memcpy_h2d_raw(
            dst_ptr: usize,
            src_ptr: usize,
            size: usize,
            stream: usize,
        ) -> Result<()>;

        /// Block until all operations on `stream` complete.
        fn cuda_stream_synchronize_raw(stream: usize) -> Result<()>;

        /// Batched D2H: fires all copies in a single `cudaMemcpyBatchAsync` call.
        fn cuda_memcpy_batch_d2h(
            dst_ptrs: &[usize],
            src_ptrs: &[usize],
            sizes: &[usize],
            stream: usize,
        ) -> Result<()>;

        /// Batched H2D: fires all copies in a single `cudaMemcpyBatchAsync` call.
        fn cuda_memcpy_batch_h2d(
            dst_ptrs: &[usize],
            src_ptrs: &[usize],
            sizes: &[usize],
            stream: usize,
        ) -> Result<()>;

        // ---- ScopedDevice ----

        /// RAII guard that sets the CUDA device on construction and restores
        /// the previous device when dropped.
        ///
        /// See: [`rmm::cuda_set_device_raii`](https://docs.rapids.ai/api/rmm/stable/structrmm_1_1cuda__set__device__raii.html)
        type ScopedDevice;

        /// Creates a [`ScopedDevice`] that sets the current CUDA device to
        /// `device_id`. The previous device is restored when the returned
        /// `UniquePtr` is dropped.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if `device_id` is invalid.
        fn scoped_device_new(device_id: i32) -> Result<UniquePtr<ScopedDevice>>;

        // ---- Prefetch ----

        /// Prefetches memory to the specified device on the given stream.
        ///
        /// This is a no-op if the pointer does not refer to CUDA managed memory
        /// or if concurrent managed access is not supported.
        ///
        /// # Parameters
        ///
        /// - `ptr`: Device pointer as `usize`.
        /// - `size`: Number of bytes to prefetch.
        /// - `device`: CUDA device ID.
        /// - `stream`: Raw `cudaStream_t` as `usize`.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if the prefetch fails.
        fn prefetch(ptr: usize, size: usize, device: i32, stream: usize) -> Result<()>;

        // ---- Alignment utilities ----

        /// Returns the CUDA allocation alignment constant (256 bytes).
        fn cuda_allocation_alignment() -> usize;

        /// Aligns `value` up to the nearest multiple of `alignment`.
        ///
        /// `alignment` must be a power of two.
        fn align_up(value: usize, alignment: usize) -> usize;

        /// Aligns `value` down to the nearest multiple of `alignment`.
        ///
        /// `alignment` must be a power of two.
        fn align_down(value: usize, alignment: usize) -> usize;

        /// Returns `true` if `value` is aligned to `alignment`.
        ///
        /// `alignment` must be a power of two.
        fn is_aligned(value: usize, alignment: usize) -> bool;
    }
}

// SAFETY: DeviceBuffer owns GPU memory via UniquePtr. Ownership can be
// transferred across threads; the GPU driver serializes access.
unsafe impl Send for ffi::DeviceBuffer {}
// SAFETY: &DeviceBuffer only allows immutable queries (size, capacity, data pointer).
unsafe impl Sync for ffi::DeviceBuffer {}

// SAFETY: CudaStream owns a CUDA stream handle. Ownership can be
// transferred across threads.
unsafe impl Send for ffi::CudaStream {}
// SAFETY: &CudaStream only allows immutable queries (view, is_valid) and
// synchronize which is thread-safe in the CUDA runtime.
unsafe impl Sync for ffi::CudaStream {}

// SAFETY: CudaStreamPool is internally synchronized by the CUDA runtime.
unsafe impl Send for ffi::CudaStreamPool {}
// SAFETY: &CudaStreamPool only allows round-robin stream retrieval which is
// thread-safe.
unsafe impl Sync for ffi::CudaStreamPool {}

// SAFETY: CudaMemoryResource wraps cudaMalloc/Free which are thread-safe.
unsafe impl Send for ffi::CudaMemoryResource {}
// SAFETY: device_memory_resource::allocate/deallocate are thread-safe.
unsafe impl Sync for ffi::CudaMemoryResource {}

// SAFETY: CudaAsyncMemoryResource wraps cudaMallocAsync/FreeAsync which are
// thread-safe.
unsafe impl Send for ffi::CudaAsyncMemoryResource {}
// SAFETY: device_memory_resource::allocate/deallocate are thread-safe.
unsafe impl Sync for ffi::CudaAsyncMemoryResource {}

// SAFETY: ManagedMemoryResource wraps cudaMallocManaged/Free which are
// thread-safe.
unsafe impl Send for ffi::ManagedMemoryResource {}
// SAFETY: device_memory_resource::allocate/deallocate are thread-safe.
unsafe impl Sync for ffi::ManagedMemoryResource {}

// SAFETY: PoolMemoryResource uses internal locking for thread safety.
unsafe impl Send for ffi::PoolMemoryResource {}
// SAFETY: pool_memory_resource uses std::mutex internally.
unsafe impl Sync for ffi::PoolMemoryResource {}

// MemoryResourceRef is intentionally !Send and !Sync: it is a non-owning
// pointer to a caller-managed allocator, and moving or sharing that raw handle
// across threads would overstate the lifetime guarantees encoded by the safe
// rmm wrapper layer.
