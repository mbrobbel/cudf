// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Low-level CXX FFI bindings for RMM (RAPIDS Memory Manager).
//!
//! This crate provides device queries, GPU memory allocation
//! ([`DeviceBuffer`](ffi::DeviceBuffer)), CUDA stream management
//! ([`CudaStream`](ffi::CudaStream), [`CudaStreamPool`](ffi::CudaStreamPool)),
//! prefetch, and alignment utilities.

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
        fn device_buffer_new(size: usize, stream: usize) -> UniquePtr<DeviceBuffer>;

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
        fn device_buffer_resize(buf: Pin<&mut DeviceBuffer>, new_size: usize, stream: usize);

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
        fn cuda_stream_new() -> UniquePtr<CudaStream>;

        /// Returns the raw `cudaStream_t` handle as `usize`.
        fn cuda_stream_view(stream: &CudaStream) -> usize;

        /// Synchronizes the stream, blocking until all preceding work completes.
        ///
        /// # Errors
        ///
        /// Throws a C++ exception if synchronization fails.
        fn cuda_stream_synchronize(stream: &CudaStream);

        /// Returns `true` if the stream handle is valid (non-null).
        fn cuda_stream_is_valid(stream: &CudaStream) -> bool;

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
        fn cuda_stream_pool_new(pool_size: usize) -> UniquePtr<CudaStreamPool>;

        /// Returns a stream handle (`cudaStream_t` as `usize`) from the pool.
        ///
        /// Successive calls cycle through the pool in round-robin order.
        fn cuda_stream_pool_get_stream(pool: &CudaStreamPool) -> usize;

        /// Returns the number of streams in the pool.
        fn cuda_stream_pool_get_pool_size(pool: &CudaStreamPool) -> usize;

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
        fn prefetch(ptr: usize, size: usize, device: i32, stream: usize);

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
