// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/prefetch.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "rust/cxx.h"

namespace rmm_sys {

// DeviceId is a CXX shared struct — defined in the generated header.
struct DeviceId;

// ---------- Device queries ----------

int32_t get_num_cuda_devices();
DeviceId get_current_device();
std::size_t available_device_memory();
std::size_t total_device_memory();
std::size_t percent_of_free_device_memory(int32_t percent);

// ---------- DeviceBuffer wrapper ----------

/// Opaque wrapper around `rmm::device_buffer`.
class DeviceBuffer {
 public:
  DeviceBuffer(std::size_t size, std::size_t stream)
      : buf_(size, rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)}) {}

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&) = default;
  DeviceBuffer& operator=(DeviceBuffer&&) = default;
  ~DeviceBuffer() = default;

  std::size_t size() const { return buf_.size(); }
  std::size_t capacity() const { return buf_.capacity(); }
  bool is_empty() const { return buf_.is_empty(); }

  void resize(std::size_t new_size, std::size_t stream) {
    buf_.resize(new_size, rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)});
  }

  void reserve(std::size_t new_capacity, std::size_t stream) {
    buf_.reserve(new_capacity,
                 rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)});
  }

  void shrink_to_fit(std::size_t stream) {
    buf_.shrink_to_fit(rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)});
  }

  std::size_t data() const {
    return reinterpret_cast<std::size_t>(buf_.data());
  }

 private:
  rmm::device_buffer buf_;
};

std::unique_ptr<DeviceBuffer> device_buffer_new(std::size_t size, std::size_t stream);
std::size_t device_buffer_size(const DeviceBuffer& buf);
std::size_t device_buffer_capacity(const DeviceBuffer& buf);
void device_buffer_resize(DeviceBuffer& buf, std::size_t new_size, std::size_t stream);
void device_buffer_reserve(DeviceBuffer& buf, std::size_t new_capacity, std::size_t stream);
void device_buffer_shrink_to_fit(DeviceBuffer& buf, std::size_t stream);
std::size_t device_buffer_data(const DeviceBuffer& buf);
bool device_buffer_is_empty(const DeviceBuffer& buf);

// ---------- CudaStream wrapper ----------

/// Opaque wrapper around `rmm::cuda_stream`.
class CudaStream {
 public:
  CudaStream() : stream_() {}

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = default;
  CudaStream& operator=(CudaStream&&) = default;
  ~CudaStream() = default;

  std::size_t view() const {
    return reinterpret_cast<std::size_t>(stream_.value());
  }

  void synchronize() const { stream_.synchronize(); }
  bool is_valid() const { return stream_.is_valid(); }

 private:
  rmm::cuda_stream stream_;
};

std::unique_ptr<CudaStream> cuda_stream_new();
std::size_t cuda_stream_view(const CudaStream& stream);
void cuda_stream_synchronize(const CudaStream& stream);
bool cuda_stream_is_valid(const CudaStream& stream);

// ---------- CudaStreamPool wrapper ----------

/// Opaque wrapper around `rmm::cuda_stream_pool`.
class CudaStreamPool {
 public:
  explicit CudaStreamPool(std::size_t pool_size) : pool_(pool_size) {}

  CudaStreamPool(const CudaStreamPool&) = delete;
  CudaStreamPool& operator=(const CudaStreamPool&) = delete;
  CudaStreamPool(CudaStreamPool&&) = delete;
  CudaStreamPool& operator=(CudaStreamPool&&) = delete;
  ~CudaStreamPool() = default;

  std::size_t get_stream() const {
    auto view = pool_.get_stream();
    return reinterpret_cast<std::size_t>(view.value());
  }

  std::size_t get_pool_size() const { return pool_.get_pool_size(); }

 private:
  rmm::cuda_stream_pool pool_;
};

std::unique_ptr<CudaStreamPool> cuda_stream_pool_new(std::size_t pool_size);
std::size_t cuda_stream_pool_get_stream(const CudaStreamPool& pool);
std::size_t cuda_stream_pool_get_pool_size(const CudaStreamPool& pool);

// ---------- Memory Resources ----------

/// Opaque wrapper around `rmm::mr::cuda_memory_resource`.
///
/// Uses `cudaMalloc`/`cudaFree` for allocation and deallocation.
class CudaMemoryResource {
 public:
  CudaMemoryResource() = default;

  CudaMemoryResource(const CudaMemoryResource&) = delete;
  CudaMemoryResource& operator=(const CudaMemoryResource&) = delete;
  CudaMemoryResource(CudaMemoryResource&&) = delete;
  CudaMemoryResource& operator=(CudaMemoryResource&&) = delete;
  ~CudaMemoryResource() = default;

  rmm::mr::device_memory_resource* get() { return &mr_; }

 private:
  rmm::mr::cuda_memory_resource mr_;
};

std::unique_ptr<CudaMemoryResource> cuda_memory_resource_new();

/// Opaque wrapper around `rmm::mr::cuda_async_memory_resource`.
///
/// Uses `cudaMallocAsync`/`cudaFreeAsync` for allocation and deallocation.
class CudaAsyncMemoryResource {
 public:
  CudaAsyncMemoryResource() : mr_() {}

  CudaAsyncMemoryResource(std::size_t initial_pool_size, std::size_t release_threshold)
      : mr_(initial_pool_size, release_threshold) {}

  CudaAsyncMemoryResource(const CudaAsyncMemoryResource&) = delete;
  CudaAsyncMemoryResource& operator=(const CudaAsyncMemoryResource&) = delete;
  CudaAsyncMemoryResource(CudaAsyncMemoryResource&&) = delete;
  CudaAsyncMemoryResource& operator=(CudaAsyncMemoryResource&&) = delete;
  ~CudaAsyncMemoryResource() = default;

  rmm::mr::device_memory_resource* get() { return &mr_; }

 private:
  rmm::mr::cuda_async_memory_resource mr_;
};

std::unique_ptr<CudaAsyncMemoryResource> cuda_async_memory_resource_new();
std::unique_ptr<CudaAsyncMemoryResource> cuda_async_memory_resource_with_size(
    std::size_t initial_pool_size, std::size_t release_threshold);

/// Opaque wrapper around `rmm::mr::managed_memory_resource`.
///
/// Uses `cudaMallocManaged`/`cudaFree` for allocation and deallocation (UVM).
class ManagedMemoryResource {
 public:
  ManagedMemoryResource() = default;

  ManagedMemoryResource(const ManagedMemoryResource&) = delete;
  ManagedMemoryResource& operator=(const ManagedMemoryResource&) = delete;
  ManagedMemoryResource(ManagedMemoryResource&&) = delete;
  ManagedMemoryResource& operator=(ManagedMemoryResource&&) = delete;
  ~ManagedMemoryResource() = default;

  rmm::mr::device_memory_resource* get() { return &mr_; }

 private:
  rmm::mr::managed_memory_resource mr_;
};

std::unique_ptr<ManagedMemoryResource> managed_memory_resource_new();

/// Opaque wrapper around `rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>`.
///
/// Suballocates from a pool backed by `cudaMalloc`. This is the most commonly
/// used memory resource in production.
class PoolMemoryResource {
 public:
  PoolMemoryResource()
      : upstream_(), mr_(&upstream_, rmm::percent_of_free_device_memory(50)) {}

  PoolMemoryResource(std::size_t initial_size, std::size_t maximum_size)
      : upstream_(), mr_(&upstream_, initial_size, maximum_size) {}

  PoolMemoryResource(const PoolMemoryResource&) = delete;
  PoolMemoryResource& operator=(const PoolMemoryResource&) = delete;
  PoolMemoryResource(PoolMemoryResource&&) = delete;
  PoolMemoryResource& operator=(PoolMemoryResource&&) = delete;
  ~PoolMemoryResource() = default;

  rmm::mr::device_memory_resource* get() { return &mr_; }
  std::size_t pool_size() const { return mr_.pool_size(); }

 private:
  rmm::mr::cuda_memory_resource upstream_;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr_;
};

std::unique_ptr<PoolMemoryResource> pool_memory_resource_new();
std::unique_ptr<PoolMemoryResource> pool_memory_resource_with_size(
    std::size_t initial_size, std::size_t maximum_size);
std::size_t pool_memory_resource_pool_size(const PoolMemoryResource& mr);

// ---------- MemoryResourceRef (type-erased) ----------

/// Non-owning handle to any `rmm::mr::device_memory_resource`.
///
/// This enables passing any concrete memory resource to per-device resource
/// management functions without templates.
class MemoryResourceRef {
 public:
  explicit MemoryResourceRef(rmm::mr::device_memory_resource* p) : ptr_(p) {}

  MemoryResourceRef(const MemoryResourceRef&) = default;
  MemoryResourceRef& operator=(const MemoryResourceRef&) = default;
  MemoryResourceRef(MemoryResourceRef&&) = default;
  MemoryResourceRef& operator=(MemoryResourceRef&&) = default;
  ~MemoryResourceRef() = default;

  rmm::mr::device_memory_resource* get() const { return ptr_; }

 private:
  rmm::mr::device_memory_resource* ptr_;
};

std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_cuda(CudaMemoryResource& mr);
std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_pool(PoolMemoryResource& mr);
std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_async(CudaAsyncMemoryResource& mr);
std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_managed(ManagedMemoryResource& mr);

// ---------- Per-device resource management ----------

std::unique_ptr<MemoryResourceRef> get_current_device_resource();
std::unique_ptr<MemoryResourceRef> set_current_device_resource(const MemoryResourceRef& mr);
void reset_current_device_resource();

// ---------- ScopedDevice ----------

/// RAII wrapper around `rmm::cuda_set_device_raii`.
///
/// Sets the CUDA device on construction and restores the previous device on
/// destruction (i.e., when the `UniquePtr` is dropped on the Rust side).
class ScopedDevice {
 public:
  explicit ScopedDevice(int32_t device_id)
      : raii_(rmm::cuda_device_id{device_id}) {}

  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;
  ScopedDevice(ScopedDevice&&) = delete;
  ScopedDevice& operator=(ScopedDevice&&) = delete;
  ~ScopedDevice() = default;

 private:
  rmm::cuda_set_device_raii raii_;
};

std::unique_ptr<ScopedDevice> scoped_device_new(int32_t device_id);

// ---------- Pinned host memory ----------

class PinnedHostMemoryResource {
 public:
  PinnedHostMemoryResource() = default;
  PinnedHostMemoryResource(const PinnedHostMemoryResource&) = delete;
  PinnedHostMemoryResource& operator=(const PinnedHostMemoryResource&) = delete;
  PinnedHostMemoryResource(PinnedHostMemoryResource&&) = delete;
  PinnedHostMemoryResource& operator=(PinnedHostMemoryResource&&) = delete;
  ~PinnedHostMemoryResource() = default;
  std::size_t allocate(std::size_t size) {
    void* ptr = mr_.allocate(rmm::cuda_stream_view{}, size);
    return reinterpret_cast<std::size_t>(ptr);
  }
  void deallocate(std::size_t ptr, std::size_t size) {
    mr_.deallocate(rmm::cuda_stream_view{}, reinterpret_cast<void*>(ptr), size);
  }
 private:
  rmm::mr::pinned_host_memory_resource mr_;
};

std::unique_ptr<PinnedHostMemoryResource> pinned_host_memory_resource_new();
std::size_t pinned_host_allocate(PinnedHostMemoryResource& mr, std::size_t size);
void pinned_host_deallocate(PinnedHostMemoryResource& mr, std::size_t ptr, std::size_t size);

// ---------- Async memcpy ----------

void cuda_memcpy_d2h(rust::Slice<uint8_t> dst, std::size_t src_ptr, std::size_t stream);
void cuda_memcpy_h2d(std::size_t dst_ptr, rust::Slice<const uint8_t> src, std::size_t stream);
void cuda_stream_synchronize_raw(std::size_t stream);

// ---------- Prefetch ----------

void prefetch(std::size_t ptr, std::size_t size, int32_t device, std::size_t stream);

// ---------- Alignment utilities ----------

std::size_t cuda_allocation_alignment();
std::size_t align_up(std::size_t value, std::size_t alignment);
std::size_t align_down(std::size_t value, std::size_t alignment);
bool is_aligned(std::size_t value, std::size_t alignment);

}  // namespace rmm_sys
