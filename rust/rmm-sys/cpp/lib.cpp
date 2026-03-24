// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "rmm-sys/src/lib.rs.h"

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
#include <rmm/prefetch.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rmm_sys {

// ---- Device queries ----

int32_t get_num_cuda_devices() {
  return rmm::get_num_cuda_devices();
}

DeviceId get_current_device() {
  auto id = rmm::get_current_cuda_device();
  return DeviceId{id.value()};
}

std::size_t available_device_memory() {
  auto [free, total] = rmm::available_device_memory();
  return free;
}

std::size_t total_device_memory() {
  auto [free, total] = rmm::available_device_memory();
  return total;
}

std::size_t percent_of_free_device_memory(int32_t percent) {
  return rmm::percent_of_free_device_memory(percent);
}

// ---- DeviceBuffer free functions ----

std::unique_ptr<DeviceBuffer> device_buffer_new(std::size_t size, std::size_t stream) {
  return std::make_unique<DeviceBuffer>(size, stream);
}

std::size_t device_buffer_size(const DeviceBuffer& buf) {
  return buf.size();
}

std::size_t device_buffer_capacity(const DeviceBuffer& buf) {
  return buf.capacity();
}

void device_buffer_resize(DeviceBuffer& buf, std::size_t new_size, std::size_t stream) {
  buf.resize(new_size, stream);
}

void device_buffer_reserve(DeviceBuffer& buf, std::size_t new_capacity, std::size_t stream) {
  buf.reserve(new_capacity, stream);
}

void device_buffer_shrink_to_fit(DeviceBuffer& buf, std::size_t stream) {
  buf.shrink_to_fit(stream);
}

std::size_t device_buffer_data(const DeviceBuffer& buf) {
  return buf.data();
}

bool device_buffer_is_empty(const DeviceBuffer& buf) {
  return buf.is_empty();
}

// ---- CudaStream free functions ----

std::unique_ptr<CudaStream> cuda_stream_new() {
  return std::make_unique<CudaStream>();
}

std::size_t cuda_stream_view(const CudaStream& stream) {
  return stream.view();
}

void cuda_stream_synchronize(const CudaStream& stream) {
  stream.synchronize();
}

bool cuda_stream_is_valid(const CudaStream& stream) {
  return stream.is_valid();
}

// ---- CudaStreamPool free functions ----

std::unique_ptr<CudaStreamPool> cuda_stream_pool_new(std::size_t pool_size) {
  return std::make_unique<CudaStreamPool>(pool_size);
}

std::size_t cuda_stream_pool_get_stream(const CudaStreamPool& pool) {
  return pool.get_stream();
}

std::size_t cuda_stream_pool_get_pool_size(const CudaStreamPool& pool) {
  return pool.get_pool_size();
}

// ---- Memory Resources ----

std::unique_ptr<CudaMemoryResource> cuda_memory_resource_new() {
  return std::make_unique<CudaMemoryResource>();
}

std::unique_ptr<CudaAsyncMemoryResource> cuda_async_memory_resource_new() {
  return std::make_unique<CudaAsyncMemoryResource>();
}

std::unique_ptr<CudaAsyncMemoryResource> cuda_async_memory_resource_with_size(
    std::size_t initial_pool_size, std::size_t release_threshold) {
  return std::make_unique<CudaAsyncMemoryResource>(initial_pool_size, release_threshold);
}

std::unique_ptr<ManagedMemoryResource> managed_memory_resource_new() {
  return std::make_unique<ManagedMemoryResource>();
}

std::unique_ptr<PoolMemoryResource> pool_memory_resource_new() {
  return std::make_unique<PoolMemoryResource>();
}

std::unique_ptr<PoolMemoryResource> pool_memory_resource_with_size(
    std::size_t initial_size, std::size_t maximum_size) {
  return std::make_unique<PoolMemoryResource>(initial_size, maximum_size);
}

std::size_t pool_memory_resource_pool_size(const PoolMemoryResource& mr) {
  return mr.pool_size();
}

// ---- MemoryResourceRef ----

std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_cuda(CudaMemoryResource& mr) {
  return std::make_unique<MemoryResourceRef>(mr.get());
}

std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_pool(PoolMemoryResource& mr) {
  return std::make_unique<MemoryResourceRef>(mr.get());
}

std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_async(CudaAsyncMemoryResource& mr) {
  return std::make_unique<MemoryResourceRef>(mr.get());
}

std::unique_ptr<MemoryResourceRef> memory_resource_ref_from_managed(ManagedMemoryResource& mr) {
  return std::make_unique<MemoryResourceRef>(mr.get());
}

// ---- Per-device resource management ----

std::unique_ptr<MemoryResourceRef> get_current_device_resource() {
  auto* mr = rmm::mr::get_current_device_resource();
  return std::make_unique<MemoryResourceRef>(mr);
}

std::unique_ptr<MemoryResourceRef> set_current_device_resource(const MemoryResourceRef& mr) {
  auto* old = rmm::mr::set_current_device_resource(mr.get());
  return std::make_unique<MemoryResourceRef>(old);
}

void reset_current_device_resource() {
  rmm::mr::set_current_device_resource(nullptr);
}

// ---- ScopedDevice ----

std::unique_ptr<ScopedDevice> scoped_device_new(int32_t device_id) {
  return std::make_unique<ScopedDevice>(device_id);
}

// ---- Prefetch ----

void prefetch(std::size_t ptr, std::size_t size, int32_t device, std::size_t stream) {
  rmm::prefetch(
      reinterpret_cast<void const*>(ptr),
      size,
      rmm::cuda_device_id{device},
      rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)});
}

// ---- Alignment utilities ----

std::size_t cuda_allocation_alignment() {
  return rmm::CUDA_ALLOCATION_ALIGNMENT;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return rmm::align_up(value, alignment);
}

std::size_t align_down(std::size_t value, std::size_t alignment) {
  return rmm::align_down(value, alignment);
}

bool is_aligned(std::size_t value, std::size_t alignment) {
  return rmm::is_aligned(value, alignment);
}

}  // namespace rmm_sys
