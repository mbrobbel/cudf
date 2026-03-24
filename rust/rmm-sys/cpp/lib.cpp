// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "rmm-sys/src/lib.rs.h"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
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
