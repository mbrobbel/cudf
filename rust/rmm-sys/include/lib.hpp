// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

// ---------- Prefetch ----------

void prefetch(std::size_t ptr, std::size_t size, int32_t device, std::size_t stream);

// ---------- Alignment utilities ----------

std::size_t cuda_allocation_alignment();
std::size_t align_up(std::size_t value, std::size_t alignment);
std::size_t align_down(std::size_t value, std::size_t alignment);
bool is_aligned(std::size_t value, std::size_t alignment);

}  // namespace rmm_sys
