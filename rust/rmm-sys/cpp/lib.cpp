// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#include "rmm-sys/src/lib.rs.h"

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

namespace rmm_sys {

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

}  // namespace rmm_sys
