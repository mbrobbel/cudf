// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace rmm_sys {

// DeviceId is a CXX shared struct — defined in the generated header.
struct DeviceId;

/// Returns the number of CUDA devices available.
int32_t get_num_cuda_devices();

/// Returns the current CUDA device.
DeviceId get_current_device();

/// Returns the available (free) device memory in bytes.
std::size_t available_device_memory();

/// Returns the total device memory in bytes.
std::size_t total_device_memory();

/// Returns the specified percentage of free device memory in bytes.
std::size_t percent_of_free_device_memory(int32_t percent);

}  // namespace rmm_sys
