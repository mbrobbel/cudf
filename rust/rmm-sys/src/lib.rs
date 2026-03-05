// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#[cxx::bridge(namespace = "rmm_sys")]
pub mod ffi {
    /// A CUDA device identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct DeviceId {
        value: i32,
    }

    unsafe extern "C++" {
        include!("lib.hpp");

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
    }
}
