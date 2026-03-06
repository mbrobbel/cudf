// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

pub use rmm_sys::ffi::DeviceId;

/// Returns the number of CUDA devices available.
pub fn num_devices() -> i32 {
    rmm_sys::ffi::get_num_cuda_devices()
}

/// Returns the current CUDA device.
pub fn current_device() -> DeviceId {
    rmm_sys::ffi::get_current_device()
}

/// Returns the available (free) device memory in bytes.
pub fn available_device_memory() -> usize {
    rmm_sys::ffi::available_device_memory()
}

/// Returns the total device memory in bytes.
pub fn total_device_memory() -> usize {
    rmm_sys::ffi::total_device_memory()
}

/// Returns the specified percentage of free device memory in bytes.
pub fn percent_of_free_device_memory(percent: i32) -> usize {
    rmm_sys::ffi::percent_of_free_device_memory(percent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_at_least_one_device() {
        assert!(num_devices() > 0);
    }

    #[test]
    fn current_device_is_valid() {
        let dev = current_device();
        assert!(dev.value >= 0);
        assert!(dev.value < num_devices());
    }

    #[test]
    fn device_memory_is_nonzero() {
        assert!(total_device_memory() > 0);
        assert!(available_device_memory() > 0);
        assert!(available_device_memory() <= total_device_memory());
    }

    #[test]
    fn percent_of_free_memory() {
        let half = percent_of_free_device_memory(50);
        let full = available_device_memory();
        // 50% should be roughly half of available, with some tolerance for concurrent changes
        assert!(half > 0);
        assert!(half <= full);
    }
}
