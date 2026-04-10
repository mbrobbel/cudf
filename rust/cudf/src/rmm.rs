// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Re-export of the [`rmm`] crate for GPU memory management.
//!
//! This module re-exports the safe RMM API so cudf users can access device
//! queries, memory resources, streams, and buffers without adding a separate
//! `rmm` dependency.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::rmm::device;
//! use cudf::rmm::memory_resource::PoolMemoryResource;
//!
//! let n = device::num_devices();
//! let dev = device::current_device();
//! println!("{n} devices, current: {dev}");
//! ```

pub use rmm::aligned;
pub use rmm::buffer;
pub use rmm::device;
pub use rmm::error;
pub use rmm::gpu_context;
pub use rmm::memory_resource;
pub use rmm::prefetch;
pub use rmm::stream;

#[cfg(test)]
mod tests {
    #[test]
    fn has_at_least_one_device() {
        assert!(rmm::device::num_devices() > 0);
    }

    #[test]
    fn current_device_is_valid() {
        let dev = rmm::device::current_device();
        assert!(dev.value() >= 0);
        assert!(dev.value() < rmm::device::num_devices());
    }

    #[test]
    fn device_memory_is_nonzero() {
        assert!(rmm::device::total_memory() > 0);
        assert!(rmm::device::available_memory() > 0);
        assert!(rmm::device::available_memory() <= rmm::device::total_memory());
    }

    #[test]
    fn percent_of_free_memory() {
        let half = rmm::device::percent_of_free_memory(50);
        let full = rmm::device::available_memory();
        assert!(half > 0);
        assert!(half <= full);
    }
}
