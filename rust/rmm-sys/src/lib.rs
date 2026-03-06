// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Low-level CXX FFI bindings for RMM (RAPIDS Memory Manager).
//!
//! This crate provides device queries (memory, device count).

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
