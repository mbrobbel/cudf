// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::pedantic)]
#![deny(clippy::shadow_reuse)]
#![deny(clippy::shadow_same)]
#![deny(clippy::shadow_unrelated)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::as_conversions)]
// Allow common pedantic false positives in this codebase.
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]
// These lints are only relevant in test code.
#![cfg_attr(test, allow(clippy::unwrap_used))]

//! Safe Rust bindings for [RMM](https://github.com/rapidsai/rmm),
//! the RAPIDS Memory Manager.
//!
//! This crate provides idiomatic Rust wrappers around the core RMM types:
//!
//! - [`device::DeviceId`] — a CUDA device identifier
//! - [`gpu_context::GpuContext`] — a single-device owner for allocators and streams
//! - [`stream::Stream`] — an owning CUDA stream
//! - [`stream::StreamPool`] — a pool of CUDA streams for concurrent work
//! - [`buffer::DeviceBuffer`] — an untyped GPU memory buffer
//! - [`aligned`] — CUDA alignment utilities
//! - [`memory_resource`] — GPU memory resource allocators
//! - [`device::ScopedDevice`] — RAII guard for multi-GPU device switching
//!
//! GPU memory management is handled by RMM under the hood. All types in this
//! crate are safe wrappers — `unsafe` code lives exclusively in `rmm-sys`.
//!
//! # Recommended Safe Path
//!
//! The preferred API shape is to allocate and execute through an explicit
//! [`gpu_context::GpuContext`]:
//!
//! ```no_run
//! use rmm::buffer::DeviceBuffer;
//! use rmm::device::current_device;
//! use rmm::gpu_context::GpuContext;
//!
//! let ctx = GpuContext::<()>::new(current_device())?;
//! let alloc = ctx.default_device_allocator();
//! let exec = ctx.default_stream();
//!
//! let buf = DeviceBuffer::new_in(&alloc, 4096)?;
//! exec.synchronize()?;
//! assert_eq!(buf.size(), 4096);
//! # Ok::<(), rmm::error::Error>(())
//! ```

pub mod aligned;
pub mod buffer;
pub mod device;
pub mod error;
pub mod gpu_context;
pub mod memory_resource;
pub mod prefetch;
pub mod stream;

#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("RMM test lock poisoned")
}
