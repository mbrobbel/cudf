// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! JIT cache control for the cudf global context.
//!
//! cudf uses a JIT (just-in-time) compiler to generate optimized GPU kernels
//! at runtime. The JIT cache stores compiled kernels so they can be reused
//! across calls, avoiding repeated compilation overhead.
//!
//! These functions control the global JIT cache:
//!
//! - [`enable_jit_cache`] -- Enables or disables the JIT cache.
//! - [`clear_jit_cache`] -- Clears all cached kernels from memory and disk.

#[doc(alias = "cudf::enable_jit_cache")]
/// Enables or disables the JIT program cache.
///
/// When `enable` is `true`, the cache stores and retrieves compiled programs
/// as normal. When `false`, the cache is bypassed but its contents are
/// preserved, allowing easy re-enabling later.
///
/// It is safe to call this multiple times.
pub fn enable_jit_cache(enable: bool) {
    cudf_sys::ffi::enable_jit_cache(enable);
}

#[doc(alias = "cudf::clear_jit_cache")]
/// Clears the JIT program cache, removing all cached programs from memory
/// and disk.
///
/// This is more expensive than [`enable_jit_cache`]`(false)` since it
/// deletes cached files from disk. Prefer disabling the cache for temporary
/// bypassing.
///
/// # Safety note
///
/// While this function is not marked `unsafe`, the caller must ensure that no
/// other threads are concurrently executing cudf GPU operations when this is
/// called.
pub fn clear_jit_cache() {
    cudf_sys::ffi::clear_jit_cache();
}
