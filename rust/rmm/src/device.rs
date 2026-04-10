// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device queries and scoped device switching.
//!
//! This module provides functions to query the available CUDA devices and
//! their memory capacity. The [`DeviceId`] type identifies a specific GPU.
//! [`ScopedDevice`] is an RAII guard that temporarily switches the active
//! CUDA device.

use std::fmt;

/// A CUDA device identifier.
///
/// Wraps the integer device ID used by the CUDA runtime. Device IDs are
/// zero-based and range from `0` to [`num_devices()`]` - 1`.
///
/// # Examples
///
/// ```no_run
/// use rmm::device;
///
/// let dev = device::current_device();
/// println!("Current device: {dev}");
/// assert!(dev.value() >= 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[doc(alias = "rmm::cuda_device_id")]
pub struct DeviceId(i32);

impl DeviceId {
    /// Creates a `DeviceId` from a raw integer device ID.
    pub fn new(id: i32) -> Self {
        Self(id)
    }

    /// Returns the raw device ID value.
    pub fn value(self) -> i32 {
        self.0
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuda:{}", self.0)
    }
}

impl From<rmm_sys::ffi::DeviceId> for DeviceId {
    fn from(ffi: rmm_sys::ffi::DeviceId) -> Self {
        Self(ffi.value)
    }
}

/// Returns the number of CUDA devices available.
///
/// # Examples
///
/// ```no_run
/// use rmm::device;
///
/// let n = device::num_devices();
/// assert!(n > 0);
/// ```
pub fn num_devices() -> i32 {
    rmm_sys::ffi::get_num_cuda_devices()
}

/// Returns the current CUDA device.
///
/// The current device is set per-thread by the CUDA runtime.
///
/// # Examples
///
/// ```no_run
/// use rmm::device;
///
/// let dev = device::current_device();
/// assert!(dev.value() < device::num_devices());
/// ```
pub fn current_device() -> DeviceId {
    rmm_sys::ffi::get_current_device().into()
}

/// Returns the available (free) device memory in bytes.
///
/// The value reflects the free memory at the moment of the call and may
/// change as other processes allocate or free GPU memory.
pub fn available_memory() -> usize {
    rmm_sys::ffi::available_device_memory()
}

/// Returns the total device memory in bytes.
///
/// This is the total physical memory on the current CUDA device.
pub fn total_memory() -> usize {
    rmm_sys::ffi::total_device_memory()
}

/// Returns the specified percentage of free device memory in bytes.
///
/// # Parameters
///
/// - `percent`: The percentage of free memory to compute (0..=100).
///
/// # Examples
///
/// ```no_run
/// use rmm::device;
///
/// let half = device::percent_of_free_memory(50);
/// assert!(half > 0);
/// assert!(half <= device::available_memory());
/// ```
pub fn percent_of_free_memory(percent: i32) -> usize {
    rmm_sys::ffi::percent_of_free_device_memory(percent)
}

/// RAII guard that sets the current CUDA device on construction and restores
/// the previous device when dropped.
///
/// This is useful for multi-GPU workloads where you need to temporarily
/// operate on a different device.
///
/// # Examples
///
/// ```no_run
/// use rmm::device::ScopedDevice;
/// use rmm::device::DeviceId;
///
/// {
///     let _guard = ScopedDevice::new(DeviceId::new(1))?;
///     // Current device is now 1.
///     // ... do work on device 1 ...
/// }
/// // Previous device is restored here.
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[doc(alias = "rmm::cuda_set_device_raii")]
pub struct ScopedDevice(
    /// Held for RAII — drop restores the previous device.
    #[allow(dead_code)]
    cxx::UniquePtr<rmm_sys::ffi::ScopedDevice>,
);

impl ScopedDevice {
    /// Sets the current CUDA device to `device` and returns a guard.
    ///
    /// When the guard is dropped, the previous device is restored.
    pub fn new(device: DeviceId) -> crate::error::Result<Self> {
        let value = device.value();
        let device_count = num_devices();
        if value < 0 || value >= device_count {
            return Err(crate::error::Error::InvalidArgument(format!(
                "device id {value} is out of range 0..{}",
                device_count.saturating_sub(1)
            )));
        }
        Ok(Self(rmm_sys::ffi::scoped_device_new(device.value())?))
    }
}

impl fmt::Debug for ScopedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedDevice").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_at_least_one_device() {
        let _test_lock = crate::test_lock();
        assert!(num_devices() > 0);
    }

    #[test]
    fn current_device_is_valid() {
        let _test_lock = crate::test_lock();
        let dev = current_device();
        assert!(dev.value() >= 0);
        assert!(dev.value() < num_devices());
    }

    #[test]
    fn device_memory_is_nonzero() {
        let _test_lock = crate::test_lock();
        assert!(total_memory() > 0);
        assert!(available_memory() > 0);
        assert!(available_memory() <= total_memory());
    }

    #[test]
    fn percent_of_free() {
        let _test_lock = crate::test_lock();
        let half = percent_of_free_memory(50);
        let full = available_memory();
        assert!(half > 0);
        assert!(half <= full);
    }

    #[test]
    fn device_id_display() {
        let _test_lock = crate::test_lock();
        let dev = DeviceId::new(0);
        assert_eq!(dev.to_string(), "cuda:0");
    }

    #[test]
    fn device_id_roundtrip() {
        let _test_lock = crate::test_lock();
        let dev = DeviceId::new(42);
        assert_eq!(dev.value(), 42);
    }

    #[test]
    fn scoped_device_invalid_id_errors() {
        let _test_lock = crate::test_lock();
        let invalid = DeviceId::new(num_devices());
        assert!(ScopedDevice::new(invalid).is_err());
    }

    #[test]
    fn scoped_device_negative_id_errors() {
        let _test_lock = crate::test_lock();
        assert!(ScopedDevice::new(DeviceId::new(-1)).is_err());
    }

    #[test]
    fn scoped_device_restores_previous_device() {
        let _test_lock = crate::test_lock();
        let original = current_device();
        let target = if num_devices() > 1 {
            DeviceId::new((original.value() + 1) % num_devices())
        } else {
            original
        };

        {
            let _guard = ScopedDevice::new(target).unwrap();
            assert_eq!(current_device(), target);
        }

        assert_eq!(current_device(), original);
    }

    #[test]
    fn scoped_device_nested_guards_restore_lifo() {
        let _test_lock = crate::test_lock();
        let original = current_device();
        let first = if num_devices() > 1 {
            DeviceId::new((original.value() + 1) % num_devices())
        } else {
            original
        };

        {
            let _outer = ScopedDevice::new(first).unwrap();
            assert_eq!(current_device(), first);

            {
                let _inner = ScopedDevice::new(original).unwrap();
                assert_eq!(current_device(), original);
            }

            assert_eq!(current_device(), first);
        }

        assert_eq!(current_device(), original);
    }
}
