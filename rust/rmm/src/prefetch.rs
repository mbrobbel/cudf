// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA memory prefetching.
//!
//! Prefetching hints to the CUDA driver to migrate managed memory to a
//! specific device before it is accessed, reducing page fault overhead.

use crate::device::DeviceId;

/// Prefetches memory to the specified device on the given stream.
///
/// This is a no-op if the pointer does not refer to CUDA managed memory
/// or if concurrent managed access is not supported on the device.
///
/// # Parameters
///
/// - `ptr`: Device pointer as `usize`.
/// - `size`: Number of bytes to prefetch.
/// - `device`: Target CUDA device.
/// - `stream`: Raw `cudaStream_t` handle as `usize`.
///
/// # Examples
///
/// ```no_run
/// use rmm::device;
/// use rmm::prefetch;
///
/// # let device_ptr = 0usize;
/// let dev = device::current_device();
/// prefetch::prefetch(device_ptr, 1024, dev, 0)?;
/// # Ok::<(), rmm::error::Error>(())
/// ```
pub fn prefetch(
    ptr: usize,
    size: usize,
    device: DeviceId,
    stream: usize,
) -> crate::error::Result<()> {
    rmm_sys::ffi::prefetch(ptr, size, device.value(), stream)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::DeviceBuffer;
    use crate::device;

    #[test]
    fn prefetch_buffer() {
        let _test_lock = crate::test_lock();
        let buf = DeviceBuffer::new(1024).unwrap();
        let dev = device::current_device();
        // Should not panic; may be a no-op if not managed memory.
        prefetch(buf.as_ptr(), buf.size(), dev, 0).unwrap();
    }
}
