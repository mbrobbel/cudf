// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA memory alignment utilities.
//!
//! These functions mirror the alignment helpers in RMM and are useful for
//! computing properly aligned buffer sizes and offsets for GPU memory.

/// Returns the CUDA allocation alignment constant (256 bytes).
///
/// All RMM device allocations are aligned to at least this value.
#[doc(alias = "RMM_DEFAULT_HOST_ALIGNMENT")]
pub fn cuda_allocation_alignment() -> usize {
    rmm_sys::ffi::cuda_allocation_alignment()
}

/// Aligns `value` up to the nearest multiple of `alignment`.
///
/// `alignment` must be a power of two.
///
/// # Examples
///
/// ```
/// use rmm::aligned::align_up;
///
/// assert_eq!(align_up(100, 256), 256);
/// assert_eq!(align_up(256, 256), 256);
/// assert_eq!(align_up(257, 256), 512);
/// ```
pub fn align_up(value: usize, alignment: usize) -> usize {
    rmm_sys::ffi::align_up(value, alignment)
}

/// Aligns `value` down to the nearest multiple of `alignment`.
///
/// `alignment` must be a power of two.
///
/// # Examples
///
/// ```
/// use rmm::aligned::align_down;
///
/// assert_eq!(align_down(300, 256), 256);
/// assert_eq!(align_down(256, 256), 256);
/// assert_eq!(align_down(100, 256), 0);
/// ```
pub fn align_down(value: usize, alignment: usize) -> usize {
    rmm_sys::ffi::align_down(value, alignment)
}

/// Returns `true` if `value` is aligned to `alignment`.
///
/// `alignment` must be a power of two.
///
/// # Examples
///
/// ```
/// use rmm::aligned::is_aligned;
///
/// assert!(is_aligned(256, 256));
/// assert!(is_aligned(512, 256));
/// assert!(!is_aligned(100, 256));
/// ```
pub fn is_aligned(value: usize, alignment: usize) -> bool {
    rmm_sys::ffi::is_aligned(value, alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_alignment_is_256() {
        assert_eq!(cuda_allocation_alignment(), 256);
    }

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn align_down_basic() {
        assert_eq!(align_down(0, 256), 0);
        assert_eq!(align_down(1, 256), 0);
        assert_eq!(align_down(255, 256), 0);
        assert_eq!(align_down(256, 256), 256);
        assert_eq!(align_down(511, 256), 256);
    }

    #[test]
    fn is_aligned_basic() {
        assert!(is_aligned(0, 256));
        assert!(is_aligned(256, 256));
        assert!(is_aligned(512, 256));
        assert!(!is_aligned(1, 256));
        assert!(!is_aligned(255, 256));
    }
}
