// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA stream types for GPU work ordering.
//!
//! This module provides [`Stream`], an owning CUDA stream, and [`StreamPool`],
//! a pool of streams for concurrent GPU work. Streams provide ordering
//! guarantees: operations submitted to the same stream execute in order, while
//! operations on different streams may execute concurrently.

use std::fmt;

use crate::error::Result;
use cxx::UniquePtr;

/// An owning CUDA stream.
///
/// Unlike the lightweight `cudf::stream::Stream` (which is a non-owning
/// handle), this type owns the underlying CUDA stream and destroys it on
/// drop.
///
/// # Examples
///
/// ```no_run
/// use rmm::stream::Stream;
///
/// let stream = Stream::new()?;
/// println!("stream handle: {:#x}", stream.as_raw());
/// stream.synchronize()?;
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[doc(alias = "rmm::cuda_stream")]
pub struct Stream(UniquePtr<rmm_sys::ffi::CudaStream>);

impl Stream {
    /// Creates a new CUDA stream.
    ///
    /// The stream is destroyed when this value is dropped.
    pub fn new() -> Result<Self> {
        Ok(Self(rmm_sys::ffi::cuda_stream_new()?))
    }

    /// Returns the raw `cudaStream_t` handle as `usize`.
    ///
    /// This is useful when interacting with other CUDA libraries or FFI
    /// functions that accept raw stream handles.
    pub fn as_raw(&self) -> usize {
        rmm_sys::ffi::cuda_stream_view(&self.0)
    }

    /// Synchronizes the stream, blocking until all preceding work completes.
    ///
    /// After this call returns, all GPU operations previously submitted to
    /// this stream are guaranteed to have finished.
    pub fn synchronize(&self) -> Result<()> {
        rmm_sys::ffi::cuda_stream_synchronize(&self.0)?;
        Ok(())
    }

    /// Returns `true` if the stream handle is valid (non-null).
    pub fn is_valid(&self) -> bool {
        rmm_sys::ffi::cuda_stream_is_valid(&self.0)
    }
}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Stream").field(&self.as_raw()).finish()
    }
}

impl fmt::Display for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaStream({:#x})", self.as_raw())
    }
}

/// A pool of CUDA streams for concurrent work.
///
/// Streams are handed out in round-robin fashion via [`get_stream`](StreamPool::get_stream).
/// This is useful for overlapping independent GPU operations across multiple
/// streams without manually managing individual stream lifetimes.
///
/// # Examples
///
/// ```no_run
/// use rmm::stream::StreamPool;
///
/// let pool = StreamPool::new(4)?;
/// assert_eq!(pool.pool_size(), 4);
///
/// let handle = pool.get_stream();
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[doc(alias = "rmm::cuda_stream_pool")]
pub struct StreamPool(UniquePtr<rmm_sys::ffi::CudaStreamPool>);

impl StreamPool {
    /// Creates a new pool with `size` streams.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`](crate::error::Error::InvalidArgument) if
    /// `size` is zero.
    pub fn new(size: usize) -> crate::error::Result<Self> {
        if size == 0 {
            return Err(crate::error::Error::InvalidArgument(
                "pool_size must be > 0".into(),
            ));
        }
        Ok(Self(rmm_sys::ffi::cuda_stream_pool_new(size)?))
    }

    /// Returns a raw stream handle (`cudaStream_t` as `usize`) from the pool.
    ///
    /// Successive calls cycle through the pool in round-robin order.
    pub fn get_stream(&self) -> usize {
        rmm_sys::ffi::cuda_stream_pool_get_stream(&self.0)
    }

    /// Returns the number of streams in the pool.
    pub fn pool_size(&self) -> usize {
        rmm_sys::ffi::cuda_stream_pool_get_pool_size(&self.0)
    }
}

impl fmt::Debug for StreamPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamPool")
            .field("pool_size", &self.pool_size())
            .finish()
    }
}

impl fmt::Display for StreamPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaStreamPool(size={})", self.pool_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_creation() {
        let _test_lock = crate::test_lock();
        let stream = Stream::new().unwrap();
        assert!(stream.is_valid());
        assert_ne!(stream.as_raw(), 0);
    }

    #[test]
    fn stream_synchronize() {
        let _test_lock = crate::test_lock();
        let stream = Stream::new().unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn stream_debug_display() {
        let _test_lock = crate::test_lock();
        let stream = Stream::new().unwrap();
        let debug = format!("{stream:?}");
        assert!(debug.starts_with("Stream("));
        let display = format!("{stream}");
        assert!(display.starts_with("CudaStream(0x"));
    }

    #[test]
    fn stream_pool_creation() {
        let _test_lock = crate::test_lock();
        let pool = StreamPool::new(4).unwrap();
        assert_eq!(pool.pool_size(), 4);
    }

    #[test]
    fn stream_pool_zero_size_errors() {
        let _test_lock = crate::test_lock();
        let result = StreamPool::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn stream_pool_round_robin() {
        let _test_lock = crate::test_lock();
        let pool = StreamPool::new(2).unwrap();
        let s1 = pool.get_stream();
        let s2 = pool.get_stream();
        let s3 = pool.get_stream();
        // Round-robin: s3 should cycle back to s1
        assert_eq!(s1, s3);
        // Different streams in the pool should have different handles
        assert_ne!(s1, s2);
    }

    #[test]
    fn stream_pool_debug_display() {
        let _test_lock = crate::test_lock();
        let pool = StreamPool::new(3).unwrap();
        let debug = format!("{pool:?}");
        assert!(debug.contains("pool_size: 3"));
        let display = format!("{pool}");
        assert!(display.contains("size=3"));
    }
}
