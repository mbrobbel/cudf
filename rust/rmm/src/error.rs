// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Error types for RMM operations.
//!
//! This module defines the [`Error`] enum and the [`Result`] type alias used
//! throughout the rmm crate. Most errors originate from the underlying
//! RMM C++ library and are wrapped as [`Error::Rmm`].

use std::fmt;

/// Error type for RMM operations.
///
/// This enum is `#[non_exhaustive]`, so new variants may be added in future
/// versions without breaking existing match arms.
///
/// # Conversion
///
/// A [`cxx::Exception`] is automatically converted into `Error::Rmm` via the
/// [`From`] implementation, so most FFI call sites can use `?` directly.
///
/// # Examples
///
/// ```
/// use rmm::error::Error;
///
/// let msg = format!("{}", Error::InvalidArgument("pool_size must be > 0".into()));
/// assert!(msg.contains("pool_size must be > 0"));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An error propagated from the underlying RMM C++ library.
    ///
    /// The inner [`cxx::Exception`] contains the C++ exception message.
    Rmm(cxx::Exception),
    /// An invalid argument was provided to an RMM function.
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rmm(e) => write!(f, "rmm error: {e}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rmm(e) => Some(e),
            Self::InvalidArgument(_) => None,
        }
    }
}

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Self::Rmm(e)
    }
}

/// Result type alias for RMM operations.
///
/// Equivalent to `std::result::Result<T, rmm::error::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
