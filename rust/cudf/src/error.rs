// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Error types for cudf operations.
//!
//! This module defines the [`Error`] enum and the [`Result`] type alias used
//! throughout the cudf crate. Most errors originate from the underlying
//! libcudf C++ library and are wrapped as [`Error::Cudf`].

use std::fmt;

/// Error type for cudf operations.
///
/// This enum is `#[non_exhaustive]`, so new variants may be added in future
/// versions without breaking existing match arms.
///
/// # Conversion
///
/// A [`cxx::Exception`] is automatically converted into `Error::Cudf` via the
/// [`From`] implementation, so most FFI call sites can use `?` directly.
///
/// # Examples
///
/// ```
/// use cudf::error::Error;
///
/// let err = Error::OutOfBounds { index: 5, len: 3 };
/// assert_eq!(err.to_string(), "index 5 out of bounds for length 3");
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An error propagated from the underlying libcudf C++ library.
    ///
    /// The inner [`cxx::Exception`] contains the C++ exception message.
    Cudf(cxx::Exception),
    /// An index was out of bounds for the given container.
    OutOfBounds {
        /// The index that was out of bounds.
        index: usize,
        /// The length of the container.
        len: usize,
    },
    /// A file path could not be converted to valid UTF-8.
    InvalidPath,
    /// An argument was invalid.
    InvalidArgument(String),
    /// An Arrow data type is not supported for conversion.
    #[cfg(feature = "arrow")]
    UnsupportedArrowType(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cudf(e) => write!(f, "cudf error: {e}"),
            Error::OutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
            Error::InvalidPath => write!(f, "path is not valid UTF-8"),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            #[cfg(feature = "arrow")]
            Error::UnsupportedArrowType(msg) => {
                write!(f, "unsupported Arrow type: {msg}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Cudf(e) => Some(e),
            Error::OutOfBounds { .. } | Error::InvalidPath | Error::InvalidArgument(_) => None,
            #[cfg(feature = "arrow")]
            Error::UnsupportedArrowType(_) => None,
        }
    }
}

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Error::Cudf(e)
    }
}

/// Result type alias for cudf operations.
///
/// Equivalent to `std::result::Result<T, cudf::error::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
