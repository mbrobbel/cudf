// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

/// Error type for cudf operations.
#[derive(Debug)]
pub enum Error {
    /// An error from the underlying C++ library.
    Cudf(cxx::Exception),
    /// An index was out of bounds.
    OutOfBounds {
        /// The index that was out of bounds.
        index: usize,
        /// The length of the container.
        len: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cudf(e) => write!(f, "cudf error: {e}"),
            Error::OutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Cudf(e) => Some(e),
            Error::OutOfBounds { .. } => None,
        }
    }
}

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        Error::Cudf(e)
    }
}

/// Result type alias for cudf operations.
pub type Result<T> = std::result::Result<T, Error>;
