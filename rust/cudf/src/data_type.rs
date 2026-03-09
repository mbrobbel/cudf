// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Logical element type descriptors ([`DataType`] and [`TypeId`]).
//!
//! Every column in cudf has a [`DataType`] that describes its element type.
//! For most types, [`TypeId`] alone is sufficient; fixed-point types
//! (`DECIMAL32`, `DECIMAL64`, `DECIMAL128`) additionally carry a `scale`.

/// Identifies the logical element type of a column.
///
/// This is a C++ enum mirrored via CXX. Common variants include:
///
/// * Integers: `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`
/// * Floating point: `FLOAT32`, `FLOAT64`
/// * Boolean: `BOOL8`
/// * String: `STRING`
/// * Timestamps: `TIMESTAMP_SECONDS`, `TIMESTAMP_MILLISECONDS`,
///   `TIMESTAMP_MICROSECONDS`, `TIMESTAMP_NANOSECONDS`
/// * Durations: `DURATION_SECONDS`, `DURATION_MILLISECONDS`,
///   `DURATION_MICROSECONDS`, `DURATION_NANOSECONDS`
/// * Fixed-point decimals: `DECIMAL32`, `DECIMAL64`, `DECIMAL128`
/// * Nested: `LIST`, `STRUCT`
/// * Other: `DICTIONARY32`, `EMPTY`
#[doc(alias = "type_id")]
pub use cudf_sys::ffi::TypeId;

#[doc(alias = "data_type")]
/// A logical data type descriptor.
///
/// Wraps a [`TypeId`] and an optional `scale` for fixed-point types
/// (`DECIMAL32`, `DECIMAL64`, `DECIMAL128`). For all other types the scale
/// is 0 and can be ignored.
///
/// # Examples
///
/// ```
/// use cudf::data_type::{DataType, TypeId};
///
/// // Simple type (scale defaults to 0)
/// let dt = DataType::from(TypeId::INT32);
/// assert_eq!(dt.id(), TypeId::INT32);
/// assert_eq!(dt.scale(), 0);
///
/// // Fixed-point type with scale -3 (thousandths)
/// let decimal = DataType::new(TypeId::DECIMAL64, -3);
/// assert_eq!(decimal.scale(), -3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataType {
    id: TypeId,
    scale: i32,
}

impl DataType {
    /// Creates a new `DataType` from a type id and scale.
    ///
    /// For non-decimal types, `scale` should be `0`. For decimal types, the
    /// scale is the power-of-ten exponent (e.g. `-3` means the value is
    /// stored as `value * 10^-3`).
    pub fn new(id: TypeId, scale: i32) -> Self {
        Self { id, scale }
    }

    /// Returns the type identifier.
    pub fn id(&self) -> TypeId {
        self.id
    }

    /// Returns the scale (meaningful only for fixed-point decimal types).
    ///
    /// For non-decimal types, this always returns `0`.
    pub fn scale(&self) -> i32 {
        self.scale
    }
}

impl From<TypeId> for DataType {
    fn from(id: TypeId) -> Self {
        Self { id, scale: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_from_type_id() {
        let dt = DataType::from(TypeId::INT32);
        assert_eq!(dt.id(), TypeId::INT32);
        assert_eq!(dt.scale(), 0);
    }

    #[test]
    fn data_type_with_scale() {
        let dt = DataType::new(TypeId::DECIMAL64, -3);
        assert_eq!(dt.id(), TypeId::DECIMAL64);
        assert_eq!(dt.scale(), -3);
    }

    #[test]
    fn type_id_equality() {
        assert_eq!(TypeId::FLOAT32, TypeId::FLOAT32);
        assert_ne!(TypeId::FLOAT32, TypeId::FLOAT64);
    }

    #[test]
    fn data_type_from_impl() {
        let dt: DataType = TypeId::STRING.into();
        assert_eq!(dt.id(), TypeId::STRING);
        assert_eq!(dt.scale(), 0);
    }

    #[test]
    fn data_type_clone_eq() {
        let dt = DataType::new(TypeId::INT64, 0);
        let dt2 = dt;
        assert_eq!(dt, dt2);
    }
}
