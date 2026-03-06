// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Logical element type descriptors ([`DataType`] and [`TypeId`]).

pub use cudf_sys::ffi::TypeId;

/// A logical data type descriptor.
///
/// Wraps a [`TypeId`] and an optional scale for fixed-point types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataType {
    id: TypeId,
    scale: i32,
}

impl DataType {
    /// Creates a new `DataType` from a type id and scale.
    pub fn new(id: TypeId, scale: i32) -> Self {
        Self { id, scale }
    }

    /// Returns the type identifier.
    pub fn id(&self) -> TypeId {
        self.id
    }

    /// Returns the scale (meaningful for fixed-point types).
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
