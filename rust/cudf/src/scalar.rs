// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use cxx::UniquePtr;

use crate::data_type::TypeId;

/// An owning GPU scalar value.
///
/// Wraps a `cudf::scalar` via the CXX FFI layer.
pub struct Scalar(pub(crate) UniquePtr<cudf_sys::ffi::Scalar>);

// GPU memory is globally accessible from any CPU thread.
unsafe impl Send for Scalar {}
unsafe impl Sync for Scalar {}

impl Scalar {
    /// Creates a valid INT32 scalar.
    pub fn from_i32(value: i32) -> Self {
        Self(cudf_sys::ffi::make_int32_scalar(value, true))
    }

    /// Creates a null INT32 scalar.
    pub fn null_i32() -> Self {
        Self(cudf_sys::ffi::make_int32_scalar(0, false))
    }

    /// Creates a valid INT64 scalar.
    pub fn from_i64(value: i64) -> Self {
        Self(cudf_sys::ffi::make_int64_scalar(value, true))
    }

    /// Creates a null INT64 scalar.
    pub fn null_i64() -> Self {
        Self(cudf_sys::ffi::make_int64_scalar(0, false))
    }

    /// Creates a valid FLOAT32 scalar.
    pub fn from_f32(value: f32) -> Self {
        Self(cudf_sys::ffi::make_float32_scalar(value, true))
    }

    /// Creates a null FLOAT32 scalar.
    pub fn null_f32() -> Self {
        Self(cudf_sys::ffi::make_float32_scalar(0.0, false))
    }

    /// Creates a valid FLOAT64 scalar.
    pub fn from_f64(value: f64) -> Self {
        Self(cudf_sys::ffi::make_float64_scalar(value, true))
    }

    /// Creates a null FLOAT64 scalar.
    pub fn null_f64() -> Self {
        Self(cudf_sys::ffi::make_float64_scalar(0.0, false))
    }

    /// Creates a valid BOOL8 scalar.
    pub fn from_bool(value: bool) -> Self {
        Self(cudf_sys::ffi::make_bool_scalar(value, true))
    }

    /// Creates a null BOOL8 scalar.
    pub fn null_bool() -> Self {
        Self(cudf_sys::ffi::make_bool_scalar(false, false))
    }

    /// Creates a valid STRING scalar.
    pub fn from_string(value: &str) -> Self {
        Self(cudf_sys::ffi::make_string_scalar(value))
    }

    /// Returns `true` if the scalar holds a valid (non-null) value.
    pub fn is_valid(&self) -> bool {
        cudf_sys::ffi::scalar_is_valid(&self.0)
    }

    /// Returns the type identifier of the scalar.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::scalar_type_id(&self.0);
        // Safety: type_id values from C++ are valid TypeId discriminants
        unsafe { std::mem::transmute::<i32, TypeId>(id) }
    }

    /// Extracts the value as `i32`, or `None` if the type doesn't match.
    pub fn as_i32(&self) -> Option<i32> {
        if self.type_id() != TypeId::INT32 || !self.is_valid() {
            return None;
        }
        Some(cudf_sys::ffi::scalar_to_i32(&self.0))
    }

    /// Extracts the value as `i64`, or `None` if the type doesn't match.
    pub fn as_i64(&self) -> Option<i64> {
        if self.type_id() != TypeId::INT64 || !self.is_valid() {
            return None;
        }
        Some(cudf_sys::ffi::scalar_to_i64(&self.0))
    }

    /// Extracts the value as `f32`, or `None` if the type doesn't match.
    pub fn as_f32(&self) -> Option<f32> {
        if self.type_id() != TypeId::FLOAT32 || !self.is_valid() {
            return None;
        }
        Some(cudf_sys::ffi::scalar_to_f32(&self.0))
    }

    /// Extracts the value as `f64`, or `None` if the type doesn't match.
    pub fn as_f64(&self) -> Option<f64> {
        if self.type_id() != TypeId::FLOAT64 || !self.is_valid() {
            return None;
        }
        Some(cudf_sys::ffi::scalar_to_f64(&self.0))
    }

    /// Extracts the value as `bool`, or `None` if the type doesn't match.
    pub fn as_bool(&self) -> Option<bool> {
        if self.type_id() != TypeId::BOOL8 || !self.is_valid() {
            return None;
        }
        Some(cudf_sys::ffi::scalar_to_bool(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_i32_roundtrip() {
        let s = Scalar::from_i32(42);
        assert!(s.is_valid());
        assert_eq!(s.type_id(), TypeId::INT32);
        assert_eq!(s.as_i32(), Some(42));
    }

    #[test]
    fn scalar_i64_roundtrip() {
        let s = Scalar::from_i64(123_456_789);
        assert!(s.is_valid());
        assert_eq!(s.type_id(), TypeId::INT64);
        assert_eq!(s.as_i64(), Some(123_456_789));
    }

    #[test]
    fn scalar_f32_roundtrip() {
        let s = Scalar::from_f32(3.14);
        assert!(s.is_valid());
        assert_eq!(s.type_id(), TypeId::FLOAT32);
        assert!((s.as_f32().unwrap() - 3.14).abs() < 1e-5);
    }

    #[test]
    fn scalar_f64_roundtrip() {
        let s = Scalar::from_f64(2.718281828);
        assert!(s.is_valid());
        assert_eq!(s.type_id(), TypeId::FLOAT64);
        assert!((s.as_f64().unwrap() - 2.718281828).abs() < 1e-9);
    }

    #[test]
    fn scalar_bool_roundtrip() {
        let s = Scalar::from_bool(true);
        assert!(s.is_valid());
        assert_eq!(s.type_id(), TypeId::BOOL8);
        assert_eq!(s.as_bool(), Some(true));
    }

    #[test]
    fn scalar_null() {
        let s = Scalar::null_i32();
        assert!(!s.is_valid());
        assert_eq!(s.type_id(), TypeId::INT32);
        assert_eq!(s.as_i32(), None);
    }

    #[test]
    fn scalar_type_mismatch_returns_none() {
        let s = Scalar::from_i32(42);
        assert_eq!(s.as_i64(), None);
        assert_eq!(s.as_f64(), None);
        assert_eq!(s.as_bool(), None);
    }

    #[test]
    fn scalar_string() {
        let s = Scalar::from_string("hello");
        assert!(s.is_valid());
        assert_eq!(s.type_id(), TypeId::STRING);
    }
}
