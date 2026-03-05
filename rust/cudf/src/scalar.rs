// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use cxx::UniquePtr;

use crate::data_type::TypeId;

/// A scalar value that can be sent to the GPU.
///
/// This is a pure Rust enum; the FFI scalar is created on demand via
/// [`scalar_to_ffi`] when calling into libcudf.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    TimestampSeconds(i64),
    TimestampMilliseconds(i64),
    TimestampMicroseconds(i64),
    TimestampNanoseconds(i64),
    DurationSeconds(i64),
    DurationMilliseconds(i64),
    DurationMicroseconds(i64),
    DurationNanoseconds(i64),
    Null(TypeId),
}

impl Scalar {
    /// Creates a valid INT32 scalar.
    pub fn from_i32(value: i32) -> Self {
        Scalar::Int32(value)
    }

    /// Creates a null INT32 scalar.
    pub fn null_i32() -> Self {
        Scalar::Null(TypeId::INT32)
    }

    /// Creates a valid INT64 scalar.
    pub fn from_i64(value: i64) -> Self {
        Scalar::Int64(value)
    }

    /// Creates a null INT64 scalar.
    pub fn null_i64() -> Self {
        Scalar::Null(TypeId::INT64)
    }

    /// Creates a valid FLOAT32 scalar.
    pub fn from_f32(value: f32) -> Self {
        Scalar::Float32(value)
    }

    /// Creates a null FLOAT32 scalar.
    pub fn null_f32() -> Self {
        Scalar::Null(TypeId::FLOAT32)
    }

    /// Creates a valid FLOAT64 scalar.
    pub fn from_f64(value: f64) -> Self {
        Scalar::Float64(value)
    }

    /// Creates a null FLOAT64 scalar.
    pub fn null_f64() -> Self {
        Scalar::Null(TypeId::FLOAT64)
    }

    /// Creates a valid BOOL8 scalar.
    pub fn from_bool(value: bool) -> Self {
        Scalar::Bool(value)
    }

    /// Creates a null BOOL8 scalar.
    pub fn null_bool() -> Self {
        Scalar::Null(TypeId::BOOL8)
    }

    /// Creates a valid STRING scalar.
    pub fn from_string(value: &str) -> Self {
        Scalar::String(value.to_owned())
    }

    /// Returns `true` if the scalar holds a valid (non-null) value.
    pub fn is_valid(&self) -> bool {
        !matches!(self, Scalar::Null(_))
    }

    /// Returns the type identifier of the scalar.
    pub fn type_id(&self) -> TypeId {
        match self {
            Scalar::Int8(_) => TypeId::INT8,
            Scalar::Int16(_) => TypeId::INT16,
            Scalar::Int32(_) => TypeId::INT32,
            Scalar::Int64(_) => TypeId::INT64,
            Scalar::UInt8(_) => TypeId::UINT8,
            Scalar::UInt16(_) => TypeId::UINT16,
            Scalar::UInt32(_) => TypeId::UINT32,
            Scalar::UInt64(_) => TypeId::UINT64,
            Scalar::Float32(_) => TypeId::FLOAT32,
            Scalar::Float64(_) => TypeId::FLOAT64,
            Scalar::Bool(_) => TypeId::BOOL8,
            Scalar::String(_) => TypeId::STRING,
            Scalar::TimestampSeconds(_) => TypeId::TIMESTAMP_SECONDS,
            Scalar::TimestampMilliseconds(_) => TypeId::TIMESTAMP_MILLISECONDS,
            Scalar::TimestampMicroseconds(_) => TypeId::TIMESTAMP_MICROSECONDS,
            Scalar::TimestampNanoseconds(_) => TypeId::TIMESTAMP_NANOSECONDS,
            Scalar::DurationSeconds(_) => TypeId::DURATION_SECONDS,
            Scalar::DurationMilliseconds(_) => TypeId::DURATION_MILLISECONDS,
            Scalar::DurationMicroseconds(_) => TypeId::DURATION_MICROSECONDS,
            Scalar::DurationNanoseconds(_) => TypeId::DURATION_NANOSECONDS,
            Scalar::Null(tid) => *tid,
        }
    }

    /// Extracts the value as `i32`, or `None` if the type doesn't match.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Scalar::Int32(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `i64`, or `None` if the type doesn't match.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Scalar::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `f32`, or `None` if the type doesn't match.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Scalar::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `f64`, or `None` if the type doesn't match.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Scalar::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `bool`, or `None` if the type doesn't match.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Scalar::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Converts a Rust `Scalar` enum into an FFI scalar for libcudf calls.
pub(crate) fn scalar_to_ffi(s: &Scalar) -> UniquePtr<cudf_sys::ffi::Scalar> {
    match s {
        Scalar::Int8(v) => cudf_sys::ffi::make_int32_scalar(*v as i32, true),
        Scalar::Int16(v) => cudf_sys::ffi::make_int32_scalar(*v as i32, true),
        Scalar::Int32(v) => cudf_sys::ffi::make_int32_scalar(*v, true),
        Scalar::Int64(v) => cudf_sys::ffi::make_int64_scalar(*v, true),
        Scalar::UInt8(v) => cudf_sys::ffi::make_int32_scalar(*v as i32, true),
        Scalar::UInt16(v) => cudf_sys::ffi::make_int32_scalar(*v as i32, true),
        Scalar::UInt32(v) => cudf_sys::ffi::make_int64_scalar(*v as i64, true),
        Scalar::UInt64(v) => cudf_sys::ffi::make_int64_scalar(*v as i64, true),
        Scalar::Float32(v) => cudf_sys::ffi::make_float32_scalar(*v, true),
        Scalar::Float64(v) => cudf_sys::ffi::make_float64_scalar(*v, true),
        Scalar::Bool(v) => cudf_sys::ffi::make_bool_scalar(*v, true),
        Scalar::String(v) => cudf_sys::ffi::make_string_scalar(v),
        Scalar::TimestampSeconds(v)
        | Scalar::TimestampMilliseconds(v)
        | Scalar::TimestampMicroseconds(v)
        | Scalar::TimestampNanoseconds(v)
        | Scalar::DurationSeconds(v)
        | Scalar::DurationMilliseconds(v)
        | Scalar::DurationMicroseconds(v)
        | Scalar::DurationNanoseconds(v) => cudf_sys::ffi::make_int64_scalar(*v, true),
        Scalar::Null(tid) => match *tid {
            TypeId::INT8 | TypeId::INT16 | TypeId::INT32 | TypeId::UINT8 | TypeId::UINT16 => {
                cudf_sys::ffi::make_int32_scalar(0, false)
            }
            TypeId::INT64 | TypeId::UINT32 | TypeId::UINT64 => {
                cudf_sys::ffi::make_int64_scalar(0, false)
            }
            TypeId::FLOAT32 => cudf_sys::ffi::make_float32_scalar(0.0, false),
            TypeId::FLOAT64 => cudf_sys::ffi::make_float64_scalar(0.0, false),
            TypeId::BOOL8 => cudf_sys::ffi::make_bool_scalar(false, false),
            _ => cudf_sys::ffi::make_int32_scalar(0, false),
        },
    }
}

/// Converts an FFI scalar back into a Rust `Scalar` enum.
pub(crate) fn scalar_from_ffi(ffi: &UniquePtr<cudf_sys::ffi::Scalar>) -> Scalar {
    let type_id_raw = cudf_sys::ffi::scalar_type_id(ffi);
    // Safety: type_id values from C++ are valid TypeId discriminants
    let tid: TypeId = unsafe { std::mem::transmute::<i32, TypeId>(type_id_raw) };
    let valid = cudf_sys::ffi::scalar_is_valid(ffi);

    if !valid {
        return Scalar::Null(tid);
    }

    match tid {
        TypeId::INT8 => Scalar::Int8(cudf_sys::ffi::scalar_to_i32(ffi) as i8),
        TypeId::INT16 => Scalar::Int16(cudf_sys::ffi::scalar_to_i32(ffi) as i16),
        TypeId::INT32 => Scalar::Int32(cudf_sys::ffi::scalar_to_i32(ffi)),
        TypeId::INT64 => Scalar::Int64(cudf_sys::ffi::scalar_to_i64(ffi)),
        TypeId::UINT8 => Scalar::UInt8(cudf_sys::ffi::scalar_to_i32(ffi) as u8),
        TypeId::UINT16 => Scalar::UInt16(cudf_sys::ffi::scalar_to_i32(ffi) as u16),
        TypeId::UINT32 => Scalar::UInt32(cudf_sys::ffi::scalar_to_i64(ffi) as u32),
        TypeId::UINT64 => Scalar::UInt64(cudf_sys::ffi::scalar_to_i64(ffi) as u64),
        TypeId::FLOAT32 => Scalar::Float32(cudf_sys::ffi::scalar_to_f32(ffi)),
        TypeId::FLOAT64 => Scalar::Float64(cudf_sys::ffi::scalar_to_f64(ffi)),
        TypeId::BOOL8 => Scalar::Bool(cudf_sys::ffi::scalar_to_bool(ffi)),
        TypeId::TIMESTAMP_SECONDS => Scalar::TimestampSeconds(cudf_sys::ffi::scalar_to_i64(ffi)),
        TypeId::TIMESTAMP_MILLISECONDS => {
            Scalar::TimestampMilliseconds(cudf_sys::ffi::scalar_to_i64(ffi))
        }
        TypeId::TIMESTAMP_MICROSECONDS => {
            Scalar::TimestampMicroseconds(cudf_sys::ffi::scalar_to_i64(ffi))
        }
        TypeId::TIMESTAMP_NANOSECONDS => {
            Scalar::TimestampNanoseconds(cudf_sys::ffi::scalar_to_i64(ffi))
        }
        TypeId::DURATION_SECONDS => Scalar::DurationSeconds(cudf_sys::ffi::scalar_to_i64(ffi)),
        TypeId::DURATION_MILLISECONDS => {
            Scalar::DurationMilliseconds(cudf_sys::ffi::scalar_to_i64(ffi))
        }
        TypeId::DURATION_MICROSECONDS => {
            Scalar::DurationMicroseconds(cudf_sys::ffi::scalar_to_i64(ffi))
        }
        TypeId::DURATION_NANOSECONDS => {
            Scalar::DurationNanoseconds(cudf_sys::ffi::scalar_to_i64(ffi))
        }
        _ => Scalar::Null(tid),
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
