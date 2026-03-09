// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Scalar values for GPU operations.

use cxx::UniquePtr;

use crate::data_type::TypeId;

#[doc(alias = "scalar")]
/// A scalar value that can be sent to the GPU.
///
/// `Scalar` is a pure Rust enum that represents a single typed value (or null).
/// The FFI representation is created on demand via the internal `scalar_to_ffi`
/// conversion when calling into libcudf. This keeps the Rust side cheap and
/// avoids unnecessary GPU allocations for scalar construction.
///
/// # Null scalars
///
/// A null scalar is represented by the [`Null`](Scalar::Null) variant, which
/// carries a [`TypeId`] to preserve type information. Null scalars are created
/// via methods like [`null_i32`](Scalar::null_i32), [`null_f64`](Scalar::null_f64), etc.
///
/// # Examples
///
/// ```
/// use cudf::scalar::Scalar;
/// use cudf::data_type::TypeId;
///
/// let s = Scalar::from_i32(42);
/// assert!(s.is_valid());
/// assert_eq!(s.type_id(), TypeId::INT32);
/// assert_eq!(s.as_i32(), Some(42));
///
/// let null = Scalar::null_i32();
/// assert!(!null.is_valid());
/// assert_eq!(null.as_i32(), None);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// Signed 8-bit integer.
    Int8(i8),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Signed 32-bit integer.
    Int32(i32),
    /// Signed 64-bit integer.
    Int64(i64),
    /// Unsigned 8-bit integer.
    UInt8(u8),
    /// Unsigned 16-bit integer.
    UInt16(u16),
    /// Unsigned 32-bit integer.
    UInt32(u32),
    /// Unsigned 64-bit integer.
    UInt64(u64),
    /// 32-bit floating point.
    Float32(f32),
    /// 64-bit floating point.
    Float64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Timestamp in seconds since epoch.
    TimestampSeconds(i64),
    /// Timestamp in milliseconds since epoch.
    TimestampMilliseconds(i64),
    /// Timestamp in microseconds since epoch.
    TimestampMicroseconds(i64),
    /// Timestamp in nanoseconds since epoch.
    TimestampNanoseconds(i64),
    /// Duration in seconds.
    DurationSeconds(i64),
    /// Duration in milliseconds.
    DurationMilliseconds(i64),
    /// Duration in microseconds.
    DurationMicroseconds(i64),
    /// Duration in nanoseconds.
    DurationNanoseconds(i64),
    /// Null value of the given type.
    Null(TypeId),
}

impl Scalar {
    /// Creates a valid `INT32` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudf::scalar::Scalar;
    /// let s = Scalar::from_i32(42);
    /// assert_eq!(s.as_i32(), Some(42));
    /// ```
    pub fn from_i32(value: i32) -> Self {
        Scalar::Int32(value)
    }

    /// Creates a null `INT32` scalar.
    ///
    /// The scalar has type [`TypeId::INT32`](crate::data_type::TypeId::INT32) but
    /// carries no value. [`is_valid`](Scalar::is_valid) returns `false` and
    /// [`as_i32`](Scalar::as_i32) returns `None`.
    pub fn null_i32() -> Self {
        Scalar::Null(TypeId::INT32)
    }

    /// Creates a valid `INT64` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudf::scalar::Scalar;
    /// let s = Scalar::from_i64(100_000);
    /// assert_eq!(s.as_i64(), Some(100_000));
    /// ```
    pub fn from_i64(value: i64) -> Self {
        Scalar::Int64(value)
    }

    /// Creates a null `INT64` scalar.
    ///
    /// See [`null_i32`](Scalar::null_i32) for null scalar semantics.
    pub fn null_i64() -> Self {
        Scalar::Null(TypeId::INT64)
    }

    /// Creates a valid `FLOAT32` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudf::scalar::Scalar;
    /// let s = Scalar::from_f32(3.14);
    /// assert!((s.as_f32().unwrap() - 3.14).abs() < 1e-5);
    /// ```
    pub fn from_f32(value: f32) -> Self {
        Scalar::Float32(value)
    }

    /// Creates a null `FLOAT32` scalar.
    ///
    /// See [`null_i32`](Scalar::null_i32) for null scalar semantics.
    pub fn null_f32() -> Self {
        Scalar::Null(TypeId::FLOAT32)
    }

    /// Creates a valid `FLOAT64` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudf::scalar::Scalar;
    /// let s = Scalar::from_f64(2.718);
    /// assert!((s.as_f64().unwrap() - 2.718).abs() < 1e-9);
    /// ```
    pub fn from_f64(value: f64) -> Self {
        Scalar::Float64(value)
    }

    /// Creates a null `FLOAT64` scalar.
    ///
    /// See [`null_i32`](Scalar::null_i32) for null scalar semantics.
    pub fn null_f64() -> Self {
        Scalar::Null(TypeId::FLOAT64)
    }

    /// Creates a valid `BOOL8` scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudf::scalar::Scalar;
    /// let s = Scalar::from_bool(true);
    /// assert_eq!(s.as_bool(), Some(true));
    /// ```
    pub fn from_bool(value: bool) -> Self {
        Scalar::Bool(value)
    }

    /// Creates a null `BOOL8` scalar.
    ///
    /// See [`null_i32`](Scalar::null_i32) for null scalar semantics.
    pub fn null_bool() -> Self {
        Scalar::Null(TypeId::BOOL8)
    }

    /// Creates a valid `INT8` scalar.
    pub fn from_i8(value: i8) -> Self {
        Scalar::Int8(value)
    }

    /// Creates a valid `INT16` scalar.
    pub fn from_i16(value: i16) -> Self {
        Scalar::Int16(value)
    }

    /// Creates a valid `UINT8` scalar.
    pub fn from_u8(value: u8) -> Self {
        Scalar::UInt8(value)
    }

    /// Creates a valid `UINT16` scalar.
    pub fn from_u16(value: u16) -> Self {
        Scalar::UInt16(value)
    }

    /// Creates a valid `UINT32` scalar.
    pub fn from_u32(value: u32) -> Self {
        Scalar::UInt32(value)
    }

    /// Creates a valid `UINT64` scalar.
    pub fn from_u64(value: u64) -> Self {
        Scalar::UInt64(value)
    }

    /// Creates a valid `STRING` scalar.
    ///
    /// The string is cloned into an owned `String`. For GPU operations,
    /// the string data is transferred to device memory on demand.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudf::scalar::Scalar;
    /// let s = Scalar::from_string("hello");
    /// assert_eq!(s.as_str(), Some("hello"));
    /// ```
    pub fn from_string(value: &str) -> Self {
        Scalar::String(value.to_owned())
    }

    /// Creates a valid `TIMESTAMP_SECONDS` scalar from epoch seconds.
    ///
    /// The `value` is the number of seconds since the Unix epoch
    /// (1970-01-01 00:00:00 UTC).
    pub fn from_timestamp_s(value: i64) -> Self {
        Scalar::TimestampSeconds(value)
    }

    /// Creates a valid `TIMESTAMP_MILLISECONDS` scalar.
    ///
    /// The `value` is the number of milliseconds since the Unix epoch.
    pub fn from_timestamp_ms(value: i64) -> Self {
        Scalar::TimestampMilliseconds(value)
    }

    /// Creates a valid `TIMESTAMP_MICROSECONDS` scalar.
    ///
    /// The `value` is the number of microseconds since the Unix epoch.
    pub fn from_timestamp_us(value: i64) -> Self {
        Scalar::TimestampMicroseconds(value)
    }

    /// Creates a valid `TIMESTAMP_NANOSECONDS` scalar.
    ///
    /// The `value` is the number of nanoseconds since the Unix epoch.
    pub fn from_timestamp_ns(value: i64) -> Self {
        Scalar::TimestampNanoseconds(value)
    }

    /// Creates a valid `DURATION_SECONDS` scalar.
    ///
    /// The `value` is a duration in whole seconds.
    pub fn from_duration_s(value: i64) -> Self {
        Scalar::DurationSeconds(value)
    }

    /// Creates a valid `DURATION_MILLISECONDS` scalar.
    ///
    /// The `value` is a duration in milliseconds.
    pub fn from_duration_ms(value: i64) -> Self {
        Scalar::DurationMilliseconds(value)
    }

    /// Creates a valid `DURATION_MICROSECONDS` scalar.
    ///
    /// The `value` is a duration in microseconds.
    pub fn from_duration_us(value: i64) -> Self {
        Scalar::DurationMicroseconds(value)
    }

    /// Creates a valid `DURATION_NANOSECONDS` scalar.
    ///
    /// The `value` is a duration in nanoseconds.
    pub fn from_duration_ns(value: i64) -> Self {
        Scalar::DurationNanoseconds(value)
    }

    /// Returns `true` if the scalar holds a valid (non-null) value.
    ///
    /// Returns `false` for [`Null`](Scalar::Null) variants.
    pub fn is_valid(&self) -> bool {
        !matches!(self, Scalar::Null(_))
    }

    /// Returns the [`TypeId`] of the scalar.
    ///
    /// Every scalar carries its type, including null scalars.
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
    ///
    /// Returns `None` for null scalars and for scalars of a different type
    /// (e.g. calling `as_i32()` on a `FLOAT64` scalar).
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Scalar::Int32(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `i64`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`INT64` scalars.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Scalar::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `f32`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`FLOAT32` scalars.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Scalar::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `f64`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`FLOAT64` scalars.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Scalar::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `i8`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`INT8` scalars.
    pub fn as_i8(&self) -> Option<i8> {
        match self {
            Scalar::Int8(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `i16`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`INT16` scalars.
    pub fn as_i16(&self) -> Option<i16> {
        match self {
            Scalar::Int16(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `u8`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`UINT8` scalars.
    pub fn as_u8(&self) -> Option<u8> {
        match self {
            Scalar::UInt8(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `u16`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`UINT16` scalars.
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            Scalar::UInt16(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `u32`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`UINT32` scalars.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Scalar::UInt32(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `u64`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`UINT64` scalars.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Scalar::UInt64(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `bool`, or `None` if the type doesn't match.
    ///
    /// Returns `None` for null scalars and for non-`BOOL8` scalars.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Scalar::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the value as `&str`, or `None` if not a `STRING` scalar.
    ///
    /// Returns `None` for null scalars and for non-string scalars.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Scalar::String(v) => Some(v),
            _ => None,
        }
    }

    /// Repeats a string scalar `times` times, returning a new string scalar.
    ///
    /// The scalar must be a `STRING` variant. The result is a new `STRING`
    /// scalar whose value is the original string concatenated `times` times.
    ///
    /// # Errors
    ///
    /// Returns an error if the scalar is not a string or if the libcudf call
    /// fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::scalar::Scalar;
    ///
    /// let s = Scalar::from_string("ab");
    /// let repeated = s.repeat_string(3)?;
    /// assert_eq!(repeated.as_str(), Some("ababab"));
    /// ```
    pub fn repeat_string(&self, times: i32) -> crate::error::Result<Self> {
        let ffi = scalar_to_ffi(self);
        let result = cudf_sys::strings::ffi::repeat_string_scalar(
            &ffi,
            times,
            crate::stream::Stream::default_stream().as_raw(),
        )?;
        Ok(scalar_from_ffi(&result))
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Int8(v) => write!(f, "{v}i8"),
            Scalar::Int16(v) => write!(f, "{v}i16"),
            Scalar::Int32(v) => write!(f, "{v}i32"),
            Scalar::Int64(v) => write!(f, "{v}i64"),
            Scalar::UInt8(v) => write!(f, "{v}u8"),
            Scalar::UInt16(v) => write!(f, "{v}u16"),
            Scalar::UInt32(v) => write!(f, "{v}u32"),
            Scalar::UInt64(v) => write!(f, "{v}u64"),
            Scalar::Float32(v) => write!(f, "{v}f32"),
            Scalar::Float64(v) => write!(f, "{v}f64"),
            Scalar::Bool(v) => write!(f, "{v}"),
            Scalar::String(v) => write!(f, "\"{v}\""),
            Scalar::TimestampSeconds(v) => write!(f, "{v}ts"),
            Scalar::TimestampMilliseconds(v) => write!(f, "{v}tms"),
            Scalar::TimestampMicroseconds(v) => write!(f, "{v}tus"),
            Scalar::TimestampNanoseconds(v) => write!(f, "{v}tns"),
            Scalar::DurationSeconds(v) => write!(f, "{v}ds"),
            Scalar::DurationMilliseconds(v) => write!(f, "{v}dms"),
            Scalar::DurationMicroseconds(v) => write!(f, "{v}dus"),
            Scalar::DurationNanoseconds(v) => write!(f, "{v}dns"),
            Scalar::Null(tid) => write!(f, "null({tid})"),
        }
    }
}

/// Converts a Rust `Scalar` enum into an FFI scalar for libcudf calls.
pub(crate) fn scalar_to_ffi(s: &Scalar) -> UniquePtr<cudf_sys::ffi::Scalar> {
    match s {
        Scalar::Int8(v) => cudf_sys::ffi::make_int8_scalar(*v, true),
        Scalar::Int16(v) => cudf_sys::ffi::make_int16_scalar(*v, true),
        Scalar::Int32(v) => cudf_sys::ffi::make_int32_scalar(*v, true),
        Scalar::Int64(v) => cudf_sys::ffi::make_int64_scalar(*v, true),
        Scalar::UInt8(v) => cudf_sys::ffi::make_uint8_scalar(*v, true),
        Scalar::UInt16(v) => cudf_sys::ffi::make_uint16_scalar(*v, true),
        Scalar::UInt32(v) => cudf_sys::ffi::make_uint32_scalar(*v, true),
        Scalar::UInt64(v) => cudf_sys::ffi::make_uint64_scalar(*v, true),
        Scalar::Float32(v) => cudf_sys::ffi::make_float32_scalar(*v, true),
        Scalar::Float64(v) => cudf_sys::ffi::make_float64_scalar(*v, true),
        Scalar::Bool(v) => cudf_sys::ffi::make_bool_scalar(*v, true),
        Scalar::String(v) => cudf_sys::ffi::make_string_scalar(v),
        Scalar::TimestampSeconds(v) => cudf_sys::ffi::make_timestamp_s_scalar(*v, true),
        Scalar::TimestampMilliseconds(v) => cudf_sys::ffi::make_timestamp_ms_scalar(*v, true),
        Scalar::TimestampMicroseconds(v) => cudf_sys::ffi::make_timestamp_us_scalar(*v, true),
        Scalar::TimestampNanoseconds(v) => cudf_sys::ffi::make_timestamp_ns_scalar(*v, true),
        Scalar::DurationSeconds(v) => cudf_sys::ffi::make_duration_s_scalar(*v, true),
        Scalar::DurationMilliseconds(v) => cudf_sys::ffi::make_duration_ms_scalar(*v, true),
        Scalar::DurationMicroseconds(v) => cudf_sys::ffi::make_duration_us_scalar(*v, true),
        Scalar::DurationNanoseconds(v) => cudf_sys::ffi::make_duration_ns_scalar(*v, true),
        Scalar::Null(tid) => cudf_sys::ffi::make_default_constructed_scalar(tid.repr, 0),
    }
}

/// Converts an FFI scalar back into a Rust `Scalar` enum.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // Intentional narrowing casts at the FFI boundary.
pub(crate) fn scalar_from_ffi(ffi: &UniquePtr<cudf_sys::ffi::Scalar>) -> Scalar {
    let type_id_raw = cudf_sys::ffi::scalar_type_id(ffi);
    // C++ scalars always have a valid type_id.
    let tid: TypeId = cudf_sys::type_id_from_i32(type_id_raw).unwrap_or(TypeId::EMPTY);
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

    #[test]
    fn scalar_null_i64() {
        let s = Scalar::null_i64();
        assert!(!s.is_valid());
        assert_eq!(s.type_id(), TypeId::INT64);
        assert_eq!(s.as_i64(), None);
    }

    #[test]
    fn scalar_null_f32() {
        let s = Scalar::null_f32();
        assert!(!s.is_valid());
        assert_eq!(s.type_id(), TypeId::FLOAT32);
        assert_eq!(s.as_f32(), None);
    }

    #[test]
    fn scalar_null_f64() {
        let s = Scalar::null_f64();
        assert!(!s.is_valid());
        assert_eq!(s.type_id(), TypeId::FLOAT64);
        assert_eq!(s.as_f64(), None);
    }

    #[test]
    fn scalar_null_bool() {
        let s = Scalar::null_bool();
        assert!(!s.is_valid());
        assert_eq!(s.type_id(), TypeId::BOOL8);
        assert_eq!(s.as_bool(), None);
    }

    #[test]
    fn scalar_timestamp_type_ids() {
        assert_eq!(
            Scalar::TimestampSeconds(0).type_id(),
            TypeId::TIMESTAMP_SECONDS
        );
        assert_eq!(
            Scalar::TimestampMilliseconds(0).type_id(),
            TypeId::TIMESTAMP_MILLISECONDS
        );
        assert_eq!(
            Scalar::TimestampMicroseconds(0).type_id(),
            TypeId::TIMESTAMP_MICROSECONDS
        );
        assert_eq!(
            Scalar::TimestampNanoseconds(0).type_id(),
            TypeId::TIMESTAMP_NANOSECONDS
        );
    }

    #[test]
    fn scalar_duration_type_ids() {
        assert_eq!(
            Scalar::DurationSeconds(0).type_id(),
            TypeId::DURATION_SECONDS
        );
        assert_eq!(
            Scalar::DurationMilliseconds(0).type_id(),
            TypeId::DURATION_MILLISECONDS
        );
        assert_eq!(
            Scalar::DurationMicroseconds(0).type_id(),
            TypeId::DURATION_MICROSECONDS
        );
        assert_eq!(
            Scalar::DurationNanoseconds(0).type_id(),
            TypeId::DURATION_NANOSECONDS
        );
    }

    #[test]
    fn scalar_unsigned_type_ids() {
        assert_eq!(Scalar::UInt8(0).type_id(), TypeId::UINT8);
        assert_eq!(Scalar::UInt16(0).type_id(), TypeId::UINT16);
        assert_eq!(Scalar::UInt32(0).type_id(), TypeId::UINT32);
        assert_eq!(Scalar::UInt64(0).type_id(), TypeId::UINT64);
        assert_eq!(Scalar::Int8(0).type_id(), TypeId::INT8);
        assert_eq!(Scalar::Int16(0).type_id(), TypeId::INT16);
    }

    #[test]
    fn scalar_clone_eq() {
        let s = Scalar::from_i32(42);
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    #[test]
    fn scalar_all_valid() {
        assert!(Scalar::Int8(1).is_valid());
        assert!(Scalar::Int16(1).is_valid());
        assert!(Scalar::UInt8(1).is_valid());
        assert!(Scalar::UInt16(1).is_valid());
        assert!(Scalar::UInt32(1).is_valid());
        assert!(Scalar::UInt64(1).is_valid());
        assert!(Scalar::TimestampSeconds(0).is_valid());
        assert!(Scalar::DurationSeconds(0).is_valid());
    }
}
