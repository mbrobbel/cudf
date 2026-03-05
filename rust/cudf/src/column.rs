// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use cxx::UniquePtr;

use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::{scalar_from_ffi, Scalar};
use crate::stream::Stream;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    Stream::default_stream().as_raw()
}

/// An owning GPU column.
///
/// Wraps a `cudf::column` via the CXX FFI layer. Dropping this value
/// frees the underlying GPU memory.
pub struct Column(pub(crate) UniquePtr<cudf_sys::ffi::Column>);

// GPU memory is globally accessible from any CPU thread.
unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Column {
    /// Creates a column by repeating a scalar value `count` times.
    pub fn from_scalar(scalar: &Scalar, count: usize) -> Self {
        let ffi = crate::scalar::scalar_to_ffi(scalar);
        Self(cudf_sys::ffi::make_column_from_scalar(
            &ffi,
            count as i32,
            ds(),
        ))
    }

    /// Creates an empty column of the given type.
    pub fn empty(type_id: TypeId) -> Self {
        Self(cudf_sys::ffi::make_empty_column_by_type(type_id.repr))
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        cudf_sys::ffi::column_size(&self.0) as usize
    }

    /// Returns `true` if the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the count of null elements.
    pub fn null_count(&self) -> usize {
        cudf_sys::ffi::column_null_count(&self.0) as usize
    }

    /// Returns `true` if the column contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_has_nulls(&self.0)
    }

    /// Returns the type identifier of the column.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_type_id(&self.0);
        // Safety: type_id values from C++ are valid TypeId discriminants
        unsafe { std::mem::transmute::<i32, TypeId>(id) }
    }

    /// Returns an immutable view of the column.
    pub fn view(&self) -> ColumnView<'_> {
        ColumnView(cudf_sys::ffi::column_view_of(&self.0))
    }

    /// Copies the column data to host as `Vec<i16>`.
    pub fn to_vec_i16(&self) -> Vec<i16> {
        cudf_sys::ffi::column_to_host_i16(&self.0, ds())
    }

    /// Copies the column data to host as `Vec<i32>`.
    pub fn to_vec_i32(&self) -> Vec<i32> {
        cudf_sys::ffi::column_to_host_i32(&self.0, ds())
    }

    /// Copies the column data to host as `Vec<i64>`.
    pub fn to_vec_i64(&self) -> Vec<i64> {
        cudf_sys::ffi::column_to_host_i64(&self.0, ds())
    }

    /// Copies the column data to host as `Vec<f32>`.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        cudf_sys::ffi::column_to_host_f32(&self.0, ds())
    }

    /// Copies the column data to host as `Vec<f64>`.
    pub fn to_vec_f64(&self) -> Vec<f64> {
        cudf_sys::ffi::column_to_host_f64(&self.0, ds())
    }

    /// Copies the column data to host as `Vec<bool>`.
    pub fn to_vec_bool(&self) -> Vec<bool> {
        cudf_sys::ffi::column_to_host_bool(&self.0, ds())
    }

    /// Returns per-element validity as a host vector of bools.
    pub fn null_mask_to_host(&self) -> Vec<bool> {
        cudf_sys::ffi::column_null_mask_to_host(&self.0, ds())
    }

    /// Copies string column data to a host vector of strings.
    pub fn to_vec_string(&self) -> Vec<String> {
        cudf_sys::ffi::column_to_host_strings(&self.0, ds())
    }

    /// Creates an INT32 column from a host slice.
    pub fn from_slice_i32(data: &[i32]) -> Self {
        Self(cudf_sys::ffi::make_column_from_host_i32(data, ds()))
    }

    /// Creates an INT64 column from a host slice.
    pub fn from_slice_i64(data: &[i64]) -> Self {
        Self(cudf_sys::ffi::make_column_from_host_i64(data, ds()))
    }

    /// Creates a FLOAT64 column from a host slice.
    pub fn from_slice_f64(data: &[f64]) -> Self {
        Self(cudf_sys::ffi::make_column_from_host_f64(data, ds()))
    }

    /// Creates a BOOL8 column from a host slice.
    pub fn from_slice_bool(data: &[bool]) -> Self {
        Self(cudf_sys::ffi::make_column_from_host_bool(data, ds()))
    }

    /// Creates a TIMESTAMP_SECONDS column from epoch-second values.
    pub fn from_timestamps_s(data: &[i64]) -> Self {
        Self(cudf_sys::ffi::make_column_from_host_timestamp_s(data, ds()))
    }

    /// Creates a string column from a slice of strings.
    pub fn from_strings(values: &[&str]) -> Self {
        let strings: Vec<String> = values.iter().map(|s| s.to_string()).collect();
        Self(cudf_sys::ffi::make_string_column(strings, ds()))
    }
}

/// A non-owning, immutable view of a GPU column.
///
/// The lifetime parameter ties this view to the owning [`Table`](crate::Table).
pub struct ColumnView<'a>(pub(crate) &'a cudf_sys::ffi::column_view);

impl ColumnView<'_> {
    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        cudf_sys::ffi::column_view_size(self.0) as usize
    }

    /// Returns `true` if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the count of null elements.
    pub fn null_count(&self) -> usize {
        cudf_sys::ffi::column_view_null_count(self.0) as usize
    }

    /// Returns `true` if the view contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_view_has_nulls(self.0)
    }

    /// Returns the offset of the view into the underlying data.
    pub fn offset(&self) -> usize {
        cudf_sys::ffi::column_view_offset(self.0) as usize
    }

    /// Returns the type identifier of the column.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_view_type_id(self.0);
        // Safety: type_id values from C++ are valid TypeId discriminants
        unsafe { std::mem::transmute::<i32, TypeId>(id) }
    }

    // -- Unary ops --

    /// Casts the column to a different type.
    pub fn cast(&self, target: TypeId) -> Result<Column> {
        self.cast_on(target, Stream::default_stream())
    }

    /// Cast on a custom CUDA stream.
    pub fn cast_on(&self, target: TypeId, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_cast(self.0, target.repr, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates a null value.
    pub fn is_null(&self) -> Result<Column> {
        self.is_null_on(Stream::default_stream())
    }

    /// is_null on a custom CUDA stream.
    pub fn is_null_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_null(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates a valid value.
    pub fn is_valid(&self) -> Result<Column> {
        self.is_valid_on(Stream::default_stream())
    }

    /// is_valid on a custom CUDA stream.
    pub fn is_valid_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_valid(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates NaN.
    pub fn is_nan(&self) -> Result<Column> {
        self.is_nan_on(Stream::default_stream())
    }

    /// is_nan on a custom CUDA stream.
    pub fn is_nan_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_nan(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Negates every element.
    pub fn negate(&self) -> Result<Column> {
        self.negate_on(Stream::default_stream())
    }

    /// Negate on a custom CUDA stream.
    pub fn negate_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_negate(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Returns the absolute value of every element.
    pub fn abs(&self) -> Result<Column> {
        self.abs_on(Stream::default_stream())
    }

    /// Abs on a custom CUDA stream.
    pub fn abs_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_abs(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    // -- Reductions --

    /// Computes the sum of all elements.
    pub fn sum(&self, output_type: TypeId) -> Result<Scalar> {
        self.sum_on(output_type, Stream::default_stream())
    }

    /// Sum on a custom CUDA stream.
    pub fn sum_on(&self, output_type: TypeId, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_sum(self.0, output_type.repr, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Computes the minimum value.
    pub fn min(&self, output_type: TypeId) -> Result<Scalar> {
        self.min_on(output_type, Stream::default_stream())
    }

    /// Min on a custom CUDA stream.
    pub fn min_on(&self, output_type: TypeId, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_min(self.0, output_type.repr, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Computes the maximum value.
    pub fn max(&self, output_type: TypeId) -> Result<Scalar> {
        self.max_on(output_type, Stream::default_stream())
    }

    /// Max on a custom CUDA stream.
    pub fn max_on(&self, output_type: TypeId, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_max(self.0, output_type.repr, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Computes the product of all elements.
    pub fn product(&self, output_type: TypeId) -> Result<Scalar> {
        self.product_on(output_type, Stream::default_stream())
    }

    /// Product on a custom CUDA stream.
    pub fn product_on(&self, output_type: TypeId, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_product(self.0, output_type.repr, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Returns true if any element is non-zero.
    pub fn any(&self) -> Result<Scalar> {
        self.any_on(Stream::default_stream())
    }

    /// Any on a custom CUDA stream.
    pub fn any_on(&self, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_any(self.0, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Returns true if all elements are non-zero.
    pub fn all(&self) -> Result<Scalar> {
        self.all_on(Stream::default_stream())
    }

    /// All on a custom CUDA stream.
    pub fn all_on(&self, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_all(self.0, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    // -- Quantile --

    /// Computes quantiles of the column.
    pub fn quantile(&self, quantiles: &[f64]) -> Result<Column> {
        self.quantile_on(quantiles, Stream::default_stream())
    }

    /// Quantile on a custom CUDA stream.
    pub fn quantile_on(&self, quantiles: &[f64], stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::quantile_column(
            self.0,
            quantiles,
            crate::quantile::Interpolation::LINEAR.repr,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Computes quantiles with a specified interpolation method.
    pub fn quantile_with_interp(
        &self,
        quantiles: &[f64],
        interp: crate::quantile::Interpolation,
    ) -> Result<Column> {
        self.quantile_with_interp_on(quantiles, interp, Stream::default_stream())
    }

    /// Quantile with interpolation on a custom CUDA stream.
    pub fn quantile_with_interp_on(
        &self,
        quantiles: &[f64],
        interp: crate::quantile::Interpolation,
        stream: Stream,
    ) -> Result<Column> {
        let c =
            cudf_sys::ffi::quantile_column(self.0, quantiles, interp.repr, stream.as_raw())?;
        Ok(Column(c))
    }

    // -- Copying --

    /// Creates an empty column with the same type.
    pub fn empty_like(&self) -> Column {
        Column(cudf_sys::ffi::empty_like_column(self.0))
    }

    // -- Transform --

    /// Converts NaN values to null in a floating-point column.
    pub fn nans_to_nulls(&self) -> Result<Column> {
        self.nans_to_nulls_on(Stream::default_stream())
    }

    /// nans_to_nulls on a custom CUDA stream.
    pub fn nans_to_nulls_on(&self, stream: Stream) -> Result<Column> {
        let result = cudf_sys::ffi::nans_to_nulls(self.0, stream.as_raw())?;
        Ok(Column(result))
    }

    // -- Binary ops (convenience) --

    /// Element-wise addition with another column.
    pub fn add(&self, rhs: &ColumnView<'_>, output_type: TypeId) -> Result<Column> {
        self.add_on(rhs, output_type, Stream::default_stream())
    }

    /// Add on a custom CUDA stream.
    pub fn add_on(
        &self,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::ADD,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &ColumnView<'_>, output_type: TypeId) -> Result<Column> {
        self.sub_on(rhs, output_type, Stream::default_stream())
    }

    /// Sub on a custom CUDA stream.
    pub fn sub_on(
        &self,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::SUB,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &ColumnView<'_>, output_type: TypeId) -> Result<Column> {
        self.mul_on(rhs, output_type, Stream::default_stream())
    }

    /// Mul on a custom CUDA stream.
    pub fn mul_on(
        &self,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::MUL,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise division.
    pub fn div(&self, rhs: &ColumnView<'_>, output_type: TypeId) -> Result<Column> {
        self.div_on(rhs, output_type, Stream::default_stream())
    }

    /// Div on a custom CUDA stream.
    pub fn div_on(
        &self,
        rhs: &ColumnView<'_>,
        output_type: TypeId,
        stream: Stream,
    ) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::DIV,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise equality comparison.
    pub fn eq(&self, rhs: &ColumnView<'_>) -> Result<Column> {
        self.eq_on(rhs, Stream::default_stream())
    }

    /// Eq on a custom CUDA stream.
    pub fn eq_on(&self, rhs: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::EQUAL,
            TypeId::BOOL8.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise not-equal comparison.
    pub fn ne(&self, rhs: &ColumnView<'_>) -> Result<Column> {
        self.ne_on(rhs, Stream::default_stream())
    }

    /// Ne on a custom CUDA stream.
    pub fn ne_on(&self, rhs: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::NOT_EQUAL,
            TypeId::BOOL8.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise less-than comparison.
    pub fn lt(&self, rhs: &ColumnView<'_>) -> Result<Column> {
        self.lt_on(rhs, Stream::default_stream())
    }

    /// Lt on a custom CUDA stream.
    pub fn lt_on(&self, rhs: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::LESS,
            TypeId::BOOL8.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise greater-than comparison.
    pub fn gt(&self, rhs: &ColumnView<'_>) -> Result<Column> {
        self.gt_on(rhs, Stream::default_stream())
    }

    /// Gt on a custom CUDA stream.
    pub fn gt_on(&self, rhs: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::GREATER,
            TypeId::BOOL8.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise less-than-or-equal comparison.
    pub fn le(&self, rhs: &ColumnView<'_>) -> Result<Column> {
        self.le_on(rhs, Stream::default_stream())
    }

    /// Le on a custom CUDA stream.
    pub fn le_on(&self, rhs: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::LESS_EQUAL,
            TypeId::BOOL8.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    /// Element-wise greater-than-or-equal comparison.
    pub fn ge(&self, rhs: &ColumnView<'_>) -> Result<Column> {
        self.ge_on(rhs, Stream::default_stream())
    }

    /// Ge on a custom CUDA stream.
    pub fn ge_on(&self, rhs: &ColumnView<'_>, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::binary_operation_columns(
            self.0,
            rhs.0,
            crate::ops::BinaryOperator::GREATER_EQUAL,
            TypeId::BOOL8.repr,
            stream.as_raw(),
        )?;
        Ok(Column(col))
    }

    // -- Search --

    /// Checks if a scalar value exists in this column.
    pub fn contains_scalar(&self, needle: &Scalar) -> Result<bool> {
        self.contains_scalar_on(needle, Stream::default_stream())
    }

    /// contains_scalar on a custom CUDA stream.
    pub fn contains_scalar_on(&self, needle: &Scalar, stream: Stream) -> Result<bool> {
        let ffi = crate::scalar::scalar_to_ffi(needle);
        cudf_sys::ffi::contains_scalar(self.0, &ffi, stream.as_raw()).map_err(Into::into)
    }

    /// Checks which values from `needles` exist in this column.
    pub fn contains_column(&self, needles: &ColumnView<'_>) -> Result<Column> {
        self.contains_column_on(needles, Stream::default_stream())
    }

    /// contains_column on a custom CUDA stream.
    pub fn contains_column_on(
        &self,
        needles: &ColumnView<'_>,
        stream: Stream,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::contains_column(self.0, needles.0, stream.as_raw())?;
        Ok(Column(c))
    }

    // -- Fill --

    /// Fills the range [begin, end) with a scalar value (out-of-place).
    pub fn fill(&self, begin: usize, end: usize, value: &Scalar) -> Result<Column> {
        self.fill_on(begin, end, value, Stream::default_stream())
    }

    /// Fill on a custom CUDA stream.
    pub fn fill_on(
        &self,
        begin: usize,
        end: usize,
        value: &Scalar,
        stream: Stream,
    ) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(value);
        let c = cudf_sys::ffi::fill_column(
            self.0,
            begin as i32,
            end as i32,
            &ffi,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_from_scalar_i32() {
        let s = Scalar::from_i32(7);
        let col = Column::from_scalar(&s, 5);
        assert_eq!(col.len(), 5);
        assert_eq!(col.type_id(), TypeId::INT32);
        assert!(!col.has_nulls());
        assert_eq!(col.to_vec_i32(), vec![7, 7, 7, 7, 7]);
    }

    #[test]
    fn column_from_scalar_f64() {
        let s = Scalar::from_f64(2.5);
        let col = Column::from_scalar(&s, 3);
        assert_eq!(col.len(), 3);
        assert_eq!(col.to_vec_f64(), vec![2.5, 2.5, 2.5]);
    }

    #[test]
    fn column_from_scalar_bool() {
        let s = Scalar::from_bool(true);
        let col = Column::from_scalar(&s, 4);
        assert_eq!(col.len(), 4);
        assert_eq!(col.to_vec_bool(), vec![true, true, true, true]);
    }

    #[test]
    fn empty_column() {
        let col = Column::empty(TypeId::INT32);
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
        assert_eq!(col.type_id(), TypeId::INT32);
    }

    #[test]
    fn column_view_matches() {
        let s = Scalar::from_i32(10);
        let col = Column::from_scalar(&s, 3);
        let view = col.view();
        assert_eq!(view.len(), 3);
        assert_eq!(view.type_id(), TypeId::INT32);
        assert!(!view.has_nulls());
    }

    #[test]
    fn column_null_mask() {
        let s = Scalar::from_i32(42);
        let col = Column::from_scalar(&s, 3);
        let mask = col.null_mask_to_host();
        assert_eq!(mask, vec![true, true, true]);
    }
}
