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

/// Null mask allocation state for column factories.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskState {
    /// No null mask allocated.
    Unallocated = 0,
    /// Null mask allocated but uninitialized.
    Uninitialized = 1,
    /// All elements valid (no nulls).
    AllValid = 2,
    /// All elements null.
    AllNull = 3,
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

    /// Creates an uninitialized fixed-width column with `num_rows` elements.
    ///
    /// `mask_state` controls null mask allocation:
    /// - `MaskState::Unallocated` — no null mask
    /// - `MaskState::AllValid` — all valid (no nulls)
    /// - `MaskState::AllNull` — all null
    pub fn fixed_width(
        type_id: TypeId,
        scale: i32,
        num_rows: usize,
        mask_state: MaskState,
    ) -> Result<Self> {
        let c = cudf_sys::ffi::make_fixed_width_column(
            type_id.repr,
            scale,
            num_rows as i32,
            mask_state as i32,
            ds(),
        )?;
        Ok(Self(c))
    }

    /// Creates an empty lists column with the given child element type.
    pub fn empty_lists(child_type: TypeId) -> Result<Self> {
        let c = cudf_sys::ffi::make_empty_lists_column(child_type.repr, ds())?;
        Ok(Self(c))
    }

    /// Creates a dictionary column filled with a single scalar value.
    pub fn dictionary_from_scalar(scalar: &Scalar, count: usize) -> Result<Self> {
        let ffi = crate::scalar::scalar_to_ffi(scalar);
        let c = cudf_sys::ffi::make_dictionary_from_scalar(&ffi, count as i32, ds())?;
        Ok(Self(c))
    }

    /// Creates a LIST column from offsets and child columns (no null mask).
    ///
    /// `offsets` must be an INT32 column of length `num_rows + 1`.
    /// `child` contains the flattened list elements.
    pub fn from_lists(num_rows: usize, offsets: Column, child: Column) -> Result<Self> {
        let c = cudf_sys::ffi::make_lists_column(
            num_rows as i32,
            offsets.0,
            child.0,
            ds(),
        )?;
        Ok(Self(c))
    }

    /// Creates a STRUCT column from child columns (no null mask).
    pub fn from_structs(num_rows: usize, children: Vec<Column>) -> Result<Self> {
        let mut builder = cudf_sys::ffi::new_struct_column_builder();
        for child in children {
            cudf_sys::ffi::struct_column_builder_add(builder.pin_mut(), child.0);
        }
        let c = cudf_sys::ffi::struct_column_builder_build(
            builder.pin_mut(),
            num_rows as i32,
            ds(),
        )?;
        Ok(Self(c))
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

    // -- In-place mutations --

    /// Fills the range `[begin, end)` with a scalar value in-place.
    pub fn fill_in_place(&mut self, begin: usize, end: usize, value: &Scalar) -> Result<()> {
        let ffi = crate::scalar::scalar_to_ffi(value);
        cudf_sys::ffi::fill_in_place(
            self.0.pin_mut(),
            begin as i32,
            end as i32,
            &ffi,
            ds(),
        )?;
        Ok(())
    }

    /// Copies elements from `source[source_begin..source_end]` into `self`
    /// starting at `dest_begin`, in-place.
    pub fn copy_range_in_place(
        &mut self,
        source: &ColumnView<'_>,
        source_begin: usize,
        source_end: usize,
        dest_begin: usize,
    ) -> Result<()> {
        cudf_sys::ffi::copy_range_in_place(
            self.0.pin_mut(),
            source.0,
            source_begin as i32,
            source_end as i32,
            dest_begin as i32,
            ds(),
        )?;
        Ok(())
    }

    /// Converts this column's null mask to a BOOL8 column
    /// (true = valid, false = null). If no mask is present, returns all-true.
    pub fn null_mask_to_bools(&self) -> Result<Column> {
        let v = self.view();
        let c = cudf_sys::ffi::null_mask_to_bools(v.0, ds())?;
        Ok(Column(c))
    }

    /// Sets this column's null mask from a BOOL8 column
    /// (true = valid, false = null).
    pub fn set_null_mask_from_bools(&mut self, bools: &ColumnView<'_>) -> Result<()> {
        cudf_sys::ffi::set_null_mask_from_bools(self.0.pin_mut(), bools.0, ds())?;
        Ok(())
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

    // -- Child column access --

    /// Returns the number of child columns (e.g. struct fields, list offsets/child).
    pub fn num_children(&self) -> usize {
        cudf_sys::ffi::column_view_num_children(self.0) as usize
    }

    /// Deep-copies a child column by index.
    ///
    /// For STRUCT columns: index 0..N-1 are the struct fields.
    /// For LIST columns: index 0 is offsets, index 1 is child values.
    /// For DICTIONARY columns: index 0 is indices, index 1 is keys.
    pub fn child(&self, index: usize) -> Result<Column> {
        let c = cudf_sys::ffi::column_view_child_copy(self.0, index as i32, ds())?;
        Ok(Column(c))
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

    /// Counts the number of distinct values in this column.
    ///
    /// - `include_nulls`: whether null values count as a distinct value.
    /// - `nan_is_null`: whether NaN values are treated as null.
    pub fn distinct_count(&self, include_nulls: bool, nan_is_null: bool) -> usize {
        let null_policy = if include_nulls { 1 } else { 0 }; // INCLUDE=1, EXCLUDE=0
        cudf_sys::ffi::distinct_count_column(
            self.0,
            null_policy,
            nan_is_null,
            ds(),
        ) as usize
    }

    /// Counts consecutive unique values in the column.
    pub fn unique_count(&self, include_nulls: bool, nan_is_null: bool) -> usize {
        let null_policy = if include_nulls { 1 } else { 0 };
        cudf_sys::ffi::unique_count_column(
            self.0,
            null_policy,
            nan_is_null,
            ds(),
        ) as usize
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

    // -- Generic unary operation --

    /// Applies a unary operation to this column.
    pub fn unary_op(&self, op: crate::ops::UnaryOperator) -> Result<Column> {
        self.unary_op_on(op, Stream::default_stream())
    }

    /// unary_op on a custom CUDA stream.
    pub fn unary_op_on(&self, op: crate::ops::UnaryOperator, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_operation(self.0, op.repr, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Returns a BOOL8 column where `true` indicates a non-NaN value.
    pub fn is_not_nan(&self) -> Result<Column> {
        self.is_not_nan_on(Stream::default_stream())
    }

    /// is_not_nan on a custom CUDA stream.
    pub fn is_not_nan_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::unary_is_not_nan(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    // -- Math convenience methods --

    /// Computes the sine of each element.
    pub fn sin(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::SIN) }
    /// Computes the cosine of each element.
    pub fn cos(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::COS) }
    /// Computes the tangent of each element.
    pub fn tan(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::TAN) }
    /// Computes the arcsine of each element.
    pub fn arcsin(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::ARCSIN) }
    /// Computes the arccosine of each element.
    pub fn arccos(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::ARCCOS) }
    /// Computes the arctangent of each element.
    pub fn arctan(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::ARCTAN) }
    /// Computes the hyperbolic sine of each element.
    pub fn sinh(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::SINH) }
    /// Computes the hyperbolic cosine of each element.
    pub fn cosh(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::COSH) }
    /// Computes the hyperbolic tangent of each element.
    pub fn tanh(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::TANH) }
    /// Computes the inverse hyperbolic sine of each element.
    pub fn arcsinh(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::ARCSINH) }
    /// Computes the inverse hyperbolic cosine of each element.
    pub fn arccosh(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::ARCCOSH) }
    /// Computes the inverse hyperbolic tangent of each element.
    pub fn arctanh(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::ARCTANH) }
    /// Computes e^x for each element.
    pub fn exp(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::EXP) }
    /// Computes the natural log of each element.
    pub fn log(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::LOG) }
    /// Computes the square root of each element.
    pub fn sqrt(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::SQRT) }
    /// Computes the cube root of each element.
    pub fn cbrt(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::CBRT) }
    /// Computes the ceiling of each element.
    pub fn ceil(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::CEIL) }
    /// Computes the floor of each element.
    pub fn floor(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::FLOOR) }
    /// Rounds each element to the nearest integer (round half to even).
    pub fn rint(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::RINT) }
    /// Bitwise inversion of each element.
    pub fn bit_invert(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::BIT_INVERT) }
    /// Logical NOT of each element.
    pub fn logical_not(&self) -> Result<Column> { self.unary_op(crate::ops::UnaryOperator::NOT) }

    // -- Round --

    /// Rounds column values to the given number of decimal places.
    pub fn round(&self, decimal_places: i32) -> Result<Column> {
        self.round_on(decimal_places, Stream::default_stream())
    }

    /// round on a custom CUDA stream.
    pub fn round_on(&self, decimal_places: i32, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::round_column(self.0, decimal_places, 0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Rounds with a specific rounding method.
    pub fn round_with_method(
        &self,
        decimal_places: i32,
        method: crate::ops::RoundingMethod,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::round_column(
            self.0,
            decimal_places,
            method.repr,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    // -- Reductions (new) --

    /// Computes the mean of all elements.
    pub fn mean(&self, output_type: TypeId) -> Result<Scalar> {
        self.mean_on(output_type, Stream::default_stream())
    }

    /// Mean on a custom CUDA stream.
    pub fn mean_on(&self, output_type: TypeId, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_mean(self.0, output_type.repr, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Computes the standard deviation.
    pub fn std_dev(&self, output_type: TypeId, ddof: i32) -> Result<Scalar> {
        self.std_dev_on(output_type, ddof, Stream::default_stream())
    }

    /// std_dev on a custom CUDA stream.
    pub fn std_dev_on(&self, output_type: TypeId, ddof: i32, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_std(self.0, output_type.repr, ddof, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Computes the variance.
    pub fn variance(&self, output_type: TypeId, ddof: i32) -> Result<Scalar> {
        self.variance_on(output_type, ddof, Stream::default_stream())
    }

    /// variance on a custom CUDA stream.
    pub fn variance_on(&self, output_type: TypeId, ddof: i32, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_var(self.0, output_type.repr, ddof, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Computes the median.
    pub fn median(&self, output_type: TypeId) -> Result<Scalar> {
        self.median_on(output_type, Stream::default_stream())
    }

    /// median on a custom CUDA stream.
    pub fn median_on(&self, output_type: TypeId, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_median(self.0, output_type.repr, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Counts the number of unique elements.
    pub fn nunique(&self) -> Result<Scalar> {
        self.nunique_on(Stream::default_stream())
    }

    /// nunique on a custom CUDA stream.
    pub fn nunique_on(&self, stream: Stream) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_nunique(self.0, 0, stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Returns the minimum via minmax.
    pub fn minmax_min(&self) -> Result<Scalar> {
        let s = cudf_sys::ffi::minmax_min(self.0, Stream::default_stream().as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Returns the maximum via minmax.
    pub fn minmax_max(&self) -> Result<Scalar> {
        let s = cudf_sys::ffi::minmax_max(self.0, Stream::default_stream().as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    // -- Generic reduce --

    /// Reduces the column using any aggregation kind.
    ///
    /// `ddof` is only used for STD and VAR aggregations.
    pub fn reduce(
        &self,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
    ) -> Result<Scalar> {
        self.reduce_on(agg, output_type, ddof, Stream::default_stream())
    }

    /// Generic reduce on a custom CUDA stream.
    pub fn reduce_on(
        &self,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        stream: Stream,
    ) -> Result<Scalar> {
        let s = cudf_sys::ffi::reduce_generic(
            self.0,
            agg.repr,
            ddof,
            output_type.repr,
            stream.as_raw(),
        )?;
        Ok(crate::scalar::scalar_from_ffi(&s))
    }

    // -- Scan --

    /// Computes a prefix scan (cumulative operation).
    pub fn scan(&self, agg_kind: crate::groupby::AggregationKind, inclusive: bool) -> Result<Column> {
        self.scan_on(agg_kind, inclusive, Stream::default_stream())
    }

    /// scan on a custom CUDA stream.
    pub fn scan_on(
        &self,
        agg_kind: crate::groupby::AggregationKind,
        inclusive: bool,
        stream: Stream,
    ) -> Result<Column> {
        let scan_type = if inclusive { 0 } else { 1 };
        let c = cudf_sys::ffi::scan_column(
            self.0,
            agg_kind.repr,
            scan_type,
            0, // null_policy: EXCLUDE
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }

    // -- Copying extras --

    /// Shifts column values by offset, filling with the given scalar.
    pub fn shift(&self, offset: i32, fill_value: &Scalar) -> Result<Column> {
        self.shift_on(offset, fill_value, Stream::default_stream())
    }

    /// shift on a custom CUDA stream.
    pub fn shift_on(&self, offset: i32, fill_value: &Scalar, stream: Stream) -> Result<Column> {
        let ffi = crate::scalar::scalar_to_ffi(fill_value);
        let c = cudf_sys::ffi::shift_column(self.0, offset, &ffi, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Returns a single element as a scalar.
    pub fn get_element(&self, index: usize) -> Result<Scalar> {
        let s = cudf_sys::ffi::get_element(self.0, index as i32, Stream::default_stream().as_raw())?;
        Ok(scalar_from_ffi(&s))
    }

    /// Reverses the elements of this column.
    pub fn reverse(&self) -> Result<Column> {
        self.reverse_on(Stream::default_stream())
    }

    /// reverse on a custom CUDA stream.
    pub fn reverse_on(&self, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::reverse_column(self.0, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Extracts a slice [begin, end) as a new owned column.
    pub fn slice(&self, begin: usize, end: usize) -> Result<Column> {
        self.slice_on(begin, end, Stream::default_stream())
    }

    /// slice on a custom CUDA stream.
    pub fn slice_on(&self, begin: usize, end: usize, stream: Stream) -> Result<Column> {
        let c = cudf_sys::ffi::slice_column(self.0, begin as i32, end as i32, stream.as_raw())?;
        Ok(Column(c))
    }

    /// Selects elements from lhs (self) or rhs based on a boolean mask.
    pub fn copy_if_else(&self, rhs: &ColumnView<'_>, mask: &ColumnView<'_>) -> Result<Column> {
        let c = cudf_sys::ffi::copy_if_else_columns(
            self.0,
            rhs.0,
            mask.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    // -- Label bins --

    /// Assigns bin labels to column values.
    pub fn label_bins(
        &self,
        left_edges: &ColumnView<'_>,
        left_inclusive: crate::labeling::Inclusive,
        right_edges: &ColumnView<'_>,
        right_inclusive: crate::labeling::Inclusive,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::label_bins_column(
            self.0,
            left_edges.0,
            left_inclusive.repr,
            right_edges.0,
            right_inclusive.repr,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
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

    // -- Sorting (column-level) --

    /// Compute rank of each element. method: 0=FIRST,1=AVERAGE,2=MIN,3=MAX,4=DENSE.
    pub fn rank(
        &self,
        method: crate::sorting::RankMethod,
        order: crate::sorting::Order,
        null_handling: crate::compaction::NullPolicy,
        null_precedence: crate::sorting::NullOrder,
        percentage: bool,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::rank_column(
            self.0,
            method as i32,
            order.repr,
            null_handling.repr,
            null_precedence.repr,
            percentage,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Top k values of the column.
    pub fn top_k(&self, k: usize, order: crate::sorting::Order) -> Result<Column> {
        let c = cudf_sys::ffi::top_k(self.0, k as i32, order.repr, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Top k indices of the column.
    pub fn top_k_order(&self, k: usize, order: crate::sorting::Order) -> Result<Column> {
        let c = cudf_sys::ffi::top_k_order(self.0, k as i32, order.repr, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Segmented top k values.
    pub fn segmented_top_k(&self, segment_offsets: &ColumnView<'_>, k: usize, order: crate::sorting::Order) -> Result<Column> {
        let c = cudf_sys::ffi::segmented_top_k(self.0, segment_offsets.0, k as i32, order.repr, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Segmented top k indices.
    pub fn segmented_top_k_order(&self, segment_offsets: &ColumnView<'_>, k: usize, order: crate::sorting::Order) -> Result<Column> {
        let c = cudf_sys::ffi::segmented_top_k_order(self.0, segment_offsets.0, k as i32, order.repr, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    // -- Copying (new) --

    /// Copy range of elements from this column into target.
    pub fn copy_range_into(
        &self,
        target: &ColumnView<'_>,
        source_begin: usize,
        source_end: usize,
        target_begin: usize,
    ) -> Result<Column> {
        let c = cudf_sys::ffi::copy_range(
            self.0,
            target.0,
            source_begin as i32,
            source_end as i32,
            target_begin as i32,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Create uninitialized column of same type and size.
    pub fn allocate_like(&self) -> Result<Column> {
        let c = cudf_sys::ffi::allocate_like_column(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Check if this column has non-empty null rows (LIST/STRING).
    pub fn has_nonempty_nulls(&self) -> Result<bool> {
        cudf_sys::ffi::has_nonempty_nulls(self.0, Stream::default_stream().as_raw()).map_err(Into::into)
    }

    /// Check if this column *may* have non-empty data in null rows.
    /// This is a fast check (no stream needed) that may return false positives.
    pub fn may_have_nonempty_nulls(&self) -> bool {
        cudf_sys::ffi::may_have_nonempty_nulls(self.0)
    }

    /// Purge non-empty null row contents.
    pub fn purge_nonempty_nulls(&self) -> Result<Column> {
        let c = cudf_sys::ffi::purge_nonempty_nulls(self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    // -- Replace (new) --

    /// Replace nulls using preceding/following policy.
    pub fn replace_nulls_policy(&self, preceding: bool) -> Result<Column> {
        let policy = if preceding { 0 } else { 1 };
        let c = cudf_sys::ffi::replace_nulls_policy(self.0, policy, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Clamp with separate replacement values for lo and hi.
    pub fn clamp_with_replace(
        &self,
        lo: &Scalar,
        lo_replace: &Scalar,
        hi: &Scalar,
        hi_replace: &Scalar,
    ) -> Result<Column> {
        let lo_ffi = crate::scalar::scalar_to_ffi(lo);
        let lo_r_ffi = crate::scalar::scalar_to_ffi(lo_replace);
        let hi_ffi = crate::scalar::scalar_to_ffi(hi);
        let hi_r_ffi = crate::scalar::scalar_to_ffi(hi_replace);
        let c = cudf_sys::ffi::clamp_column_with_replace(
            self.0, &lo_ffi, &lo_r_ffi, &hi_ffi, &hi_r_ffi,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Normalize NaNs and zeros (convert -NaN to NaN, -0.0 to 0.0).
    pub fn normalize_nans_and_zeros(&self) -> Result<Column> {
        let c = cudf_sys::ffi::normalize_nans_and_zeros(self.0, Stream::default_stream().as_raw())?;
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

    #[test]
    fn column_from_slice_i32() {
        let col = Column::from_slice_i32(&[10, 20, 30]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::INT32);
        assert_eq!(col.to_vec_i32(), vec![10, 20, 30]);
    }

    #[test]
    fn column_from_slice_i64() {
        let col = Column::from_slice_i64(&[100, 200, 300]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::INT64);
        assert_eq!(col.to_vec_i64(), vec![100, 200, 300]);
    }

    #[test]
    fn column_from_slice_f64() {
        let col = Column::from_slice_f64(&[1.5, 2.5, 3.5]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::FLOAT64);
        assert_eq!(col.to_vec_f64(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn column_from_slice_bool() {
        let col = Column::from_slice_bool(&[true, false, true]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::BOOL8);
        assert_eq!(col.to_vec_bool(), vec![true, false, true]);
    }

    #[test]
    fn column_from_strings() {
        let col = Column::from_strings(&["hello", "world"]);
        assert_eq!(col.len(), 2);
        assert_eq!(col.type_id(), TypeId::STRING);
        assert_eq!(col.to_vec_string(), vec!["hello", "world"]);
    }

    #[test]
    fn column_from_timestamps_s() {
        let col = Column::from_timestamps_s(&[1704067200, 1718443845]);
        assert_eq!(col.len(), 2);
        assert_eq!(col.type_id(), TypeId::TIMESTAMP_SECONDS);
        assert_eq!(col.to_vec_i64(), vec![1704067200, 1718443845]);
    }

    #[test]
    fn column_null_scalar() {
        let s = Scalar::null_i32();
        let col = Column::from_scalar(&s, 3);
        assert_eq!(col.len(), 3);
        assert!(col.has_nulls());
        assert_eq!(col.null_count(), 3);
    }

    #[test]
    fn column_nullable() {
        let col = Column::from_slice_i32(&[1, 2, 3]);
        // from_slice creates non-nullable columns
        assert!(!col.has_nulls());
    }

    #[test]
    fn column_view_offset() {
        let col = Column::from_slice_i32(&[1, 2, 3]);
        assert_eq!(col.view().offset(), 0);
    }

    #[test]
    fn column_to_vec_f32() {
        let s = Scalar::from_f32(1.5);
        let col = Column::from_scalar(&s, 2);
        assert_eq!(col.to_vec_f32(), vec![1.5, 1.5]);
    }
}
