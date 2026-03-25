// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU column types: owning [`Column`] and borrowed [`ColumnView`].
//!
//! A [`Column`] owns GPU memory and frees it on drop. A [`ColumnView`] borrows
//! device data without owning it, tied by lifetime to the parent [`Column`] or
//! [`Table`](crate::table::Table).
//!
//! Columns are **untyped**: the element type is tracked at runtime via
//! [`TypeId`] rather than as a Rust generic parameter. Use
//! [`Column::type_id`] to query the type, and the appropriate `to_vec_*`
//! method to copy data back to the host.
//!
//! All GPU operations follow the builder pattern and implement
//! [`GpuOp`](crate::stream::GpuOp). Call [`.call()`](crate::stream::GpuOp::call)
//! to execute, or chain [`.stream()`](crate::stream::GpuOp::stream) first to
//! run on a non-default CUDA stream.

use cxx::UniquePtr;

use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::{Scalar, scalar_from_ffi};
use crate::stream::Stream;
use crate::{i32_to_usize, usize_to_i32};

#[doc(alias = "mask_state")]
/// Controls null mask allocation when constructing columns via
/// [`Column::fixed_width`].
///
/// The null mask (also called validity mask) is a bitmask indicating
/// which elements are valid and which are null. This enum determines
/// the initial state of that bitmask when a new column is created.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskState {
    /// No null mask is allocated. The column reports zero nulls and
    /// [`Column::has_nulls`] returns `false`. Use this when you know
    /// the column will never contain nulls.
    Unallocated = 0,
    /// A null mask is allocated but its contents are uninitialized.
    /// Useful when you plan to write the mask yourself via
    /// [`Column::set_null_mask_from_bools`].
    Uninitialized = 1,
    /// All elements are marked valid (no nulls). The mask is allocated
    /// and filled with all-ones.
    AllValid = 2,
    /// All elements are marked null. The mask is allocated and filled
    /// with all-zeros. [`Column::null_count`] will equal [`Column::len`].
    AllNull = 3,
}

#[allow(clippy::as_conversions)]
impl From<MaskState> for i32 {
    fn from(state: MaskState) -> Self {
        state as Self
    }
}

// ---------------------------------------------------------------------------
// Macro-generated builders for host readback and host-to-GPU constructors
// ---------------------------------------------------------------------------

macro_rules! to_host_builder {
    ($name:ident, $elem:ty, $ffi_fn:path, $doc:expr) => {
        #[doc = $doc]
        pub struct $name<'a> {
            view: &'a cudf_sys::ffi::column_view,
            stream: Stream,
        }
        impl crate::stream::GpuOp for $name<'_> {
            type Output = Vec<$elem>;
            fn stream(mut self, stream: Stream) -> Self {
                self.stream = stream;
                self
            }
            fn call(self) -> Result<Self::Output> {
                Ok($ffi_fn(self.view, self.stream.as_raw()))
            }
        }
    };
}

to_host_builder!(
    ToVecI8,
    i8,
    cudf_sys::ffi::view_to_host_i8,
    "Builder for copying GPU data to host as `Vec<i8>`."
);
to_host_builder!(
    ToVecI16,
    i16,
    cudf_sys::ffi::view_to_host_i16,
    "Builder for copying GPU data to host as `Vec<i16>`."
);
to_host_builder!(
    ToVecI32,
    i32,
    cudf_sys::ffi::view_to_host_i32,
    "Builder for copying GPU data to host as `Vec<i32>`."
);
to_host_builder!(
    ToVecI64,
    i64,
    cudf_sys::ffi::view_to_host_i64,
    "Builder for copying GPU data to host as `Vec<i64>`."
);
to_host_builder!(
    ToVecF32,
    f32,
    cudf_sys::ffi::view_to_host_f32,
    "Builder for copying GPU data to host as `Vec<f32>`."
);
to_host_builder!(
    ToVecF64,
    f64,
    cudf_sys::ffi::view_to_host_f64,
    "Builder for copying GPU data to host as `Vec<f64>`."
);
to_host_builder!(
    ToVecU8,
    u8,
    cudf_sys::ffi::view_to_host_u8,
    "Builder for copying GPU data to host as `Vec<u8>`."
);
to_host_builder!(
    ToVecU16,
    u16,
    cudf_sys::ffi::view_to_host_u16,
    "Builder for copying GPU data to host as `Vec<u16>`."
);
to_host_builder!(
    ToVecU32,
    u32,
    cudf_sys::ffi::view_to_host_u32,
    "Builder for copying GPU data to host as `Vec<u32>`."
);
to_host_builder!(
    ToVecU64,
    u64,
    cudf_sys::ffi::view_to_host_u64,
    "Builder for copying GPU data to host as `Vec<u64>`."
);
to_host_builder!(
    ToVecBool,
    bool,
    cudf_sys::ffi::view_to_host_bool,
    "Builder for copying GPU data to host as `Vec<bool>`."
);
to_host_builder!(
    NullMaskToHost,
    bool,
    cudf_sys::ffi::view_null_mask_to_host,
    "Builder for copying the null mask to host as `Vec<bool>`."
);
to_host_builder!(
    ToVecString,
    String,
    cudf_sys::strings::ffi::view_to_host_strings,
    "Builder for copying string GPU data to host as `Vec<String>`."
);

macro_rules! from_host_builder {
    ($name:ident, $elem:ty, $ffi_fn:path, $doc:expr) => {
        #[doc = $doc]
        pub struct $name<'a> {
            data: &'a [$elem],
            stream: Stream,
        }
        impl crate::stream::GpuOp for $name<'_> {
            type Output = Column;
            fn stream(mut self, stream: Stream) -> Self {
                self.stream = stream;
                self
            }
            fn call(self) -> Result<Self::Output> {
                Ok(Column($ffi_fn(self.data, self.stream.as_raw())))
            }
        }
    };
}

from_host_builder!(
    FromSliceI8,
    i8,
    cudf_sys::ffi::make_column_from_host_i8,
    "Builder for creating an `INT8` column from a host slice."
);
from_host_builder!(
    FromSliceI16,
    i16,
    cudf_sys::ffi::make_column_from_host_i16,
    "Builder for creating an `INT16` column from a host slice."
);
from_host_builder!(
    FromSliceI32,
    i32,
    cudf_sys::ffi::make_column_from_host_i32,
    "Builder for creating an `INT32` column from a host slice."
);
from_host_builder!(
    FromSliceI64,
    i64,
    cudf_sys::ffi::make_column_from_host_i64,
    "Builder for creating an `INT64` column from a host slice."
);
from_host_builder!(
    FromSliceF32,
    f32,
    cudf_sys::ffi::make_column_from_host_f32,
    "Builder for creating a `FLOAT32` column from a host slice."
);
from_host_builder!(
    FromSliceF64,
    f64,
    cudf_sys::ffi::make_column_from_host_f64,
    "Builder for creating a `FLOAT64` column from a host slice."
);
from_host_builder!(
    FromSliceU8,
    u8,
    cudf_sys::ffi::make_column_from_host_u8,
    "Builder for creating a `UINT8` column from a host slice."
);
from_host_builder!(
    FromSliceU16,
    u16,
    cudf_sys::ffi::make_column_from_host_u16,
    "Builder for creating a `UINT16` column from a host slice."
);
from_host_builder!(
    FromSliceU32,
    u32,
    cudf_sys::ffi::make_column_from_host_u32,
    "Builder for creating a `UINT32` column from a host slice."
);
from_host_builder!(
    FromSliceU64,
    u64,
    cudf_sys::ffi::make_column_from_host_u64,
    "Builder for creating a `UINT64` column from a host slice."
);
from_host_builder!(
    FromSliceBool,
    bool,
    cudf_sys::ffi::make_column_from_host_bool,
    "Builder for creating a `BOOL8` column from a host slice."
);
from_host_builder!(
    FromTimestampsS,
    i64,
    cudf_sys::ffi::make_column_from_host_timestamp_s,
    "Builder for creating a `TIMESTAMP_SECONDS` column."
);
from_host_builder!(
    FromTimestampsMs,
    i64,
    cudf_sys::ffi::make_column_from_host_timestamp_ms,
    "Builder for creating a `TIMESTAMP_MILLISECONDS` column."
);
from_host_builder!(
    FromTimestampsUs,
    i64,
    cudf_sys::ffi::make_column_from_host_timestamp_us,
    "Builder for creating a `TIMESTAMP_MICROSECONDS` column."
);
from_host_builder!(
    FromTimestampsNs,
    i64,
    cudf_sys::ffi::make_column_from_host_timestamp_ns,
    "Builder for creating a `TIMESTAMP_NANOSECONDS` column."
);
from_host_builder!(
    FromDurationsS,
    i64,
    cudf_sys::ffi::make_column_from_host_duration_s,
    "Builder for creating a `DURATION_SECONDS` column."
);
from_host_builder!(
    FromDurationsMs,
    i64,
    cudf_sys::ffi::make_column_from_host_duration_ms,
    "Builder for creating a `DURATION_MILLISECONDS` column."
);
from_host_builder!(
    FromDurationsUs,
    i64,
    cudf_sys::ffi::make_column_from_host_duration_us,
    "Builder for creating a `DURATION_MICROSECONDS` column."
);
from_host_builder!(
    FromDurationsNs,
    i64,
    cudf_sys::ffi::make_column_from_host_duration_ns,
    "Builder for creating a `DURATION_NANOSECONDS` column."
);

// ---------------------------------------------------------------------------
// Hand-written builders for unique constructors
// ---------------------------------------------------------------------------

/// Builder for [`Column::from_scalar`].
pub struct FromScalar<'a> {
    scalar: &'a Scalar,
    count: usize,
    stream: Stream,
}
impl crate::stream::GpuOp for FromScalar<'_> {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.scalar);
        Ok(Column(cudf_sys::ffi::make_column_from_scalar(
            &ffi,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )))
    }
}

/// Builder for [`Column::fixed_width`].
pub struct FixedWidth {
    type_id: TypeId,
    scale: i32,
    num_rows: usize,
    mask_state: MaskState,
    stream: Stream,
}
impl crate::stream::GpuOp for FixedWidth {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::make_fixed_width_column(
            self.type_id.repr,
            self.scale,
            usize_to_i32(self.num_rows),
            i32::from(self.mask_state),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Column::empty_lists`].
pub struct EmptyLists {
    child_type: TypeId,
    stream: Stream,
}
impl crate::stream::GpuOp for EmptyLists {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::make_empty_lists_column(self.child_type.repr, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Column::dictionary_from_scalar`].
pub struct DictionaryFromScalar<'a> {
    scalar: &'a Scalar,
    count: usize,
    stream: Stream,
}
impl crate::stream::GpuOp for DictionaryFromScalar<'_> {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.scalar);
        let c = cudf_sys::ffi::make_dictionary_from_scalar(
            &ffi,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Column::from_lists`].
pub struct FromLists {
    num_rows: usize,
    offsets: Column,
    child: Column,
    stream: Stream,
}
impl crate::stream::GpuOp for FromLists {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::make_lists_column(
            usize_to_i32(self.num_rows),
            self.offsets.0,
            self.child.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Column::from_structs`].
pub struct FromStructs {
    num_rows: usize,
    children: Vec<Column>,
    stream: Stream,
}
impl crate::stream::GpuOp for FromStructs {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let mut builder = cudf_sys::ffi::new_struct_column_builder();
        for child in self.children {
            cudf_sys::ffi::struct_column_builder_add(builder.pin_mut(), child.0);
        }
        let c = cudf_sys::ffi::struct_column_builder_build(
            builder.pin_mut(),
            usize_to_i32(self.num_rows),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Column::from_strings`].
pub struct FromStrings<'a> {
    values: &'a [&'a str],
    stream: Stream,
}
impl crate::stream::GpuOp for FromStrings<'_> {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let strings: Vec<String> = self
            .values
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        Ok(Column(cudf_sys::strings::ffi::make_string_column(
            strings,
            self.stream.as_raw(),
        )))
    }
}

/// Builder for [`Column::fill_in_place`].
pub struct FillInPlace<'a> {
    col: &'a mut Column,
    begin: usize,
    end: usize,
    value: &'a Scalar,
    stream: Stream,
}
impl crate::stream::GpuOp for FillInPlace<'_> {
    type Output = ();
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.value);
        cudf_sys::copying::ffi::fill_in_place(
            self.col.0.pin_mut(),
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            &ffi,
            self.stream.as_raw(),
        )?;
        Ok(())
    }
}

/// Builder for [`Column::copy_range_in_place`].
pub struct CopyRangeInPlace<'a> {
    col: &'a mut Column,
    source: &'a ColumnView<'a>,
    source_begin: usize,
    source_end: usize,
    dest_begin: usize,
    stream: Stream,
}
impl crate::stream::GpuOp for CopyRangeInPlace<'_> {
    type Output = ();
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        cudf_sys::copying::ffi::copy_range_in_place(
            self.col.0.pin_mut(),
            self.source.0,
            usize_to_i32(self.source_begin),
            usize_to_i32(self.source_end),
            usize_to_i32(self.dest_begin),
            self.stream.as_raw(),
        )?;
        Ok(())
    }
}

/// Builder for [`Column::null_mask_to_bools`].
pub struct NullMaskToBools<'a> {
    col: &'a Column,
    stream: Stream,
}
impl crate::stream::GpuOp for NullMaskToBools<'_> {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let v = self.col.view();
        let c = cudf_sys::ffi::null_mask_to_bools(v.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Column::set_null_mask_from_bools`].
pub struct SetNullMaskFromBools<'a> {
    col: &'a mut Column,
    bools: &'a ColumnView<'a>,
    stream: Stream,
}
impl crate::stream::GpuOp for SetNullMaskFromBools<'_> {
    type Output = ();
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        cudf_sys::ffi::set_null_mask_from_bools(
            self.col.0.pin_mut(),
            self.bools.0,
            self.stream.as_raw(),
        )?;
        Ok(())
    }
}

/// Builder for [`Column::with_null_mask_from_bools`].
pub struct WithNullMaskFromBools<'a> {
    col: &'a Column,
    validity: &'a ColumnView<'a>,
    stream: Stream,
}
impl crate::stream::GpuOp for WithNullMaskFromBools<'_> {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::column_with_null_mask_from_bools(
            &self.col.0,
            self.validity.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::to_owned_column`].
pub struct ToOwnedColumn<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
impl crate::stream::GpuOp for ToOwnedColumn<'_> {
    type Output = Column;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::copy_column(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

#[doc(alias = "column")]
/// An owning GPU column backed by device memory.
///
/// `Column` wraps a `cudf::column` through the CXX FFI layer. It owns the
/// underlying GPU buffers (element data, null mask, and any child columns)
/// and frees them when dropped.
///
/// Columns are **untyped** -- the element type is stored as a runtime
/// [`TypeId`] rather than a Rust generic parameter. Query the type with
/// [`type_id`](Column::type_id) and use the matching `to_vec_*` method to
/// copy data to the host.
///
/// # Construction
///
/// | Method | Creates |
/// |--------|---------|
/// | [`from_scalar`](Column::from_scalar) | Repeat a scalar N times |
/// | [`from_slice_i32`](Column::from_slice_i32), etc. | Upload host data |
/// | [`from_strings`](Column::from_strings) | String column from `&[&str]` |
/// | [`fixed_width`](Column::fixed_width) | Uninitialized fixed-width column |
/// | [`empty`](Column::empty) | Zero-length column |
/// | [`from_lists`](Column::from_lists) | LIST column from offsets + child |
/// | [`from_structs`](Column::from_structs) | STRUCT column from child columns |
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::GpuOp;
///
/// // Create from host data
/// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// assert_eq!(col.len(), 3);
///
/// // Create by repeating a scalar
/// let s = Scalar::from_i32(42);
/// let repeated = Column::from_scalar(&s, 5).call()?;
/// assert_eq!(repeated.to_vec_i32().call()?, [42, 42, 42, 42, 42]);
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct Column(pub(crate) UniquePtr<cudf_sys::ffi::Column>);

impl Column {
    #[doc(alias = "make_column_from_scalar")]
    /// Creates a column by repeating `scalar` for `count` rows.
    ///
    /// The resulting column has the same [`TypeId`] as the scalar.
    /// If the scalar is null (e.g. [`Scalar::null_i32`]), every element in
    /// the column will be null.
    ///
    /// Returns a [`FromScalar`] builder -- call [`.call()`](crate::stream::GpuOp::call)
    /// to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let s = Scalar::from_i32(42);
    /// let col = Column::from_scalar(&s, 5).call()?;
    /// assert_eq!(col.len(), 5);
    /// assert_eq!(col.to_vec_i32().call()?, [42, 42, 42, 42, 42]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn from_scalar(scalar: &Scalar, count: usize) -> FromScalar<'_> {
        FromScalar {
            scalar,
            count,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_empty_column")]
    /// Creates an empty (zero-length) column of the given `type_id`.
    ///
    /// The column has no data and no null mask. This is useful as a
    /// starting point for accumulation or as a placeholder.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    ///
    /// let col = Column::empty(TypeId::INT32);
    /// assert!(col.is_empty());
    /// assert_eq!(col.type_id(), TypeId::INT32);
    /// ```
    pub fn empty(type_id: TypeId) -> Self {
        Self(cudf_sys::ffi::make_empty_column_by_type(type_id.repr))
    }

    #[doc(alias = "make_fixed_width_column")]
    /// Creates an uninitialized fixed-width column with `num_rows` elements.
    ///
    /// The element data buffer is allocated but **not initialized**; you must
    /// fill it (e.g. via [`fill_in_place`](Column::fill_in_place) or
    /// [`copy_range_in_place`](Column::copy_range_in_place)) before reading.
    ///
    /// `type_id` must be a fixed-width type (integer, float, bool, timestamp,
    /// duration, or decimal). String and list types are not supported here.
    ///
    /// `scale` is the decimal scale factor, used only for `DECIMAL32`,
    /// `DECIMAL64`, and `DECIMAL128` types. Pass `0` for all other types.
    ///
    /// `mask_state` controls the null mask -- see [`MaskState`] for details.
    ///
    /// Returns a [`FixedWidth`] builder.
    ///
    /// # Errors
    ///
    /// Returns an error if `type_id` is not a fixed-width type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::{Column, MaskState};
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::fixed_width(TypeId::FLOAT64, 0, 100, MaskState::AllValid).call()?;
    /// assert_eq!(col.len(), 100);
    /// assert!(!col.has_nulls());
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn fixed_width(
        type_id: TypeId,
        scale: i32,
        num_rows: usize,
        mask_state: MaskState,
    ) -> FixedWidth {
        FixedWidth {
            type_id,
            scale,
            num_rows,
            mask_state,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_empty_lists_column")]
    /// Creates an empty (zero-length) LIST column whose child elements have
    /// the given `child_type`.
    ///
    /// This is useful when you need a typed LIST column placeholder (e.g. as
    /// an empty accumulator for concatenation) without any rows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::empty_lists(TypeId::INT32).call()?;
    /// assert!(col.is_empty());
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn empty_lists(child_type: TypeId) -> EmptyLists {
        EmptyLists {
            child_type,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_dictionary_from_scalar")]
    /// Creates a dictionary-encoded column filled with `scalar` repeated
    /// `count` times.
    ///
    /// The result is a `DICTIONARY32` column with a single-element keys
    /// column containing `scalar`, and an indices column pointing every row
    /// to that key. Dictionary encoding is efficient when a column has many
    /// repeated values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let s = Scalar::from_i32(99);
    /// let col = Column::dictionary_from_scalar(&s, 3).call()?;
    /// assert_eq!(col.len(), 3);
    /// assert_eq!(col.type_id(), TypeId::DICTIONARY32);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn dictionary_from_scalar(scalar: &Scalar, count: usize) -> DictionaryFromScalar<'_> {
        DictionaryFromScalar {
            scalar,
            count,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_lists_column")]
    /// Creates a LIST column from `offsets` and a `child` column, with no
    /// null mask.
    ///
    /// `num_rows` is the number of list elements (rows) in the resulting
    /// column.
    ///
    /// `offsets` must be an `INT32` column of length `num_rows + 1`. Each
    /// consecutive pair `offsets[i]..offsets[i+1]` defines the slice of
    /// `child` that forms the i-th list. `offsets[0]` is typically `0` and
    /// `offsets[num_rows]` equals `child.len()`.
    ///
    /// `child` contains all list elements flattened into a single column.
    ///
    /// Both `offsets` and `child` are consumed (moved) into the new column.
    ///
    /// # Errors
    ///
    /// Returns an error if `offsets` is not `INT32` or has the wrong length.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// // Build [[10, 20], [30]]
    /// let offsets = Column::from_slice_i32(&[0, 2, 3]).call()?;
    /// let child = Column::from_slice_i32(&[10, 20, 30]).call()?;
    /// let lists = Column::from_lists(2, offsets, child).call()?;
    /// assert_eq!(lists.len(), 2);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn from_lists(num_rows: usize, offsets: Column, child: Column) -> FromLists {
        FromLists {
            num_rows,
            offsets,
            child,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_structs_column")]
    /// Creates a STRUCT column from `children` columns, with no null mask.
    ///
    /// `num_rows` is the number of struct rows. Every column in `children`
    /// must have exactly `num_rows` elements; they become the struct fields.
    ///
    /// All child columns are consumed (moved) into the new column.
    ///
    /// # Errors
    ///
    /// Returns an error if any child column has a different length than
    /// `num_rows`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let names = Column::from_strings(&["Alice", "Bob"]).call()?;
    /// let ages = Column::from_slice_i32(&[30, 25]).call()?;
    /// let structs = Column::from_structs(2, vec![names, ages]).call()?;
    /// assert_eq!(structs.len(), 2);
    /// assert_eq!(structs.type_id(), TypeId::STRUCT);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn from_structs(num_rows: usize, children: Vec<Column>) -> FromStructs {
        FromStructs {
            num_rows,
            children,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "size")]
    /// Returns the number of elements in the column, including nulls.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// assert_eq!(col.len(), 3);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_size(&self.0))
    }

    /// Returns `true` if the column has zero elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    ///
    /// let col = Column::empty(TypeId::FLOAT64);
    /// assert!(col.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of null elements in the column.
    ///
    /// If the column has no null mask (i.e. it was created without one),
    /// this returns `0`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let s = Scalar::null_i32();
    /// let col = Column::from_scalar(&s, 3).call()?;
    /// assert_eq!(col.null_count(), 3);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn null_count(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_null_count(&self.0))
    }

    /// Returns `true` if the column contains any null elements.
    ///
    /// Equivalent to `self.null_count() > 0`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// assert!(!col.has_nulls());
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_has_nulls(&self.0)
    }

    #[doc(alias = "type")]
    /// Returns the [`TypeId`] of this column's elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, 2.0]).call()?;
    /// assert_eq!(col.type_id(), TypeId::FLOAT64);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_type_id(&self.0);
        // C++ columns always have a valid type_id; fall back to EMPTY
        // for the theoretically-impossible out-of-range case.
        cudf_sys::type_id_from_i32(id).unwrap_or(TypeId::EMPTY)
    }

    /// Returns an immutable [`ColumnView`] borrowing this column's data.
    ///
    /// The view is tied to this column's lifetime and provides access to
    /// all GPU operations (arithmetic, reductions, comparisons, etc.)
    /// without copying data.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let view = col.view();
    /// assert_eq!(view.len(), 3);
    /// assert_eq!(view.type_id(), col.type_id());
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn view(&self) -> ColumnView<'_> {
        ColumnView(cudf_sys::ffi::column_view_of(&self.0))
    }

    /// Copies the column data from GPU to host as `Vec<i8>`.
    ///
    /// The column should have type [`TypeId::INT8`]. The returned vector
    /// has one element per row; null values are included but their values
    /// are unspecified. Use [`null_mask_to_host`](Column::null_mask_to_host)
    /// to determine which elements are valid.
    pub fn to_vec_i8(&self) -> ToVecI8<'_> {
        ToVecI8 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<i16>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_i16(&self) -> ToVecI16<'_> {
        ToVecI16 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<i32>`.
    ///
    /// The column should have type [`TypeId::INT32`]. See
    /// [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[10, 20, 30]).call()?;
    /// assert_eq!(col.to_vec_i32().call()?, [10, 20, 30]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn to_vec_i32(&self) -> ToVecI32<'_> {
        ToVecI32 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<i64>`.
    ///
    /// Also suitable for timestamp and duration columns, which are stored
    /// as `i64` internally. See [`to_vec_i8`](Column::to_vec_i8) for
    /// details on null handling.
    pub fn to_vec_i64(&self) -> ToVecI64<'_> {
        ToVecI64 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<f32>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_f32(&self) -> ToVecF32<'_> {
        ToVecF32 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<f64>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.5, 2.5]).call()?;
    /// assert_eq!(col.to_vec_f64().call()?, [1.5, 2.5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn to_vec_f64(&self) -> ToVecF64<'_> {
        ToVecF64 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u8>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u8(&self) -> ToVecU8<'_> {
        ToVecU8 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u16>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u16(&self) -> ToVecU16<'_> {
        ToVecU16 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u32>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u32(&self) -> ToVecU32<'_> {
        ToVecU32 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u64>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u64(&self) -> ToVecU64<'_> {
        ToVecU64 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<bool>`.
    ///
    /// The column should have type [`TypeId::BOOL8`]. See
    /// [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_bool(&self) -> ToVecBool<'_> {
        ToVecBool {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the per-element null mask from GPU to host as `Vec<bool>`.
    ///
    /// Each element is `true` if the corresponding row is valid (non-null)
    /// and `false` if it is null. If the column has no null mask, every
    /// element is `true`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let mask = col.null_mask_to_host().call()?;
    /// assert_eq!(mask, [true, true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn null_mask_to_host(&self) -> NullMaskToHost<'_> {
        NullMaskToHost {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies string column data from GPU to host as `Vec<String>`.
    ///
    /// The column should have type [`TypeId::STRING`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_strings(&["hello", "world"]).call()?;
    /// assert_eq!(col.to_vec_string().call()?, ["hello", "world"]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn to_vec_string(&self) -> ToVecString<'_> {
        ToVecString {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i8")]
    /// Creates an `INT8` column by copying `data` from host memory to
    /// the GPU.
    ///
    /// The resulting column has [`TypeId::INT8`] and `data.len()` rows,
    /// with no null mask.
    pub fn from_slice_i8(data: &[i8]) -> FromSliceI8<'_> {
        FromSliceI8 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i16")]
    /// Creates an `INT16` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_i16(data: &[i16]) -> FromSliceI16<'_> {
        FromSliceI16 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i32")]
    /// Creates an `INT32` column by copying `data` from host to GPU.
    ///
    /// The resulting column has `data.len()` rows and no null mask.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// assert_eq!(col.len(), 3);
    /// assert_eq!(col.to_vec_i32().call()?, [1, 2, 3]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn from_slice_i32(data: &[i32]) -> FromSliceI32<'_> {
        FromSliceI32 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i64")]
    /// Creates an `INT64` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_i64(data: &[i64]) -> FromSliceI64<'_> {
        FromSliceI64 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_f64")]
    /// Creates a `FLOAT64` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_f64(data: &[f64]) -> FromSliceF64<'_> {
        FromSliceF64 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_f32")]
    /// Creates a `FLOAT32` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_f32(data: &[f32]) -> FromSliceF32<'_> {
        FromSliceF32 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u8")]
    /// Creates a `UINT8` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_u8(data: &[u8]) -> FromSliceU8<'_> {
        FromSliceU8 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u16")]
    /// Creates a `UINT16` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_u16(data: &[u16]) -> FromSliceU16<'_> {
        FromSliceU16 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u32")]
    /// Creates a `UINT32` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_u32(data: &[u32]) -> FromSliceU32<'_> {
        FromSliceU32 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u64")]
    /// Creates a `UINT64` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_u64(data: &[u64]) -> FromSliceU64<'_> {
        FromSliceU64 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_bool")]
    /// Creates a `BOOL8` column by copying `data` from host to GPU.
    ///
    /// See [`from_slice_i8`](Column::from_slice_i8) for semantics.
    pub fn from_slice_bool(data: &[bool]) -> FromSliceBool<'_> {
        FromSliceBool {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_s")]
    /// Creates a `TIMESTAMP_SECONDS` column from `data` containing
    /// Unix epoch seconds (i.e. seconds since 1970-01-01T00:00:00Z).
    pub fn from_timestamps_s(data: &[i64]) -> FromTimestampsS<'_> {
        FromTimestampsS {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_ms")]
    /// Creates a `TIMESTAMP_MILLISECONDS` column from `data` containing
    /// milliseconds since the Unix epoch.
    pub fn from_timestamps_ms(data: &[i64]) -> FromTimestampsMs<'_> {
        FromTimestampsMs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_us")]
    /// Creates a `TIMESTAMP_MICROSECONDS` column from `data` containing
    /// microseconds since the Unix epoch.
    pub fn from_timestamps_us(data: &[i64]) -> FromTimestampsUs<'_> {
        FromTimestampsUs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_ns")]
    /// Creates a `TIMESTAMP_NANOSECONDS` column from `data` containing
    /// nanoseconds since the Unix epoch.
    pub fn from_timestamps_ns(data: &[i64]) -> FromTimestampsNs<'_> {
        FromTimestampsNs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_s")]
    /// Creates a `DURATION_SECONDS` column from `data` containing
    /// duration values in whole seconds.
    pub fn from_durations_s(data: &[i64]) -> FromDurationsS<'_> {
        FromDurationsS {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_ms")]
    /// Creates a `DURATION_MILLISECONDS` column from `data` containing
    /// duration values in milliseconds.
    pub fn from_durations_ms(data: &[i64]) -> FromDurationsMs<'_> {
        FromDurationsMs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_us")]
    /// Creates a `DURATION_MICROSECONDS` column from `data` containing
    /// duration values in microseconds.
    pub fn from_durations_us(data: &[i64]) -> FromDurationsUs<'_> {
        FromDurationsUs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_ns")]
    /// Creates a `DURATION_NANOSECONDS` column from `data` containing
    /// duration values in nanoseconds.
    pub fn from_durations_ns(data: &[i64]) -> FromDurationsNs<'_> {
        FromDurationsNs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_string_column")]
    /// Creates a `STRING` column by copying `values` from host to GPU.
    ///
    /// Each element in `values` becomes one row. The resulting column has
    /// no null mask; all rows are valid.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_strings(&["hello", "world"]).call()?;
    /// assert_eq!(col.type_id(), TypeId::STRING);
    /// assert_eq!(col.to_vec_string().call()?, ["hello", "world"]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn from_strings<'a>(values: &'a [&'a str]) -> FromStrings<'a> {
        FromStrings {
            values,
            stream: Stream::default_stream(),
        }
    }

    // -- In-place mutations --

    /// Fills elements in the half-open range `[begin, end)` with `value`,
    /// modifying this column in-place.
    ///
    /// `value` must have the same type as the column. The range must
    /// satisfy `begin <= end <= self.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds or the scalar type
    /// does not match.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let mut col = Column::from_slice_i32(&[0, 0, 0, 0]).call()?;
    /// let val = Scalar::from_i32(7);
    /// col.fill_in_place(1, 3, &val).call()?;
    /// assert_eq!(col.to_vec_i32().call()?, [0, 7, 7, 0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn fill_in_place<'a>(
        &'a mut self,
        begin: usize,
        end: usize,
        value: &'a Scalar,
    ) -> FillInPlace<'a> {
        FillInPlace {
            col: self,
            begin,
            end,
            value,
            stream: Stream::default_stream(),
        }
    }

    /// Copies elements from `source[source_begin..source_end]` into
    /// `self` starting at `dest_begin`, modifying this column in-place.
    ///
    /// The source range `[source_begin, source_end)` is copied element by
    /// element into `self[dest_begin..]`. Null values in the source
    /// overwrite the corresponding positions in this column (both data and
    /// null mask).
    ///
    /// # Errors
    ///
    /// Returns an error if either range is out of bounds or the types
    /// do not match.
    pub fn copy_range_in_place<'a>(
        &'a mut self,
        source: &'a ColumnView<'a>,
        source_begin: usize,
        source_end: usize,
        dest_begin: usize,
    ) -> CopyRangeInPlace<'a> {
        CopyRangeInPlace {
            col: self,
            source,
            source_begin,
            source_end,
            dest_begin,
            stream: Stream::default_stream(),
        }
    }

    /// Converts this column's null mask into a new `BOOL8` column.
    ///
    /// Each element is `true` if the corresponding row is valid and
    /// `false` if it is null. If the column has no null mask, all
    /// elements are `true`.
    ///
    /// Unlike [`null_mask_to_host`](Column::null_mask_to_host), the
    /// result stays on the GPU as a column, which is useful for further
    /// GPU operations.
    pub fn null_mask_to_bools(&self) -> NullMaskToBools<'_> {
        NullMaskToBools {
            col: self,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces this column's null mask in-place using a `BOOL8` column.
    ///
    /// `bools` must be a `BOOL8` column with the same length as `self`.
    /// Elements where `bools` is `true` become valid; elements where
    /// `bools` is `false` become null.
    ///
    /// # Errors
    ///
    /// Returns an error if `bools` has the wrong type or length.
    pub fn set_null_mask_from_bools<'a>(
        &'a mut self,
        bools: &'a ColumnView<'a>,
    ) -> SetNullMaskFromBools<'a> {
        SetNullMaskFromBools {
            col: self,
            bools,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a new column with the same data as `self` but a null mask
    /// derived from the `BOOL8` column `validity`.
    ///
    /// This is an out-of-place version of
    /// [`set_null_mask_from_bools`](Column::set_null_mask_from_bools):
    /// `self` is not modified.
    ///
    /// `validity` must be `BOOL8` with the same length as `self`.
    /// `true` marks valid, `false` marks null.
    ///
    /// # Errors
    ///
    /// Returns an error if `validity` has the wrong type or length.
    pub fn with_null_mask_from_bools<'a>(
        &'a self,
        validity: &'a ColumnView<'a>,
    ) -> WithNullMaskFromBools<'a> {
        WithNullMaskFromBools {
            col: self,
            validity,
            stream: Stream::default_stream(),
        }
    }
}

// ---------------------------------------------------------------------------
// Builder structs for ColumnView operations
// ---------------------------------------------------------------------------

/// Builder for [`ColumnView::cast`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`]
/// of the target type.
pub struct Cast<'a> {
    view: &'a ColumnView<'a>,
    target: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Cast<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::unary::ffi::unary_cast(self.view.0, self.target.repr, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::is_null`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct IsNull<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for IsNull<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_null(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::is_valid`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct IsValid<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for IsValid<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_valid(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::is_nan`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct IsNan<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for IsNan<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_nan(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::negate`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Negate<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Negate<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_negate(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::abs`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Abs<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Abs<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_abs(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::sum`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a [`Scalar`].
pub struct Sum<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Sum<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_sum(
            self.view.0,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::min`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a [`Scalar`].
pub struct Min<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Min<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_min(
            self.view.0,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::max`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a [`Scalar`].
pub struct Max<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Max<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_max(
            self.view.0,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::product`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a [`Scalar`].
pub struct Product<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Product<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_product(
            self.view.0,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::any`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a `BOOL8`
/// [`Scalar`].
pub struct Any<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Any<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_any(self.view.0, self.stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::all`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a `BOOL8`
/// [`Scalar`].
pub struct All<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for All<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_all(self.view.0, self.stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::quantile`].
pub struct Quantile<'a> {
    view: &'a ColumnView<'a>,
    quantiles: &'a [f64],
    stream: Stream,
}

impl crate::stream::GpuOp for Quantile<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::quantile::ffi::quantile_column(
            self.view.0,
            self.quantiles,
            crate::quantile::Interpolation::LINEAR.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::quantile_with_interp`].
pub struct QuantileWithInterp<'a> {
    view: &'a ColumnView<'a>,
    quantiles: &'a [f64],
    interp: crate::quantile::Interpolation,
    stream: Stream,
}

impl crate::stream::GpuOp for QuantileWithInterp<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::quantile::ffi::quantile_column(
            self.view.0,
            self.quantiles,
            self.interp.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::nans_to_nulls`]. See that method for details.
pub struct NansToNulls<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for NansToNulls<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let result = cudf_sys::transform::ffi::nans_to_nulls(self.view.0, self.stream.as_raw())?;
        Ok(Column(result))
    }
}

/// Builder for [`ColumnView::add`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Add<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Add<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::ADD,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::sub`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Sub<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Sub<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::SUB,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::mul`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Mul<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Mul<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::MUL,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::div`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Div<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Div<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::DIV,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::eq`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct Eq<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Eq<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::EQUAL,
            TypeId::BOOL8.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::ne`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct Ne<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Ne<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::NOT_EQUAL,
            TypeId::BOOL8.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::lt`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct Lt<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Lt<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::LESS,
            TypeId::BOOL8.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::gt`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct Gt<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Gt<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::GREATER,
            TypeId::BOOL8.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::le`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct Le<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Le<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::LESS_EQUAL,
            TypeId::BOOL8.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::ge`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct Ge<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Ge<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let col = cudf_sys::binaryop::ffi::binary_operation_columns(
            self.view.0,
            self.rhs.0,
            crate::ops::BinaryOperator::GREATER_EQUAL,
            TypeId::BOOL8.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`ColumnView::unary_op`] and the math convenience methods
/// (`sin`, `cos`, `ceil`, `floor`, etc.). Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct UnaryOp<'a> {
    view: &'a ColumnView<'a>,
    op: crate::ops::UnaryOperator,
    stream: Stream,
}

impl crate::stream::GpuOp for UnaryOp<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::unary::ffi::unary_operation(self.view.0, self.op.repr, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::is_not_nan`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing a `BOOL8`
/// [`Column`].
pub struct IsNotNan<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for IsNotNan<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_not_nan(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::round`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Round<'a> {
    view: &'a ColumnView<'a>,
    decimal_places: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for Round<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::round_column(
            self.view.0,
            self.decimal_places,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::mean`].
pub struct Mean<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Mean<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_mean(
            self.view.0,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::std_dev`].
pub struct StdDev<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    ddof: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for StdDev<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_std(
            self.view.0,
            self.output_type.repr,
            self.ddof,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::variance`].
pub struct Variance<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    ddof: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for Variance<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_var(
            self.view.0,
            self.output_type.repr,
            self.ddof,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::median`].
pub struct Median<'a> {
    view: &'a ColumnView<'a>,
    output_type: TypeId,
    stream: Stream,
}

impl crate::stream::GpuOp for Median<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_median(
            self.view.0,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::nunique`].
pub struct Nunique<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Nunique<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_nunique(self.view.0, 0, self.stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::reduce`].
pub struct Reduce<'a> {
    view: &'a ColumnView<'a>,
    agg: crate::groupby::AggregationKind,
    output_type: TypeId,
    ddof: i32,
    stream: Stream,
}

impl crate::stream::GpuOp for Reduce<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_generic(
            self.view.0,
            self.agg.repr,
            self.ddof,
            self.output_type.repr,
            self.stream.as_raw(),
        )?;
        Ok(crate::scalar::scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::reduce_with_init`].
pub struct ReduceWithInit<'a> {
    view: &'a ColumnView<'a>,
    agg: crate::groupby::AggregationKind,
    output_type: TypeId,
    ddof: i32,
    init: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for ReduceWithInit<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::reduce_with_init(
            self.view.0,
            self.agg.repr,
            self.ddof,
            self.output_type.repr,
            &crate::scalar::scalar_to_ffi(self.init),
            self.stream.as_raw(),
        )?;
        Ok(crate::scalar::scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::segmented_reduce`].
pub struct SegmentedReduce<'a> {
    view: &'a ColumnView<'a>,
    offsets: &'a ColumnView<'a>,
    agg: crate::groupby::AggregationKind,
    output_type: TypeId,
    ddof: i32,
    exclude_nulls: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedReduce<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let null_handling = i32::from(!self.exclude_nulls);
        let c = cudf_sys::reduction::ffi::segmented_reduce(
            self.view.0,
            self.offsets.0,
            self.agg.repr,
            self.ddof,
            self.output_type.repr,
            null_handling,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::segmented_reduce_with_init`].
pub struct SegmentedReduceWithInit<'a> {
    view: &'a ColumnView<'a>,
    offsets: &'a ColumnView<'a>,
    agg: crate::groupby::AggregationKind,
    output_type: TypeId,
    ddof: i32,
    exclude_nulls: bool,
    init: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedReduceWithInit<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let null_handling = i32::from(!self.exclude_nulls);
        let init_ffi = crate::scalar::scalar_to_ffi(self.init);
        let c = cudf_sys::reduction::ffi::segmented_reduce_with_init(
            self.view.0,
            self.offsets.0,
            self.agg.repr,
            self.ddof,
            self.output_type.repr,
            null_handling,
            &init_ffi,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::scan`].
pub struct Scan<'a> {
    view: &'a ColumnView<'a>,
    agg_kind: crate::groupby::AggregationKind,
    inclusive: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for Scan<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let scan_type = i32::from(!self.inclusive);
        let c = cudf_sys::reduction::ffi::scan_column(
            self.view.0,
            self.agg_kind.repr,
            scan_type,
            0, // null_policy: EXCLUDE
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::shift`].
pub struct Shift<'a> {
    view: &'a ColumnView<'a>,
    offset: i32,
    fill_value: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for Shift<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.fill_value);
        let c = cudf_sys::copying::ffi::shift_column(
            self.view.0,
            self.offset,
            &ffi,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::reverse`].
pub struct Reverse<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Reverse<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::reverse_column(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::slice`].
pub struct Slice<'a> {
    view: &'a ColumnView<'a>,
    begin: usize,
    end: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Slice<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::slice_column(
            self.view.0,
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::contains_scalar`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning `bool`.
pub struct ContainsScalar<'a> {
    view: &'a ColumnView<'a>,
    needle: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for ContainsScalar<'_> {
    type Output = bool;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.needle);
        cudf_sys::search::ffi::contains_scalar(self.view.0, &ffi, self.stream.as_raw())
            .map_err(Into::into)
    }
}

/// Builder for [`ColumnView::contains_column`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), returning a `BOOL8`
/// [`Column`].
pub struct ContainsColumn<'a> {
    view: &'a ColumnView<'a>,
    needles: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for ContainsColumn<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::search::ffi::contains_column(
            self.view.0,
            self.needles.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::fill`].
pub struct Fill<'a> {
    view: &'a ColumnView<'a>,
    begin: usize,
    end: usize,
    value: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for Fill<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.value);
        let c = cudf_sys::filling::ffi::fill_column(
            self.view.0,
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            &ffi,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::get_element`].
pub struct GetElement<'a> {
    view: &'a ColumnView<'a>,
    index: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for GetElement<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::copying::ffi::get_element(
            self.view.0,
            usize_to_i32(self.index),
            self.stream.as_raw(),
        )?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::copy_if_else`].
pub struct CopyIfElse<'a> {
    view: &'a ColumnView<'a>,
    rhs: &'a ColumnView<'a>,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for CopyIfElse<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::copy_if_else_columns(
            self.view.0,
            self.rhs.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::label_bins`].
pub struct LabelBins<'a> {
    view: &'a ColumnView<'a>,
    left_edges: &'a ColumnView<'a>,
    left_inclusive: crate::labeling::Inclusive,
    right_edges: &'a ColumnView<'a>,
    right_inclusive: crate::labeling::Inclusive,
    stream: Stream,
}

impl crate::stream::GpuOp for LabelBins<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::labeling::ffi::label_bins_column(
            self.view.0,
            self.left_edges.0,
            self.left_inclusive.repr,
            self.right_edges.0,
            self.right_inclusive.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::rank`]. See that method for details.
pub struct Rank<'a> {
    view: &'a ColumnView<'a>,
    method: crate::sorting::RankMethod,
    order: crate::sorting::Order,
    null_handling: crate::compaction::NullPolicy,
    null_precedence: crate::sorting::NullOrder,
    percentage: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for Rank<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::sorting::ffi::rank_column(
            self.view.0,
            i32::from(self.method),
            self.order.repr,
            self.null_handling.repr,
            self.null_precedence.repr,
            self.percentage,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::top_k`].
pub struct TopK<'a> {
    view: &'a ColumnView<'a>,
    k: usize,
    order: crate::sorting::Order,
    stream: Stream,
}

impl crate::stream::GpuOp for TopK<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::sorting::ffi::top_k(
            self.view.0,
            usize_to_i32(self.k),
            self.order.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::top_k_order`].
pub struct TopKOrder<'a> {
    view: &'a ColumnView<'a>,
    k: usize,
    order: crate::sorting::Order,
    stream: Stream,
}

impl crate::stream::GpuOp for TopKOrder<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::sorting::ffi::top_k_order(
            self.view.0,
            usize_to_i32(self.k),
            self.order.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::segmented_top_k`].
pub struct SegmentedTopK<'a> {
    view: &'a ColumnView<'a>,
    segment_offsets: &'a ColumnView<'a>,
    k: usize,
    order: crate::sorting::Order,
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedTopK<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::sorting::ffi::segmented_top_k(
            self.view.0,
            self.segment_offsets.0,
            usize_to_i32(self.k),
            self.order.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::segmented_top_k_order`].
pub struct SegmentedTopKOrder<'a> {
    view: &'a ColumnView<'a>,
    segment_offsets: &'a ColumnView<'a>,
    k: usize,
    order: crate::sorting::Order,
    stream: Stream,
}

impl crate::stream::GpuOp for SegmentedTopKOrder<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::sorting::ffi::segmented_top_k_order(
            self.view.0,
            self.segment_offsets.0,
            usize_to_i32(self.k),
            self.order.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::copy_range_into`].
pub struct CopyRangeInto<'a> {
    view: &'a ColumnView<'a>,
    target: &'a ColumnView<'a>,
    source_begin: usize,
    source_end: usize,
    target_begin: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for CopyRangeInto<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::copy_range(
            self.view.0,
            self.target.0,
            usize_to_i32(self.source_begin),
            usize_to_i32(self.source_end),
            usize_to_i32(self.target_begin),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::allocate_like`].
pub struct AllocateLike<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for AllocateLike<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::allocate_like_column(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::has_nonempty_nulls`].
pub struct HasNonemptyNulls<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for HasNonemptyNulls<'_> {
    type Output = bool;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        cudf_sys::copying::ffi::has_nonempty_nulls(self.view.0, self.stream.as_raw())
            .map_err(Into::into)
    }
}

/// Builder for [`ColumnView::purge_nonempty_nulls`].
pub struct PurgeNonemptyNulls<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for PurgeNonemptyNulls<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::purge_nonempty_nulls(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::replace_nulls_policy`].
pub struct ReplaceNullsPolicy<'a> {
    view: &'a ColumnView<'a>,
    preceding: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ReplaceNullsPolicy<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let policy = i32::from(!self.preceding);
        let c = cudf_sys::replace::ffi::replace_nulls_policy(
            self.view.0,
            policy,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::clamp_with_replace`].
pub struct ClampWithReplace<'a> {
    view: &'a ColumnView<'a>,
    lo: &'a Scalar,
    lo_replace: &'a Scalar,
    hi: &'a Scalar,
    hi_replace: &'a Scalar,
    stream: Stream,
}

impl crate::stream::GpuOp for ClampWithReplace<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let lo_ffi = crate::scalar::scalar_to_ffi(self.lo);
        let lo_r_ffi = crate::scalar::scalar_to_ffi(self.lo_replace);
        let hi_ffi = crate::scalar::scalar_to_ffi(self.hi);
        let hi_r_ffi = crate::scalar::scalar_to_ffi(self.hi_replace);
        let c = cudf_sys::replace::ffi::clamp_column_with_replace(
            self.view.0,
            &lo_ffi,
            &lo_r_ffi,
            &hi_ffi,
            &hi_r_ffi,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::normalize_nans_and_zeros`].
pub struct NormalizeNansAndZeros<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for NormalizeNansAndZeros<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::replace::ffi::normalize_nans_and_zeros(self.view.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::minmax_min`].
pub struct MinmaxMin<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for MinmaxMin<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::minmax_min(self.view.0, self.stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::minmax_max`].
pub struct MinmaxMax<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for MinmaxMax<'_> {
    type Output = Scalar;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let s = cudf_sys::reduction::ffi::minmax_max(self.view.0, self.stream.as_raw())?;
        Ok(scalar_from_ffi(&s))
    }
}

/// Builder for [`ColumnView::round_with_method`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct RoundWithMethod<'a> {
    view: &'a ColumnView<'a>,
    decimal_places: i32,
    method: crate::ops::RoundingMethod,
    stream: Stream,
}

impl crate::stream::GpuOp for RoundWithMethod<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::round_column(
            self.view.0,
            self.decimal_places,
            self.method.repr,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::child`].
pub struct Child<'a> {
    view: &'a ColumnView<'a>,
    index: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Child<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::column_view_child_copy(
            self.view.0,
            usize_to_i32(self.index),
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`ColumnView::distinct_count`].
pub struct ColumnDistinctCount<'a> {
    view: &'a ColumnView<'a>,
    include_nulls: bool,
    nan_is_null: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ColumnDistinctCount<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let null_policy = i32::from(self.include_nulls);
        Ok(i32_to_usize(
            cudf_sys::compaction::ffi::distinct_count_column(
                self.view.0,
                null_policy,
                self.nan_is_null,
                self.stream.as_raw(),
            ),
        ))
    }
}

/// Builder for [`ColumnView::unique_count`].
pub struct ColumnUniqueCount<'a> {
    view: &'a ColumnView<'a>,
    include_nulls: bool,
    nan_is_null: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ColumnUniqueCount<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let null_policy = i32::from(self.include_nulls);
        Ok(i32_to_usize(
            cudf_sys::compaction::ffi::unique_count_column(
                self.view.0,
                null_policy,
                self.nan_is_null,
                self.stream.as_raw(),
            ),
        ))
    }
}

/// Builder for [`ColumnView::percentile_approx`].
pub struct PercentileApprox<'a> {
    view: &'a ColumnView<'a>,
    percentiles: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for PercentileApprox<'_> {
    type Output = Column;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::quantile::ffi::percentile_approx(
            self.view.0,
            self.percentiles.0,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

// ---------------------------------------------------------------------------

#[doc(alias = "column_view")]
/// A non-owning, immutable view of a GPU column.
///
/// `ColumnView` borrows device data from a [`Column`] or
/// [`Table`](crate::table::Table) without taking ownership. The
/// lifetime parameter ensures the view cannot outlive the data it
/// references.
///
/// Most GPU operations (arithmetic, reductions, comparisons, etc.) are
/// methods on `ColumnView`. They return builder structs implementing
/// [`GpuOp`](crate::stream::GpuOp); call
/// [`.call()`](crate::stream::GpuOp::call) to execute.
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::data_type::TypeId;
/// use cudf::stream::GpuOp;
///
/// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// let view = col.view();
///
/// // Compute the sum on the GPU
/// let total = view.sum(TypeId::INT64).call()?;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct ColumnView<'a>(pub(crate) &'a cudf_sys::ffi::column_view);

impl ColumnView<'_> {
    #[doc(alias = "size")]
    /// Returns the number of elements in this view, including nulls.
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_size(self.0))
    }

    /// Returns `true` if this view has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of null elements in this view.
    pub fn null_count(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_null_count(self.0))
    }

    /// Returns `true` if this view contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_view_has_nulls(self.0)
    }

    /// Returns the offset of this view into the underlying data buffer.
    ///
    /// Sliced views may have a non-zero offset. For columns created
    /// directly, this is always `0`.
    pub fn offset(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_offset(self.0))
    }

    #[doc(alias = "type")]
    /// Returns the [`TypeId`] of this view's elements.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_view_type_id(self.0);
        // C++ column views always have a valid type_id.
        cudf_sys::type_id_from_i32(id).unwrap_or(TypeId::EMPTY)
    }

    // -- Child column access --

    /// Returns the number of child columns.
    ///
    /// For `STRUCT` columns this is the number of fields. For `LIST`
    /// columns this is 2 (offsets and child values). For `DICTIONARY32`
    /// columns this is 2 (indices and keys). For primitive types this
    /// is 0.
    pub fn num_children(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_num_children(self.0))
    }

    /// Deep-copies a child column by `index`, returning an owned
    /// [`Column`].
    ///
    /// The child index meaning depends on the column type:
    /// - **STRUCT**: `0..N-1` are the struct fields.
    /// - **LIST**: `0` is the offsets column, `1` is the child values.
    /// - **DICTIONARY32**: `0` is indices, `1` is keys.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range.
    pub fn child(&self, index: usize) -> Child<'_> {
        Child {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }

    // -- Device pointer access (for direct D2H/H2D) --

    /// Returns the raw device pointer to this column's data buffer.
    ///
    /// For fixed-width types, this points to the contiguous array of elements.
    /// For STRING columns, this points to the character data. For STRUCT/LIST
    /// columns, this returns 0 (data is in child columns).
    ///
    /// The returned `usize` is an opaque device pointer. Pass it to
    /// [`rmm::memory_resource::memcpy_d2h`] for host readback.
    pub fn data_ptr(&self) -> usize {
        cudf_sys::ffi::column_view_data_ptr(self.0)
    }

    /// Returns the raw device pointer to the null mask (validity bitmap).
    ///
    /// Returns 0 if this column has no null mask (i.e. `!has_nulls()`).
    pub fn null_mask_ptr(&self) -> usize {
        cudf_sys::ffi::column_view_null_mask_ptr(self.0)
    }

    /// Returns the size in bytes of one element of this column's type.
    ///
    /// For example, `INT64` returns 8, `FLOAT32` returns 4.
    /// Returns 0 for variable-width types (STRING, LIST, STRUCT).
    pub fn type_byte_size(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_type_size(self.0))
    }

    /// Returns the size of the character data in bytes (STRING columns only).
    ///
    /// This is the total byte length of all strings in the column, not the
    /// number of elements. For non-STRING columns, this is meaningless.
    pub fn chars_size(&self, stream: Stream) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_chars_size(self.0, stream.as_raw()))
    }

    // -- Unary ops --

    /// Casts every element in this column to a different type.
    ///
    /// The `target` [`TypeId`] specifies the desired output type. For
    /// example, casting an `INT32` column to `FLOAT64` converts each integer
    /// to its floating-point equivalent.
    ///
    /// Returns a [`Cast`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the cast between the source and target types is
    /// not supported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(42), 2).call()?;
    /// let f64_col = col.view().cast(TypeId::FLOAT64).call()?;
    /// assert_eq!(f64_col.type_id(), TypeId::FLOAT64);
    /// assert_eq!(f64_col.to_vec_f64().call()?, [42.0, 42.0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn cast(&self, target: TypeId) -> Cast<'_> {
        Cast {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a null element.
    ///
    /// The output column has the same length as this column and no null mask
    /// of its own. Elements that are null in the source map to `true`;
    /// valid elements map to `false`.
    ///
    /// Returns an [`IsNull`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
    /// let nulls = col.view().is_null().call()?;
    /// assert_eq!(nulls.to_vec_bool().call()?, [false, false, false]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn is_null(&self) -> IsNull<'_> {
        IsNull {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a valid (non-null)
    /// element.
    ///
    /// This is the logical inverse of [`is_null`](Self::is_null).
    ///
    /// Returns an [`IsValid`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
    /// let valid = col.view().is_valid().call()?;
    /// assert_eq!(valid.to_vec_bool().call()?, [true, true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn is_valid(&self) -> IsValid<'_> {
        IsValid {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a NaN element.
    ///
    /// Only applicable to floating-point columns (`FLOAT32` / `FLOAT64`).
    /// Null elements produce `false` (not NaN). For the inverse check, see
    /// [`is_not_nan`](Self::is_not_nan).
    ///
    /// Returns an [`IsNan`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a floating-point type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]).call()?;
    /// let nans = col.view().is_nan().call()?;
    /// assert_eq!(nans.to_vec_bool().call()?, [false, true, false]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn is_nan(&self) -> IsNan<'_> {
        IsNan {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Negates every element (unary minus).
    ///
    /// Applicable to numeric columns. The result column has the same type
    /// as the input.
    ///
    /// Returns a [`Negate`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let neg = col.view().negate().call()?;
    /// assert_eq!(neg.to_vec_i32().call()?, [-5, -5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn negate(&self) -> Negate<'_> {
        Negate {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the absolute value of every element.
    ///
    /// Applicable to signed numeric columns. The result column has the same
    /// type as the input.
    ///
    /// Returns an [`Abs`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(-7), 2).call()?;
    /// let a = col.view().abs().call()?;
    /// assert_eq!(a.to_vec_i32().call()?, [7, 7]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn abs(&self) -> Abs<'_> {
        Abs {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Reductions --

    /// Reduces the column to a single scalar by summing all elements.
    ///
    /// Null values are skipped during the reduction. The `output_type`
    /// controls the [`TypeId`] of the returned [`Scalar`] -- for example,
    /// summing an `INT32` column with `output_type` set to `INT64` avoids
    /// overflow for large sums.
    ///
    /// Returns a [`Sum`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible with the column
    /// type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(10), 4).call()?;
    /// let total = col.view().sum(TypeId::INT32).call()?;
    /// assert_eq!(total.as_i32(), Some(40));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn sum(&self, output_type: TypeId) -> Sum<'_> {
        Sum {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column to the minimum element.
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned [`Scalar`].
    ///
    /// Returns a [`Min`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[5, 2, 8]).call()?;
    /// let m = col.view().min(TypeId::INT32).call()?;
    /// assert_eq!(m.as_i32(), Some(2));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn min(&self, output_type: TypeId) -> Min<'_> {
        Min {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column to the maximum element.
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned [`Scalar`].
    ///
    /// Returns a [`Max`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[5, 2, 8]).call()?;
    /// let m = col.view().max(TypeId::INT32).call()?;
    /// assert_eq!(m.as_i32(), Some(8));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn max(&self, output_type: TypeId) -> Max<'_> {
        Max {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column to a single scalar by multiplying all elements.
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned [`Scalar`].
    ///
    /// Returns a [`Product`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(2), 3).call()?;
    /// let p = col.view().product(TypeId::INT32).call()?;
    /// assert_eq!(p.as_i32(), Some(8)); // 2 * 2 * 2
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn product(&self, output_type: TypeId) -> Product<'_> {
        Product {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces a `BOOL8` column, returning `true` if any element is true.
    ///
    /// Null values are skipped. The result is a [`Scalar`] of type `BOOL8`.
    /// An empty column (or one with all nulls) yields `false`.
    ///
    /// Returns an [`Any`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_bool(true), 3).call()?;
    /// let result = col.view().any().call()?;
    /// assert_eq!(result.as_bool(), Some(true));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn any(&self) -> Any<'_> {
        Any {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces a `BOOL8` column, returning `true` only if every element is
    /// true.
    ///
    /// Null values are skipped. The result is a [`Scalar`] of type `BOOL8`.
    /// An empty column (or one with all nulls) yields `true` (vacuous truth).
    ///
    /// Returns an [`All`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_bool(false), 3).call()?;
    /// let result = col.view().all().call()?;
    /// assert_eq!(result.as_bool(), Some(false));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn all(&self) -> All<'_> {
        All {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Quantile --

    /// Computes quantiles of this column using linear interpolation.
    ///
    /// `quantiles` is a slice of values in `[0.0, 1.0]`. Returns a `FLOAT64`
    /// column with one row per requested quantile.
    ///
    /// For custom interpolation, use
    /// [`quantile_with_interp`](ColumnView::quantile_with_interp).
    ///
    /// # Arguments
    ///
    /// * `quantiles` -- Slice of quantile values, each in `[0.0, 1.0]`.
    ///   For example, `&[0.25, 0.5, 0.75]` computes the quartiles.
    ///
    /// # Returns
    ///
    /// A `FLOAT64` [`Column`] with `quantiles.len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// let median = col.view().quantile(&[0.5]).call()?;
    /// // median contains [3.0]
    /// ```
    pub fn quantile<'a>(&'a self, quantiles: &'a [f64]) -> Quantile<'a> {
        Quantile {
            view: self,
            quantiles,
            stream: Stream::default_stream(),
        }
    }

    /// Computes quantiles with a specified
    /// [`Interpolation`](crate::quantile::Interpolation) method.
    ///
    /// See [`quantile`](ColumnView::quantile) for details.
    ///
    /// # Arguments
    ///
    /// * `quantiles` -- Slice of quantile values in `[0.0, 1.0]`.
    /// * `interp` -- Controls how values between data points are estimated
    ///   (e.g. `LINEAR`, `LOWER`, `HIGHER`, `MIDPOINT`, `NEAREST`).
    ///
    /// # Returns
    ///
    /// A `FLOAT64` [`Column`] with `quantiles.len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::quantile::Interpolation;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 4]).call()?;
    /// let q = col.view().quantile_with_interp(&[0.5], Interpolation::LOWER).call()?;
    /// // q contains [2.0] (lower of the two middle elements)
    /// ```
    pub fn quantile_with_interp<'a>(
        &'a self,
        quantiles: &'a [f64],
        interp: crate::quantile::Interpolation,
    ) -> QuantileWithInterp<'a> {
        QuantileWithInterp {
            view: self,
            quantiles,
            interp,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying --

    /// Creates an empty (zero-length) column with the same type as this
    /// view. The result has no data and no null mask.
    pub fn empty_like(&self) -> Column {
        Column(cudf_sys::copying::ffi::empty_like_column(self.0))
    }

    /// Creates an owning deep copy of this view's data.
    ///
    /// This copies all device memory (element data, null mask, and child
    /// columns for nested types) into a new independently-owned
    /// [`Column`]. The original data is not modified.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let copy = col.view().to_owned_column().call()?;
    /// assert_eq!(copy.to_vec_i32().call()?, [1, 2, 3]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn to_owned_column(&self) -> ToOwnedColumn<'_> {
        ToOwnedColumn {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Host data extraction (operates directly on view, no deep copy) --

    /// Copies the view's data from GPU to host as `Vec<i8>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i8(&self) -> ToVecI8<'_> {
        ToVecI8 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<i16>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i16(&self) -> ToVecI16<'_> {
        ToVecI16 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<i32>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i32(&self) -> ToVecI32<'_> {
        ToVecI32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<i64>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i64(&self) -> ToVecI64<'_> {
        ToVecI64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<f32>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_f32(&self) -> ToVecF32<'_> {
        ToVecF32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<f64>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_f64(&self) -> ToVecF64<'_> {
        ToVecF64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u8>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u8(&self) -> ToVecU8<'_> {
        ToVecU8 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u16>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u16(&self) -> ToVecU16<'_> {
        ToVecU16 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u32>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u32(&self) -> ToVecU32<'_> {
        ToVecU32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u64>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u64(&self) -> ToVecU64<'_> {
        ToVecU64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<bool>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_bool(&self) -> ToVecBool<'_> {
        ToVecBool {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the per-element null mask from GPU to host as `Vec<bool>`.
    ///
    /// See [`Column::null_mask_to_host`] for details.
    pub fn null_mask_to_host(&self) -> NullMaskToHost<'_> {
        NullMaskToHost {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies string data from GPU to host as `Vec<String>`.
    ///
    /// See [`Column::to_vec_string`] for details.
    pub fn to_vec_string(&self) -> ToVecString<'_> {
        ToVecString {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }

    // -- Transform --

    /// Converts NaN values to null in a floating-point column.
    ///
    /// Returns a new [`Column`] where every NaN
    /// element has been replaced by a null. Non-NaN values (including
    /// existing nulls) are preserved. The column must have a floating-point
    /// type (`FLOAT32` or `FLOAT64`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]).call()?;
    /// let clean = col.view().nans_to_nulls().call()?;
    /// assert_eq!(clean.null_count(), 2);
    /// assert_eq!(clean.len(), 5);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a floating-point type or a GPU
    /// error occurs.
    pub fn nans_to_nulls(&self) -> NansToNulls<'_> {
        NansToNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct values in this column.
    ///
    /// When `include_nulls` is `true`, null is counted as one distinct
    /// value (regardless of how many null rows exist). When
    /// `nan_is_null` is `true`, NaN values are treated as null for
    /// counting purposes.
    pub fn distinct_count(
        &self,
        include_nulls: bool,
        nan_is_null: bool,
    ) -> ColumnDistinctCount<'_> {
        ColumnDistinctCount {
            view: self,
            include_nulls,
            nan_is_null,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of consecutive groups of unique values.
    ///
    /// Unlike [`distinct_count`](ColumnView::distinct_count), this
    /// only counts transitions between consecutive distinct values,
    /// so the column should typically be sorted first.
    ///
    /// `include_nulls` and `nan_is_null` have the same meaning as in
    /// [`distinct_count`](ColumnView::distinct_count).
    pub fn unique_count(&self, include_nulls: bool, nan_is_null: bool) -> ColumnUniqueCount<'_> {
        ColumnUniqueCount {
            view: self,
            include_nulls,
            nan_is_null,
            stream: Stream::default_stream(),
        }
    }

    /// Computes approximate percentiles from a pre-built t-digest column.
    ///
    /// `self` must be a t-digest column (a STRUCT column produced by
    /// the t-digest aggregation). `percentiles` is a `FLOAT64` column
    /// of values in `[0.0, 1.0]`.
    pub fn percentile_approx<'a>(
        &'a self,
        percentiles: &'a ColumnView<'a>,
    ) -> PercentileApprox<'a> {
        PercentileApprox {
            view: self,
            percentiles,
            stream: Stream::default_stream(),
        }
    }

    // -- Binary ops (convenience) --

    /// Computes the element-wise sum of this column and `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column -- for example, adding two
    /// `INT32` columns with `output_type` set to `FLOAT64` produces a
    /// `FLOAT64` result.
    ///
    /// This is a convenience wrapper around
    /// [`binary_op`](crate::ops) with [`BinaryOperator::ADD`](crate::ops::BinaryOperator::ADD).
    ///
    /// Returns an [`Add`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(10), 3).call()?;
    /// let sum = a.view().add(&b.view(), TypeId::INT32).call()?;
    /// assert_eq!(sum.to_vec_i32().call()?, [11, 11, 11]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn add<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Add<'a> {
        Add {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the element-wise difference of this column minus `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column.
    ///
    /// Returns a [`Sub`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(10), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 3).call()?;
    /// let diff = a.view().sub(&b.view(), TypeId::INT32).call()?;
    /// assert_eq!(diff.to_vec_i32().call()?, [7, 7, 7]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn sub<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Sub<'a> {
        Sub {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the element-wise product of this column and `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column.
    ///
    /// Returns a [`Mul`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_f64(2.0), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_f64(3.0), 3).call()?;
    /// let prod = a.view().mul(&b.view(), TypeId::FLOAT64).call()?;
    /// assert_eq!(prod.to_vec_f64().call()?, [6.0, 6.0, 6.0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn mul<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Mul<'a> {
        Mul {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the element-wise division of this column by `rhs`.
    ///
    /// Both columns must have the same length. The `output_type` controls
    /// the [`TypeId`] of the resulting column. For integer types this
    /// performs truncating division; use `FLOAT64` output for exact results.
    ///
    /// Returns a [`Div`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the columns have different lengths or if the
    /// type combination is unsupported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_f64(10.0), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_f64(4.0), 2).call()?;
    /// let quot = a.view().div(&b.view(), TypeId::FLOAT64).call()?;
    /// assert_eq!(quot.to_vec_f64().call()?, [2.5, 2.5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn div<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Div<'a> {
        Div {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise equality comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when the corresponding elements of `self` and `rhs` are equal.
    ///
    /// Returns an [`struct@Eq`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 3).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(5), 3).call()?;
    /// let mask = a.view().eq(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn eq<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Eq<'a> {
        Eq {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise not-equal comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when the corresponding elements differ.
    ///
    /// Returns a [`Ne`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let mask = a.view().ne(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn ne<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Ne<'a> {
        Ne {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise less-than comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] < rhs[i]`.
    ///
    /// Returns an [`Lt`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let mask = a.view().lt(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn lt<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Lt<'a> {
        Lt {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise greater-than comparison, producing a `BOOL8` column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] > rhs[i]`.
    ///
    /// Returns a [`Gt`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let mask = a.view().gt(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn gt<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Gt<'a> {
        Gt {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise less-than-or-equal comparison, producing a `BOOL8`
    /// column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] <= rhs[i]`.
    ///
    /// Returns an [`Le`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(3), 2).call()?;
    /// let mask = a.view().le(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn le<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Le<'a> {
        Le {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise greater-than-or-equal comparison, producing a `BOOL8`
    /// column.
    ///
    /// Both columns must have the same length. Each output element is `true`
    /// when `self[i] >= rhs[i]`.
    ///
    /// Returns a [`Ge`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let a = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let b = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let mask = a.view().ge(&b.view()).call()?;
    /// assert_eq!(mask.to_vec_bool().call()?, [true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn ge<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Ge<'a> {
        Ge {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    // -- Generic unary operation --

    /// Applies a generic unary operation to every element of this column.
    ///
    /// The `op` parameter selects the operation (see
    /// [`UnaryOperator`](crate::ops::UnaryOperator) for all variants).
    /// Convenience wrappers such as [`sin`](Self::sin), [`abs`](Self::abs),
    /// and [`ceil`](Self::ceil) delegate to this method.
    ///
    /// Returns a [`UnaryOp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn unary_op(&self, op: crate::ops::UnaryOperator) -> UnaryOp<'_> {
        UnaryOp {
            view: self,
            op,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a non-NaN element.
    ///
    /// Only applicable to floating-point columns. This is the logical
    /// inverse of [`is_nan`](Self::is_nan).
    ///
    /// Returns an [`IsNotNan`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    pub fn is_not_nan(&self) -> IsNotNan<'_> {
        IsNotNan {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Math convenience methods --

    /// Computes the sine of each element (radians), returning a new
    /// column. Applicable to floating-point types.
    pub fn sin(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SIN)
    }
    /// Computes the cosine of each element (radians).
    pub fn cos(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::COS)
    }
    /// Computes the tangent of each element (radians).
    pub fn tan(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::TAN)
    }
    /// Computes the arcsine (inverse sine) of each element.
    pub fn arcsin(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCSIN)
    }
    /// Computes the arccosine (inverse cosine) of each element.
    pub fn arccos(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCCOS)
    }
    /// Computes the arctangent (inverse tangent) of each element.
    pub fn arctan(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCTAN)
    }
    /// Computes the hyperbolic sine of each element.
    pub fn sinh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SINH)
    }
    /// Computes the hyperbolic cosine of each element.
    pub fn cosh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::COSH)
    }
    /// Computes the hyperbolic tangent of each element.
    pub fn tanh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::TANH)
    }
    /// Computes the inverse hyperbolic sine of each element.
    pub fn arcsinh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCSINH)
    }
    /// Computes the inverse hyperbolic cosine of each element.
    pub fn arccosh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCCOSH)
    }
    /// Computes the inverse hyperbolic tangent of each element.
    pub fn arctanh(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCTANH)
    }
    /// Computes `e^x` for each element.
    pub fn exp(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::EXP)
    }
    /// Computes the natural logarithm (`ln`) of each element.
    pub fn log(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::LOG)
    }
    /// Computes the square root of each element.
    pub fn sqrt(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SQRT)
    }
    /// Computes the cube root of each element.
    pub fn cbrt(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::CBRT)
    }
    /// Rounds each element up to the smallest integer not less than the value.
    ///
    /// Applicable to floating-point columns. The result has the same type as
    /// the input.
    ///
    /// Returns a [`UnaryOp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn ceil(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::CEIL)
    }
    /// Rounds each element down to the largest integer not greater than the
    /// value.
    ///
    /// Applicable to floating-point columns. The result has the same type as
    /// the input.
    ///
    /// Returns a [`UnaryOp`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn floor(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::FLOOR)
    }
    /// Rounds each element to the nearest integer (round half to even),
    /// returning a floating-point column.
    pub fn rint(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::RINT)
    }
    /// Bitwise inversion of each element.
    pub fn bit_invert(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::BIT_INVERT)
    }
    /// Logical NOT of each element.
    pub fn logical_not(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::NOT)
    }

    // -- Round --

    /// Rounds column values to the given number of decimal places.
    ///
    /// Uses the `HALF_UP` rounding method by default. A positive
    /// `decimal_places` rounds to that many digits after the decimal point;
    /// a negative value rounds to digits before the decimal point (e.g.,
    /// `-1` rounds to the nearest 10).
    ///
    /// For control over the rounding strategy, see
    /// [`round_with_method`](Self::round_with_method).
    ///
    /// Returns a [`Round`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.15, 2.25, 3.35]).call()?;
    /// let rounded = col.view().round(1).call()?;
    /// // [1.2, 2.3, 3.4] with HALF_UP rounding
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn round(&self, decimal_places: i32) -> Round<'_> {
        Round {
            view: self,
            decimal_places,
            stream: Stream::default_stream(),
        }
    }

    /// Rounds column values using a specific [`RoundingMethod`](crate::ops::RoundingMethod).
    ///
    /// - `decimal_places` -- number of decimal places to round to (see
    ///   [`round`](Self::round) for sign semantics).
    /// - `method` -- either
    ///   [`HALF_UP`](crate::ops::RoundingMethod::HALF_UP) or
    ///   [`HALF_EVEN`](crate::ops::RoundingMethod::HALF_EVEN) (banker's
    ///   rounding).
    ///
    /// Returns a [`RoundWithMethod`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn round_with_method(
        &self,
        decimal_places: i32,
        method: crate::ops::RoundingMethod,
    ) -> RoundWithMethod<'_> {
        RoundWithMethod {
            view: self,
            decimal_places,
            method,
            stream: Stream::default_stream(),
        }
    }

    // -- Reductions (new) --

    /// Computes the arithmetic mean of all non-null elements, returning
    /// a [`Scalar`].
    ///
    /// `output_type` is typically `FLOAT64`.
    pub fn mean(&self, output_type: TypeId) -> Mean<'_> {
        Mean {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the standard deviation of all non-null elements.
    ///
    /// `ddof` is the delta degrees of freedom (0 for population, 1 for
    /// sample standard deviation). `output_type` is typically `FLOAT64`.
    pub fn std_dev(&self, output_type: TypeId, ddof: i32) -> StdDev<'_> {
        StdDev {
            view: self,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the variance of all non-null elements.
    ///
    /// `ddof` is the delta degrees of freedom (0 for population, 1 for
    /// sample variance). `output_type` is typically `FLOAT64`.
    pub fn variance(&self, output_type: TypeId, ddof: i32) -> Variance<'_> {
        Variance {
            view: self,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the median of all non-null elements, returning a
    /// [`Scalar`].
    ///
    /// `output_type` is typically `FLOAT64`.
    pub fn median(&self, output_type: TypeId) -> Median<'_> {
        Median {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct non-null values, returning a
    /// [`Scalar`] of type `INT32`.
    ///
    /// Null values are excluded from the count.
    pub fn nunique(&self) -> Nunique<'_> {
        Nunique {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the minimum value using the minmax kernel, which finds
    /// both min and max in a single pass. Returns just the minimum.
    ///
    /// Prefer this over [`min`](ColumnView::min) when you also need
    /// [`minmax_max`](ColumnView::minmax_max), since the kernel
    /// computes both simultaneously.
    pub fn minmax_min(&self) -> MinmaxMin<'_> {
        MinmaxMin {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the maximum value using the minmax kernel. Returns just
    /// the maximum. See [`minmax_min`](ColumnView::minmax_min).
    pub fn minmax_max(&self) -> MinmaxMax<'_> {
        MinmaxMax {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Generic reduce --

    /// Reduces the column using a generic
    /// [`AggregationKind`](crate::groupby::AggregationKind), returning
    /// a [`Scalar`].
    ///
    /// `output_type` specifies the result scalar type. `ddof` (delta
    /// degrees of freedom) is only used for `STD` and `VAR`
    /// aggregations; pass `0` for others.
    pub fn reduce(
        &self,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
    ) -> Reduce<'_> {
        Reduce {
            view: self,
            agg,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column with an explicit initial value.
    ///
    /// Like [`reduce`](ColumnView::reduce), but the reduction starts
    /// from `init` instead of the identity element. Supports `SUM`,
    /// `PRODUCT`, `MIN`, `MAX`, `ANY`, and `ALL` aggregations.
    pub fn reduce_with_init<'a>(
        &'a self,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        init: &'a Scalar,
    ) -> ReduceWithInit<'a> {
        ReduceWithInit {
            view: self,
            agg,
            output_type,
            ddof,
            init,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces each segment independently, returning one result per
    /// segment in a new column.
    ///
    /// `offsets` is an `INT32` column of length `N+1` defining `N`
    /// segments, similar to list offsets. `exclude_nulls` controls
    /// whether null values are skipped during aggregation. `ddof` is
    /// used only for `STD` and `VAR` aggregations.
    pub fn segmented_reduce<'a>(
        &'a self,
        offsets: &'a ColumnView<'_>,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        exclude_nulls: bool,
    ) -> SegmentedReduce<'a> {
        SegmentedReduce {
            view: self,
            offsets,
            agg,
            output_type,
            ddof,
            exclude_nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Segmented reduce with an explicit initial value.
    ///
    /// Like [`segmented_reduce`](ColumnView::segmented_reduce), but
    /// each segment's reduction starts from `init`. Only `SUM`,
    /// `PRODUCT`, `MIN`, `MAX`, `ANY`, and `ALL` aggregations are
    /// supported.
    #[doc(alias = "segmented_reduce")]
    pub fn segmented_reduce_with_init<'a>(
        &'a self,
        offsets: &'a ColumnView<'_>,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        exclude_nulls: bool,
        init: &'a Scalar,
    ) -> SegmentedReduceWithInit<'a> {
        SegmentedReduceWithInit {
            view: self,
            offsets,
            agg,
            output_type,
            ddof,
            exclude_nulls,
            init,
            stream: Stream::default_stream(),
        }
    }

    // -- Scan --

    /// Computes a prefix scan (cumulative operation) on this column.
    ///
    /// `agg_kind` specifies the scan operation (e.g. `SUM` for a
    /// cumulative sum, `MIN` for a running minimum). When `inclusive`
    /// is `true`, element `i` includes itself; when `false`, element
    /// `i` is the result of the first `i` elements (exclusive scan).
    pub fn scan(&self, agg_kind: crate::groupby::AggregationKind, inclusive: bool) -> Scan<'_> {
        Scan {
            view: self,
            agg_kind,
            inclusive,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying extras --

    /// Shifts column elements by `offset` positions, filling vacated
    /// positions with `fill_value`.
    ///
    /// Positive `offset` shifts elements to the right (later indices);
    /// negative shifts to the left. The result has the same length as
    /// the input.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let fill = Scalar::from_i32(0);
    /// let shifted = col.view().shift(1, &fill).call()?;
    /// assert_eq!(shifted.to_vec_i32().call()?, [0, 1, 2]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn shift<'a>(&'a self, offset: i32, fill_value: &'a Scalar) -> Shift<'a> {
        Shift {
            view: self,
            offset,
            fill_value,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the element at `index` as a [`Scalar`].
    ///
    /// If the element is null, the returned scalar is null. This
    /// involves a GPU-to-host transfer of a single value.
    ///
    /// # Errors
    ///
    /// Returns an error if `index >= self.len()`.
    pub fn get_element(&self, index: usize) -> GetElement<'_> {
        GetElement {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }

    /// Reverses the order of elements, returning a new column.
    pub fn reverse(&self) -> Reverse<'_> {
        Reverse {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Extracts the half-open range `[begin, end)` as a new owned
    /// column.
    ///
    /// The result is an independent deep copy of the specified range.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds.
    pub fn slice(&self, begin: usize, end: usize) -> Slice<'_> {
        Slice {
            view: self,
            begin,
            end,
            stream: Stream::default_stream(),
        }
    }

    /// Selects elements from `self` where `mask` is `true`, and from
    /// `rhs` where `mask` is `false`, returning a new column.
    ///
    /// All three columns (`self`, `rhs`, `mask`) must have the same
    /// length. `mask` must be a `BOOL8` column. `self` and `rhs` must
    /// have compatible types.
    pub fn copy_if_else<'a>(
        &'a self,
        rhs: &'a ColumnView<'_>,
        mask: &'a ColumnView<'_>,
    ) -> CopyIfElse<'a> {
        CopyIfElse {
            view: self,
            rhs,
            mask,
            stream: Stream::default_stream(),
        }
    }

    // -- Label bins --

    /// Assigns integer bin labels to each element based on bin edges.
    ///
    /// Bin *i* is defined by `[left_edges[i], right_edges[i]]` with the
    /// inclusivity of each edge controlled by `left_inclusive` and
    /// `right_inclusive`. Elements that fall outside all bins receive `-1`.
    ///
    /// # Arguments
    ///
    /// * `left_edges` -- Column of left boundaries, one per bin. Must be
    ///   sorted in ascending order.
    /// * `left_inclusive` -- Whether the left edge of each bin is inclusive
    ///   ([`Inclusive::YES`](crate::labeling::Inclusive::YES)) or exclusive.
    /// * `right_edges` -- Column of right boundaries, one per bin. Must
    ///   have the same length as `left_edges`.
    /// * `right_inclusive` -- Whether the right edge of each bin is inclusive
    ///   or exclusive.
    ///
    /// # Returns
    ///
    /// An `INT32` column with the same length as `self`, where each element
    /// is the index of the bin it falls into, or `-1` if it falls outside
    /// all bins.
    ///
    /// # Errors
    ///
    /// Returns an error if edge columns have mismatched lengths or the
    /// libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::labeling::Inclusive;
    /// use cudf::stream::GpuOp;
    ///
    /// let values = Column::from_slice_i32(&[1, 5, 15]).call()?;
    /// let left = Column::from_slice_i32(&[0, 10]).call()?;
    /// let right = Column::from_slice_i32(&[10, 20]).call()?;
    /// let labels = values.view().label_bins(
    ///     &left.view(), Inclusive::YES,
    ///     &right.view(), Inclusive::NO,
    /// ).call()?;
    /// // labels: [0, 0, 1] (1 and 5 in bin 0, 15 in bin 1)
    /// ```
    pub fn label_bins<'a>(
        &'a self,
        left_edges: &'a ColumnView<'_>,
        left_inclusive: crate::labeling::Inclusive,
        right_edges: &'a ColumnView<'_>,
        right_inclusive: crate::labeling::Inclusive,
    ) -> LabelBins<'a> {
        LabelBins {
            view: self,
            left_edges,
            left_inclusive,
            right_edges,
            right_inclusive,
            stream: Stream::default_stream(),
        }
    }

    // -- Search --

    /// Searches this column for a scalar value.
    ///
    /// Returns `true` if `needle` appears anywhere in the column, `false`
    /// otherwise. The column does not need to be sorted.
    ///
    /// Returns a [`ContainsScalar`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[10, 20, 30]).call()?;
    /// assert!(col.view().contains_scalar(&Scalar::from_i32(20)).call()?);
    /// assert!(!col.view().contains_scalar(&Scalar::from_i32(25)).call()?);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn contains_scalar<'a>(&'a self, needle: &'a Scalar) -> ContainsScalar<'a> {
        ContainsScalar {
            view: self,
            needle,
            stream: Stream::default_stream(),
        }
    }

    /// Checks which values from `needles` exist in this column.
    ///
    /// Returns a `BOOL8` column with the same length as `needles`. Each
    /// output element is `true` if the corresponding needle is found
    /// anywhere in `self`, and `false` otherwise. Neither column needs to
    /// be sorted.
    ///
    /// Returns a [`ContainsColumn`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let haystack = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    /// let needles = Column::from_slice_i32(&[20, 60]).call()?;
    /// let found = haystack.view()
    ///     .contains_column(&needles.view())
    ///     .call()?;
    /// assert_eq!(found.to_vec_bool().call()?, [true, false]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn contains_column<'a>(&'a self, needles: &'a ColumnView<'_>) -> ContainsColumn<'a> {
        ContainsColumn {
            view: self,
            needles,
            stream: Stream::default_stream(),
        }
    }

    // -- Sorting (column-level) --

    /// Computes the rank of each element in this column.
    ///
    /// Ranks are 1-based by default. The ranking strategy is controlled by
    /// `method` (see [`RankMethod`](crate::sorting::RankMethod)):
    ///
    /// - `method` -- how to resolve ties among equal values.
    /// - `order` -- [`Order::ASCENDING`](crate::sorting::Order::ASCENDING)
    ///   ranks smallest values first;
    ///   [`Order::DESCENDING`](crate::sorting::Order::DESCENDING) ranks
    ///   largest values first.
    /// - `null_handling` -- whether null elements receive a rank
    ///   ([`NullPolicy::INCLUDE`](crate::compaction::NullPolicy::INCLUDE))
    ///   or are left as null
    ///   ([`NullPolicy::EXCLUDE`](crate::compaction::NullPolicy::EXCLUDE)).
    /// - `null_precedence` -- where nulls sort relative to non-null values.
    /// - `percentage` -- when `true`, ranks are normalized to the range
    ///   `[0.0, 1.0]` and the output type is `FLOAT64`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::compaction::NullPolicy;
    /// use cudf::sorting::{Order, NullOrder, RankMethod};
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[30, 10, 20, 10]).call()?;
    /// let view = col.view();
    ///
    /// // Dense rank ascending: [3, 1, 2, 1]
    /// let ranks = view.rank(
    ///     RankMethod::Dense,
    ///     Order::ASCENDING,
    ///     NullPolicy::EXCLUDE,
    ///     NullOrder::AFTER,
    ///     false,
    /// ).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn rank(
        &self,
        method: crate::sorting::RankMethod,
        order: crate::sorting::Order,
        null_handling: crate::compaction::NullPolicy,
        null_precedence: crate::sorting::NullOrder,
        percentage: bool,
    ) -> Rank<'_> {
        Rank {
            view: self,
            method,
            order,
            null_handling,
            null_precedence,
            percentage,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the `k` largest (or smallest) values as a new column.
    ///
    /// `order` controls the direction: `ASCENDING` returns the `k`
    /// smallest, `DESCENDING` returns the `k` largest.
    pub fn top_k(&self, k: usize, order: crate::sorting::Order) -> TopK<'_> {
        TopK {
            view: self,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the row indices of the `k` largest (or smallest)
    /// values as an `INT32` column.
    ///
    /// See [`top_k`](ColumnView::top_k) for the `order` semantics.
    pub fn top_k_order(&self, k: usize, order: crate::sorting::Order) -> TopKOrder<'_> {
        TopKOrder {
            view: self,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the `k` largest (or smallest) values within each segment.
    ///
    /// `segment_offsets` is an `INT32` column defining segment
    /// boundaries (same format as list offsets).
    pub fn segmented_top_k<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'_>,
        k: usize,
        order: crate::sorting::Order,
    ) -> SegmentedTopK<'a> {
        SegmentedTopK {
            view: self,
            segment_offsets,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the row indices of the `k` largest (or smallest) values
    /// within each segment.
    ///
    /// See [`segmented_top_k`](ColumnView::segmented_top_k) for details.
    pub fn segmented_top_k_order<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'_>,
        k: usize,
        order: crate::sorting::Order,
    ) -> SegmentedTopKOrder<'a> {
        SegmentedTopKOrder {
            view: self,
            segment_offsets,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying (new) --

    /// Copies `self[source_begin..source_end]` into `target` starting
    /// at `target_begin`, returning a new column based on `target`.
    ///
    /// This is an out-of-place operation: neither `self` nor `target`
    /// is modified. The result is a copy of `target` with the specified
    /// range overwritten by elements from `self`.
    ///
    /// # Arguments
    ///
    /// * `target` -- The column to copy into (used as the base).
    /// * `source_begin` -- Start index in `self` (inclusive).
    /// * `source_end` -- End index in `self` (exclusive).
    /// * `target_begin` -- Start index in `target` where copied elements
    ///   are written.
    ///
    /// # Returns
    ///
    /// A new [`Column`] that is a copy of `target` with the range
    /// `[target_begin, target_begin + (source_end - source_begin))` replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if ranges are out of bounds, types are mismatched,
    /// or the libcudf call fails.
    pub fn copy_range_into<'a>(
        &'a self,
        target: &'a ColumnView<'_>,
        source_begin: usize,
        source_end: usize,
        target_begin: usize,
    ) -> CopyRangeInto<'a> {
        CopyRangeInto {
            view: self,
            target,
            source_begin,
            source_end,
            target_begin,
            stream: Stream::default_stream(),
        }
    }

    /// Creates an uninitialized column with the same type and size as
    /// this view. The data buffer is allocated but not initialized.
    pub fn allocate_like(&self) -> AllocateLike<'_> {
        AllocateLike {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Checks whether this column has null rows that contain non-empty
    /// data (relevant for variable-width types like `LIST` and
    /// `STRING`).
    ///
    /// This requires a GPU kernel. For a fast host-side check that may
    /// return false positives, use
    /// [`may_have_nonempty_nulls`](ColumnView::may_have_nonempty_nulls).
    pub fn has_nonempty_nulls(&self) -> HasNonemptyNulls<'_> {
        HasNonemptyNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Fast host-side check for whether this column might have non-empty
    /// data in null rows.
    ///
    /// May return `true` even when no non-empty nulls exist (false
    /// positive), but never returns `false` when they do exist. Use
    /// [`has_nonempty_nulls`](ColumnView::has_nonempty_nulls) for an
    /// exact check.
    pub fn may_have_nonempty_nulls(&self) -> bool {
        cudf_sys::copying::ffi::may_have_nonempty_nulls(self.0)
    }

    /// Returns a new column with non-empty null row data cleared.
    ///
    /// For variable-width types (`LIST`, `STRING`), null rows may
    /// still contain data. This method produces a column where null
    /// rows have zero-length content, which can be required for
    /// certain interop scenarios.
    pub fn purge_nonempty_nulls(&self) -> PurgeNonemptyNulls<'_> {
        PurgeNonemptyNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Replace (new) --

    /// Replaces null values using a fill policy, returning a new column.
    ///
    /// When `preceding` is `true`, each null is replaced by the last
    /// non-null value before it (forward fill). When `false`, each null
    /// is replaced by the next non-null value after it (backward fill).
    /// Leading/trailing nulls that have no fill source remain null.
    pub fn replace_nulls_policy(&self, preceding: bool) -> ReplaceNullsPolicy<'_> {
        ReplaceNullsPolicy {
            view: self,
            preceding,
            stream: Stream::default_stream(),
        }
    }

    /// Clamps column values to `[lo, hi]`, replacing out-of-range
    /// values with separate replacement scalars.
    ///
    /// Elements less than `lo` are replaced with `lo_replace`. Elements
    /// greater than `hi` are replaced with `hi_replace`. Elements
    /// within the range are unchanged.
    pub fn clamp_with_replace<'a>(
        &'a self,
        lo: &'a Scalar,
        lo_replace: &'a Scalar,
        hi: &'a Scalar,
        hi_replace: &'a Scalar,
    ) -> ClampWithReplace<'a> {
        ClampWithReplace {
            view: self,
            lo,
            lo_replace,
            hi,
            hi_replace,
            stream: Stream::default_stream(),
        }
    }

    /// Normalizes NaN and zero values in a floating-point column.
    ///
    /// Converts all negative NaN representations to the canonical
    /// positive NaN, and `-0.0` to `+0.0`. This is useful before
    /// operations that require consistent equality semantics.
    pub fn normalize_nans_and_zeros(&self) -> NormalizeNansAndZeros<'_> {
        NormalizeNansAndZeros {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Fill --

    /// Returns a new column with elements in `[begin, end)` replaced
    /// by `value`, leaving elements outside the range unchanged.
    ///
    /// This is an out-of-place version of
    /// [`Column::fill_in_place`]. `value` must have the same type as
    /// the column.
    ///
    /// # Arguments
    ///
    /// * `begin` -- Start index of the fill range (inclusive).
    /// * `end` -- End index of the fill range (exclusive). The range
    ///   must satisfy `begin <= end <= self.len()`.
    /// * `value` -- The scalar value to fill with. Its type must match
    ///   the column's [`TypeId`].
    ///
    /// # Returns
    ///
    /// A new [`Column`] with the range filled.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds or the scalar type
    /// does not match.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(1), 5).call()?;
    /// let result = col.view().fill(1, 3, &Scalar::from_i32(99)).call()?;
    /// // result contains [1, 99, 99, 1, 1]
    /// ```
    pub fn fill<'a>(&'a self, begin: usize, end: usize, value: &'a Scalar) -> Fill<'a> {
        Fill {
            view: self,
            begin,
            end,
            value,
            stream: Stream::default_stream(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::GpuOp;

    #[test]
    fn column_from_scalar_i32() {
        let s = Scalar::from_i32(7);
        let col = Column::from_scalar(&s, 5).call().unwrap();
        assert_eq!(col.len(), 5);
        assert_eq!(col.type_id(), TypeId::INT32);
        assert!(!col.has_nulls());
        assert_eq!(col.to_vec_i32().call().unwrap(), vec![7, 7, 7, 7, 7]);
    }

    #[test]
    fn column_from_scalar_f64() {
        let s = Scalar::from_f64(2.5);
        let col = Column::from_scalar(&s, 3).call().unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col.to_vec_f64().call().unwrap(), vec![2.5, 2.5, 2.5]);
    }

    #[test]
    fn column_from_scalar_bool() {
        let s = Scalar::from_bool(true);
        let col = Column::from_scalar(&s, 4).call().unwrap();
        assert_eq!(col.len(), 4);
        assert_eq!(
            col.to_vec_bool().call().unwrap(),
            vec![true, true, true, true]
        );
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
        let col = Column::from_scalar(&s, 3).call().unwrap();
        let view = col.view();
        assert_eq!(view.len(), 3);
        assert_eq!(view.type_id(), TypeId::INT32);
        assert!(!view.has_nulls());
    }

    #[test]
    fn column_null_mask() {
        let s = Scalar::from_i32(42);
        let col = Column::from_scalar(&s, 3).call().unwrap();
        let mask = col.null_mask_to_host().call().unwrap();
        assert_eq!(mask, vec![true, true, true]);
    }

    #[test]
    fn column_from_slice_i32() {
        let col = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::INT32);
        assert_eq!(col.to_vec_i32().call().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn column_from_slice_i64() {
        let col = Column::from_slice_i64(&[100, 200, 300]).call().unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::INT64);
        assert_eq!(col.to_vec_i64().call().unwrap(), vec![100, 200, 300]);
    }

    #[test]
    fn column_from_slice_f64() {
        let col = Column::from_slice_f64(&[1.5, 2.5, 3.5]).call().unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::FLOAT64);
        assert_eq!(col.to_vec_f64().call().unwrap(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn column_from_slice_bool() {
        let col = Column::from_slice_bool(&[true, false, true])
            .call()
            .unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_id(), TypeId::BOOL8);
        assert_eq!(col.to_vec_bool().call().unwrap(), vec![true, false, true]);
    }

    #[test]
    fn column_from_strings() {
        let col = Column::from_strings(&["hello", "world"]).call().unwrap();
        assert_eq!(col.len(), 2);
        assert_eq!(col.type_id(), TypeId::STRING);
        assert_eq!(col.to_vec_string().call().unwrap(), vec!["hello", "world"]);
    }

    #[test]
    fn column_from_timestamps_s() {
        let col = Column::from_timestamps_s(&[1704067200, 1718443845])
            .call()
            .unwrap();
        assert_eq!(col.len(), 2);
        assert_eq!(col.type_id(), TypeId::TIMESTAMP_SECONDS);
        assert_eq!(
            col.to_vec_i64().call().unwrap(),
            vec![1704067200, 1718443845]
        );
    }

    #[test]
    fn column_null_scalar() {
        let s = Scalar::null_i32();
        let col = Column::from_scalar(&s, 3).call().unwrap();
        assert_eq!(col.len(), 3);
        assert!(col.has_nulls());
        assert_eq!(col.null_count(), 3);
    }

    #[test]
    fn column_nullable() {
        let col = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
        // from_slice creates non-nullable columns
        assert!(!col.has_nulls());
    }

    #[test]
    fn column_view_offset() {
        let col = Column::from_slice_i32(&[1, 2, 3]).call().unwrap();
        assert_eq!(col.view().offset(), 0);
    }

    #[test]
    fn column_to_vec_f32() {
        let s = Scalar::from_f32(1.5);
        let col = Column::from_scalar(&s, 2).call().unwrap();
        assert_eq!(col.to_vec_f32().call().unwrap(), vec![1.5, 1.5]);
    }
}
