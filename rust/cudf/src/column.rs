// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU column types: owning [`Column`] and borrowed [`ColumnView`].

use cxx::UniquePtr;

use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::{Scalar, scalar_from_ffi};
use crate::stream::Stream;
use crate::{i32_to_usize, usize_to_i32};

#[doc(alias = "mask_state")]
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
/// An owning GPU column.
///
/// Wraps a `cudf::column` via the CXX FFI layer. Dropping this value
/// frees the underlying GPU memory.
pub struct Column(pub(crate) UniquePtr<cudf_sys::ffi::Column>);

impl Column {
    #[doc(alias = "make_column_from_scalar")]
    /// Creates a column by repeating a scalar value `count` times.
    pub fn from_scalar(scalar: &Scalar, count: usize) -> FromScalar<'_> {
        FromScalar {
            scalar,
            count,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_empty_column")]
    /// Creates an empty column of the given type.
    pub fn empty(type_id: TypeId) -> Self {
        Self(cudf_sys::ffi::make_empty_column_by_type(type_id.repr))
    }

    #[doc(alias = "make_fixed_width_column")]
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
    /// Creates an empty lists column with the given child element type.
    pub fn empty_lists(child_type: TypeId) -> EmptyLists {
        EmptyLists {
            child_type,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_dictionary_from_scalar")]
    /// Creates a dictionary column filled with a single scalar value.
    pub fn dictionary_from_scalar(scalar: &Scalar, count: usize) -> DictionaryFromScalar<'_> {
        DictionaryFromScalar {
            scalar,
            count,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_lists_column")]
    /// Creates a LIST column from offsets and child columns (no null mask).
    ///
    /// `offsets` must be an INT32 column of length `num_rows + 1`.
    /// `child` contains the flattened list elements.
    pub fn from_lists(num_rows: usize, offsets: Column, child: Column) -> FromLists {
        FromLists {
            num_rows,
            offsets,
            child,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_structs_column")]
    /// Creates a STRUCT column from child columns (no null mask).
    pub fn from_structs(num_rows: usize, children: Vec<Column>) -> FromStructs {
        FromStructs {
            num_rows,
            children,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "size")]
    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_size(&self.0))
    }

    /// Returns `true` if the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the count of null elements.
    pub fn null_count(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_null_count(&self.0))
    }

    /// Returns `true` if the column contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_has_nulls(&self.0)
    }

    #[doc(alias = "type")]
    /// Returns the type identifier of the column.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_type_id(&self.0);
        // C++ columns always have a valid type_id; fall back to EMPTY
        // for the theoretically-impossible out-of-range case.
        cudf_sys::type_id_from_i32(id).unwrap_or(TypeId::EMPTY)
    }

    /// Returns an immutable view of the column.
    pub fn view(&self) -> ColumnView<'_> {
        ColumnView(cudf_sys::ffi::column_view_of(&self.0))
    }

    /// Copies the column data to host as `Vec<i8>`.
    pub fn to_vec_i8(&self) -> ToVecI8<'_> {
        ToVecI8 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<i16>`.
    pub fn to_vec_i16(&self) -> ToVecI16<'_> {
        ToVecI16 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<i32>`.
    pub fn to_vec_i32(&self) -> ToVecI32<'_> {
        ToVecI32 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<i64>`.
    pub fn to_vec_i64(&self) -> ToVecI64<'_> {
        ToVecI64 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<f32>`.
    pub fn to_vec_f32(&self) -> ToVecF32<'_> {
        ToVecF32 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<f64>`.
    pub fn to_vec_f64(&self) -> ToVecF64<'_> {
        ToVecF64 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<u8>`.
    pub fn to_vec_u8(&self) -> ToVecU8<'_> {
        ToVecU8 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<u16>`.
    pub fn to_vec_u16(&self) -> ToVecU16<'_> {
        ToVecU16 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<u32>`.
    pub fn to_vec_u32(&self) -> ToVecU32<'_> {
        ToVecU32 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<u64>`.
    pub fn to_vec_u64(&self) -> ToVecU64<'_> {
        ToVecU64 {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data to host as `Vec<bool>`.
    pub fn to_vec_bool(&self) -> ToVecBool<'_> {
        ToVecBool {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Returns per-element validity as a host vector of bools.
    pub fn null_mask_to_host(&self) -> NullMaskToHost<'_> {
        NullMaskToHost {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    /// Copies string column data to a host vector of strings.
    pub fn to_vec_string(&self) -> ToVecString<'_> {
        ToVecString {
            view: cudf_sys::ffi::column_view_of(&self.0),
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i8")]
    /// Creates an INT8 column from a host slice.
    pub fn from_slice_i8(data: &[i8]) -> FromSliceI8<'_> {
        FromSliceI8 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i16")]
    /// Creates an INT16 column from a host slice.
    pub fn from_slice_i16(data: &[i16]) -> FromSliceI16<'_> {
        FromSliceI16 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i32")]
    /// Creates an INT32 column from a host slice.
    pub fn from_slice_i32(data: &[i32]) -> FromSliceI32<'_> {
        FromSliceI32 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_i64")]
    /// Creates an INT64 column from a host slice.
    pub fn from_slice_i64(data: &[i64]) -> FromSliceI64<'_> {
        FromSliceI64 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_f64")]
    /// Creates a FLOAT64 column from a host slice.
    pub fn from_slice_f64(data: &[f64]) -> FromSliceF64<'_> {
        FromSliceF64 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_f32")]
    /// Creates a FLOAT32 column from a host slice.
    pub fn from_slice_f32(data: &[f32]) -> FromSliceF32<'_> {
        FromSliceF32 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u8")]
    /// Creates a UINT8 column from a host slice.
    pub fn from_slice_u8(data: &[u8]) -> FromSliceU8<'_> {
        FromSliceU8 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u16")]
    /// Creates a UINT16 column from a host slice.
    pub fn from_slice_u16(data: &[u16]) -> FromSliceU16<'_> {
        FromSliceU16 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u32")]
    /// Creates a UINT32 column from a host slice.
    pub fn from_slice_u32(data: &[u32]) -> FromSliceU32<'_> {
        FromSliceU32 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_u64")]
    /// Creates a UINT64 column from a host slice.
    pub fn from_slice_u64(data: &[u64]) -> FromSliceU64<'_> {
        FromSliceU64 {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_bool")]
    /// Creates a BOOL8 column from a host slice.
    pub fn from_slice_bool(data: &[bool]) -> FromSliceBool<'_> {
        FromSliceBool {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_s")]
    /// Creates a `TIMESTAMP_SECONDS` column from epoch-second values.
    pub fn from_timestamps_s(data: &[i64]) -> FromTimestampsS<'_> {
        FromTimestampsS {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_ms")]
    /// Creates a `TIMESTAMP_MILLISECONDS` column from epoch-millisecond values.
    pub fn from_timestamps_ms(data: &[i64]) -> FromTimestampsMs<'_> {
        FromTimestampsMs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_us")]
    /// Creates a `TIMESTAMP_MICROSECONDS` column from epoch-microsecond values.
    pub fn from_timestamps_us(data: &[i64]) -> FromTimestampsUs<'_> {
        FromTimestampsUs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_timestamp_ns")]
    /// Creates a `TIMESTAMP_NANOSECONDS` column from epoch-nanosecond values.
    pub fn from_timestamps_ns(data: &[i64]) -> FromTimestampsNs<'_> {
        FromTimestampsNs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_s")]
    /// Creates a `DURATION_SECONDS` column from host data.
    pub fn from_durations_s(data: &[i64]) -> FromDurationsS<'_> {
        FromDurationsS {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_ms")]
    /// Creates a `DURATION_MILLISECONDS` column from host data.
    pub fn from_durations_ms(data: &[i64]) -> FromDurationsMs<'_> {
        FromDurationsMs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_us")]
    /// Creates a `DURATION_MICROSECONDS` column from host data.
    pub fn from_durations_us(data: &[i64]) -> FromDurationsUs<'_> {
        FromDurationsUs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_column_from_host_duration_ns")]
    /// Creates a `DURATION_NANOSECONDS` column from host data.
    pub fn from_durations_ns(data: &[i64]) -> FromDurationsNs<'_> {
        FromDurationsNs {
            data,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "make_string_column")]
    /// Creates a string column from a slice of strings.
    pub fn from_strings<'a>(values: &'a [&'a str]) -> FromStrings<'a> {
        FromStrings {
            values,
            stream: Stream::default_stream(),
        }
    }

    // -- In-place mutations --

    /// Fills the range `[begin, end)` with a scalar value in-place.
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

    /// Copies elements from `source[source_begin..source_end]` into `self`
    /// starting at `dest_begin`, in-place.
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

    /// Converts this column's null mask to a BOOL8 column
    /// (true = valid, false = null). If no mask is present, returns all-true.
    pub fn null_mask_to_bools(&self) -> NullMaskToBools<'_> {
        NullMaskToBools {
            col: self,
            stream: Stream::default_stream(),
        }
    }

    /// Sets this column's null mask from a BOOL8 column
    /// (true = valid, false = null).
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

    /// Returns a copy of this column with a null mask from a BOOL8 validity column.
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

/// Builder for [`ColumnView::cast`].
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

/// Builder for [`ColumnView::is_null`].
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

/// Builder for [`ColumnView::is_valid`].
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

/// Builder for [`ColumnView::is_nan`].
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

/// Builder for [`ColumnView::negate`].
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

/// Builder for [`ColumnView::abs`].
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

/// Builder for [`ColumnView::sum`].
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

/// Builder for [`ColumnView::min`].
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

/// Builder for [`ColumnView::max`].
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

/// Builder for [`ColumnView::product`].
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

/// Builder for [`ColumnView::any`].
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

/// Builder for [`ColumnView::all`].
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

/// Builder for [`ColumnView::nans_to_nulls`].
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

/// Builder for [`ColumnView::add`].
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

/// Builder for [`ColumnView::sub`].
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

/// Builder for [`ColumnView::mul`].
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

/// Builder for [`ColumnView::div`].
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

/// Builder for [`ColumnView::eq`].
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

/// Builder for [`ColumnView::ne`].
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

/// Builder for [`ColumnView::lt`].
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

/// Builder for [`ColumnView::gt`].
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

/// Builder for [`ColumnView::le`].
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

/// Builder for [`ColumnView::ge`].
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

/// Builder for [`ColumnView::unary_op`].
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

/// Builder for [`ColumnView::is_not_nan`].
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

/// Builder for [`ColumnView::round`].
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

/// Builder for [`ColumnView::contains_scalar`].
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

/// Builder for [`ColumnView::contains_column`].
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

/// Builder for [`ColumnView::rank`].
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

/// Builder for [`ColumnView::round_with_method`].
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
/// The lifetime parameter ties this view to the owning [`Table`](crate::table::Table).
pub struct ColumnView<'a>(pub(crate) &'a cudf_sys::ffi::column_view);

impl ColumnView<'_> {
    #[doc(alias = "size")]
    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_size(self.0))
    }

    /// Returns `true` if the view has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the count of null elements.
    pub fn null_count(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_null_count(self.0))
    }

    /// Returns `true` if the view contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_view_has_nulls(self.0)
    }

    /// Returns the offset of the view into the underlying data.
    pub fn offset(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_offset(self.0))
    }

    #[doc(alias = "type")]
    /// Returns the type identifier of the column.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_view_type_id(self.0);
        // C++ column views always have a valid type_id.
        cudf_sys::type_id_from_i32(id).unwrap_or(TypeId::EMPTY)
    }

    // -- Child column access --

    /// Returns the number of child columns (e.g. struct fields, list offsets/child).
    pub fn num_children(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_num_children(self.0))
    }

    /// Deep-copies a child column by index.
    ///
    /// For STRUCT columns: index 0..N-1 are the struct fields.
    /// For LIST columns: index 0 is offsets, index 1 is child values.
    /// For DICTIONARY columns: index 0 is indices, index 1 is keys.
    pub fn child(&self, index: usize) -> Child<'_> {
        Child {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }

    // -- Unary ops --

    /// Casts the column to a different type.
    pub fn cast(&self, target: TypeId) -> Cast<'_> {
        Cast {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a BOOL8 column where `true` indicates a null value.
    pub fn is_null(&self) -> IsNull<'_> {
        IsNull {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a BOOL8 column where `true` indicates a valid value.
    pub fn is_valid(&self) -> IsValid<'_> {
        IsValid {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a BOOL8 column where `true` indicates NaN.
    pub fn is_nan(&self) -> IsNan<'_> {
        IsNan {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Negates every element.
    pub fn negate(&self) -> Negate<'_> {
        Negate {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the absolute value of every element.
    pub fn abs(&self) -> Abs<'_> {
        Abs {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Reductions --

    /// Computes the sum of all elements.
    pub fn sum(&self, output_type: TypeId) -> Sum<'_> {
        Sum {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the minimum value.
    pub fn min(&self, output_type: TypeId) -> Min<'_> {
        Min {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the maximum value.
    pub fn max(&self, output_type: TypeId) -> Max<'_> {
        Max {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the product of all elements.
    pub fn product(&self, output_type: TypeId) -> Product<'_> {
        Product {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Returns true if any element is non-zero.
    pub fn any(&self) -> Any<'_> {
        Any {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns true if all elements are non-zero.
    pub fn all(&self) -> All<'_> {
        All {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Quantile --

    /// Computes quantiles of the column.
    pub fn quantile<'a>(&'a self, quantiles: &'a [f64]) -> Quantile<'a> {
        Quantile {
            view: self,
            quantiles,
            stream: Stream::default_stream(),
        }
    }

    /// Computes quantiles with a specified interpolation method.
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

    /// Creates an empty column with the same type.
    pub fn empty_like(&self) -> Column {
        Column(cudf_sys::copying::ffi::empty_like_column(self.0))
    }

    /// Creates an owning deep copy of this column view.
    ///
    /// This copies all device data (values, null mask, child columns)
    /// into a new independently-owned [`Column`].
    pub fn to_owned_column(&self) -> ToOwnedColumn<'_> {
        ToOwnedColumn {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Host data extraction (operates directly on view, no deep copy) --

    /// Copies the view data to host as `Vec<i8>`.
    pub fn to_vec_i8(&self) -> ToVecI8<'_> {
        ToVecI8 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<i16>`.
    pub fn to_vec_i16(&self) -> ToVecI16<'_> {
        ToVecI16 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<i32>`.
    pub fn to_vec_i32(&self) -> ToVecI32<'_> {
        ToVecI32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<i64>`.
    pub fn to_vec_i64(&self) -> ToVecI64<'_> {
        ToVecI64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<f32>`.
    pub fn to_vec_f32(&self) -> ToVecF32<'_> {
        ToVecF32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<f64>`.
    pub fn to_vec_f64(&self) -> ToVecF64<'_> {
        ToVecF64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<u8>`.
    pub fn to_vec_u8(&self) -> ToVecU8<'_> {
        ToVecU8 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<u16>`.
    pub fn to_vec_u16(&self) -> ToVecU16<'_> {
        ToVecU16 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<u32>`.
    pub fn to_vec_u32(&self) -> ToVecU32<'_> {
        ToVecU32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<u64>`.
    pub fn to_vec_u64(&self) -> ToVecU64<'_> {
        ToVecU64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view data to host as `Vec<bool>`.
    pub fn to_vec_bool(&self) -> ToVecBool<'_> {
        ToVecBool {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Returns per-element validity as a host vector of bools.
    pub fn null_mask_to_host(&self) -> NullMaskToHost<'_> {
        NullMaskToHost {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies string view data to a host vector of strings.
    pub fn to_vec_string(&self) -> ToVecString<'_> {
        ToVecString {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }

    // -- Transform --

    /// Converts NaN values to null in a floating-point column.
    pub fn nans_to_nulls(&self) -> NansToNulls<'_> {
        NansToNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct values in this column.
    ///
    /// - `include_nulls`: whether null values count as a distinct value.
    /// - `nan_is_null`: whether NaN values are treated as null.
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

    /// Counts consecutive unique values in the column.
    pub fn unique_count(&self, include_nulls: bool, nan_is_null: bool) -> ColumnUniqueCount<'_> {
        ColumnUniqueCount {
            view: self,
            include_nulls,
            nan_is_null,
            stream: Stream::default_stream(),
        }
    }

    /// Computes approximate percentiles from a t-digest column.
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

    /// Element-wise addition with another column.
    pub fn add<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Add<'a> {
        Add {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise subtraction.
    pub fn sub<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Sub<'a> {
        Sub {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise multiplication.
    pub fn mul<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Mul<'a> {
        Mul {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise division.
    pub fn div<'a>(&'a self, rhs: &'a ColumnView<'_>, output_type: TypeId) -> Div<'a> {
        Div {
            view: self,
            rhs,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise equality comparison.
    pub fn eq<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Eq<'a> {
        Eq {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise not-equal comparison.
    pub fn ne<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Ne<'a> {
        Ne {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise less-than comparison.
    pub fn lt<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Lt<'a> {
        Lt {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise greater-than comparison.
    pub fn gt<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Gt<'a> {
        Gt {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise less-than-or-equal comparison.
    pub fn le<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Le<'a> {
        Le {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    /// Element-wise greater-than-or-equal comparison.
    pub fn ge<'a>(&'a self, rhs: &'a ColumnView<'_>) -> Ge<'a> {
        Ge {
            view: self,
            rhs,
            stream: Stream::default_stream(),
        }
    }

    // -- Generic unary operation --

    /// Applies a unary operation to this column.
    pub fn unary_op(&self, op: crate::ops::UnaryOperator) -> UnaryOp<'_> {
        UnaryOp {
            view: self,
            op,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a BOOL8 column where `true` indicates a non-NaN value.
    pub fn is_not_nan(&self) -> IsNotNan<'_> {
        IsNotNan {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Math convenience methods --

    /// Computes the sine of each element.
    pub fn sin(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::SIN)
    }
    /// Computes the cosine of each element.
    pub fn cos(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::COS)
    }
    /// Computes the tangent of each element.
    pub fn tan(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::TAN)
    }
    /// Computes the arcsine of each element.
    pub fn arcsin(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCSIN)
    }
    /// Computes the arccosine of each element.
    pub fn arccos(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::ARCCOS)
    }
    /// Computes the arctangent of each element.
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
    /// Computes e^x for each element.
    pub fn exp(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::EXP)
    }
    /// Computes the natural log of each element.
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
    /// Computes the ceiling of each element.
    pub fn ceil(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::CEIL)
    }
    /// Computes the floor of each element.
    pub fn floor(&self) -> UnaryOp<'_> {
        self.unary_op(crate::ops::UnaryOperator::FLOOR)
    }
    /// Rounds each element to the nearest integer (round half to even).
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
    pub fn round(&self, decimal_places: i32) -> Round<'_> {
        Round {
            view: self,
            decimal_places,
            stream: Stream::default_stream(),
        }
    }

    /// Rounds with a specific rounding method.
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

    /// Computes the mean of all elements.
    pub fn mean(&self, output_type: TypeId) -> Mean<'_> {
        Mean {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the standard deviation.
    pub fn std_dev(&self, output_type: TypeId, ddof: i32) -> StdDev<'_> {
        StdDev {
            view: self,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the variance.
    pub fn variance(&self, output_type: TypeId, ddof: i32) -> Variance<'_> {
        Variance {
            view: self,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the median.
    pub fn median(&self, output_type: TypeId) -> Median<'_> {
        Median {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of unique elements.
    pub fn nunique(&self) -> Nunique<'_> {
        Nunique {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the minimum via minmax.
    pub fn minmax_min(&self) -> MinmaxMin<'_> {
        MinmaxMin {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the maximum via minmax.
    pub fn minmax_max(&self) -> MinmaxMax<'_> {
        MinmaxMax {
            view: self,
            stream: Stream::default_stream(),
        }
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
    ) -> Reduce<'_> {
        Reduce {
            view: self,
            agg,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Generic reduce with an initial value (supports SUM, PRODUCT, MIN, MAX, ANY, ALL).
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

    /// Segmented reduce: reduces each segment defined by offsets.
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

    /// Segmented reduce with an initial value.
    ///
    /// Only SUM, PRODUCT, MIN, MAX, ANY, and ALL aggregations are supported.
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

    /// Computes a prefix scan (cumulative operation).
    pub fn scan(&self, agg_kind: crate::groupby::AggregationKind, inclusive: bool) -> Scan<'_> {
        Scan {
            view: self,
            agg_kind,
            inclusive,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying extras --

    /// Shifts column values by offset, filling with the given scalar.
    pub fn shift<'a>(&'a self, offset: i32, fill_value: &'a Scalar) -> Shift<'a> {
        Shift {
            view: self,
            offset,
            fill_value,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a single element as a scalar.
    pub fn get_element(&self, index: usize) -> GetElement<'_> {
        GetElement {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }

    /// Reverses the elements of this column.
    pub fn reverse(&self) -> Reverse<'_> {
        Reverse {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Extracts a slice [begin, end) as a new owned column.
    pub fn slice(&self, begin: usize, end: usize) -> Slice<'_> {
        Slice {
            view: self,
            begin,
            end,
            stream: Stream::default_stream(),
        }
    }

    /// Selects elements from lhs (self) or rhs based on a boolean mask.
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

    /// Assigns bin labels to column values.
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

    /// Checks if a scalar value exists in this column.
    pub fn contains_scalar<'a>(&'a self, needle: &'a Scalar) -> ContainsScalar<'a> {
        ContainsScalar {
            view: self,
            needle,
            stream: Stream::default_stream(),
        }
    }

    /// Checks which values from `needles` exist in this column.
    pub fn contains_column<'a>(&'a self, needles: &'a ColumnView<'_>) -> ContainsColumn<'a> {
        ContainsColumn {
            view: self,
            needles,
            stream: Stream::default_stream(),
        }
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

    /// Top k values of the column.
    pub fn top_k(&self, k: usize, order: crate::sorting::Order) -> TopK<'_> {
        TopK {
            view: self,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Top k indices of the column.
    pub fn top_k_order(&self, k: usize, order: crate::sorting::Order) -> TopKOrder<'_> {
        TopKOrder {
            view: self,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Segmented top k values.
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

    /// Segmented top k indices.
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

    /// Copy range of elements from this column into target.
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

    /// Create uninitialized column of same type and size.
    pub fn allocate_like(&self) -> AllocateLike<'_> {
        AllocateLike {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Check if this column has non-empty null rows (LIST/STRING).
    pub fn has_nonempty_nulls(&self) -> HasNonemptyNulls<'_> {
        HasNonemptyNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Check if this column *may* have non-empty data in null rows.
    /// This is a fast check (no stream needed) that may return false positives.
    pub fn may_have_nonempty_nulls(&self) -> bool {
        cudf_sys::copying::ffi::may_have_nonempty_nulls(self.0)
    }

    /// Purge non-empty null row contents.
    pub fn purge_nonempty_nulls(&self) -> PurgeNonemptyNulls<'_> {
        PurgeNonemptyNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Replace (new) --

    /// Replace nulls using preceding/following policy.
    pub fn replace_nulls_policy(&self, preceding: bool) -> ReplaceNullsPolicy<'_> {
        ReplaceNullsPolicy {
            view: self,
            preceding,
            stream: Stream::default_stream(),
        }
    }

    /// Clamp with separate replacement values for lo and hi.
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

    /// Normalize NaNs and zeros (convert -NaN to NaN, -0.0 to 0.0).
    pub fn normalize_nans_and_zeros(&self) -> NormalizeNansAndZeros<'_> {
        NormalizeNansAndZeros {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Fill --

    /// Fills the range [begin, end) with a scalar value (out-of-place).
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
