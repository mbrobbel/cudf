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
//! The preferred safe path is to allocate columns from an explicit
//! [`rmm::gpu_context::GpuContext`] via `*_in` constructors. The
//! [`UnboundColumn`] type remains available as a legacy/raw-owner escape hatch
//! for the older ambient-allocation model.
//!
//! All GPU operations follow the builder pattern and implement
//! [`GpuOp`](crate::stream::GpuOp). Call [`.call()`](crate::stream::GpuOp::call)
//! to execute, or bind an allocator first and use
//! [`.call_on()`](crate::stream::AllocatedGpuOp::call_on) to execute on an
//! explicit context stream.
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::stream::GpuOp;
//! use cudf::stream::GpuOpExt;
//! use rmm::device::current_device;
//! use rmm::gpu_context::GpuContext;
//!
//! let ctx = GpuContext::<()>::new(current_device())?;
//! let alloc = ctx.default_device_allocator();
//! let exec = ctx.default_stream();
//!
//! let col = Column::from_slice_i32_in(&alloc, &[1, 2, 3])?;
//! let reversed = col.view().reverse().in_alloc(&alloc).call_on(&exec)?;
//! assert_eq!(reversed.to_vec_i32().call()?, vec![3, 2, 1]);
//! # Ok::<(), cudf::error::Error>(())
//! ```

use cxx::UniquePtr;
use rmm::gpu_context::{Allocator, ContextBound};

use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::{Scalar, scalar_from_ffi};
use crate::stream::GpuOpExt;
use crate::stream::Stream;
use crate::{i32_to_usize, usize_to_i32};

/// A column bound to an explicit allocator/context lifetime.
pub type Column<'ctx> = OwnedColumn<'ctx>;

/// Backward-compatible alias for the bound safe-column surface.
pub type OwnedColumn<'ctx> = BoundColumn<'ctx, ()>;

/// A legacy unbound owning column.
///
/// Prefer [`Column<'_>`] for the safe context-bound surface. `UnboundColumn`
/// is kept for compatibility with the older ambient-allocation model and for
/// bridging APIs that have not yet been fully migrated.
pub type UnboundColumn = RawColumn<UniquePtr<cudf_sys::ffi::Column>>;

#[doc(hidden)]
pub type BoundColumn<'ctx, Brand> =
    RawColumn<ContextBound<'ctx, Brand, UniquePtr<cudf_sys::ffi::Column>>>;

#[doc(hidden)]
pub trait ColumnOwner {
    fn as_unique_ptr(&self) -> &UniquePtr<cudf_sys::ffi::Column>;
}

impl ColumnOwner for UniquePtr<cudf_sys::ffi::Column> {
    fn as_unique_ptr(&self) -> &UniquePtr<cudf_sys::ffi::Column> {
        self
    }
}

impl<Brand> ColumnOwner for ContextBound<'_, Brand, UniquePtr<cudf_sys::ffi::Column>> {
    fn as_unique_ptr(&self) -> &UniquePtr<cudf_sys::ffi::Column> {
        self
    }
}

#[doc(hidden)]
pub trait IntoColumnRaw {
    fn into_unique_ptr(self) -> UniquePtr<cudf_sys::ffi::Column>;
}

impl IntoColumnRaw for UniquePtr<cudf_sys::ffi::Column> {
    fn into_unique_ptr(self) -> UniquePtr<cudf_sys::ffi::Column> {
        self
    }
}

impl<Brand> IntoColumnRaw for ContextBound<'_, Brand, UniquePtr<cudf_sys::ffi::Column>> {
    fn into_unique_ptr(self) -> UniquePtr<cudf_sys::ffi::Column> {
        self.into_inner()
    }
}

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
                Ok($ffi_fn(self.view, self.stream.as_raw())?)
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
            type Output = UnboundColumn;
            fn stream(mut self, stream: Stream) -> Self {
                self.stream = stream;
                self
            }
            fn call(self) -> Result<Self::Output> {
                Ok(RawColumn($ffi_fn(self.data, self.stream.as_raw())?))
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

macro_rules! allocator_constructor {
    ($(fn $name:ident => $builder:ident($($arg:ident : $argty:ty),*);)*) => {
        $(
            #[doc = concat!(
                "Executes [`",
                stringify!($builder),
                "`](Self::",
                stringify!($builder),
                ") in an explicit allocator context."
            )]
            pub fn $name<'ctx, Brand>(
                alloc: &Allocator<'ctx, Brand>,
                $($arg: $argty),*
            ) -> Result<BoundColumn<'ctx, Brand>> {
                Self::$builder($($arg),*).call_in(alloc)
            }
        )*
    };
}

macro_rules! allocator_forwarding_constructor {
    ($(fn $name:ident => $builder:ident($($arg:ident : $argty:ty),*);)*) => {
        $(
            #[doc = concat!(
                "Executes [`",
                stringify!($builder),
                "`](UnboundColumn::",
                stringify!($builder),
                ") in an explicit allocator context."
            )]
            pub fn $name<'ctx, Brand>(
                alloc: &Allocator<'ctx, Brand>,
                $($arg: $argty),*
            ) -> Result<BoundColumn<'ctx, Brand>> {
                UnboundColumn::$builder($($arg),*).call_in(alloc)
            }
        )*
    };
}

macro_rules! legacy_builder_forwarder {
    ($(fn $name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)*) => {
        $(
            #[doc = concat!(
                "Returns the legacy builder produced by [`UnboundColumn::",
                stringify!($name),
                "`]."
            )]
            pub fn $name($($arg: $argty),*) -> $ret {
                UnboundColumn::$name($($arg),*)
            }
        )*
    };
}

/// Builder for [`Column::from_scalar`].
pub struct FromScalar<'a> {
    scalar: &'a Scalar,
    count: usize,
    stream: Stream,
}
impl crate::stream::GpuOp for FromScalar<'_> {
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.scalar)?;
        Ok(RawColumn(cudf_sys::ffi::make_column_from_scalar(
            &ffi,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?))
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
    type Output = UnboundColumn;
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
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::empty_lists`].
pub struct EmptyLists {
    child_type: TypeId,
    stream: Stream,
}
impl crate::stream::GpuOp for EmptyLists {
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::make_empty_lists_column(self.child_type.repr, self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::dictionary_from_scalar`].
pub struct DictionaryFromScalar<'a> {
    scalar: &'a Scalar,
    count: usize,
    stream: Stream,
}
impl crate::stream::GpuOp for DictionaryFromScalar<'_> {
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.scalar)?;
        let c = cudf_sys::ffi::make_dictionary_from_scalar(
            &ffi,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::from_lists`].
pub struct FromLists {
    num_rows: usize,
    offsets: UnboundColumn,
    child: UnboundColumn,
    stream: Stream,
}
impl crate::stream::GpuOp for FromLists {
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        if self.offsets.type_id() != TypeId::INT32 {
            return Err(crate::error::Error::InvalidArgument(
                "offsets column must have type INT32".into(),
            ));
        }
        let expected_len = self.num_rows.saturating_add(1);
        if self.offsets.len() != expected_len {
            return Err(crate::error::Error::InvalidArgument(format!(
                "offsets column length must be num_rows + 1 (expected {expected_len}, got {})",
                self.offsets.len()
            )));
        }
        let c = cudf_sys::ffi::make_lists_column(
            usize_to_i32(self.num_rows),
            self.offsets.0,
            self.child.0,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::from_structs`].
pub struct FromStructs {
    num_rows: usize,
    children: Vec<UnboundColumn>,
    stream: Stream,
}
impl crate::stream::GpuOp for FromStructs {
    type Output = UnboundColumn;
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
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::from_strings`].
pub struct FromStrings<'a> {
    values: &'a [&'a str],
    stream: Stream,
}
impl crate::stream::GpuOp for FromStrings<'_> {
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        // Build a flat chars buffer and an offsets array so that the FFI
        // function creates the GPU column in one shot (two H2D copies)
        // instead of N scalar allocations + concatenation.
        let total_bytes: usize = self.values.iter().map(|s| s.len()).sum();
        let mut chars = Vec::with_capacity(total_bytes);
        let mut offsets = Vec::with_capacity(self.values.len() + 1);
        let mut offset: i32 = 0;
        for s in self.values {
            offsets.push(offset);
            chars.extend_from_slice(s.as_bytes());
            offset = offset
                .checked_add(i32::try_from(s.len()).map_err(|_| {
                    crate::error::Error::InvalidArgument("string length exceeds i32::MAX".into())
                })?)
                .ok_or_else(|| {
                    crate::error::Error::InvalidArgument(
                        "total string bytes exceed i32::MAX".into(),
                    )
                })?;
        }
        offsets.push(offset);
        Ok(RawColumn(
            cudf_sys::strings::ffi::make_string_column_from_offsets(
                &chars,
                &offsets,
                self.stream.as_raw(),
            )?,
        ))
    }
}

/// Builder for [`Column::fill_in_place`].
pub struct FillInPlace<'a> {
    col: &'a mut UnboundColumn,
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
        let ffi = crate::scalar::scalar_to_ffi(self.value)?;
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
    col: &'a mut UnboundColumn,
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
pub struct NullMaskToBools<'a, Raw = UniquePtr<cudf_sys::ffi::Column>> {
    col: &'a RawColumn<Raw>,
    stream: Stream,
}
impl<Raw> crate::stream::GpuOp for NullMaskToBools<'_, Raw>
where
    Raw: ColumnOwner,
{
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let v = self.col.view();
        let c = cudf_sys::ffi::null_mask_to_bools(v.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::set_null_mask_from_bools`].
pub struct SetNullMaskFromBools<'a> {
    col: &'a mut UnboundColumn,
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
pub struct WithNullMaskFromBools<'a, Raw = UniquePtr<cudf_sys::ffi::Column>> {
    col: &'a RawColumn<Raw>,
    validity: &'a ColumnView<'a>,
    stream: Stream,
}
impl<Raw> crate::stream::GpuOp for WithNullMaskFromBools<'_, Raw>
where
    Raw: ColumnOwner,
{
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::column_with_null_mask_from_bools(
            self.col.0.as_unique_ptr(),
            self.validity.0,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`Column::with_null_mask`].
pub struct WithNullMask<'a, Raw = UniquePtr<cudf_sys::ffi::Column>> {
    col: &'a RawColumn<Raw>,
    mask_bytes: &'a [u8],
    null_count: i32,
    stream: Stream,
}
impl<Raw> crate::stream::GpuOp for WithNullMask<'_, Raw>
where
    Raw: ColumnOwner,
{
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::ffi::column_with_null_mask(
            self.col.0.as_unique_ptr(),
            self.mask_bytes,
            self.null_count,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::to_owned_column`].
pub struct ToOwnedColumn<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}
impl crate::stream::GpuOp for ToOwnedColumn<'_> {
    type Output = UnboundColumn;
    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }
    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::copy_column(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
/// ```no_run
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
pub struct RawColumn<Raw = UniquePtr<cudf_sys::ffi::Column>>(pub(crate) Raw);

impl UnboundColumn {
    allocator_constructor! {
        fn from_scalar_in => from_scalar(scalar: &Scalar, count: usize);
        fn fixed_width_in => fixed_width(type_id: TypeId, scale: i32, num_rows: usize, mask_state: MaskState);
        fn empty_lists_in => empty_lists(child_type: TypeId);
        fn dictionary_from_scalar_in => dictionary_from_scalar(scalar: &Scalar, count: usize);
        fn from_slice_i8_in => from_slice_i8(data: &[i8]);
        fn from_slice_i16_in => from_slice_i16(data: &[i16]);
        fn from_slice_i32_in => from_slice_i32(data: &[i32]);
        fn from_slice_i64_in => from_slice_i64(data: &[i64]);
        fn from_slice_f32_in => from_slice_f32(data: &[f32]);
        fn from_slice_f64_in => from_slice_f64(data: &[f64]);
        fn from_slice_u8_in => from_slice_u8(data: &[u8]);
        fn from_slice_u16_in => from_slice_u16(data: &[u16]);
        fn from_slice_u32_in => from_slice_u32(data: &[u32]);
        fn from_slice_u64_in => from_slice_u64(data: &[u64]);
        fn from_slice_bool_in => from_slice_bool(data: &[bool]);
        fn from_timestamps_s_in => from_timestamps_s(data: &[i64]);
        fn from_timestamps_ms_in => from_timestamps_ms(data: &[i64]);
        fn from_timestamps_us_in => from_timestamps_us(data: &[i64]);
        fn from_timestamps_ns_in => from_timestamps_ns(data: &[i64]);
        fn from_durations_s_in => from_durations_s(data: &[i64]);
        fn from_durations_ms_in => from_durations_ms(data: &[i64]);
        fn from_durations_us_in => from_durations_us(data: &[i64]);
        fn from_durations_ns_in => from_durations_ns(data: &[i64]);
        fn from_strings_in => from_strings(values: &[&str]);
    }
}

impl RawColumn<ContextBound<'_, (), UniquePtr<cudf_sys::ffi::Column>>> {
    allocator_forwarding_constructor! {
        fn from_scalar_in => from_scalar(scalar: &Scalar, count: usize);
        fn fixed_width_in => fixed_width(type_id: TypeId, scale: i32, num_rows: usize, mask_state: MaskState);
        fn empty_lists_in => empty_lists(child_type: TypeId);
        fn dictionary_from_scalar_in => dictionary_from_scalar(scalar: &Scalar, count: usize);
        fn from_slice_i8_in => from_slice_i8(data: &[i8]);
        fn from_slice_i16_in => from_slice_i16(data: &[i16]);
        fn from_slice_i32_in => from_slice_i32(data: &[i32]);
        fn from_slice_i64_in => from_slice_i64(data: &[i64]);
        fn from_slice_f32_in => from_slice_f32(data: &[f32]);
        fn from_slice_f64_in => from_slice_f64(data: &[f64]);
        fn from_slice_u8_in => from_slice_u8(data: &[u8]);
        fn from_slice_u16_in => from_slice_u16(data: &[u16]);
        fn from_slice_u32_in => from_slice_u32(data: &[u32]);
        fn from_slice_u64_in => from_slice_u64(data: &[u64]);
        fn from_slice_bool_in => from_slice_bool(data: &[bool]);
        fn from_timestamps_s_in => from_timestamps_s(data: &[i64]);
        fn from_timestamps_ms_in => from_timestamps_ms(data: &[i64]);
        fn from_timestamps_us_in => from_timestamps_us(data: &[i64]);
        fn from_timestamps_ns_in => from_timestamps_ns(data: &[i64]);
        fn from_durations_s_in => from_durations_s(data: &[i64]);
        fn from_durations_ms_in => from_durations_ms(data: &[i64]);
        fn from_durations_us_in => from_durations_us(data: &[i64]);
        fn from_durations_ns_in => from_durations_ns(data: &[i64]);
        fn from_strings_in => from_strings(values: &[&str]);
    }

    legacy_builder_forwarder! {
        fn from_scalar(scalar: &Scalar, count: usize) -> FromScalar<'_>;
        fn fixed_width(type_id: TypeId, scale: i32, num_rows: usize, mask_state: MaskState) -> FixedWidth;
        fn empty_lists(child_type: TypeId) -> EmptyLists;
        fn dictionary_from_scalar(scalar: &Scalar, count: usize) -> DictionaryFromScalar<'_>;
        fn from_slice_i8(data: &[i8]) -> FromSliceI8<'_>;
        fn from_slice_i16(data: &[i16]) -> FromSliceI16<'_>;
        fn from_slice_i32(data: &[i32]) -> FromSliceI32<'_>;
        fn from_slice_i64(data: &[i64]) -> FromSliceI64<'_>;
        fn from_slice_f32(data: &[f32]) -> FromSliceF32<'_>;
        fn from_slice_f64(data: &[f64]) -> FromSliceF64<'_>;
        fn from_slice_u8(data: &[u8]) -> FromSliceU8<'_>;
        fn from_slice_u16(data: &[u16]) -> FromSliceU16<'_>;
        fn from_slice_u32(data: &[u32]) -> FromSliceU32<'_>;
        fn from_slice_u64(data: &[u64]) -> FromSliceU64<'_>;
        fn from_slice_bool(data: &[bool]) -> FromSliceBool<'_>;
        fn from_timestamps_s(data: &[i64]) -> FromTimestampsS<'_>;
        fn from_timestamps_ms(data: &[i64]) -> FromTimestampsMs<'_>;
        fn from_timestamps_us(data: &[i64]) -> FromTimestampsUs<'_>;
        fn from_timestamps_ns(data: &[i64]) -> FromTimestampsNs<'_>;
        fn from_durations_s(data: &[i64]) -> FromDurationsS<'_>;
        fn from_durations_ms(data: &[i64]) -> FromDurationsMs<'_>;
        fn from_durations_us(data: &[i64]) -> FromDurationsUs<'_>;
        fn from_durations_ns(data: &[i64]) -> FromDurationsNs<'_>;
    }

    /// Returns the legacy builder produced by [`UnboundColumn::from_strings`].
    pub fn from_strings<'a>(values: &'a [&'a str]) -> FromStrings<'a> {
        UnboundColumn::from_strings(values)
    }

    /// Creates an empty unbound column of `type_id`.
    pub fn empty(type_id: TypeId) -> Result<UnboundColumn> {
        UnboundColumn::empty(type_id)
    }

    /// Returns the legacy builder produced by [`UnboundColumn::from_lists`].
    pub fn from_lists(num_rows: usize, offsets: UnboundColumn, child: UnboundColumn) -> FromLists {
        UnboundColumn::from_lists(num_rows, offsets, child)
    }

    /// Returns the legacy builder produced by [`UnboundColumn::from_structs`].
    pub fn from_structs(num_rows: usize, children: Vec<UnboundColumn>) -> FromStructs {
        UnboundColumn::from_structs(num_rows, children)
    }
}

impl UnboundColumn {
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
    /// ```no_run
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
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    ///
    /// let col = Column::empty(TypeId::INT32)?;
    /// assert!(col.is_empty());
    /// assert_eq!(col.type_id(), TypeId::INT32);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn empty(type_id: TypeId) -> Result<UnboundColumn> {
        Ok(RawColumn(cudf_sys::ffi::make_empty_column_by_type(
            type_id.repr,
        )?))
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    /// ```no_run
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
    pub fn from_lists(num_rows: usize, offsets: UnboundColumn, child: UnboundColumn) -> FromLists {
        FromLists {
            num_rows,
            offsets,
            child,
            stream: Stream::default_stream(),
        }
    }

    /// Creates a LIST column in `alloc` from allocator-bound offsets and child
    /// columns.
    pub fn from_lists_in<'ctx, Brand>(
        alloc: &Allocator<'ctx, Brand>,
        num_rows: usize,
        offsets: BoundColumn<'ctx, Brand>,
        child: BoundColumn<'ctx, Brand>,
    ) -> Result<BoundColumn<'ctx, Brand>> {
        let unbound_offsets = offsets.0.into_inner();
        let unbound_child = child.0.into_inner();
        UnboundColumn::from_lists(
            num_rows,
            RawColumn(unbound_offsets),
            RawColumn(unbound_child),
        )
        .call_in(alloc)
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
    pub fn from_structs(num_rows: usize, children: Vec<UnboundColumn>) -> FromStructs {
        FromStructs {
            num_rows,
            children,
            stream: Stream::default_stream(),
        }
    }

    /// Binds this unbound column to `alloc`'s context lifetime.
    ///
    /// This is a transitional bridge from the legacy unbound owner surface to
    /// the explicit allocator-bound safe path.
    pub fn into_owned_in<'ctx>(self, alloc: &Allocator<'ctx>) -> OwnedColumn<'ctx> {
        RawColumn(alloc.bind(self.0))
    }
}

impl<Raw> RawColumn<Raw>
where
    Raw: ColumnOwner,
{
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
        i32_to_usize(cudf_sys::ffi::column_size(self.0.as_unique_ptr()))
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
        i32_to_usize(cudf_sys::ffi::column_null_count(self.0.as_unique_ptr()))
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
        cudf_sys::ffi::column_has_nulls(self.0.as_unique_ptr())
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
        let id = cudf_sys::ffi::column_type_id(self.0.as_unique_ptr());
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
        ColumnView(cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()))
    }

    /// Copies the column data from GPU to host as `Vec<i8>`.
    ///
    /// The column should have type [`TypeId::INT8`]. The returned vector
    /// has one element per row; null values are included but their values
    /// are unspecified. Use [`null_mask_to_host`](Column::null_mask_to_host)
    /// to determine which elements are valid.
    pub fn to_vec_i8(&self) -> ToVecI8<'_> {
        ToVecI8 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<i16>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_i16(&self) -> ToVecI16<'_> {
        ToVecI16 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
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
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
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
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<f32>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_f32(&self) -> ToVecF32<'_> {
        ToVecF32 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
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
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u8>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u8(&self) -> ToVecU8<'_> {
        ToVecU8 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u16>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u16(&self) -> ToVecU16<'_> {
        ToVecU16 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u32>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u32(&self) -> ToVecU32<'_> {
        ToVecU32 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<u64>`.
    ///
    /// See [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_u64(&self) -> ToVecU64<'_> {
        ToVecU64 {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
            stream: Stream::default_stream(),
        }
    }

    /// Copies the column data from GPU to host as `Vec<bool>`.
    ///
    /// The column should have type [`TypeId::BOOL8`]. See
    /// [`to_vec_i8`](Column::to_vec_i8) for details on null handling.
    pub fn to_vec_bool(&self) -> ToVecBool<'_> {
        ToVecBool {
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
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
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
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
            view: cudf_sys::ffi::column_view_of(self.0.as_unique_ptr()),
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
    pub fn null_mask_to_bools(&self) -> NullMaskToBools<'_, Raw> {
        NullMaskToBools {
            col: self,
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
    ) -> WithNullMaskFromBools<'a, Raw> {
        WithNullMaskFromBools {
            col: self,
            validity,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a new column with the same data as `self` but a null mask
    /// set from a raw bitmask buffer.
    ///
    /// `mask_bytes` is an Arrow-compatible LSB-first validity bitmask: bit
    /// *i* is set when row *i* is valid. `null_count` is the pre-computed
    /// number of null (unset) bits.
    ///
    /// This is more efficient than [`with_null_mask_from_bools`](Column::with_null_mask_from_bools)
    /// when the caller already has a packed bitmask (e.g. from Arrow),
    /// because it avoids creating an intermediate boolean column and the
    /// `bools_to_mask` GPU kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    #[doc(alias = "set_null_mask")]
    pub fn with_null_mask<'a>(
        &'a self,
        mask_bytes: &'a [u8],
        null_count: i32,
    ) -> WithNullMask<'a, Raw> {
        WithNullMask {
            col: self,
            mask_bytes,
            null_count,
            stream: Stream::default_stream(),
        }
    }
}

impl UnboundColumn {
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
    #[doc(alias = "make_string_column_from_offsets")]
    /// Creates a `STRING` column by copying `values` from host to GPU.
    ///
    /// All strings are concatenated into a single chars buffer and paired
    /// with an offsets array, so the GPU column is created in one shot
    /// (two host-to-device copies) rather than one allocation per string.
    ///
    /// Each element in `values` becomes one row. The resulting column has
    /// no null mask; all rows are valid.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidArgument`](crate::error::Error::InvalidArgument)
    /// if the total byte length of all strings exceeds `i32::MAX`.
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::unary::ffi::unary_cast(self.view.0, self.target.repr, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_null(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_valid(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_nan(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::negate`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Negate<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Negate<'_> {
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_negate(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::abs`]. Executes via
/// [`.call()`](crate::stream::GpuOp::call), producing an owned [`Column`].
pub struct Abs<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Abs<'_> {
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_abs(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::nans_to_nulls`]. See that method for details.
pub struct NansToNulls<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for NansToNulls<'_> {
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let result = cudf_sys::transform::ffi::nans_to_nulls(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(result))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(col))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::unary::ffi::unary_operation(self.view.0, self.op.repr, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::unary::ffi::unary_is_not_nan(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
        let init_ffi = crate::scalar::scalar_to_ffi(self.init)?;
        let s = cudf_sys::reduction::ffi::reduce_with_init(
            self.view.0,
            self.agg.repr,
            self.ddof,
            self.output_type.repr,
            &init_ffi,
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let null_handling = i32::from(!self.exclude_nulls);
        let init_ffi = crate::scalar::scalar_to_ffi(self.init)?;
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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.fill_value)?;
        let c = cudf_sys::copying::ffi::shift_column(
            self.view.0,
            self.offset,
            &ffi,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::reverse`].
pub struct Reverse<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for Reverse<'_> {
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::reverse_column(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
        let ffi = crate::scalar::scalar_to_ffi(self.needle)?;
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let ffi = crate::scalar::scalar_to_ffi(self.value)?;
        let c = cudf_sys::filling::ffi::fill_column(
            self.view.0,
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            &ffi,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::allocate_like`].
pub struct AllocateLike<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for AllocateLike<'_> {
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::allocate_like_column(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c = cudf_sys::copying::ffi::purge_nonempty_nulls(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::replace_nulls_policy`].
pub struct ReplaceNullsPolicy<'a> {
    view: &'a ColumnView<'a>,
    preceding: bool,
    stream: Stream,
}

impl crate::stream::GpuOp for ReplaceNullsPolicy<'_> {
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let lo_ffi = crate::scalar::scalar_to_ffi(self.lo)?;
        let lo_r_ffi = crate::scalar::scalar_to_ffi(self.lo_replace)?;
        let hi_ffi = crate::scalar::scalar_to_ffi(self.hi)?;
        let hi_r_ffi = crate::scalar::scalar_to_ffi(self.hi_replace)?;
        let c = cudf_sys::replace::ffi::clamp_column_with_replace(
            self.view.0,
            &lo_ffi,
            &lo_r_ffi,
            &hi_ffi,
            &hi_r_ffi,
            self.stream.as_raw(),
        )?;
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::normalize_nans_and_zeros`].
pub struct NormalizeNansAndZeros<'a> {
    view: &'a ColumnView<'a>,
    stream: Stream,
}

impl crate::stream::GpuOp for NormalizeNansAndZeros<'_> {
    type Output = UnboundColumn;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let c =
            cudf_sys::replace::ffi::normalize_nans_and_zeros(self.view.0, self.stream.as_raw())?;
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
    }
}

/// Builder for [`ColumnView::child`].
pub struct Child<'a> {
    view: &'a ColumnView<'a>,
    index: usize,
    stream: Stream,
}

impl crate::stream::GpuOp for Child<'_> {
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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
    type Output = UnboundColumn;

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
        Ok(RawColumn(c))
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

#[path = "column/view_impl.rs"]
mod view_impl;
#[path = "column/view_math_impl.rs"]
mod view_math_impl;
#[path = "column/view_reduce_impl.rs"]
mod view_reduce_impl;

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
        let col = Column::empty(TypeId::INT32).unwrap();
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

    #[test]
    fn fixed_width_rejects_non_fixed_width_type() {
        let result = Column::fixed_width(TypeId::STRING, 0, 4, MaskState::AllValid).call();
        assert!(result.is_err());
    }

    #[test]
    fn from_lists_rejects_non_int32_offsets() {
        let offsets = Column::from_slice_i64(&[0, 2, 3]).call().unwrap();
        let child = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        assert!(Column::from_lists(2, offsets, child).call().is_err());
    }

    #[test]
    fn from_lists_rejects_wrong_offset_length() {
        let offsets = Column::from_slice_i32(&[0, 2]).call().unwrap();
        let child = Column::from_slice_i32(&[10, 20, 30]).call().unwrap();
        assert!(Column::from_lists(2, offsets, child).call().is_err());
    }
}
