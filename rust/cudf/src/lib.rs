// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::pedantic)]
#![deny(clippy::shadow_reuse)]
#![deny(clippy::shadow_same)]
#![deny(clippy::shadow_unrelated)]
// TODO: Enforce as_conversions once i32↔usize casts at the FFI boundary are
// wrapped in explicit conversion helpers. Tracked separately.
#![allow(clippy::as_conversions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
// Allow common pedantic false positives in this codebase.
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]
// These lints are only relevant in test code.
#![cfg_attr(test, allow(clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::unreadable_literal))]
#![cfg_attr(test, allow(clippy::approx_constant))]

//! Safe Rust bindings for [libcudf](https://github.com/rapidsai/cudf),
//! the RAPIDS GPU `DataFrame` library.
//!
//! This crate provides idiomatic Rust wrappers around the core cudf types:
//!
//! - [`column::ColumnView`] — a non-owning view of a GPU column
//! - [`table::Table`] — an owning GPU table (a set of equal-length columns)
//! - [`data_type::DataType`] / [`data_type::TypeId`] — logical element type descriptors
//!
//! GPU memory management is handled by [RMM](https://github.com/rapidsai/rmm)
//! and exposed via the [`rmm`] module.

pub mod column;
pub mod compaction;
pub mod concatenate;
pub mod copying;
pub mod data_type;
pub mod datetime;
pub mod dictionary;
pub mod error;
pub mod fill;
pub mod filter;
pub mod groupby;
pub mod hashing;
pub mod io;
pub mod join;
pub mod labeling;
pub mod lists;
pub mod merge;
pub mod ops;
pub mod partitioning;
pub mod quantile;
pub mod reduction;
pub mod replace;
pub mod reshape;
pub mod rmm;
pub mod rolling;
pub mod scalar;
pub mod search;
pub mod sorting;
pub mod stream;
pub mod strings;
pub mod table;
pub mod transform;

use column::{Column, ColumnView, MaskState};
use data_type::DataType;
use error::Result;
use scalar::Scalar;
use stream::Stream;

// -- Null mask utilities --

/// Compute the bytes required for a bitmask of the given number of bits.
pub fn bitmask_allocation_size_bytes(number_of_bits: usize) -> usize {
    cudf_sys::ffi::bitmask_allocation_size_bytes(number_of_bits as i32)
}

/// Compute the number of bitmask words needed for the given number of bits.
pub fn num_bitmask_words(number_of_bits: usize) -> usize {
    cudf_sys::ffi::num_bitmask_words(number_of_bits as i32) as usize
}

/// Returns the null count implied by a mask state for a given number of rows.
pub fn state_null_count(mask_state: MaskState, num_rows: usize) -> usize {
    cudf_sys::ffi::state_null_count(mask_state as i32, num_rows as i32) as usize
}

// -- Type checking utilities --

/// Check if two column types are equivalent (ignoring scale for fixed-point).
pub fn column_types_equivalent(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> bool {
    cudf_sys::ffi::column_types_equivalent(lhs.0, rhs.0)
}

/// Check if two columns have the same types (including scale and nested types).
pub fn columns_have_same_types(lhs: &ColumnView<'_>, rhs: &ColumnView<'_>) -> bool {
    cudf_sys::ffi::columns_have_same_types(lhs.0, rhs.0)
}

/// Check if two tables have columns of the same types.
pub fn tables_have_same_types(lhs: &table::Table, rhs: &table::Table) -> bool {
    cudf_sys::ffi::tables_have_same_types(&lhs.0, &rhs.0)
}

/// Check if a cast between two data types is supported.
pub fn is_supported_cast(from: DataType, to: DataType) -> bool {
    cudf_sys::ffi::is_supported_cast(from.id().repr, from.scale(), to.id().repr, to.scale())
}

// -- Fill utilities --

/// Generate a calendrical month sequence starting from `init`, adding `months` each step.
pub fn calendrical_month_sequence(count: usize, init: &Scalar, months: i32) -> Result<Column> {
    let ffi = scalar::scalar_to_ffi(init);
    let c = cudf_sys::ffi::calendrical_month_sequence(
        count as i32,
        &ffi,
        months,
        Stream::default_stream().as_raw(),
    )?;
    Ok(Column(c))
}

/// Select from two scalars based on boolean mask.
pub fn copy_if_else_scalars(lhs: &Scalar, rhs: &Scalar, mask: &ColumnView<'_>) -> Result<Column> {
    let lhs_ffi = scalar::scalar_to_ffi(lhs);
    let rhs_ffi = scalar::scalar_to_ffi(rhs);
    let c = cudf_sys::ffi::copy_if_else_scalars(
        &lhs_ffi,
        &rhs_ffi,
        mask.0,
        Stream::default_stream().as_raw(),
    )?;
    Ok(Column(c))
}
