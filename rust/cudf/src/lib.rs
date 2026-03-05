// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Safe Rust bindings for [libcudf](https://github.com/rapidsai/cudf),
//! the RAPIDS GPU DataFrame library.
//!
//! This crate provides idiomatic Rust wrappers around the core cudf types:
//!
//! - [`ColumnView`] — a non-owning view of a GPU column
//! - [`Table`] — an owning GPU table (a set of equal-length columns)
//! - [`DataType`] / [`TypeId`] — logical element type descriptors
//!
//! GPU memory management is handled by [RMM](https://github.com/rapidsai/rmm)
//! and exposed via the [`rmm`] module.

pub mod column;
pub mod concatenate;
pub mod copying;
pub mod data_type;
pub mod datetime;
pub mod error;
pub mod fill;
pub mod filter;
pub mod groupby;
pub mod hashing;
pub mod io;
pub mod join;
pub mod merge;
pub mod ops;
pub mod partitioning;
pub mod quantile;
pub mod reduction;
pub mod replace;
pub mod reshape;
pub mod rmm;
pub mod scalar;
pub mod search;
pub mod sorting;
pub mod strings;
pub mod table;
pub mod transform;

pub use column::{Column, ColumnView};
pub use data_type::{DataType, TypeId};
pub use error::{Error, Result};
pub use groupby::AggregationKind;
pub use ops::BinaryOperator;
pub use scalar::Scalar;
pub use sorting::{NullOrder, Order};
pub use table::{Table, TableBuilder};
