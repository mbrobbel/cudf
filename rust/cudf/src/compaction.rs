// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Stream compaction operations: unique, distinct, `drop_nans`, `drop_nulls`.
//!
//! These operations remove rows from a [`Table`](crate::table::Table) based on
//! null/NaN content or duplicate key combinations. Available as methods on
//! `Table`:
//!
//! * [`drop_nans`](crate::table::Table::drop_nans) -- drop rows with NaN in
//!   specified columns.
//! * [`drop_nans_with_threshold`](crate::table::Table::drop_nans_with_threshold) --
//!   drop rows below a non-NaN threshold.
//! * [`drop_nulls`](crate::table::Table::drop_nulls) -- drop rows where all
//!   columns are null.
//! * [`drop_nulls_with_threshold`](crate::table::Table::drop_nulls_with_threshold) --
//!   drop rows below a non-null threshold.
//! * [`unique`](crate::table::Table::unique) -- keep first occurrence of
//!   consecutive duplicate key rows.
//! * [`distinct`](crate::table::Table::distinct) -- keep one occurrence of each
//!   distinct key combination.
//! * [`stable_distinct`](crate::table::Table::stable_distinct) -- like `distinct`
//!   but preserves input order.

/// Policy controlling which duplicate row to keep in
/// [`unique`](crate::table::Table::unique) and
/// [`distinct`](crate::table::Table::distinct).
///
/// Variants: `KEEP_FIRST`, `KEEP_LAST`, `KEEP_NONE`, `KEEP_ANY`.
#[doc(alias = "duplicate_keep_option")]
pub use cudf_sys::ffi::DuplicateKeepOption;

/// Policy controlling whether NaN values are compared as equal.
///
/// Used with distinct/unique operations. Variants: `ALL_EQUAL`, `UNEQUAL`.
#[doc(alias = "nan_equality")]
pub use cudf_sys::ffi::NanEquality;

/// Policy controlling whether null values are compared as equal.
///
/// Used with distinct/unique operations. Variants: `EQUAL`, `UNEQUAL`.
#[doc(alias = "null_equality")]
pub use cudf_sys::ffi::NullEquality;

/// Policy controlling whether nulls are included or excluded.
///
/// Variants: `INCLUDE`, `EXCLUDE`.
#[doc(alias = "null_policy")]
pub use cudf_sys::ffi::NullPolicy;

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::stream::GpuOp;
    use crate::table::TableBuilder;

    #[test]
    fn drop_nans_basic() {
        let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0])
            .call()
            .unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.drop_nans(&[0]).call().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn unique_basic() {
        let col = Column::from_slice_i32(&[1, 1, 2, 2, 3]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.unique(&[0]).call().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn distinct_basic() {
        let col = Column::from_slice_i32(&[3, 1, 2, 1, 3]).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.distinct(&[0]).call().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn drop_nulls_with_threshold() {
        let col = Column::from_scalar(&Scalar::null_i32(), 5).call().unwrap();
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        // threshold 1 means at least 1 non-null required
        let result = table.drop_nulls_with_threshold(&[0], 1).call().unwrap();
        assert_eq!(result.len(), 0);
    }
}
