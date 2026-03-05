// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Search operations on GPU columns and tables.

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::scalar::Scalar;
use crate::sorting::{NullOrder, Order};
use crate::table::Table;

/// Checks if a scalar value exists in a column.
pub fn contains(col: &ColumnView<'_>, needle: &Scalar) -> Result<bool> {
    cudf_sys::ffi::contains_scalar(col.0, &needle.0).map_err(Into::into)
}

/// Checks which values from `needles` exist in `haystack`.
///
/// Returns a BOOL8 column of the same size as `needles`.
pub fn contains_column(
    haystack: &ColumnView<'_>,
    needles: &ColumnView<'_>,
) -> Result<Column> {
    let c = cudf_sys::ffi::contains_column(haystack.0, needles.0)?;
    Ok(Column(c))
}

/// Finds lower bound insertion points in a sorted table.
pub fn lower_bound(
    haystack: &Table,
    needles: &Table,
    column_orders: &[Order],
    null_orders: &[NullOrder],
) -> Result<Column> {
    let orders_i32: Vec<i32> = column_orders.iter().map(|o| o.repr).collect();
    let nulls_i32: Vec<i32> = null_orders.iter().map(|n| n.repr).collect();
    let c = cudf_sys::ffi::lower_bound(&haystack.0, &needles.0, &orders_i32, &nulls_i32)?;
    Ok(Column(c))
}

/// Finds upper bound insertion points in a sorted table.
pub fn upper_bound(
    haystack: &Table,
    needles: &Table,
    column_orders: &[Order],
    null_orders: &[NullOrder],
) -> Result<Column> {
    let orders_i32: Vec<i32> = column_orders.iter().map(|o| o.repr).collect();
    let nulls_i32: Vec<i32> = null_orders.iter().map(|n| n.repr).collect();
    let c = cudf_sys::ffi::upper_bound(&haystack.0, &needles.0, &orders_i32, &nulls_i32)?;
    Ok(Column(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;

    #[test]
    fn contains_found() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50]));
        let needle = Scalar::from_i32(20);
        assert!(contains(&col.view(), &needle).unwrap());
    }

    #[test]
    fn contains_not_found() {
        let col = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50]));
        let needle = Scalar::from_i32(25);
        assert!(!contains(&col.view(), &needle).unwrap());
    }

    #[test]
    fn contains_column_test() {
        let haystack = Column(cudf_sys::ffi::make_column_from_host_i32(&[10, 20, 30, 40, 50]));
        let needles = Column(cudf_sys::ffi::make_column_from_host_i32(&[20, 40, 60, 80]));
        let result = contains_column(&haystack.view(), &needles.view()).unwrap();
        assert_eq!(result.to_vec_bool(), vec![true, true, false, false]);
    }
}
