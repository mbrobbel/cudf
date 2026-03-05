// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reduction operations on GPU columns.

use crate::column::ColumnView;
use crate::data_type::TypeId;
use crate::error::Result;
use crate::scalar::Scalar;

/// Default stream shorthand for internal use.
fn ds() -> usize {
    crate::stream::Stream::default_stream().as_raw()
}

/// Computes the sum of all elements in a column.
pub fn sum(col: &ColumnView<'_>, output_type: TypeId) -> Result<Scalar> {
    let s = cudf_sys::ffi::reduce_sum(col.0, output_type.repr, ds())?;
    Ok(Scalar(s))
}

/// Computes the minimum value in a column.
pub fn min(col: &ColumnView<'_>, output_type: TypeId) -> Result<Scalar> {
    let s = cudf_sys::ffi::reduce_min(col.0, output_type.repr, ds())?;
    Ok(Scalar(s))
}

/// Computes the maximum value in a column.
pub fn max(col: &ColumnView<'_>, output_type: TypeId) -> Result<Scalar> {
    let s = cudf_sys::ffi::reduce_max(col.0, output_type.repr, ds())?;
    Ok(Scalar(s))
}

/// Computes the product of all elements in a column.
pub fn product(col: &ColumnView<'_>, output_type: TypeId) -> Result<Scalar> {
    let s = cudf_sys::ffi::reduce_product(col.0, output_type.repr, ds())?;
    Ok(Scalar(s))
}

/// Returns true (as a BOOL8 scalar) if any element is non-zero.
pub fn any(col: &ColumnView<'_>) -> Result<Scalar> {
    let s = cudf_sys::ffi::reduce_any(col.0, ds())?;
    Ok(Scalar(s))
}

/// Returns true (as a BOOL8 scalar) if all elements are non-zero.
pub fn all(col: &ColumnView<'_>) -> Result<Scalar> {
    let s = cudf_sys::ffi::reduce_all(col.0, ds())?;
    Ok(Scalar(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar as ScalarVal;

    #[test]
    fn sum_i32() {
        // sum of [10, 10, 10, 10] = 40
        let col = Column::from_scalar(&ScalarVal::from_i32(10), 4);
        let result = sum(&col.view(), TypeId::INT32).unwrap();
        assert!(result.is_valid());
        assert_eq!(result.as_i32(), Some(40));
    }

    #[test]
    fn min_i32() {
        // All same value
        let col = Column::from_scalar(&ScalarVal::from_i32(3), 5);
        let result = min(&col.view(), TypeId::INT32).unwrap();
        assert_eq!(result.as_i32(), Some(3));
    }

    #[test]
    fn max_f64() {
        let col = Column::from_scalar(&ScalarVal::from_f64(3.7), 3);
        let result = max(&col.view(), TypeId::FLOAT64).unwrap();
        assert!((result.as_f64().unwrap() - 3.7).abs() < 1e-9);
    }

    #[test]
    fn product_i32() {
        // product of [2, 2, 2] = 8
        let col = Column::from_scalar(&ScalarVal::from_i32(2), 3);
        let result = product(&col.view(), TypeId::INT32).unwrap();
        assert_eq!(result.as_i32(), Some(8));
    }

    #[test]
    fn any_bool_true() {
        let col = Column::from_scalar(&ScalarVal::from_bool(true), 3);
        let result = any(&col.view()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn all_bool_true() {
        let col = Column::from_scalar(&ScalarVal::from_bool(true), 3);
        let result = all(&col.view()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn all_bool_false() {
        let col = Column::from_scalar(&ScalarVal::from_bool(false), 3);
        let result = all(&col.view()).unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }
}
