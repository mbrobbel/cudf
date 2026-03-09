// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reduction operations on GPU columns.
//!
//! All reductions are available as methods on
//! [`ColumnView`](crate::column::ColumnView). Each reduces an entire column
//! to a single [`Scalar`](crate::scalar::Scalar). Null values are skipped
//! during the reduction.
//!
//! - [`sum`](crate::column::ColumnView::sum) -- sum of all elements.
//! - [`min`](crate::column::ColumnView::min) -- minimum element.
//! - [`max`](crate::column::ColumnView::max) -- maximum element.
//! - [`product`](crate::column::ColumnView::product) -- product of all
//!   elements.
//! - [`any`](crate::column::ColumnView::any) -- `true` if any element is
//!   true (`BOOL8` columns).
//! - [`all`](crate::column::ColumnView::all) -- `true` only if every element
//!   is true (`BOOL8` columns).

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar as ScalarVal;
    use crate::stream::GpuOp;

    #[test]
    fn sum_i32() {
        let col = Column::from_scalar(&ScalarVal::from_i32(10), 4)
            .call()
            .unwrap();
        let result = col.view().sum(TypeId::INT32).call().unwrap();
        assert!(result.is_valid());
        assert_eq!(result.as_i32(), Some(40));
    }

    #[test]
    fn min_i32() {
        let col = Column::from_scalar(&ScalarVal::from_i32(3), 5)
            .call()
            .unwrap();
        let result = col.view().min(TypeId::INT32).call().unwrap();
        assert_eq!(result.as_i32(), Some(3));
    }

    #[test]
    fn max_f64() {
        let col = Column::from_scalar(&ScalarVal::from_f64(3.7), 3)
            .call()
            .unwrap();
        let result = col.view().max(TypeId::FLOAT64).call().unwrap();
        assert!((result.as_f64().unwrap() - 3.7).abs() < 1e-9);
    }

    #[test]
    fn product_i32() {
        let col = Column::from_scalar(&ScalarVal::from_i32(2), 3)
            .call()
            .unwrap();
        let result = col.view().product(TypeId::INT32).call().unwrap();
        assert_eq!(result.as_i32(), Some(8));
    }

    #[test]
    fn any_bool_true() {
        let col = Column::from_scalar(&ScalarVal::from_bool(true), 3)
            .call()
            .unwrap();
        let result = col.view().any().call().unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn all_bool_true() {
        let col = Column::from_scalar(&ScalarVal::from_bool(true), 3)
            .call()
            .unwrap();
        let result = col.view().all().call().unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn all_bool_false() {
        let col = Column::from_scalar(&ScalarVal::from_bool(false), 3)
            .call()
            .unwrap();
        let result = col.view().all().call().unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }
}
