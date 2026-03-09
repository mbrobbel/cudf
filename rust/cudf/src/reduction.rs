// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Reduction operations on GPU columns.
//!
//! Available as methods on [`ColumnView`](crate::ColumnView):
//! `col.sum(...)`, `col.min(...)`, `col.max(...)`, etc.

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar as ScalarVal;
    use crate::stream::GpuOp;

    #[test]
    fn sum_i32() {
        let col = Column::from_scalar(&ScalarVal::from_i32(10), 4);
        let result = col.view().sum(TypeId::INT32).call().unwrap();
        assert!(result.is_valid());
        assert_eq!(result.as_i32(), Some(40));
    }

    #[test]
    fn min_i32() {
        let col = Column::from_scalar(&ScalarVal::from_i32(3), 5);
        let result = col.view().min(TypeId::INT32).call().unwrap();
        assert_eq!(result.as_i32(), Some(3));
    }

    #[test]
    fn max_f64() {
        let col = Column::from_scalar(&ScalarVal::from_f64(3.7), 3);
        let result = col.view().max(TypeId::FLOAT64).call().unwrap();
        assert!((result.as_f64().unwrap() - 3.7).abs() < 1e-9);
    }

    #[test]
    fn product_i32() {
        let col = Column::from_scalar(&ScalarVal::from_i32(2), 3);
        let result = col.view().product(TypeId::INT32).call().unwrap();
        assert_eq!(result.as_i32(), Some(8));
    }

    #[test]
    fn any_bool_true() {
        let col = Column::from_scalar(&ScalarVal::from_bool(true), 3);
        let result = col.view().any().call().unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn all_bool_true() {
        let col = Column::from_scalar(&ScalarVal::from_bool(true), 3);
        let result = col.view().all().call().unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn all_bool_false() {
        let col = Column::from_scalar(&ScalarVal::from_bool(false), 3);
        let result = col.view().all().call().unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }
}
