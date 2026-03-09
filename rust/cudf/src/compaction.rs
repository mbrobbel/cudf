// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Stream compaction operations: unique, distinct, `drop_nans`, `drop_nulls`.

#[doc(alias = "duplicate_keep_option")]
pub use cudf_sys::ffi::DuplicateKeepOption;
#[doc(alias = "nan_equality")]
pub use cudf_sys::ffi::NanEquality;
#[doc(alias = "null_equality")]
pub use cudf_sys::ffi::NullEquality;
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
        let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.drop_nans(&[0]).call().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn unique_basic() {
        let col = Column::from_slice_i32(&[1, 1, 2, 2, 3]);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.unique(&[0]).call().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn distinct_basic() {
        let col = Column::from_slice_i32(&[3, 1, 2, 1, 3]);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.distinct(&[0]).call().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn drop_nulls_with_threshold() {
        let col = Column::from_scalar(&Scalar::null_i32(), 5);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        // threshold 1 means at least 1 non-null required
        let result = table.drop_nulls_with_threshold(&[0], 1).call().unwrap();
        assert_eq!(result.len(), 0);
    }
}
