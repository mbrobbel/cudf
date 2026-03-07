// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Label bins operations.

#[doc(alias = "inclusive")]
pub use cudf_sys::ffi::Inclusive;

#[cfg(test)]
mod tests {
    use crate::column::Column;

    #[test]
    fn label_bins_basic() {
        let col = Column::from_slice_i32(&[1, 5, 10, 15, 20]);
        let left_edges = Column::from_slice_i32(&[0, 10]);
        let right_edges = Column::from_slice_i32(&[10, 20]);
        let result = col
            .view()
            .label_bins(
                &left_edges.view(),
                super::Inclusive::YES,
                &right_edges.view(),
                super::Inclusive::NO,
            )
            .call()
            .unwrap();
        assert_eq!(result.len(), 5);
    }
}
