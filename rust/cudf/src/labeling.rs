// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Bin labeling operations.
//!
//! Assigns integer labels to column values based on which bin they fall into.
//! Available as a method on [`ColumnView`](crate::column::ColumnView):
//!
//! * [`ColumnView::label_bins`](crate::column::ColumnView::label_bins) -- label
//!   each element with the index of the bin it belongs to, or `-1` if it falls
//!   outside all bins.

/// Controls whether a bin edge is inclusive or exclusive.
///
/// Variants: `YES` (inclusive), `NO` (exclusive).
#[doc(alias = "inclusive")]
pub use cudf_sys::ffi::Inclusive;

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::stream::GpuOp;

    #[test]
    fn label_bins_basic() {
        let col = Column::from_slice_i32(&[1, 5, 10, 15, 20]).call().unwrap();
        let left_edges = Column::from_slice_i32(&[0, 10]).call().unwrap();
        let right_edges = Column::from_slice_i32(&[10, 20]).call().unwrap();
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
