// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Fill, repeat, and sequence operations on GPU columns and tables.
//!
//! - `column_view.fill(begin, end, value).call()` fills a range with a scalar.
//! - `table.repeat(count).call()` repeats each row.
//! - `sequence(count, init, step).call()` generates an arithmetic sequence.

use crate::column::Column;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Builder for generating an arithmetic sequence: \[init, init+step, init+2*step, ...\].
///
/// Created by [`sequence()`]. Call [`.call()`](Sequence::call) to execute.
pub struct Sequence<'a> {
    count: usize,
    init: &'a Scalar,
    step: &'a Scalar,
    stream: Stream,
}

/// Generates an arithmetic sequence: \[init, init+step, init+2*step, ...\].
///
/// Returns a [`Sequence`] builder. Use `.stream()` to set a custom CUDA stream,
/// then `.call()` to execute.
///
/// # Example
/// ```ignore
/// let col = sequence(5, &Scalar::from_i32(0), &Scalar::from_i32(2)).call()?;
/// ```
pub fn sequence<'a>(count: usize, init: &'a Scalar, step: &'a Scalar) -> Sequence<'a> {
    Sequence {
        count,
        init,
        step,
        stream: Stream::default_stream(),
    }
}

impl Sequence<'_> {
    /// Sets the CUDA stream for this operation.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the sequence generation and returns the resulting column.
    pub fn call(self) -> Result<Column> {
        let init_ffi = crate::scalar::scalar_to_ffi(self.init);
        let step_ffi = crate::scalar::scalar_to_ffi(self.step);
        let c = cudf_sys::filling::ffi::sequence_column(
            crate::usize_to_i32(self.count),
            &init_ffi,
            &step_ffi,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::table::TableBuilder;

    #[test]
    fn fill_range() {
        let col = Column::from_scalar(&Scalar::from_i32(1), 5);
        let value = Scalar::from_i32(99);
        let result = col.view().fill(1, 3, &value).call().unwrap();
        assert_eq!(result.to_vec_i32(), vec![1, 99, 99, 1, 1]);
    }

    #[test]
    fn repeat_table_test() {
        let col = Column::from_slice_i32(&[10, 20, 30]);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.repeat(2).call().unwrap();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn sequence_i32() {
        let init = Scalar::from_i32(0);
        let step = Scalar::from_i32(2);
        let result = sequence(5, &init, &step).call().unwrap();
        assert_eq!(result.to_vec_i32(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn sequence_f64() {
        let init = Scalar::from_f64(1.0);
        let step = Scalar::from_f64(0.5);
        let result = sequence(3, &init, &step).call().unwrap();
        let data = result.to_vec_f64();
        assert!((data[0] - 1.0).abs() < 1e-9);
        assert!((data[1] - 1.5).abs() < 1e-9);
        assert!((data[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn sequence_with_stream() {
        use crate::stream::Stream;
        let init = Scalar::from_i32(10);
        let step = Scalar::from_i32(5);
        let result = sequence(3, &init, &step)
            .stream(Stream::default_stream())
            .call()
            .unwrap();
        assert_eq!(result.to_vec_i32(), vec![10, 15, 20]);
    }
}
