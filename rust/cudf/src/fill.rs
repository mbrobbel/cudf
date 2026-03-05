// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Fill, repeat, and sequence operations on GPU columns and tables.
//!
//! - `column_view.fill(begin, end, value)` fills a range with a scalar.
//! - `table.repeat(count)` repeats each row.
//! - `sequence(count, init, step)` remains a free function.

use crate::column::Column;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::stream::Stream;

/// Generates an arithmetic sequence: [init, init+step, init+2*step, ...].
pub fn sequence(count: usize, init: &Scalar, step: &Scalar) -> Result<Column> {
    sequence_on(count, init, step, Stream::default_stream())
}

/// Sequence on a custom CUDA stream.
pub fn sequence_on(count: usize, init: &Scalar, step: &Scalar, stream: Stream) -> Result<Column> {
    let init_ffi = crate::scalar::scalar_to_ffi(init);
    let step_ffi = crate::scalar::scalar_to_ffi(step);
    let c = cudf_sys::ffi::sequence_column(count as i32, &init_ffi, &step_ffi, stream.as_raw())?;
    Ok(Column(c))
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
        let result = col.view().fill(1, 3, &value).unwrap();
        assert_eq!(result.to_vec_i32(), vec![1, 99, 99, 1, 1]);
    }

    #[test]
    fn repeat_table_test() {
        let col = Column::from_slice_i32(&[10, 20, 30]);
        let mut builder = TableBuilder::new();
        builder.push_column(col);
        let table = builder.build().unwrap();
        let result = table.repeat(2).unwrap();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn sequence_i32() {
        let init = Scalar::from_i32(0);
        let step = Scalar::from_i32(2);
        let result = sequence(5, &init, &step).unwrap();
        assert_eq!(result.to_vec_i32(), vec![0, 2, 4, 6, 8]);
    }
}
