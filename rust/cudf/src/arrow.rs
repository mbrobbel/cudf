// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Arrow interop: convert between cudf GPU columns/tables and Arrow arrays.
//!
//! This module is behind the `arrow` feature flag.
//!
//! # Examples
//!
//! ```ignore
//! use cudf::column::Column;
//! use arrow_array::Int32Array;
//!
//! // Arrow → GPU
//! let arrow_array = Int32Array::from(vec![1, 2, 3]);
//! let gpu_col = Column::from_arrow(&arrow_array).unwrap();
//!
//! // GPU → Arrow
//! let back = gpu_col.to_arrow().unwrap();
//! let typed = back.as_any().downcast_ref::<Int32Array>().unwrap();
//! ```

use std::sync::Arc;

use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType as ArrowDataType, Field, Schema};

use crate::column::Column;
use crate::data_type::TypeId;
use crate::error::{Error, Result};
use crate::stream::GpuOp;
use crate::table::Table;

/// Maps a cudf [`TypeId`] to an Arrow [`DataType`](ArrowDataType).
pub fn type_id_to_arrow(tid: TypeId) -> Result<ArrowDataType> {
    match tid {
        TypeId::INT8 => Ok(ArrowDataType::Int8),
        TypeId::INT16 => Ok(ArrowDataType::Int16),
        TypeId::INT32 => Ok(ArrowDataType::Int32),
        TypeId::INT64 => Ok(ArrowDataType::Int64),
        TypeId::UINT8 => Ok(ArrowDataType::UInt8),
        TypeId::UINT16 => Ok(ArrowDataType::UInt16),
        TypeId::UINT32 => Ok(ArrowDataType::UInt32),
        TypeId::UINT64 => Ok(ArrowDataType::UInt64),
        TypeId::FLOAT32 => Ok(ArrowDataType::Float32),
        TypeId::FLOAT64 => Ok(ArrowDataType::Float64),
        TypeId::BOOL8 => Ok(ArrowDataType::Boolean),
        TypeId::STRING => Ok(ArrowDataType::Utf8),
        TypeId::TIMESTAMP_SECONDS => Ok(ArrowDataType::Timestamp(
            arrow_schema::TimeUnit::Second,
            None,
        )),
        TypeId::TIMESTAMP_MILLISECONDS => Ok(ArrowDataType::Timestamp(
            arrow_schema::TimeUnit::Millisecond,
            None,
        )),
        TypeId::TIMESTAMP_MICROSECONDS => Ok(ArrowDataType::Timestamp(
            arrow_schema::TimeUnit::Microsecond,
            None,
        )),
        TypeId::TIMESTAMP_NANOSECONDS => Ok(ArrowDataType::Timestamp(
            arrow_schema::TimeUnit::Nanosecond,
            None,
        )),
        TypeId::DURATION_SECONDS => Ok(ArrowDataType::Duration(arrow_schema::TimeUnit::Second)),
        TypeId::DURATION_MILLISECONDS => {
            Ok(ArrowDataType::Duration(arrow_schema::TimeUnit::Millisecond))
        }
        TypeId::DURATION_MICROSECONDS => {
            Ok(ArrowDataType::Duration(arrow_schema::TimeUnit::Microsecond))
        }
        TypeId::DURATION_NANOSECONDS => {
            Ok(ArrowDataType::Duration(arrow_schema::TimeUnit::Nanosecond))
        }
        _ => Err(Error::UnsupportedArrowType(format!("{tid}"))),
    }
}

/// Maps an Arrow [`DataType`](ArrowDataType) to a cudf [`TypeId`].
pub fn arrow_to_type_id(dt: &ArrowDataType) -> Result<TypeId> {
    match dt {
        ArrowDataType::Int8 => Ok(TypeId::INT8),
        ArrowDataType::Int16 => Ok(TypeId::INT16),
        ArrowDataType::Int32 => Ok(TypeId::INT32),
        ArrowDataType::Int64 => Ok(TypeId::INT64),
        ArrowDataType::UInt8 => Ok(TypeId::UINT8),
        ArrowDataType::UInt16 => Ok(TypeId::UINT16),
        ArrowDataType::UInt32 => Ok(TypeId::UINT32),
        ArrowDataType::UInt64 => Ok(TypeId::UINT64),
        ArrowDataType::Float32 => Ok(TypeId::FLOAT32),
        ArrowDataType::Float64 => Ok(TypeId::FLOAT64),
        ArrowDataType::Boolean => Ok(TypeId::BOOL8),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => Ok(TypeId::STRING),
        ArrowDataType::Timestamp(arrow_schema::TimeUnit::Second, _) => {
            Ok(TypeId::TIMESTAMP_SECONDS)
        }
        ArrowDataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            Ok(TypeId::TIMESTAMP_MILLISECONDS)
        }
        ArrowDataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            Ok(TypeId::TIMESTAMP_MICROSECONDS)
        }
        ArrowDataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            Ok(TypeId::TIMESTAMP_NANOSECONDS)
        }
        ArrowDataType::Duration(arrow_schema::TimeUnit::Second) => Ok(TypeId::DURATION_SECONDS),
        ArrowDataType::Duration(arrow_schema::TimeUnit::Millisecond) => {
            Ok(TypeId::DURATION_MILLISECONDS)
        }
        ArrowDataType::Duration(arrow_schema::TimeUnit::Microsecond) => {
            Ok(TypeId::DURATION_MICROSECONDS)
        }
        ArrowDataType::Duration(arrow_schema::TimeUnit::Nanosecond) => {
            Ok(TypeId::DURATION_NANOSECONDS)
        }
        other => Err(Error::UnsupportedArrowType(format!("{other}"))),
    }
}

/// Copies an Arrow primitive array's values to a host slice and uploads to GPU.
///
/// This handles the common pattern: extract arrow values → `Column::from_slice_*`.
/// For nullable arrays, it also sets the null mask on the GPU column.
macro_rules! primitive_to_gpu {
    ($array:expr, $arrow_ty:ty, $from_fn:ident) => {{
        let typed = $array
            .as_any()
            .downcast_ref::<arrow_array::PrimitiveArray<$arrow_ty>>()
            .ok_or_else(|| Error::UnsupportedArrowType("type mismatch".into()))?;
        let values = typed.values();
        let col = Column::$from_fn(values).call()?;
        apply_arrow_nulls(col, $array)
    }};
}

/// If the Arrow array has nulls, create a boolean validity mask and apply it.
fn apply_arrow_nulls(col: Column, array: &dyn Array) -> Result<Column> {
    let Some(null_buf) = array.nulls() else {
        return Ok(col);
    };
    // Use the NullBuffer's iterator to bulk-extract validity bits
    // instead of calling is_valid(i) per element.
    let validity: Vec<bool> = null_buf.iter().collect();
    let mask_col = Column::from_slice_bool(&validity).call()?;
    col.with_null_mask_from_bools(&mask_col.view()).call()
}

impl Column {
    /// Creates a GPU column from an Arrow array.
    ///
    /// Copies data from host (Arrow) to device (GPU). Supports all primitive
    /// types, booleans, and UTF-8 strings. Null masks are preserved.
    pub fn from_arrow(array: &dyn Array) -> Result<Self> {
        match array.data_type() {
            ArrowDataType::Int8 => primitive_to_gpu!(array, Int8Type, from_slice_i8),
            ArrowDataType::Int16 => primitive_to_gpu!(array, Int16Type, from_slice_i16),
            ArrowDataType::Int32 => primitive_to_gpu!(array, Int32Type, from_slice_i32),
            ArrowDataType::Int64 => primitive_to_gpu!(array, Int64Type, from_slice_i64),
            ArrowDataType::UInt8 => primitive_to_gpu!(array, UInt8Type, from_slice_u8),
            ArrowDataType::UInt16 => primitive_to_gpu!(array, UInt16Type, from_slice_u16),
            ArrowDataType::UInt32 => primitive_to_gpu!(array, UInt32Type, from_slice_u32),
            ArrowDataType::UInt64 => primitive_to_gpu!(array, UInt64Type, from_slice_u64),
            ArrowDataType::Float32 => primitive_to_gpu!(array, Float32Type, from_slice_f32),
            ArrowDataType::Float64 => primitive_to_gpu!(array, Float64Type, from_slice_f64),
            ArrowDataType::Boolean => {
                let typed = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| Error::UnsupportedArrowType("BooleanArray mismatch".into()))?;
                let bools: Vec<bool> = typed.iter().map(|v| v.unwrap_or(false)).collect();
                let col = Column::from_slice_bool(&bools).call()?;
                apply_arrow_nulls(col, array)
            }
            ArrowDataType::Utf8 => {
                let typed = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::UnsupportedArrowType("StringArray mismatch".into()))?;
                let strings: Vec<&str> = typed.iter().map(|v| v.unwrap_or("")).collect();
                let col = Column::from_strings(&strings).call()?;
                apply_arrow_nulls(col, array)
            }
            ArrowDataType::Timestamp(arrow_schema::TimeUnit::Second, _) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::TimestampSecondType,
                    from_timestamps_s
                )
            }
            ArrowDataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::TimestampMillisecondType,
                    from_timestamps_ms
                )
            }
            ArrowDataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::TimestampMicrosecondType,
                    from_timestamps_us
                )
            }
            ArrowDataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::TimestampNanosecondType,
                    from_timestamps_ns
                )
            }
            ArrowDataType::Duration(arrow_schema::TimeUnit::Second) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::DurationSecondType,
                    from_durations_s
                )
            }
            ArrowDataType::Duration(arrow_schema::TimeUnit::Millisecond) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::DurationMillisecondType,
                    from_durations_ms
                )
            }
            ArrowDataType::Duration(arrow_schema::TimeUnit::Microsecond) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::DurationMicrosecondType,
                    from_durations_us
                )
            }
            ArrowDataType::Duration(arrow_schema::TimeUnit::Nanosecond) => {
                primitive_to_gpu!(
                    array,
                    arrow_array::types::DurationNanosecondType,
                    from_durations_ns
                )
            }
            other => Err(Error::UnsupportedArrowType(format!("{other}"))),
        }
    }

    /// Copies GPU column data to an Arrow array on the host.
    ///
    /// Supports all primitive types, booleans, and UTF-8 strings.
    /// Null masks are preserved.
    pub fn to_arrow(&self) -> Result<ArrayRef> {
        self.view().to_arrow()
    }
}

use crate::column::ColumnView;

impl ColumnView<'_> {
    /// Copies GPU column view data to an Arrow array on the host.
    ///
    /// This operates directly on the view without a GPU deep copy,
    /// making it more efficient than converting through an owning Column.
    pub fn to_arrow(&self) -> Result<ArrayRef> {
        let tid = self.type_id();
        let nulls = self.arrow_nulls();
        to_arrow_inner(self, tid, nulls)
    }

    fn arrow_nulls(&self) -> Option<arrow_buffer::NullBuffer> {
        if self.has_nulls() {
            Some(arrow_buffer::NullBuffer::from(
                self.null_mask_to_host().call().unwrap(),
            ))
        } else {
            None
        }
    }
}

/// Helper to convert a primitive GPU column to Arrow.
macro_rules! gpu_to_primitive {
    ($col:expr, $to_fn:ident, $arrow_array:ty, $nulls:expr) => {{
        let values = $col.$to_fn().call()?;
        Ok(Arc::new(<$arrow_array>::new(values.into(), $nulls)))
    }};
}

/// Core conversion logic, split out to keep `to_arrow` under the line limit.
fn to_arrow_inner(
    col: &ColumnView<'_>,
    tid: TypeId,
    nulls: Option<arrow_buffer::NullBuffer>,
) -> Result<ArrayRef> {
    match tid {
        TypeId::INT8 => gpu_to_primitive!(col, to_vec_i8, Int8Array, nulls),
        TypeId::INT16 => gpu_to_primitive!(col, to_vec_i16, Int16Array, nulls),
        TypeId::INT32 => gpu_to_primitive!(col, to_vec_i32, Int32Array, nulls),
        TypeId::INT64 => gpu_to_primitive!(col, to_vec_i64, Int64Array, nulls),
        TypeId::UINT8 => gpu_to_primitive!(col, to_vec_u8, UInt8Array, nulls),
        TypeId::UINT16 => gpu_to_primitive!(col, to_vec_u16, UInt16Array, nulls),
        TypeId::UINT32 => gpu_to_primitive!(col, to_vec_u32, UInt32Array, nulls),
        TypeId::UINT64 => gpu_to_primitive!(col, to_vec_u64, UInt64Array, nulls),
        TypeId::FLOAT32 => gpu_to_primitive!(col, to_vec_f32, Float32Array, nulls),
        TypeId::FLOAT64 => gpu_to_primitive!(col, to_vec_f64, Float64Array, nulls),
        TypeId::BOOL8 => Ok(bool_to_arrow(col, nulls)),
        TypeId::STRING => Ok(string_to_arrow(col)),
        TypeId::TIMESTAMP_SECONDS => {
            gpu_to_primitive!(col, to_vec_i64, arrow_array::TimestampSecondArray, nulls)
        }
        TypeId::TIMESTAMP_MILLISECONDS => {
            gpu_to_primitive!(
                col,
                to_vec_i64,
                arrow_array::TimestampMillisecondArray,
                nulls
            )
        }
        TypeId::TIMESTAMP_MICROSECONDS => {
            gpu_to_primitive!(
                col,
                to_vec_i64,
                arrow_array::TimestampMicrosecondArray,
                nulls
            )
        }
        TypeId::TIMESTAMP_NANOSECONDS => {
            gpu_to_primitive!(
                col,
                to_vec_i64,
                arrow_array::TimestampNanosecondArray,
                nulls
            )
        }
        TypeId::DURATION_SECONDS => {
            gpu_to_primitive!(col, to_vec_i64, arrow_array::DurationSecondArray, nulls)
        }
        TypeId::DURATION_MILLISECONDS => {
            gpu_to_primitive!(
                col,
                to_vec_i64,
                arrow_array::DurationMillisecondArray,
                nulls
            )
        }
        TypeId::DURATION_MICROSECONDS => {
            gpu_to_primitive!(
                col,
                to_vec_i64,
                arrow_array::DurationMicrosecondArray,
                nulls
            )
        }
        TypeId::DURATION_NANOSECONDS => {
            gpu_to_primitive!(col, to_vec_i64, arrow_array::DurationNanosecondArray, nulls)
        }
        _ => Err(Error::UnsupportedArrowType(format!("{tid}"))),
    }
}

/// Convert a BOOL8 GPU column to Arrow `BooleanArray`.
fn bool_to_arrow(col: &ColumnView<'_>, nulls: Option<arrow_buffer::NullBuffer>) -> ArrayRef {
    let values = col.to_vec_bool().call().unwrap();
    // Build the values buffer directly, then attach the null buffer separately
    // to avoid an intermediate Vec<Option<bool>> allocation.
    let values_buf = arrow_buffer::BooleanBuffer::from(values);
    Arc::new(BooleanArray::new(values_buf, nulls))
}

/// Convert a STRING GPU column to Arrow `StringArray`.
fn string_to_arrow(col: &ColumnView<'_>) -> ArrayRef {
    let values = col.to_vec_string().call().unwrap();
    if col.has_nulls() {
        let validity = col.null_mask_to_host().call().unwrap();
        let arr: StringArray = values
            .into_iter()
            .zip(validity)
            .map(|(s, valid)| if valid { Some(s) } else { None })
            .collect();
        Arc::new(arr)
    } else {
        Arc::new(StringArray::from(values))
    }
}

impl Table {
    /// Creates a GPU table from an Arrow [`RecordBatch`].
    ///
    /// Each column in the batch is uploaded to the GPU. Field names
    /// are not preserved (cudf tables are positional, not named).
    pub fn from_record_batch(batch: &RecordBatch) -> Result<Self> {
        let columns: Vec<Column> = batch
            .columns()
            .iter()
            .map(|arr| Column::from_arrow(arr.as_ref()))
            .collect::<Result<_>>()?;
        Table::from_columns(columns)
    }

    /// Copies the GPU table to an Arrow [`RecordBatch`] on the host.
    ///
    /// Column names are generated as `"c0"`, `"c1"`, etc.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let ncols = self.columns_len();
        let mut fields = Vec::with_capacity(ncols);
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(ncols);

        for i in 0..ncols {
            let view = self.column(i)?;
            let tid = view.type_id();
            let arrow_dt = type_id_to_arrow(tid)?;
            let nullable = view.has_nulls();
            fields.push(Field::new(format!("c{i}"), arrow_dt, nullable));
            arrays.push(view.to_arrow()?);
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays).map_err(|e| Error::UnsupportedArrowType(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_id_to_arrow_roundtrip() {
        let types = [
            TypeId::INT8,
            TypeId::INT16,
            TypeId::INT32,
            TypeId::INT64,
            TypeId::UINT8,
            TypeId::UINT16,
            TypeId::UINT32,
            TypeId::UINT64,
            TypeId::FLOAT32,
            TypeId::FLOAT64,
            TypeId::BOOL8,
            TypeId::STRING,
        ];
        for tid in types {
            let arrow_dt = type_id_to_arrow(tid).unwrap();
            let back = arrow_to_type_id(&arrow_dt).unwrap();
            assert_eq!(tid, back);
        }
    }

    #[test]
    fn unsupported_type_errors() {
        assert!(type_id_to_arrow(TypeId::LIST).is_err());
        assert!(arrow_to_type_id(&ArrowDataType::Binary).is_err());
    }
}
