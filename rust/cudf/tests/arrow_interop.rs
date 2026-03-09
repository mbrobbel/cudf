// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Arrow interop (requires `arrow` feature).

#![cfg(feature = "arrow")]
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType as ArrowDataType, Field, Schema};

use cudf::column::Column;
use cudf::table::Table;

// ===========================================================================
// Primitive round-trips: Arrow → GPU → Arrow
// ===========================================================================

#[test]
fn i32_round_trip() {
    let arrow = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.len(), 5);
    assert_eq!(gpu.to_vec_i32(), [1, 2, 3, 4, 5]);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[1, 2, 3, 4, 5]);
}

#[test]
fn i8_round_trip() {
    let arrow = Int8Array::from(vec![10i8, -20, 30]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.to_vec_i8(), [10, -20, 30]);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Int8Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[10i8, -20, 30]);
}

#[test]
fn i16_round_trip() {
    let arrow = Int16Array::from(vec![100i16, 200, 300]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Int16Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[100i16, 200, 300]);
}

#[test]
fn i64_round_trip() {
    let arrow = Int64Array::from(vec![1_000_000i64, 2_000_000]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[1_000_000i64, 2_000_000]);
}

#[test]
fn u8_round_trip() {
    let arrow = UInt8Array::from(vec![1u8, 2, 255]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<UInt8Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[1u8, 2, 255]);
}

#[test]
fn u16_round_trip() {
    let arrow = UInt16Array::from(vec![1000u16, 2000]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<UInt16Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[1000u16, 2000]);
}

#[test]
fn u32_round_trip() {
    let arrow = UInt32Array::from(vec![42u32, 99]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[42u32, 99]);
}

#[test]
fn u64_round_trip() {
    let arrow = UInt64Array::from(vec![100u64, 200]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<UInt64Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[100u64, 200]);
}

#[test]
fn f32_round_trip() {
    let arrow = Float32Array::from(vec![1.5f32, 2.5, 3.5]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[1.5f32, 2.5, 3.5]);
}

#[test]
fn f64_round_trip() {
    let arrow = Float64Array::from(vec![1.1f64, 2.2, 3.3]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[1.1f64, 2.2, 3.3]);
}

// ===========================================================================
// Boolean round-trip
// ===========================================================================

#[test]
fn bool_round_trip() {
    let arrow = BooleanArray::from(vec![true, false, true, true]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.to_vec_bool(), [true, false, true, true]);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(typed.value(0));
    assert!(!typed.value(1));
    assert!(typed.value(2));
}

// ===========================================================================
// String round-trip
// ===========================================================================

#[test]
fn string_round_trip() {
    let arrow = StringArray::from(vec!["hello", "world", "!"]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.to_vec_string(), ["hello", "world", "!"]);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(typed.value(0), "hello");
    assert_eq!(typed.value(1), "world");
    assert_eq!(typed.value(2), "!");
}

// ===========================================================================
// Nullable arrays
// ===========================================================================

#[test]
fn nullable_i32_round_trip() {
    let arrow = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
    assert_eq!(arrow.null_count(), 2);

    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.len(), 5);
    assert_eq!(gpu.null_count(), 2);
    assert!(gpu.has_nulls());

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(typed.null_count(), 2);
    assert!(typed.is_valid(0));
    assert!(typed.is_null(1));
    assert!(typed.is_valid(2));
    assert!(typed.is_null(3));
    assert!(typed.is_valid(4));
    assert_eq!(typed.value(0), 1);
    assert_eq!(typed.value(2), 3);
    assert_eq!(typed.value(4), 5);
}

#[test]
fn nullable_string_round_trip() {
    let arrow = StringArray::from(vec![Some("a"), None, Some("c")]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.null_count(), 1);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(typed.null_count(), 1);
    assert!(typed.is_valid(0));
    assert!(typed.is_null(1));
    assert!(typed.is_valid(2));
    assert_eq!(typed.value(0), "a");
    assert_eq!(typed.value(2), "c");
}

#[test]
fn nullable_f64_round_trip() {
    let arrow = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.null_count(), 1);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<Float64Array>().unwrap();
    assert!(typed.is_null(1));
    assert!((typed.value(0) - 1.0).abs() < 1e-9);
    assert!((typed.value(2) - 3.0).abs() < 1e-9);
}

#[test]
fn nullable_bool_round_trip() {
    let arrow = BooleanArray::from(vec![Some(true), None, Some(false)]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.null_count(), 1);

    let back = gpu.to_arrow().unwrap();
    let typed = back.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(typed.is_null(1));
    assert!(typed.value(0));
    assert!(!typed.value(2));
}

// ===========================================================================
// RecordBatch round-trip
// ===========================================================================

#[test]
fn record_batch_round_trip() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ids", ArrowDataType::Int32, false),
        Field::new("values", ArrowDataType::Float64, false),
        Field::new("names", ArrowDataType::Utf8, false),
    ]));
    let ids = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let values = Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0]));
    let names = Arc::new(StringArray::from(vec!["a", "b", "c"]));
    let batch = RecordBatch::try_new(schema, vec![ids, values, names]).unwrap();

    let table = Table::from_record_batch(&batch).unwrap();
    assert_eq!(table.len(), 3);
    assert_eq!(table.columns_len(), 3);

    let back = table.to_record_batch().unwrap();
    assert_eq!(back.num_rows(), 3);
    assert_eq!(back.num_columns(), 3);

    // Verify data
    let ids_back = back
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids_back.values().as_ref(), &[1, 2, 3]);

    let vals_back = back
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(vals_back.values().as_ref(), &[10.0, 20.0, 30.0]);

    let names_back = back
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names_back.value(0), "a");
    assert_eq!(names_back.value(1), "b");
    assert_eq!(names_back.value(2), "c");
}

#[test]
fn record_batch_with_nulls() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowDataType::Int32, true),
        Field::new("y", ArrowDataType::Float64, true),
    ]));
    let x = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
    let y = Arc::new(Float64Array::from(vec![Some(10.0), Some(20.0), None]));
    let batch = RecordBatch::try_new(schema, vec![x, y]).unwrap();

    let table = Table::from_record_batch(&batch).unwrap();
    assert_eq!(table.len(), 3);

    let back = table.to_record_batch().unwrap();
    let x_back = back
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert!(x_back.is_null(1));
    assert_eq!(x_back.value(0), 1);

    let y_back = back
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(y_back.is_null(2));
}

// ===========================================================================
// GPU computation on Arrow data
// ===========================================================================

#[test]
fn arrow_to_gpu_compute_to_arrow() {
    use cudf::data_type::TypeId;

    // Start with Arrow data
    let a = Int32Array::from(vec![10, 20, 30, 40, 50]);
    let b = Int32Array::from(vec![1, 2, 3, 4, 5]);

    // Upload to GPU
    let gpu_a = Column::from_arrow(&a).unwrap();
    let gpu_b = Column::from_arrow(&b).unwrap();

    // Compute on GPU
    let gpu_sum = gpu_a
        .view()
        .add(&gpu_b.view(), TypeId::INT32)
        .call()
        .unwrap();

    // Download back to Arrow
    let result = gpu_sum.to_arrow().unwrap();
    let typed = result.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(typed.values().as_ref(), &[11, 22, 33, 44, 55]);
}

// ===========================================================================
// Empty arrays
// ===========================================================================

#[test]
fn empty_array_round_trip() {
    let arrow = Int32Array::from(Vec::<i32>::new());
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.len(), 0);

    let back = gpu.to_arrow().unwrap();
    assert_eq!(back.len(), 0);
}

// ===========================================================================
// Unsupported type error
// ===========================================================================

#[test]
fn unsupported_type_returns_error() {
    use arrow_array::BinaryArray;
    let arrow = BinaryArray::from(vec![b"hello" as &[u8]]);
    let result = Column::from_arrow(&arrow);
    assert!(result.is_err());
}

// ===========================================================================
// Timestamp round-trip
// ===========================================================================

#[test]
fn timestamp_round_trip() {
    use arrow_array::TimestampSecondArray;
    let arrow = TimestampSecondArray::from(vec![1_710_498_645i64, 1_719_792_000]);
    let gpu = Column::from_arrow(&arrow).unwrap();
    assert_eq!(gpu.to_vec_i64(), [1_710_498_645, 1_719_792_000]);

    let back = gpu.to_arrow().unwrap();
    let typed = back
        .as_any()
        .downcast_ref::<TimestampSecondArray>()
        .unwrap();
    assert_eq!(typed.values().as_ref(), &[1_710_498_645i64, 1_719_792_000]);
}
