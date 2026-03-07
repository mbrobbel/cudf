// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! GPU table types: owning [`Table`] and [`TableBuilder`].

use cxx::UniquePtr;

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::groupby::AggregationKind;
use crate::scalar::Scalar;
use crate::sorting::{NullOrder, Order};
use crate::stream::Stream;
use crate::{i32_to_usize, usize_to_i32};

#[doc(alias = "table")]
/// An owning GPU table (a set of columns of the same size).
///
/// Wraps a `cudf::table` via the CXX FFI layer. Dropping this value
/// frees the underlying GPU memory for all columns.
pub struct Table(pub(crate) UniquePtr<cudf_sys::ffi::Table>);

impl Default for Table {
    /// Creates an empty table with zero columns and zero rows.
    fn default() -> Self {
        Self(cudf_sys::ffi::table_empty())
    }
}

// ---------------------------------------------------------------------------
// Builder structs for Table operations
// ---------------------------------------------------------------------------

/// Builder for [`Table::sort`].
pub struct Sort<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl Sort<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::sort_table(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::sort_ascending`].
pub struct SortAscending<'a> {
    table: &'a Table,
    stream: Stream,
}

impl SortAscending<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        self.table.sort(&[], &[]).stream(self.stream).call()
    }
}

/// Builder for [`Table::sorted_order`].
pub struct SortedOrder<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl SortedOrder<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let col = cudf_sys::sorting::ffi::sorted_order(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(col))
    }
}

/// Builder for [`Table::is_sorted`].
pub struct IsSorted<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl IsSorted<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<bool> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        cudf_sys::sorting::ffi::is_sorted_table(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )
        .map_err(Into::into)
    }
}

/// Builder for [`Table::inner_join`].
pub struct InnerJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl InnerJoin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::join::ffi::inner_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_join`].
pub struct LeftJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl LeftJoin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::join::ffi::left_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::full_join`].
pub struct FullJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl FullJoin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::join::ffi::full_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_semi_join`].
pub struct LeftSemiJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl LeftSemiJoin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::join::ffi::left_semi_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::left_anti_join`].
pub struct LeftAntiJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    left_on: &'a [i32],
    right_on: &'a [i32],
    stream: Stream,
}

impl LeftAntiJoin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::join::ffi::left_anti_join(
            &self.table.0,
            &self.right.0,
            self.left_on,
            self.right_on,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::groupby`].
pub struct Groupby<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_column: i32,
    agg: AggregationKind,
    stream: Stream,
}

impl Groupby<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let result = cudf_sys::groupby::ffi::groupby_single(
            &self.table.0,
            self.key_columns,
            self.value_column,
            self.agg.repr,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::groupby_multi`].
pub struct GroupbyMulti<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    aggs: &'a [AggregationKind],
    stream: Stream,
}

impl GroupbyMulti<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        if self.value_columns.len() != self.aggs.len() {
            return Err(crate::error::Error::OutOfBounds {
                index: self.aggs.len(),
                len: self.value_columns.len(),
            });
        }
        let agg_kinds = cudf_sys::agg_kinds_as_i32(self.aggs);
        let result = cudf_sys::groupby::ffi::groupby_multi(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            agg_kinds,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::filter`].
pub struct Filter<'a> {
    table: &'a Table,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl Filter<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::apply_boolean_mask(
            &self.table.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::drop_nulls`].
pub struct DropNulls<'a> {
    table: &'a Table,
    stream: Stream,
}

impl DropNulls<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::drop_nulls_all(&self.table.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::gather`].
pub struct Gather<'a> {
    table: &'a Table,
    indices: &'a ColumnView<'a>,
    stream: Stream,
}

impl Gather<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::gather_table(
            &self.table.0,
            self.indices.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::murmur3`].
pub struct Murmur3<'a> {
    table: &'a Table,
    seed: u32,
    stream: Stream,
}

impl Murmur3<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Column {
        Column(cudf_sys::hashing::ffi::hash_murmur3(
            &self.table.0,
            self.seed,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::xxhash64`].
pub struct Xxhash64<'a> {
    table: &'a Table,
    seed: u64,
    stream: Stream,
}

impl Xxhash64<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Column {
        Column(cudf_sys::hashing::ffi::hash_xxhash64(
            &self.table.0,
            self.seed,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::md5`].
pub struct Md5<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Md5<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Column {
        Column(cudf_sys::hashing::ffi::hash_md5(
            &self.table.0,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::sha256`].
pub struct Sha256<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Sha256<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Column {
        Column(cudf_sys::hashing::ffi::hash_sha256(
            &self.table.0,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::merge`].
pub struct Merge<'a> {
    table: &'a Table,
    right: &'a Table,
    key_columns: &'a [i32],
    orders: &'a [Order],
    null_orders: &'a [NullOrder],
    stream: Stream,
}

impl Merge<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.null_orders);
        let tbl = cudf_sys::merge::ffi::merge_tables(
            &self.table.0,
            &self.right.0,
            self.key_columns,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::hash_partition`].
pub struct HashPartition<'a> {
    table: &'a Table,
    columns: &'a [i32],
    num_partitions: usize,
    stream: Stream,
}

impl HashPartition<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let tbl = cudf_sys::partitioning::ffi::hash_partition_table(
            &self.table.0,
            self.columns,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::hash_partition_offsets`].
pub struct HashPartitionOffsets<'a> {
    table: &'a Table,
    columns: &'a [i32],
    num_partitions: usize,
    stream: Stream,
}

impl HashPartitionOffsets<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Vec<usize>> {
        let offsets = cudf_sys::partitioning::ffi::hash_partition_offsets(
            &self.table.0,
            self.columns,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::round_robin`].
pub struct RoundRobin<'a> {
    table: &'a Table,
    num_partitions: usize,
    start: usize,
    stream: Stream,
}

impl RoundRobin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let tbl = cudf_sys::partitioning::ffi::round_robin_partition_table(
            &self.table.0,
            usize_to_i32(self.num_partitions),
            usize_to_i32(self.start),
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::round_robin_offsets`].
pub struct RoundRobinOffsets<'a> {
    table: &'a Table,
    num_partitions: usize,
    start: usize,
    stream: Stream,
}

impl RoundRobinOffsets<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Vec<usize>> {
        let offsets = cudf_sys::partitioning::ffi::round_robin_partition_offsets(
            &self.table.0,
            usize_to_i32(self.num_partitions),
            usize_to_i32(self.start),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::interleave_columns`].
pub struct InterleaveColumns<'a> {
    table: &'a Table,
    stream: Stream,
}

impl InterleaveColumns<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let col = cudf_sys::reshape::ffi::interleave_columns(&self.table.0, self.stream.as_raw())?;
        Ok(Column(col))
    }
}

/// Builder for [`Table::tile`].
pub struct Tile<'a> {
    table: &'a Table,
    count: usize,
    stream: Stream,
}

impl Tile<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let tbl = cudf_sys::reshape::ffi::tile_table(
            &self.table.0,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::repeat`].
pub struct Repeat<'a> {
    table: &'a Table,
    count: usize,
    stream: Stream,
}

impl Repeat<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::filling::ffi::repeat_table(
            &self.table.0,
            usize_to_i32(self.count),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::encode`].
pub struct Encode<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Encode<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let col = cudf_sys::transform::ffi::encode_table(&self.table.0, self.stream.as_raw())?;
        Ok(Column(col))
    }
}

/// Builder for [`Table::encode_keys`].
pub struct EncodeKeys<'a> {
    table: &'a Table,
    stream: Stream,
}

impl EncodeKeys<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let tbl = cudf_sys::transform::ffi::encode_keys(&self.table.0, self.stream.as_raw())?;
        Ok(Table(tbl))
    }
}

/// Builder for [`Table::drop_nans`].
pub struct DropNans<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl DropNans<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t =
            cudf_sys::compaction::ffi::drop_nans(&self.table.0, self.keys, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::drop_nulls_with_threshold`].
pub struct DropNullsWithThreshold<'a> {
    table: &'a Table,
    keys: &'a [i32],
    threshold: usize,
    stream: Stream,
}

impl DropNullsWithThreshold<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::drop_nulls_with_threshold(
            &self.table.0,
            self.keys,
            usize_to_i32(self.threshold),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::unique`].
pub struct Unique<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl Unique<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::unique_table(
            &self.table.0,
            self.keys,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::distinct`].
pub struct Distinct<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl Distinct<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::distinct_table(
            &self.table.0,
            self.keys,
            0,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::stable_distinct`].
pub struct StableDistinct<'a> {
    table: &'a Table,
    keys: &'a [i32],
    stream: Stream,
}

impl StableDistinct<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::stable_distinct_table(
            &self.table.0,
            self.keys,
            0,
            0,
            0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::scatter`].
pub struct Scatter<'a> {
    table: &'a Table,
    source: &'a Table,
    scatter_map: &'a ColumnView<'a>,
    stream: Stream,
}

impl Scatter<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::scatter_table(
            &self.source.0,
            self.scatter_map.0,
            &self.table.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::reverse`].
pub struct Reverse<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Reverse<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::reverse_table(&self.table.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::slice`].
pub struct Slice<'a> {
    table: &'a Table,
    begin: usize,
    end: usize,
    stream: Stream,
}

impl Slice<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::slice_table(
            &self.table.0,
            usize_to_i32(self.begin),
            usize_to_i32(self.end),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::sample`].
pub struct Sample<'a> {
    table: &'a Table,
    n: usize,
    with_replacement: bool,
    seed: i64,
    stream: Stream,
}

impl Sample<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::sample_table(
            &self.table.0,
            usize_to_i32(self.n),
            self.with_replacement,
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::transpose`].
pub struct Transpose<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Transpose<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::reshape::ffi::transpose_table(&self.table.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::lower_bound`].
pub struct LowerBound<'a> {
    table: &'a Table,
    needles: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl LowerBound<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::search::ffi::lower_bound(
            &self.table.0,
            &self.needles.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::upper_bound`].
pub struct UpperBound<'a> {
    table: &'a Table,
    needles: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl UpperBound<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::search::ffi::upper_bound(
            &self.table.0,
            &self.needles.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::explode`].
pub struct Explode<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl Explode<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::lists::ffi::explode_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::explode_position`].
pub struct ExplodePosition<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl ExplodePosition<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::lists::ffi::explode_position_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::explode_outer`].
pub struct ExplodeOuter<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl ExplodeOuter<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::lists::ffi::explode_outer_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::explode_outer_position`].
pub struct ExplodeOuterPosition<'a> {
    table: &'a Table,
    column_idx: usize,
    stream: Stream,
}

impl ExplodeOuterPosition<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::lists::ffi::explode_outer_position_table(
            &self.table.0,
            usize_to_i32(self.column_idx),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::gather_checked`].
pub struct GatherChecked<'a> {
    table: &'a Table,
    indices: &'a ColumnView<'a>,
    nullify_oob: bool,
    stream: Stream,
}

impl GatherChecked<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::gather_table_checked(
            &self.table.0,
            self.indices.0,
            self.nullify_oob,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::cross_join`].
pub struct CrossJoin<'a> {
    table: &'a Table,
    right: &'a Table,
    stream: Stream,
}

impl CrossJoin<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t =
            cudf_sys::join::ffi::cross_join(&self.table.0, &self.right.0, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::partition_by_map`].
pub struct PartitionByMap<'a> {
    table: &'a Table,
    partition_map: &'a ColumnView<'a>,
    num_partitions: usize,
    stream: Stream,
}

impl PartitionByMap<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::partitioning::ffi::partition_by_map(
            &self.table.0,
            self.partition_map.0,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::partition_by_map_offsets`].
pub struct PartitionByMapOffsets<'a> {
    table: &'a Table,
    partition_map: &'a ColumnView<'a>,
    num_partitions: usize,
    stream: Stream,
}

impl PartitionByMapOffsets<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Vec<usize>> {
        let offsets = cudf_sys::partitioning::ffi::partition_by_map_offsets(
            &self.table.0,
            self.partition_map.0,
            usize_to_i32(self.num_partitions),
            self.stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(i32_to_usize).collect())
    }
}

/// Builder for [`Table::groupby_scan`].
pub struct GroupbyScan<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    aggs: &'a [AggregationKind],
    stream: Stream,
}

impl GroupbyScan<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        if self.value_columns.len() != self.aggs.len() {
            return Err(crate::error::Error::OutOfBounds {
                index: self.aggs.len(),
                len: self.value_columns.len(),
            });
        }
        let agg_kinds = cudf_sys::agg_kinds_as_i32(self.aggs);
        let result = cudf_sys::groupby::ffi::groupby_scan(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            agg_kinds,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::groupby_shift`].
pub struct GroupbyShift<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    offsets: &'a [i32],
    fill_values: &'a [Scalar],
    stream: Stream,
}

impl GroupbyShift<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.fill_values {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let result = cudf_sys::groupby::ffi::groupby_shift(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            self.offsets,
            list.pin_mut(),
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::groupby_replace_nulls`].
pub struct GroupbyReplaceNulls<'a> {
    table: &'a Table,
    key_columns: &'a [i32],
    value_columns: &'a [i32],
    policies: &'a [i32],
    stream: Stream,
}

impl GroupbyReplaceNulls<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let result = cudf_sys::groupby::ffi::groupby_replace_nulls(
            &self.table.0,
            self.key_columns,
            self.value_columns,
            self.policies,
            self.stream.as_raw(),
        )?;
        Ok(Table(result))
    }
}

/// Builder for [`Table::sha1`].
pub struct Sha1<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Sha1<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::hashing::ffi::hash_sha1(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::sha224`].
pub struct Sha224<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Sha224<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::hashing::ffi::hash_sha224(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::sha384`].
pub struct Sha384<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Sha384<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::hashing::ffi::hash_sha384(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::sha512`].
pub struct Sha512<'a> {
    table: &'a Table,
    stream: Stream,
}

impl Sha512<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::hashing::ffi::hash_sha512(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::murmurhash3_x64_128`].
pub struct MurmurHash3X64_128<'a> {
    table: &'a Table,
    seed: u64,
    stream: Stream,
}

impl MurmurHash3X64_128<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::hashing::ffi::hash_murmurhash3_x64_128(
            &self.table.0,
            self.seed,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::xxhash_32`].
pub struct XxHash32<'a> {
    table: &'a Table,
    seed: u32,
    stream: Stream,
}

impl XxHash32<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::hashing::ffi::hash_xxhash_32(&self.table.0, self.seed, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::row_bit_count`].
pub struct RowBitCount<'a> {
    table: &'a Table,
    stream: Stream,
}

impl RowBitCount<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::transform::ffi::row_bit_count(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::segmented_row_bit_count`].
pub struct SegmentedRowBitCount<'a> {
    table: &'a Table,
    segment_length: i32,
    stream: Stream,
}

impl SegmentedRowBitCount<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::transform::ffi::segmented_row_bit_count(
            &self.table.0,
            self.segment_length,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::byte_cast`].
pub struct ByteCast<'a> {
    col: &'a ColumnView<'a>,
    flip_endian: bool,
    stream: Stream,
}

impl ByteCast<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::reshape::ffi::byte_cast_column(
            self.col.0,
            self.flip_endian,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::stable_sorted_order`].
pub struct StableSortedOrder<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl StableSortedOrder<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::stable_sorted_order(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::stable_sort`].
pub struct StableSort<'a> {
    table: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl StableSort<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_sort_table(
            &self.table.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::sort_by_key`].
pub struct SortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl SortByKey<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::sort_by_key(
            &self.table.0,
            &self.keys.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::stable_sort_by_key`].
pub struct StableSortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl StableSortByKey<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_sort_by_key(
            &self.table.0,
            &self.keys.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::segmented_sorted_order`].
pub struct SegmentedSortedOrder<'a> {
    table: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl SegmentedSortedOrder<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::segmented_sorted_order(
            &self.table.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::stable_segmented_sorted_order`].
pub struct StableSegmentedSortedOrder<'a> {
    table: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl StableSegmentedSortedOrder<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let c = cudf_sys::sorting::ffi::stable_segmented_sorted_order(
            &self.table.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::segmented_sort_by_key`].
pub struct SegmentedSortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl SegmentedSortByKey<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::segmented_sort_by_key(
            &self.table.0,
            &self.keys.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::stable_segmented_sort_by_key`].
pub struct StableSegmentedSortByKey<'a> {
    table: &'a Table,
    keys: &'a Table,
    segment_offsets: &'a ColumnView<'a>,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl StableSegmentedSortByKey<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::sorting::ffi::stable_segmented_sort_by_key(
            &self.table.0,
            &self.keys.0,
            self.segment_offsets.0,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::quantiles`].
pub struct Quantiles<'a> {
    table: &'a Table,
    q: &'a [f64],
    interp: crate::quantile::Interpolation,
    is_sorted: bool,
    orders: &'a [Order],
    nulls: &'a [NullOrder],
    stream: Stream,
}

impl Quantiles<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let orders_i32 = cudf_sys::orders_as_i32(self.orders);
        let nulls_i32 = cudf_sys::null_orders_as_i32(self.nulls);
        let t = cudf_sys::quantile::ffi::quantiles_table(
            &self.table.0,
            self.q,
            self.interp.repr,
            self.is_sorted,
            orders_i32,
            nulls_i32,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::boolean_mask_scatter`].
pub struct BooleanMaskScatter<'a> {
    table: &'a Table,
    source: &'a Table,
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl BooleanMaskScatter<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::copying::ffi::boolean_mask_scatter_table(
            &self.source.0,
            &self.table.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::scatter_scalars`].
pub struct ScatterScalars<'a> {
    table: &'a Table,
    scalars: &'a [Scalar],
    scatter_map: &'a ColumnView<'a>,
    stream: Stream,
}

impl ScatterScalars<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.scalars {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::copying::ffi::scatter_scalars(
            list.pin_mut(),
            self.scatter_map.0,
            &self.table.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::boolean_mask_scatter_scalars`].
pub struct BooleanMaskScatterScalars<'a> {
    table: &'a Table,
    scalars: &'a [Scalar],
    mask: &'a ColumnView<'a>,
    stream: Stream,
}

impl BooleanMaskScatterScalars<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in self.scalars {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::copying::ffi::boolean_mask_scatter_scalars(
            list.pin_mut(),
            &self.table.0,
            self.mask.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::repeat_by_column`].
pub struct RepeatByColumn<'a> {
    table: &'a Table,
    counts: &'a ColumnView<'a>,
    stream: Stream,
}

impl RepeatByColumn<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::filling::ffi::repeat_table_column(
            &self.table.0,
            self.counts.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::distinct_count`].
pub struct DistinctCount<'a> {
    table: &'a Table,
    nulls_equal: bool,
    stream: Stream,
}

impl DistinctCount<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> usize {
        let ne = i32::from(!self.nulls_equal); // EQUAL=0, UNEQUAL=1
        i32_to_usize(cudf_sys::compaction::ffi::distinct_count_table(
            &self.table.0,
            ne,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::bitmask_and_to_bools`].
pub struct BitmaskAndToBools<'a> {
    table: &'a Table,
    stream: Stream,
}

impl BitmaskAndToBools<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::ffi::bitmask_and_to_bools(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::bitmask_or_to_bools`].
pub struct BitmaskOrToBools<'a> {
    table: &'a Table,
    stream: Stream,
}

impl BitmaskOrToBools<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c = cudf_sys::ffi::bitmask_or_to_bools(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::unique_count`].
pub struct UniqueCount<'a> {
    table: &'a Table,
    nulls_equal: bool,
    stream: Stream,
}

impl UniqueCount<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> usize {
        let ne = i32::from(!self.nulls_equal);
        i32_to_usize(cudf_sys::compaction::ffi::unique_count_table(
            &self.table.0,
            ne,
            self.stream.as_raw(),
        ))
    }
}

/// Builder for [`Table::drop_nans_with_threshold`].
pub struct DropNansWithThreshold<'a> {
    table: &'a Table,
    keys: &'a [i32],
    threshold: usize,
    stream: Stream,
}

impl DropNansWithThreshold<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::compaction::ffi::drop_nans_with_threshold(
            &self.table.0,
            self.keys,
            usize_to_i32(self.threshold),
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::approx_distinct_count`].
pub struct ApproxDistinctCount<'a> {
    table: &'a Table,
    precision: i32,
    stream: Stream,
}

impl ApproxDistinctCount<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> usize {
        cudf_sys::compaction::ffi::approx_distinct_count(
            &self.table.0,
            self.precision,
            self.stream.as_raw(),
        )
    }
}

/// Builder for [`Table::from_dlpack`].
pub struct FromDlpack {
    managed_tensor_ptr: usize,
    stream: Stream,
}

impl FromDlpack {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::io::ffi::from_dlpack(self.managed_tensor_ptr, self.stream.as_raw())?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::to_dlpack`].
pub struct ToDlpack<'a> {
    table: &'a Table,
    stream: Stream,
}

impl ToDlpack<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> usize {
        cudf_sys::io::ffi::to_dlpack(&self.table.0, self.stream.as_raw())
    }
}

/// Builder for [`Table::concatenate_columns`].
pub struct ConcatenateTableColumns<'a> {
    table: &'a Table,
    stream: Stream,
}

impl ConcatenateTableColumns<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Column> {
        let c =
            cudf_sys::concatenate::ffi::concatenate_columns(&self.table.0, self.stream.as_raw())?;
        Ok(Column(c))
    }
}

/// Builder for [`Table::concatenate`].
pub struct ConcatenateWith<'a> {
    table: &'a Table,
    other: &'a Table,
    stream: Stream,
}

impl ConcatenateWith<'_> {
    /// Sets the CUDA stream.
    pub fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    /// Executes the operation.
    pub fn call(self) -> Result<Table> {
        let t = cudf_sys::concatenate::ffi::concatenate_tables(
            &self.table.0,
            &self.other.0,
            self.stream.as_raw(),
        )?;
        Ok(Table(t))
    }
}

// ---------------------------------------------------------------------------
// impl Table
// ---------------------------------------------------------------------------

impl Table {
    #[doc(alias = "num_columns")]
    /// Returns the number of columns.
    pub fn columns_len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::table_num_columns(&self.0))
    }

    #[doc(alias = "num_rows")]
    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::table_num_rows(&self.0))
    }

    /// Returns `true` if the table has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns_len() == 0
    }

    /// Returns the total GPU allocation size in bytes.
    pub fn alloc_bytes(&self) -> usize {
        cudf_sys::ffi::table_alloc_size(&self.0)
    }

    /// Returns a view of the column at the given index, or `None` if out of bounds.
    pub fn column(&self, index: usize) -> Option<ColumnView<'_>> {
        if index >= self.columns_len() {
            return None;
        }
        let view = cudf_sys::ffi::table_get_column_view(&self.0, usize_to_i32(index)).ok()?;
        Some(ColumnView(view))
    }

    /// Returns an iterator over all columns as [`ColumnView`]s.
    pub fn columns(&self) -> Columns<'_> {
        Columns {
            table: self,
            index: 0,
            len: self.columns_len(),
        }
    }

    // -- Sorting --

    /// Sorts the table by all columns using the given orders.
    pub fn sort<'a>(&'a self, orders: &'a [Order], nulls: &'a [NullOrder]) -> Sort<'a> {
        Sort {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    #[doc(alias = "sort")]
    /// Sorts the table ascending with nulls before.
    pub fn sort_ascending(&self) -> SortAscending<'_> {
        SortAscending {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the sorted row indices.
    pub fn sorted_order<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SortedOrder<'a> {
        SortedOrder {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Returns whether the table rows are sorted according to the given orders.
    pub fn is_sorted<'a>(&'a self, orders: &'a [Order], nulls: &'a [NullOrder]) -> IsSorted<'a> {
        IsSorted {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    // -- Joins --

    /// Performs an inner join with another table.
    pub fn inner_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> InnerJoin<'a> {
        InnerJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a left join with another table.
    pub fn left_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftJoin<'a> {
        LeftJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a full outer join with another table.
    pub fn full_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> FullJoin<'a> {
        FullJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a left semi join with another table.
    pub fn left_semi_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftSemiJoin<'a> {
        LeftSemiJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    /// Performs a left anti join with another table.
    pub fn left_anti_join<'a>(
        &'a self,
        right: &'a Table,
        left_on: &'a [i32],
        right_on: &'a [i32],
    ) -> LeftAntiJoin<'a> {
        LeftAntiJoin {
            table: self,
            right,
            left_on,
            right_on,
            stream: Stream::default_stream(),
        }
    }

    // -- GroupBy --

    /// Performs a single groupby aggregation on one value column.
    pub fn groupby<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_column: i32,
        agg: AggregationKind,
    ) -> Groupby<'a> {
        Groupby {
            table: self,
            key_columns,
            value_column,
            agg,
            stream: Stream::default_stream(),
        }
    }

    /// Performs groupby with multiple aggregations on multiple value columns.
    pub fn groupby_multi<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        aggs: &'a [AggregationKind],
    ) -> GroupbyMulti<'a> {
        GroupbyMulti {
            table: self,
            key_columns,
            value_columns,
            aggs,
            stream: Stream::default_stream(),
        }
    }

    // -- Filter --

    /// Filters the table by a boolean mask column.
    pub fn filter<'a>(&'a self, mask: &'a ColumnView<'a>) -> Filter<'a> {
        Filter {
            table: self,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Drops rows where all columns are null.
    pub fn drop_nulls(&self) -> DropNulls<'_> {
        DropNulls {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying --

    /// Gathers rows using an index column.
    pub fn gather<'a>(&'a self, indices: &'a ColumnView<'a>) -> Gather<'a> {
        Gather {
            table: self,
            indices,
            stream: Stream::default_stream(),
        }
    }

    /// Creates an empty table with the same column types.
    pub fn empty_like(&self) -> Table {
        Table(cudf_sys::copying::ffi::empty_like_table(&self.0))
    }

    // -- Hashing --

    /// Computes `MurmurHash3` 32-bit hash of each row.
    pub fn murmur3(&self, seed: u32) -> Murmur3<'_> {
        Murmur3 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// Computes `XXHash64` hash of each row.
    pub fn xxhash64(&self, seed: u64) -> Xxhash64<'_> {
        Xxhash64 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// Computes MD5 hash of each row.
    pub fn md5(&self) -> Md5<'_> {
        Md5 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes SHA-256 hash of each row.
    pub fn sha256(&self) -> Sha256<'_> {
        Sha256 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// SHA-1 hash of each row (returns STRING column with 40-char hex).
    pub fn sha1(&self) -> Sha1<'_> {
        Sha1 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// `MurmurHash3` 128-bit hash (returns Table of two UINT64 columns).
    pub fn murmurhash3_x64_128(&self, seed: u64) -> MurmurHash3X64_128<'_> {
        MurmurHash3X64_128 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// `XXHash` 32-bit hash of each row.
    pub fn xxhash_32(&self, seed: u32) -> XxHash32<'_> {
        XxHash32 {
            table: self,
            seed,
            stream: Stream::default_stream(),
        }
    }

    /// SHA-224 hash of each row (returns STRING column).
    pub fn sha224(&self) -> Sha224<'_> {
        Sha224 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// SHA-384 hash of each row (returns STRING column).
    pub fn sha384(&self) -> Sha384<'_> {
        Sha384 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// SHA-512 hash of each row (returns STRING column).
    pub fn sha512(&self) -> Sha512<'_> {
        Sha512 {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Approximate per-row bit count.
    pub fn row_bit_count(&self) -> RowBitCount<'_> {
        RowBitCount {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Per-segment cumulative row bit count.
    pub fn segmented_row_bit_count(&self, segment_length: i32) -> SegmentedRowBitCount<'_> {
        SegmentedRowBitCount {
            table: self,
            segment_length,
            stream: Stream::default_stream(),
        }
    }

    /// Convert column elements to lists of bytes.
    pub fn byte_cast<'a>(col: &'a ColumnView<'a>, flip_endian: bool) -> ByteCast<'a> {
        ByteCast {
            col,
            flip_endian,
            stream: Stream::default_stream(),
        }
    }

    // -- Merge --

    /// Merges two sorted tables maintaining sort order.
    pub fn merge<'a>(
        &'a self,
        right: &'a Table,
        key_columns: &'a [i32],
        orders: &'a [Order],
        null_orders: &'a [NullOrder],
    ) -> Merge<'a> {
        Merge {
            table: self,
            right,
            key_columns,
            orders,
            null_orders,
            stream: Stream::default_stream(),
        }
    }

    // -- Partitioning --

    /// Hash-partitions the table.
    pub fn hash_partition<'a>(
        &'a self,
        columns: &'a [i32],
        num_partitions: usize,
    ) -> HashPartition<'a> {
        HashPartition {
            table: self,
            columns,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Returns partition offsets for hash partitioning.
    pub fn hash_partition_offsets<'a>(
        &'a self,
        columns: &'a [i32],
        num_partitions: usize,
    ) -> HashPartitionOffsets<'a> {
        HashPartitionOffsets {
            table: self,
            columns,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Round-robin partitions the table.
    pub fn round_robin(&self, num_partitions: usize, start: usize) -> RoundRobin<'_> {
        RoundRobin {
            table: self,
            num_partitions,
            start,
            stream: Stream::default_stream(),
        }
    }

    /// Returns partition offsets for round-robin partitioning.
    pub fn round_robin_offsets(
        &self,
        num_partitions: usize,
        start: usize,
    ) -> RoundRobinOffsets<'_> {
        RoundRobinOffsets {
            table: self,
            num_partitions,
            start,
            stream: Stream::default_stream(),
        }
    }

    // -- Reshape --

    /// Interleaves columns into a single column.
    pub fn interleave_columns(&self) -> InterleaveColumns<'_> {
        InterleaveColumns {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Tiles (repeats) the rows `count` times.
    pub fn tile(&self, count: usize) -> Tile<'_> {
        Tile {
            table: self,
            count,
            stream: Stream::default_stream(),
        }
    }

    // -- Fill (table-level) --

    /// Repeats each row `count` times.
    pub fn repeat(&self, count: usize) -> Repeat<'_> {
        Repeat {
            table: self,
            count,
            stream: Stream::default_stream(),
        }
    }

    // -- Transform --

    /// Encodes table rows as integer indices into sorted distinct rows.
    pub fn encode(&self) -> Encode<'_> {
        Encode {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the sorted distinct key rows from encoding.
    pub fn encode_keys(&self) -> EncodeKeys<'_> {
        EncodeKeys {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Stream compaction --

    /// Drops rows where any of the specified key columns contain NaN.
    pub fn drop_nans<'a>(&'a self, keys: &'a [i32]) -> DropNans<'a> {
        DropNans {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    /// Drops rows where the number of non-null key values is below the threshold.
    pub fn drop_nulls_with_threshold<'a>(
        &'a self,
        keys: &'a [i32],
        threshold: usize,
    ) -> DropNullsWithThreshold<'a> {
        DropNullsWithThreshold {
            table: self,
            keys,
            threshold,
            stream: Stream::default_stream(),
        }
    }

    /// Returns unique consecutive rows based on key columns.
    pub fn unique<'a>(&'a self, keys: &'a [i32]) -> Unique<'a> {
        Unique {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    /// Returns distinct rows based on key columns.
    pub fn distinct<'a>(&'a self, keys: &'a [i32]) -> Distinct<'a> {
        Distinct {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    /// Returns distinct rows preserving input order.
    pub fn stable_distinct<'a>(&'a self, keys: &'a [i32]) -> StableDistinct<'a> {
        StableDistinct {
            table: self,
            keys,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying extras --

    /// Scatters source table rows into this table at given indices.
    pub fn scatter<'a>(
        &'a self,
        source: &'a Table,
        scatter_map: &'a ColumnView<'a>,
    ) -> Scatter<'a> {
        Scatter {
            table: self,
            source,
            scatter_map,
            stream: Stream::default_stream(),
        }
    }

    /// Reverses the rows of the table.
    pub fn reverse(&self) -> Reverse<'_> {
        Reverse {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Extracts a slice [begin, end) of the table as a new owned table.
    pub fn slice(&self, begin: usize, end: usize) -> Slice<'_> {
        Slice {
            table: self,
            begin,
            end,
            stream: Stream::default_stream(),
        }
    }

    /// Randomly samples rows from the table.
    pub fn sample(&self, n: usize, with_replacement: bool, seed: i64) -> Sample<'_> {
        Sample {
            table: self,
            n,
            with_replacement,
            seed,
            stream: Stream::default_stream(),
        }
    }

    // -- Transpose --

    /// Transposes the table (rows become columns).
    pub fn transpose(&self) -> Transpose<'_> {
        Transpose {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Search --

    /// Finds lower bound insertion points in this sorted table.
    pub fn lower_bound<'a>(
        &'a self,
        needles: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> LowerBound<'a> {
        LowerBound {
            table: self,
            needles,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Finds upper bound insertion points in this sorted table.
    pub fn upper_bound<'a>(
        &'a self,
        needles: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> UpperBound<'a> {
        UpperBound {
            table: self,
            needles,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    // -- Sorting (new) --

    /// Stable sorted order (preserves order of equal elements).
    pub fn stable_sorted_order<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSortedOrder<'a> {
        StableSortedOrder {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable sort (preserves order of equal elements).
    pub fn stable_sort<'a>(
        &'a self,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSort<'a> {
        StableSort {
            table: self,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Sort this (values) table by a separate keys table.
    pub fn sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SortByKey<'a> {
        SortByKey {
            table: self,
            keys,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable sort this (values) table by a separate keys table.
    pub fn stable_sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSortByKey<'a> {
        StableSortByKey {
            table: self,
            keys,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Segmented sorted order (sort within segments).
    pub fn segmented_sorted_order<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SegmentedSortedOrder<'a> {
        SegmentedSortedOrder {
            table: self,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable segmented sorted order (preserves relative order of equal elements).
    pub fn stable_segmented_sorted_order<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSegmentedSortedOrder<'a> {
        StableSegmentedSortedOrder {
            table: self,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Segmented sort by key.
    pub fn segmented_sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> SegmentedSortByKey<'a> {
        SegmentedSortByKey {
            table: self,
            keys,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Stable segmented sort by key.
    pub fn stable_segmented_sort_by_key<'a>(
        &'a self,
        keys: &'a Table,
        segment_offsets: &'a ColumnView<'a>,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> StableSegmentedSortByKey<'a> {
        StableSegmentedSortByKey {
            table: self,
            keys,
            segment_offsets,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Table-level quantile rows.
    pub fn quantiles<'a>(
        &'a self,
        q: &'a [f64],
        interp: crate::quantile::Interpolation,
        is_sorted: bool,
        orders: &'a [Order],
        nulls: &'a [NullOrder],
    ) -> Quantiles<'a> {
        Quantiles {
            table: self,
            q,
            interp,
            is_sorted,
            orders,
            nulls,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying (new) --

    /// Scatter rows from source table into this table using boolean mask.
    pub fn boolean_mask_scatter<'a>(
        &'a self,
        source: &'a Table,
        mask: &'a ColumnView<'a>,
    ) -> BooleanMaskScatter<'a> {
        BooleanMaskScatter {
            table: self,
            source,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Scatter scalar values (one per column) to specified indices in this table.
    pub fn scatter_scalars<'a>(
        &'a self,
        scalars: &'a [Scalar],
        scatter_map: &'a ColumnView<'a>,
    ) -> ScatterScalars<'a> {
        ScatterScalars {
            table: self,
            scalars,
            scatter_map,
            stream: Stream::default_stream(),
        }
    }

    /// Scatter scalar values (one per column) into rows where mask is true.
    pub fn boolean_mask_scatter_scalars<'a>(
        &'a self,
        scalars: &'a [Scalar],
        mask: &'a ColumnView<'a>,
    ) -> BooleanMaskScatterScalars<'a> {
        BooleanMaskScatterScalars {
            table: self,
            scalars,
            mask,
            stream: Stream::default_stream(),
        }
    }

    /// Repeat table rows using per-row counts from a column.
    pub fn repeat_by_column<'a>(&'a self, counts: &'a ColumnView<'a>) -> RepeatByColumn<'a> {
        RepeatByColumn {
            table: self,
            counts,
            stream: Stream::default_stream(),
        }
    }

    // -- Explode --

    /// Explodes a list column, expanding each list element into its own row.
    pub fn explode(&self, column_idx: usize) -> Explode<'_> {
        Explode {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Explodes a list column with a position column added.
    pub fn explode_position(&self, column_idx: usize) -> ExplodePosition<'_> {
        ExplodePosition {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Explodes a list column, keeping null/empty list rows as null rows.
    pub fn explode_outer(&self, column_idx: usize) -> ExplodeOuter<'_> {
        ExplodeOuter {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Explode outer with position column.
    pub fn explode_outer_position(&self, column_idx: usize) -> ExplodeOuterPosition<'_> {
        ExplodeOuterPosition {
            table: self,
            column_idx,
            stream: Stream::default_stream(),
        }
    }

    /// Gather rows with out-of-bounds policy.
    /// If `nullify_oob` is true, out-of-bounds indices produce null rows;
    /// otherwise behavior is undefined for out-of-bounds indices.
    pub fn gather_checked<'a>(
        &'a self,
        indices: &'a ColumnView<'a>,
        nullify_oob: bool,
    ) -> GatherChecked<'a> {
        GatherChecked {
            table: self,
            indices,
            nullify_oob,
            stream: Stream::default_stream(),
        }
    }

    /// Compute the cross join (Cartesian product) with another table.
    pub fn cross_join<'a>(&'a self, right: &'a Table) -> CrossJoin<'a> {
        CrossJoin {
            table: self,
            right,
            stream: Stream::default_stream(),
        }
    }

    /// Partitions the table by a map column that assigns each row to a partition.
    pub fn partition_by_map<'a>(
        &'a self,
        partition_map: &'a ColumnView<'a>,
        num_partitions: usize,
    ) -> PartitionByMap<'a> {
        PartitionByMap {
            table: self,
            partition_map,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Returns partition offsets for partition-by-map.
    pub fn partition_by_map_offsets<'a>(
        &'a self,
        partition_map: &'a ColumnView<'a>,
        num_partitions: usize,
    ) -> PartitionByMapOffsets<'a> {
        PartitionByMapOffsets {
            table: self,
            partition_map,
            num_partitions,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct rows in this table.
    ///
    /// `nulls_equal`: if true, all nulls are considered equal (count as one distinct value).
    pub fn distinct_count(&self, nulls_equal: bool) -> DistinctCount<'_> {
        DistinctCount {
            table: self,
            nulls_equal,
            stream: Stream::default_stream(),
        }
    }

    // -- Bitmask combining --

    /// Bitwise AND of all column null masks. Returns a BOOL8 column where
    /// `true` means the row is valid in ALL columns.
    pub fn bitmask_and_to_bools(&self) -> BitmaskAndToBools<'_> {
        BitmaskAndToBools {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Bitwise OR of all column null masks. Returns a BOOL8 column where
    /// `true` means the row is valid in ANY column.
    pub fn bitmask_or_to_bools(&self) -> BitmaskOrToBools<'_> {
        BitmaskOrToBools {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Groupby scan/shift/replace_nulls --

    /// Performs cumulative (scan) aggregation within groups.
    pub fn groupby_scan<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        aggs: &'a [AggregationKind],
    ) -> GroupbyScan<'a> {
        GroupbyScan {
            table: self,
            key_columns,
            value_columns,
            aggs,
            stream: Stream::default_stream(),
        }
    }

    /// Shifts values within groups by specified offsets, filling with scalars.
    pub fn groupby_shift<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        offsets: &'a [i32],
        fill_values: &'a [Scalar],
    ) -> GroupbyShift<'a> {
        GroupbyShift {
            table: self,
            key_columns,
            value_columns,
            offsets,
            fill_values,
            stream: Stream::default_stream(),
        }
    }

    /// Replaces null values within groups using forward or backward fill.
    ///
    /// `policies` should be 0 (PRECEDING/forward) or 1 (FOLLOWING/backward)
    /// for each value column.
    pub fn groupby_replace_nulls<'a>(
        &'a self,
        key_columns: &'a [i32],
        value_columns: &'a [i32],
        policies: &'a [i32],
    ) -> GroupbyReplaceNulls<'a> {
        GroupbyReplaceNulls {
            table: self,
            key_columns,
            value_columns,
            policies,
            stream: Stream::default_stream(),
        }
    }

    // -- Unique count --

    /// Counts consecutive unique rows in the table.
    pub fn unique_count(&self, nulls_equal: bool) -> UniqueCount<'_> {
        UniqueCount {
            table: self,
            nulls_equal,
            stream: Stream::default_stream(),
        }
    }

    // -- Drop NaNs with threshold --

    /// Drops rows with NaN values, keeping rows with at least `threshold` non-NaN values.
    pub fn drop_nans_with_threshold<'a>(
        &'a self,
        keys: &'a [i32],
        threshold: usize,
    ) -> DropNansWithThreshold<'a> {
        DropNansWithThreshold {
            table: self,
            keys,
            threshold,
            stream: Stream::default_stream(),
        }
    }

    // -- Approximate distinct count --

    /// Estimates the approximate number of distinct rows using `HyperLogLog`.
    ///
    /// `precision` controls accuracy vs memory (4-18, default 12).
    /// Standard error ≈ 1.04 / sqrt(2^precision).
    pub fn approx_distinct_count(&self, precision: i32) -> ApproxDistinctCount<'_> {
        ApproxDistinctCount {
            table: self,
            precision,
            stream: Stream::default_stream(),
        }
    }

    /// Returns true if this table contains any nested columns (LIST, STRUCT).
    pub fn has_nested_columns(&self) -> bool {
        cudf_sys::ffi::table_has_nested_columns(&self.0)
    }

    /// Returns true if any nested column in this table has null values.
    pub fn has_nested_nulls(&self) -> bool {
        cudf_sys::ffi::table_has_nested_nulls(&self.0)
    }

    /// Returns true if any nested column in this table is nullable.
    pub fn has_nested_nullable_columns(&self) -> bool {
        cudf_sys::ffi::table_has_nested_nullable_columns(&self.0)
    }

    /// Creates a table from a `DLPack` `DLManagedTensor` pointer.
    ///
    /// The `managed_tensor_ptr` must point to a valid `DLManagedTensor`.
    pub fn from_dlpack(managed_tensor_ptr: usize) -> FromDlpack {
        FromDlpack {
            managed_tensor_ptr,
            stream: Stream::default_stream(),
        }
    }

    /// Converts this table into a `DLPack` `DLManagedTensor` pointer.
    ///
    /// All columns must have the same numeric type and zero null count.
    /// Returns the pointer as `usize`; the caller is responsible for calling
    /// the `DLManagedTensor`'s `deleter` to free it.
    pub fn to_dlpack(&self) -> ToDlpack<'_> {
        ToDlpack {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Concatenates all columns in this table into a single column.
    ///
    /// All columns must have the same data type.
    pub fn concatenate_columns(&self) -> ConcatenateTableColumns<'_> {
        ConcatenateTableColumns {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Concatenates two tables vertically (row-wise append).
    ///
    /// Both tables must have the same number of columns with matching types.
    pub fn concatenate<'a>(&'a self, other: &'a Table) -> ConcatenateWith<'a> {
        ConcatenateWith {
            table: self,
            other,
            stream: Stream::default_stream(),
        }
    }
}

/// An iterator over the columns of a [`Table`].
pub struct Columns<'a> {
    table: &'a Table,
    index: usize,
    len: usize,
}

impl<'a> Iterator for Columns<'a> {
    type Item = ColumnView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let col = self.table.column(self.index);
        if col.is_some() {
            self.index += 1;
        }
        col
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Columns<'_> {}

/// A builder for constructing a [`Table`] from individual [`Column`]s.
pub struct TableBuilder(UniquePtr<cudf_sys::ffi::TableBuilder>);

impl Default for TableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TableBuilder {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self(cudf_sys::ffi::new_table_builder())
    }

    /// Adds a column to the builder, transferring ownership.
    pub fn push_column(&mut self, col: Column) {
        cudf_sys::ffi::table_builder_add_column(self.0.pin_mut(), col.0);
    }

    /// Consumes the builder and returns a [`Table`].
    pub fn build(mut self) -> crate::Result<Table> {
        let tbl = cudf_sys::ffi::table_builder_build(self.0.pin_mut())?;
        Ok(Table(tbl))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::TypeId;
    use crate::scalar::Scalar;

    #[test]
    fn empty_table_default() {
        let table = Table::default();
        assert_eq!(table.columns_len(), 0);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn empty_table_column_out_of_bounds() {
        let table = Table::default();
        assert!(table.column(0).is_none());
    }

    #[test]
    fn empty_table_columns_iterator() {
        let table = Table::default();
        let cols: Vec<_> = table.columns().collect();
        assert!(cols.is_empty());
        assert_eq!(table.columns().len(), 0);
    }

    #[test]
    fn table_from_columns() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 3);
        let c2 = Column::from_scalar(&Scalar::from_f64(2.5), 3);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        let table = builder.build().unwrap();
        assert_eq!(table.columns_len(), 2);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn table_column_views() {
        let c1 = Column::from_scalar(&Scalar::from_i32(10), 4);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();

        let view = table.column(0).unwrap();
        assert_eq!(view.len(), 4);
        assert_eq!(view.type_id(), TypeId::INT32);
    }

    #[test]
    fn table_columns_iterator_with_data() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 2);
        let c2 = Column::from_scalar(&Scalar::from_f64(3.0), 2);
        let c3 = Column::from_scalar(&Scalar::from_bool(true), 2);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        builder.push_column(c2);
        builder.push_column(c3);
        let table = builder.build().unwrap();

        let cols: Vec<_> = table.columns().collect();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].type_id(), TypeId::INT32);
        assert_eq!(cols[1].type_id(), TypeId::FLOAT64);
        assert_eq!(cols[2].type_id(), TypeId::BOOL8);
    }

    #[test]
    fn table_alloc_bytes() {
        let c1 = Column::from_scalar(&Scalar::from_i32(1), 100);
        let mut builder = TableBuilder::new();
        builder.push_column(c1);
        let table = builder.build().unwrap();
        assert!(table.alloc_bytes() > 0);
    }

    #[test]
    fn table_builder_default() {
        let builder = TableBuilder::default();
        let table = builder.build().unwrap();
        assert_eq!(table.columns_len(), 0);
    }
}
