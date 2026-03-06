// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

use cxx::UniquePtr;

use crate::column::{Column, ColumnView};
use crate::error::Result;
use crate::groupby::AggregationKind;
use crate::scalar::Scalar;
use crate::sorting::{NullOrder, Order};
use crate::stream::Stream;

/// An owning GPU table (a set of columns of the same size).
///
/// Wraps a `cudf::table` via the CXX FFI layer. Dropping this value
/// frees the underlying GPU memory for all columns.
pub struct Table(pub(crate) UniquePtr<cudf_sys::ffi::Table>);

// GPU memory is globally accessible from any CPU thread.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Default for Table {
    /// Creates an empty table with zero columns and zero rows.
    fn default() -> Self {
        Self(cudf_sys::ffi::table_empty())
    }
}

impl Table {
    /// Returns the number of columns.
    pub fn columns_len(&self) -> usize {
        cudf_sys::ffi::table_num_columns(&self.0) as usize
    }

    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        cudf_sys::ffi::table_num_rows(&self.0) as usize
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
        let view = cudf_sys::ffi::table_get_column_view(&self.0, index as i32).ok()?;
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
    pub fn sort(&self, orders: &[Order], nulls: &[NullOrder]) -> Result<Table> {
        self.sort_on(orders, nulls, Stream::default_stream())
    }

    /// Sorts the table on a custom CUDA stream.
    pub fn sort_on(&self, orders: &[Order], nulls: &[NullOrder], stream: Stream) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::sort_table(&self.0, orders_i32, nulls_i32, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Sorts the table ascending with nulls before.
    pub fn sort_ascending(&self) -> Result<Table> {
        self.sort(&[], &[])
    }

    /// Sorts the table ascending on a custom CUDA stream.
    pub fn sort_ascending_on(&self, stream: Stream) -> Result<Table> {
        self.sort_on(&[], &[], stream)
    }

    /// Returns the sorted row indices.
    pub fn sorted_order(&self, orders: &[Order], nulls: &[NullOrder]) -> Result<Column> {
        self.sorted_order_on(orders, nulls, Stream::default_stream())
    }

    /// Returns the sorted row indices on a custom CUDA stream.
    pub fn sorted_order_on(
        &self,
        orders: &[Order],
        nulls: &[NullOrder],
        stream: Stream,
    ) -> Result<Column> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let col = cudf_sys::ffi::sorted_order(&self.0, orders_i32, nulls_i32, stream.as_raw())?;
        Ok(Column(col))
    }

    /// Returns whether the table rows are sorted according to the given orders.
    pub fn is_sorted(&self, orders: &[Order], nulls: &[NullOrder]) -> Result<bool> {
        self.is_sorted_on(orders, nulls, Stream::default_stream())
    }

    /// Checks sort order on a custom CUDA stream.
    pub fn is_sorted_on(
        &self,
        orders: &[Order],
        nulls: &[NullOrder],
        stream: Stream,
    ) -> Result<bool> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        cudf_sys::ffi::is_sorted_table(&self.0, orders_i32, nulls_i32, stream.as_raw())
            .map_err(Into::into)
    }

    // -- Joins --

    /// Performs an inner join with another table.
    pub fn inner_join(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
    ) -> Result<Table> {
        self.inner_join_on(right, left_on, right_on, Stream::default_stream())
    }

    /// Inner join on a custom CUDA stream.
    pub fn inner_join_on(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
        stream: Stream,
    ) -> Result<Table> {
        let t =
            cudf_sys::ffi::inner_join(&self.0, &right.0, left_on, right_on, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Performs a left join with another table.
    pub fn left_join(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
    ) -> Result<Table> {
        self.left_join_on(right, left_on, right_on, Stream::default_stream())
    }

    /// Left join on a custom CUDA stream.
    pub fn left_join_on(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
        stream: Stream,
    ) -> Result<Table> {
        let t =
            cudf_sys::ffi::left_join(&self.0, &right.0, left_on, right_on, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Performs a full outer join with another table.
    pub fn full_join(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
    ) -> Result<Table> {
        self.full_join_on(right, left_on, right_on, Stream::default_stream())
    }

    /// Full outer join on a custom CUDA stream.
    pub fn full_join_on(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
        stream: Stream,
    ) -> Result<Table> {
        let t =
            cudf_sys::ffi::full_join(&self.0, &right.0, left_on, right_on, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Performs a left semi join with another table.
    pub fn left_semi_join(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
    ) -> Result<Table> {
        self.left_semi_join_on(right, left_on, right_on, Stream::default_stream())
    }

    /// Left semi join on a custom CUDA stream.
    pub fn left_semi_join_on(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
        stream: Stream,
    ) -> Result<Table> {
        let t = cudf_sys::ffi::left_semi_join(
            &self.0,
            &right.0,
            left_on,
            right_on,
            stream.as_raw(),
        )?;
        Ok(Table(t))
    }

    /// Performs a left anti join with another table.
    pub fn left_anti_join(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
    ) -> Result<Table> {
        self.left_anti_join_on(right, left_on, right_on, Stream::default_stream())
    }

    /// Left anti join on a custom CUDA stream.
    pub fn left_anti_join_on(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
        stream: Stream,
    ) -> Result<Table> {
        let t = cudf_sys::ffi::left_anti_join(
            &self.0,
            &right.0,
            left_on,
            right_on,
            stream.as_raw(),
        )?;
        Ok(Table(t))
    }

    // -- GroupBy --

    /// Performs a single groupby aggregation on one value column.
    pub fn groupby(
        &self,
        key_columns: &[i32],
        value_column: i32,
        agg: AggregationKind,
    ) -> Result<Table> {
        self.groupby_on(key_columns, value_column, agg, Stream::default_stream())
    }

    /// Single groupby on a custom CUDA stream.
    pub fn groupby_on(
        &self,
        key_columns: &[i32],
        value_column: i32,
        agg: AggregationKind,
        stream: Stream,
    ) -> Result<Table> {
        let result = cudf_sys::ffi::groupby_single(
            &self.0,
            key_columns,
            value_column,
            agg.repr,
            stream.as_raw(),
        )?;
        Ok(Table(result))
    }

    /// Performs groupby with multiple aggregations on multiple value columns.
    pub fn groupby_multi(
        &self,
        key_columns: &[i32],
        value_columns: &[i32],
        aggs: &[AggregationKind],
    ) -> Result<Table> {
        self.groupby_multi_on(key_columns, value_columns, aggs, Stream::default_stream())
    }

    /// Multi groupby on a custom CUDA stream.
    pub fn groupby_multi_on(
        &self,
        key_columns: &[i32],
        value_columns: &[i32],
        aggs: &[AggregationKind],
        stream: Stream,
    ) -> Result<Table> {
        if value_columns.len() != aggs.len() {
            return Err(crate::error::Error::OutOfBounds {
                index: aggs.len(),
                len: value_columns.len(),
            });
        }
        let agg_kinds = unsafe { crate::enum_slice_as_i32(aggs) };
        let result = cudf_sys::ffi::groupby_multi(
            &self.0,
            key_columns,
            value_columns,
            agg_kinds,
            stream.as_raw(),
        )?;
        Ok(Table(result))
    }

    // -- Filter --

    /// Filters the table by a boolean mask column.
    pub fn filter(&self, mask: &ColumnView<'_>) -> Result<Table> {
        self.filter_on(mask, Stream::default_stream())
    }

    /// Filters on a custom CUDA stream.
    pub fn filter_on(&self, mask: &ColumnView<'_>, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::apply_boolean_mask(&self.0, mask.0, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Drops rows where all columns are null.
    pub fn drop_nulls(&self) -> Result<Table> {
        self.drop_nulls_on(Stream::default_stream())
    }

    /// Drops nulls on a custom CUDA stream.
    pub fn drop_nulls_on(&self, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::drop_nulls_all(&self.0, stream.as_raw())?;
        Ok(Table(t))
    }

    // -- Copying --

    /// Gathers rows using an index column.
    pub fn gather(&self, indices: &ColumnView<'_>) -> Result<Table> {
        self.gather_on(indices, Stream::default_stream())
    }

    /// Gathers on a custom CUDA stream.
    pub fn gather_on(&self, indices: &ColumnView<'_>, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::gather_table(&self.0, indices.0, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Creates an empty table with the same column types.
    pub fn empty_like(&self) -> Table {
        Table(cudf_sys::ffi::empty_like_table(&self.0))
    }

    // -- Hashing --

    /// Computes MurmurHash3 32-bit hash of each row.
    pub fn murmur3(&self, seed: u32) -> Column {
        self.murmur3_on(seed, Stream::default_stream())
    }

    /// MurmurHash3 on a custom CUDA stream.
    pub fn murmur3_on(&self, seed: u32, stream: Stream) -> Column {
        Column(cudf_sys::ffi::hash_murmur3(&self.0, seed, stream.as_raw()))
    }

    /// Computes XXHash64 hash of each row.
    pub fn xxhash64(&self, seed: u64) -> Column {
        self.xxhash64_on(seed, Stream::default_stream())
    }

    /// XXHash64 on a custom CUDA stream.
    pub fn xxhash64_on(&self, seed: u64, stream: Stream) -> Column {
        Column(cudf_sys::ffi::hash_xxhash64(&self.0, seed, stream.as_raw()))
    }

    /// Computes MD5 hash of each row.
    pub fn md5(&self) -> Column {
        self.md5_on(Stream::default_stream())
    }

    /// MD5 on a custom CUDA stream.
    pub fn md5_on(&self, stream: Stream) -> Column {
        Column(cudf_sys::ffi::hash_md5(&self.0, stream.as_raw()))
    }

    /// Computes SHA-256 hash of each row.
    pub fn sha256(&self) -> Column {
        self.sha256_on(Stream::default_stream())
    }

    /// SHA-256 on a custom CUDA stream.
    pub fn sha256_on(&self, stream: Stream) -> Column {
        Column(cudf_sys::ffi::hash_sha256(&self.0, stream.as_raw()))
    }

    /// SHA-1 hash of each row (returns STRING column with 40-char hex).
    pub fn sha1(&self) -> Result<Column> {
        let c = cudf_sys::ffi::hash_sha1(&self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// MurmurHash3 128-bit hash (returns Table of two UINT64 columns).
    pub fn murmurhash3_x64_128(&self, seed: u64) -> Result<Table> {
        let t = cudf_sys::ffi::hash_murmurhash3_x64_128(&self.0, seed, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// XXHash 32-bit hash of each row.
    pub fn xxhash_32(&self, seed: u32) -> Result<Column> {
        let c = cudf_sys::ffi::hash_xxhash_32(&self.0, seed, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// SHA-224 hash of each row (returns STRING column).
    pub fn sha224(&self) -> Result<Column> {
        let c = cudf_sys::ffi::hash_sha224(&self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// SHA-384 hash of each row (returns STRING column).
    pub fn sha384(&self) -> Result<Column> {
        let c = cudf_sys::ffi::hash_sha384(&self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// SHA-512 hash of each row (returns STRING column).
    pub fn sha512(&self) -> Result<Column> {
        let c = cudf_sys::ffi::hash_sha512(&self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Approximate per-row bit count.
    pub fn row_bit_count(&self) -> Result<Column> {
        let c = cudf_sys::ffi::row_bit_count(&self.0, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Per-segment cumulative row bit count.
    pub fn segmented_row_bit_count(&self, segment_length: i32) -> Result<Column> {
        let c = cudf_sys::ffi::segmented_row_bit_count(&self.0, segment_length, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Convert column elements to lists of bytes.
    pub fn byte_cast(col: &ColumnView<'_>, flip_endian: bool) -> Result<Column> {
        let c = cudf_sys::ffi::byte_cast_column(col.0, flip_endian, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    // -- Merge --

    /// Merges two sorted tables maintaining sort order.
    pub fn merge(
        &self,
        right: &Table,
        key_columns: &[i32],
        orders: &[Order],
        null_orders: &[NullOrder],
    ) -> Result<Table> {
        self.merge_on(right, key_columns, orders, null_orders, Stream::default_stream())
    }

    /// Merge on a custom CUDA stream.
    pub fn merge_on(
        &self,
        right: &Table,
        key_columns: &[i32],
        orders: &[Order],
        null_orders: &[NullOrder],
        stream: Stream,
    ) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(null_orders) };
        let tbl = cudf_sys::ffi::merge_tables(
            &self.0,
            &right.0,
            key_columns,
            orders_i32,
            nulls_i32,
            stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }

    // -- Partitioning --

    /// Hash-partitions the table.
    pub fn hash_partition(&self, columns: &[i32], num_partitions: usize) -> Result<Table> {
        self.hash_partition_on(columns, num_partitions, Stream::default_stream())
    }

    /// Hash partition on a custom CUDA stream.
    pub fn hash_partition_on(
        &self,
        columns: &[i32],
        num_partitions: usize,
        stream: Stream,
    ) -> Result<Table> {
        let tbl = cudf_sys::ffi::hash_partition_table(
            &self.0,
            columns,
            num_partitions as i32,
            stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }

    /// Returns partition offsets for hash partitioning.
    pub fn hash_partition_offsets(
        &self,
        columns: &[i32],
        num_partitions: usize,
    ) -> Result<Vec<usize>> {
        self.hash_partition_offsets_on(columns, num_partitions, Stream::default_stream())
    }

    /// Hash partition offsets on a custom CUDA stream.
    pub fn hash_partition_offsets_on(
        &self,
        columns: &[i32],
        num_partitions: usize,
        stream: Stream,
    ) -> Result<Vec<usize>> {
        let offsets = cudf_sys::ffi::hash_partition_offsets(
            &self.0,
            columns,
            num_partitions as i32,
            stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(|o| o as usize).collect())
    }

    /// Round-robin partitions the table.
    pub fn round_robin(&self, num_partitions: usize, start: usize) -> Result<Table> {
        self.round_robin_on(num_partitions, start, Stream::default_stream())
    }

    /// Round-robin on a custom CUDA stream.
    pub fn round_robin_on(
        &self,
        num_partitions: usize,
        start: usize,
        stream: Stream,
    ) -> Result<Table> {
        let tbl = cudf_sys::ffi::round_robin_partition_table(
            &self.0,
            num_partitions as i32,
            start as i32,
            stream.as_raw(),
        )?;
        Ok(Table(tbl))
    }

    /// Returns partition offsets for round-robin partitioning.
    pub fn round_robin_offsets(
        &self,
        num_partitions: usize,
        start: usize,
    ) -> Result<Vec<usize>> {
        self.round_robin_offsets_on(num_partitions, start, Stream::default_stream())
    }

    /// Round-robin offsets on a custom CUDA stream.
    pub fn round_robin_offsets_on(
        &self,
        num_partitions: usize,
        start: usize,
        stream: Stream,
    ) -> Result<Vec<usize>> {
        let offsets = cudf_sys::ffi::round_robin_partition_offsets(
            &self.0,
            num_partitions as i32,
            start as i32,
            stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(|o| o as usize).collect())
    }

    // -- Reshape --

    /// Interleaves columns into a single column.
    pub fn interleave_columns(&self) -> Result<Column> {
        self.interleave_columns_on(Stream::default_stream())
    }

    /// Interleave on a custom CUDA stream.
    pub fn interleave_columns_on(&self, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::interleave_columns(&self.0, stream.as_raw())?;
        Ok(Column(col))
    }

    /// Tiles (repeats) the rows `count` times.
    pub fn tile(&self, count: usize) -> Result<Table> {
        self.tile_on(count, Stream::default_stream())
    }

    /// Tile on a custom CUDA stream.
    pub fn tile_on(&self, count: usize, stream: Stream) -> Result<Table> {
        let tbl = cudf_sys::ffi::tile_table(&self.0, count as i32, stream.as_raw())?;
        Ok(Table(tbl))
    }

    // -- Fill (table-level) --

    /// Repeats each row `count` times.
    pub fn repeat(&self, count: usize) -> Result<Table> {
        self.repeat_on(count, Stream::default_stream())
    }

    /// Repeat on a custom CUDA stream.
    pub fn repeat_on(&self, count: usize, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::repeat_table(&self.0, count as i32, stream.as_raw())?;
        Ok(Table(t))
    }

    // -- Transform --

    /// Encodes table rows as integer indices into sorted distinct rows.
    pub fn encode(&self) -> Result<Column> {
        self.encode_on(Stream::default_stream())
    }

    /// Encode on a custom CUDA stream.
    pub fn encode_on(&self, stream: Stream) -> Result<Column> {
        let col = cudf_sys::ffi::encode_table(&self.0, stream.as_raw())?;
        Ok(Column(col))
    }

    /// Returns the sorted distinct key rows from encoding.
    pub fn encode_keys(&self) -> Result<Table> {
        self.encode_keys_on(Stream::default_stream())
    }

    /// Encode keys on a custom CUDA stream.
    pub fn encode_keys_on(&self, stream: Stream) -> Result<Table> {
        let tbl = cudf_sys::ffi::encode_keys(&self.0, stream.as_raw())?;
        Ok(Table(tbl))
    }

    // -- Stream compaction --

    /// Drops rows where any of the specified key columns contain NaN.
    pub fn drop_nans(&self, keys: &[i32]) -> Result<Table> {
        self.drop_nans_on(keys, Stream::default_stream())
    }

    /// drop_nans on a custom CUDA stream.
    pub fn drop_nans_on(&self, keys: &[i32], stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::drop_nans(&self.0, keys, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Drops rows where the number of non-null key values is below the threshold.
    pub fn drop_nulls_with_threshold(&self, keys: &[i32], threshold: usize) -> Result<Table> {
        self.drop_nulls_with_threshold_on(keys, threshold, Stream::default_stream())
    }

    /// drop_nulls_with_threshold on a custom CUDA stream.
    pub fn drop_nulls_with_threshold_on(
        &self,
        keys: &[i32],
        threshold: usize,
        stream: Stream,
    ) -> Result<Table> {
        let t = cudf_sys::ffi::drop_nulls_with_threshold(
            &self.0,
            keys,
            threshold as i32,
            stream.as_raw(),
        )?;
        Ok(Table(t))
    }

    /// Returns unique consecutive rows based on key columns.
    pub fn unique(&self, keys: &[i32]) -> Result<Table> {
        self.unique_on(keys, Stream::default_stream())
    }

    /// unique on a custom CUDA stream.
    pub fn unique_on(&self, keys: &[i32], stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::unique_table(&self.0, keys, 0, 0, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Returns distinct rows based on key columns.
    pub fn distinct(&self, keys: &[i32]) -> Result<Table> {
        self.distinct_on(keys, Stream::default_stream())
    }

    /// distinct on a custom CUDA stream.
    pub fn distinct_on(&self, keys: &[i32], stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::distinct_table(&self.0, keys, 0, 0, 0, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Returns distinct rows preserving input order.
    pub fn stable_distinct(&self, keys: &[i32]) -> Result<Table> {
        self.stable_distinct_on(keys, Stream::default_stream())
    }

    /// stable_distinct on a custom CUDA stream.
    pub fn stable_distinct_on(&self, keys: &[i32], stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::stable_distinct_table(&self.0, keys, 0, 0, 0, stream.as_raw())?;
        Ok(Table(t))
    }

    // -- Copying extras --

    /// Scatters source table rows into this table at given indices.
    pub fn scatter(&self, source: &Table, scatter_map: &ColumnView<'_>) -> Result<Table> {
        self.scatter_on(source, scatter_map, Stream::default_stream())
    }

    /// scatter on a custom CUDA stream.
    pub fn scatter_on(
        &self,
        source: &Table,
        scatter_map: &ColumnView<'_>,
        stream: Stream,
    ) -> Result<Table> {
        let t = cudf_sys::ffi::scatter_table(
            &source.0,
            scatter_map.0,
            &self.0,
            stream.as_raw(),
        )?;
        Ok(Table(t))
    }

    /// Reverses the rows of the table.
    pub fn reverse(&self) -> Result<Table> {
        self.reverse_on(Stream::default_stream())
    }

    /// reverse on a custom CUDA stream.
    pub fn reverse_on(&self, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::reverse_table(&self.0, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Extracts a slice [begin, end) of the table as a new owned table.
    pub fn slice(&self, begin: usize, end: usize) -> Result<Table> {
        self.slice_on(begin, end, Stream::default_stream())
    }

    /// slice on a custom CUDA stream.
    pub fn slice_on(&self, begin: usize, end: usize, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::slice_table(&self.0, begin as i32, end as i32, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Randomly samples rows from the table.
    pub fn sample(&self, n: usize, with_replacement: bool, seed: i64) -> Result<Table> {
        self.sample_on(n, with_replacement, seed, Stream::default_stream())
    }

    /// sample on a custom CUDA stream.
    pub fn sample_on(
        &self,
        n: usize,
        with_replacement: bool,
        seed: i64,
        stream: Stream,
    ) -> Result<Table> {
        let t = cudf_sys::ffi::sample_table(
            &self.0,
            n as i32,
            with_replacement,
            seed,
            stream.as_raw(),
        )?;
        Ok(Table(t))
    }

    // -- Transpose --

    /// Transposes the table (rows become columns).
    pub fn transpose(&self) -> Result<Table> {
        self.transpose_on(Stream::default_stream())
    }

    /// transpose on a custom CUDA stream.
    pub fn transpose_on(&self, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::transpose_table(&self.0, stream.as_raw())?;
        Ok(Table(t))
    }

    // -- Search --

    /// Finds lower bound insertion points in this sorted table.
    pub fn lower_bound(
        &self,
        needles: &Table,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Column> {
        self.lower_bound_on(needles, orders, nulls, Stream::default_stream())
    }

    /// Lower bound on a custom CUDA stream.
    pub fn lower_bound_on(
        &self,
        needles: &Table,
        orders: &[Order],
        nulls: &[NullOrder],
        stream: Stream,
    ) -> Result<Column> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let c = cudf_sys::ffi::lower_bound(
            &self.0,
            &needles.0,
            orders_i32,
            nulls_i32,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Finds upper bound insertion points in this sorted table.
    pub fn upper_bound(
        &self,
        needles: &Table,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Column> {
        self.upper_bound_on(needles, orders, nulls, Stream::default_stream())
    }

    /// Upper bound on a custom CUDA stream.
    pub fn upper_bound_on(
        &self,
        needles: &Table,
        orders: &[Order],
        nulls: &[NullOrder],
        stream: Stream,
    ) -> Result<Column> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let c = cudf_sys::ffi::upper_bound(
            &self.0,
            &needles.0,
            orders_i32,
            nulls_i32,
            stream.as_raw(),
        )?;
        Ok(Column(c))
    }

    // -- Sorting (new) --

    /// Stable sorted order (preserves order of equal elements).
    pub fn stable_sorted_order(&self, orders: &[Order], nulls: &[NullOrder]) -> Result<Column> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let c = cudf_sys::ffi::stable_sorted_order(&self.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Stable sort (preserves order of equal elements).
    pub fn stable_sort(&self, orders: &[Order], nulls: &[NullOrder]) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::stable_sort_table(&self.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// Sort this (values) table by a separate keys table.
    pub fn sort_by_key(&self, keys: &Table, orders: &[Order], nulls: &[NullOrder]) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::sort_by_key(&self.0, &keys.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// Stable sort this (values) table by a separate keys table.
    pub fn stable_sort_by_key(&self, keys: &Table, orders: &[Order], nulls: &[NullOrder]) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::stable_sort_by_key(&self.0, &keys.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// Segmented sorted order (sort within segments).
    pub fn segmented_sorted_order(
        &self,
        segment_offsets: &ColumnView<'_>,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Column> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let c = cudf_sys::ffi::segmented_sorted_order(&self.0, segment_offsets.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Stable segmented sorted order (preserves relative order of equal elements).
    pub fn stable_segmented_sorted_order(
        &self,
        segment_offsets: &ColumnView<'_>,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Column> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let c = cudf_sys::ffi::stable_segmented_sorted_order(&self.0, segment_offsets.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Column(c))
    }

    /// Segmented sort by key.
    pub fn segmented_sort_by_key(
        &self,
        keys: &Table,
        segment_offsets: &ColumnView<'_>,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::segmented_sort_by_key(&self.0, &keys.0, segment_offsets.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// Stable segmented sort by key.
    pub fn stable_segmented_sort_by_key(
        &self,
        keys: &Table,
        segment_offsets: &ColumnView<'_>,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::stable_segmented_sort_by_key(&self.0, &keys.0, segment_offsets.0, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// Table-level quantile rows.
    pub fn quantiles(
        &self,
        q: &[f64],
        interp: crate::quantile::Interpolation,
        is_sorted: bool,
        orders: &[Order],
        nulls: &[NullOrder],
    ) -> Result<Table> {
        let orders_i32 = unsafe { crate::enum_slice_as_i32(orders) };
        let nulls_i32 = unsafe { crate::enum_slice_as_i32(nulls) };
        let t = cudf_sys::ffi::quantiles_table(&self.0, q, interp.repr, is_sorted, orders_i32, nulls_i32, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    // -- Copying (new) --

    /// Scatter rows from source table into this table using boolean mask.
    pub fn boolean_mask_scatter(&self, source: &Table, mask: &ColumnView<'_>) -> Result<Table> {
        let t = cudf_sys::ffi::boolean_mask_scatter_table(&source.0, &self.0, mask.0, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    /// Scatter scalar values (one per column) to specified indices in this table.
    pub fn scatter_scalars(
        &self,
        scalars: &[Scalar],
        scatter_map: &ColumnView<'_>,
    ) -> Result<Table> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in scalars {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::ffi::scatter_scalars(
            list.pin_mut(),
            scatter_map.0,
            &self.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Table(t))
    }

    /// Scatter scalar values (one per column) into rows where mask is true.
    pub fn boolean_mask_scatter_scalars(
        &self,
        scalars: &[Scalar],
        mask: &ColumnView<'_>,
    ) -> Result<Table> {
        let mut list = cudf_sys::ffi::new_scalar_list();
        for s in scalars {
            let ffi = crate::scalar::scalar_to_ffi(s);
            cudf_sys::ffi::scalar_list_add(list.pin_mut(), ffi);
        }
        let t = cudf_sys::ffi::boolean_mask_scatter_scalars(
            list.pin_mut(),
            &self.0,
            mask.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Table(t))
    }

    /// Repeat table rows using per-row counts from a column.
    pub fn repeat_by_column(&self, counts: &ColumnView<'_>) -> Result<Table> {
        let t = cudf_sys::ffi::repeat_table_column(&self.0, counts.0, Stream::default_stream().as_raw())?;
        Ok(Table(t))
    }

    // -- Explode --

    /// Explodes a list column, expanding each list element into its own row.
    pub fn explode(&self, column_idx: usize) -> Result<Table> {
        self.explode_on(column_idx, Stream::default_stream())
    }

    /// explode on a custom CUDA stream.
    pub fn explode_on(&self, column_idx: usize, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::explode_table(&self.0, column_idx as i32, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Explodes a list column with a position column added.
    pub fn explode_position(&self, column_idx: usize) -> Result<Table> {
        self.explode_position_on(column_idx, Stream::default_stream())
    }

    /// explode_position on a custom CUDA stream.
    pub fn explode_position_on(&self, column_idx: usize, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::explode_position_table(&self.0, column_idx as i32, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Explodes a list column, keeping null/empty list rows as null rows.
    pub fn explode_outer(&self, column_idx: usize) -> Result<Table> {
        self.explode_outer_on(column_idx, Stream::default_stream())
    }

    /// explode_outer on a custom CUDA stream.
    pub fn explode_outer_on(&self, column_idx: usize, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::explode_outer_table(&self.0, column_idx as i32, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Explode outer with position column.
    pub fn explode_outer_position(&self, column_idx: usize) -> Result<Table> {
        self.explode_outer_position_on(column_idx, Stream::default_stream())
    }

    /// explode_outer_position on a custom CUDA stream.
    pub fn explode_outer_position_on(&self, column_idx: usize, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::explode_outer_position_table(&self.0, column_idx as i32, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Gather rows with out-of-bounds policy.
    /// If `nullify_oob` is true, out-of-bounds indices produce null rows;
    /// otherwise behavior is undefined for out-of-bounds indices.
    pub fn gather_checked(&self, indices: &ColumnView<'_>, nullify_oob: bool) -> Result<Table> {
        self.gather_checked_on(indices, nullify_oob, Stream::default_stream())
    }

    /// gather_checked on a custom CUDA stream.
    pub fn gather_checked_on(&self, indices: &ColumnView<'_>, nullify_oob: bool, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::gather_table_checked(&self.0, indices.0, nullify_oob, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Compute the cross join (Cartesian product) with another table.
    pub fn cross_join(&self, right: &Table) -> Result<Table> {
        self.cross_join_on(right, Stream::default_stream())
    }

    /// Cross join on a custom CUDA stream.
    pub fn cross_join_on(&self, right: &Table, stream: Stream) -> Result<Table> {
        let t = cudf_sys::ffi::cross_join(&self.0, &right.0, stream.as_raw())?;
        Ok(Table(t))
    }

    /// Partitions the table by a map column that assigns each row to a partition.
    pub fn partition_by_map(
        &self,
        partition_map: &ColumnView<'_>,
        num_partitions: usize,
    ) -> Result<Table> {
        self.partition_by_map_on(partition_map, num_partitions, Stream::default_stream())
    }

    /// partition_by_map on a custom CUDA stream.
    pub fn partition_by_map_on(
        &self,
        partition_map: &ColumnView<'_>,
        num_partitions: usize,
        stream: Stream,
    ) -> Result<Table> {
        let t = cudf_sys::ffi::partition_by_map(
            &self.0,
            partition_map.0,
            num_partitions as i32,
            stream.as_raw(),
        )?;
        Ok(Table(t))
    }

    /// Returns partition offsets for partition-by-map.
    pub fn partition_by_map_offsets(
        &self,
        partition_map: &ColumnView<'_>,
        num_partitions: usize,
    ) -> Result<Vec<usize>> {
        self.partition_by_map_offsets_on(partition_map, num_partitions, Stream::default_stream())
    }

    // -- Bitmask combining --

    /// Bitwise AND of all column null masks. Returns a BOOL8 column where
    /// `true` means the row is valid in ALL columns.
    pub fn bitmask_and_to_bools(&self) -> Result<Column> {
        let c = cudf_sys::ffi::bitmask_and_to_bools(
            &self.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    /// Bitwise OR of all column null masks. Returns a BOOL8 column where
    /// `true` means the row is valid in ANY column.
    pub fn bitmask_or_to_bools(&self) -> Result<Column> {
        let c = cudf_sys::ffi::bitmask_or_to_bools(
            &self.0,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Column(c))
    }

    /// partition_by_map_offsets on a custom CUDA stream.
    pub fn partition_by_map_offsets_on(
        &self,
        partition_map: &ColumnView<'_>,
        num_partitions: usize,
        stream: Stream,
    ) -> Result<Vec<usize>> {
        let offsets = cudf_sys::ffi::partition_by_map_offsets(
            &self.0,
            partition_map.0,
            num_partitions as i32,
            stream.as_raw(),
        )?;
        Ok(offsets.into_iter().map(|o| o as usize).collect())
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
