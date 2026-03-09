// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Contiguous splitting, packing, and unpacking of tables.
//!
//! Packing serialises a [`Table`] into a contiguous device buffer plus host
//! metadata, enabling efficient IPC or device-to-device transfer.

use cxx::UniquePtr;

use crate::error::Result;
use crate::stream::Stream;
use crate::table::Table;

/// A table packed into a contiguous device buffer with host metadata.
///
/// Packing serializes all column data and null masks into a single contiguous
/// GPU allocation, plus a host-side metadata buffer that describes the table
/// schema. This representation is efficient for IPC, device-to-device transfer,
/// or storage.
///
/// Created by [`Table::pack`]. Use [`unpack`](PackedColumns::unpack) to
/// reconstruct an owned [`Table`].
///
/// # Examples
///
/// ```ignore
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::GpuOp;
/// use cudf::table::TableBuilder;
///
/// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
/// let mut builder = TableBuilder::new();
/// builder.push_column(col);
/// let table = builder.build()?;
///
/// let packed = table.pack().call()?;
/// let restored = packed.unpack()?;
/// assert_eq!(restored.len(), 3);
/// ```
#[doc(alias = "packed_columns")]
pub struct PackedColumns(UniquePtr<cudf_sys::contiguous_split::ffi::PackedColumns>);

impl PackedColumns {
    /// Reconstructs an owned [`Table`] from the packed representation.
    ///
    /// The returned table has the same schema and data as the original table
    /// that was packed.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails (e.g. corrupted metadata).
    pub fn unpack(&self) -> Result<Table> {
        let t = cudf_sys::contiguous_split::ffi::unpack_packed(&self.0)?;
        Ok(Table(t))
    }

    /// Returns the host metadata as a byte vector.
    ///
    /// The metadata describes the table schema (column types, offsets, null
    /// masks) and is needed to reconstruct the table from the GPU data buffer.
    pub fn metadata(&self) -> Vec<u8> {
        self.0.metadata_to_host()
    }

    /// Returns the size of the host metadata in bytes.
    pub fn metadata_size(&self) -> usize {
        self.0.metadata_size()
    }

    /// Returns the size of the GPU data in bytes.
    pub fn gpu_data_size(&self) -> usize {
        self.0.gpu_data_size()
    }
}

/// A vector of table partitions from [`Table::contiguous_split`].
///
/// Each partition is stored contiguously in its own device memory buffer. Use
/// [`unpack`](PackedTableVec::unpack) to reconstruct individual partitions as
/// owned [`Table`] instances.
pub struct PackedTableVec(UniquePtr<cudf_sys::contiguous_split::ffi::PackedTableVec>);

impl PackedTableVec {
    /// Returns the number of partitions.
    pub fn len(&self) -> usize {
        self.0.size()
    }

    /// Returns `true` if there are no partitions.
    pub fn is_empty(&self) -> bool {
        self.0.size() == 0
    }

    /// Reconstructs partition at `index` into an owned [`Table`].
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds or the libcudf call fails.
    pub fn unpack(&self, index: usize) -> Result<Table> {
        let t = self.0.unpack_at(index)?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::pack`].
///
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct Pack<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for Pack<'_> {
    type Output = PackedColumns;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let p = cudf_sys::contiguous_split::ffi::pack_table(&self.table.0, self.stream.as_raw())?;
        Ok(PackedColumns(p))
    }
}

/// Builder for [`Table::packed_size`].
///
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct PackedSize<'a> {
    table: &'a Table,
    stream: Stream,
}

impl crate::stream::GpuOp for PackedSize<'_> {
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let n =
            cudf_sys::contiguous_split::ffi::packed_size_of(&self.table.0, self.stream.as_raw())?;
        Ok(n)
    }
}

/// Builder for [`Table::contiguous_split`].
///
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct ContiguousSplit<'a> {
    table: &'a Table,
    splits: &'a [i32],
    stream: Stream,
}

impl crate::stream::GpuOp for ContiguousSplit<'_> {
    type Output = PackedTableVec;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let v = cudf_sys::contiguous_split::ffi::contiguous_split_table(
            &self.table.0,
            self.splits,
            self.stream.as_raw(),
        )?;
        Ok(PackedTableVec(v))
    }
}

impl Table {
    /// Packs this table into a contiguous device buffer with host metadata.
    ///
    /// Serializes all column data, null masks, and child column data into a
    /// single contiguous GPU allocation. The result also contains host-side
    /// metadata describing the table schema, enabling reconstruction via
    /// [`PackedColumns::unpack`].
    ///
    /// This is useful for efficient IPC, serialization, or device-to-device
    /// transfer.
    ///
    /// # Returns
    ///
    /// A [`PackedColumns`] containing the contiguous GPU buffer and host
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::stream::GpuOp;
    ///
    /// let packed = table.pack().call()?;
    /// let restored = packed.unpack()?;
    /// assert_eq!(restored.len(), table.len());
    /// ```
    ///
    /// Returns a [`Pack`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    pub fn pack(&self) -> Pack<'_> {
        Pack {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the number of bytes required to pack this table.
    ///
    /// Computes the total GPU memory needed for [`pack`](Table::pack) without
    /// actually performing the packing. Useful for pre-allocating buffers.
    ///
    /// # Returns
    ///
    /// The size in bytes as `usize`.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// Returns a [`PackedSize`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    pub fn packed_size(&self) -> PackedSize<'_> {
        PackedSize {
            table: self,
            stream: Stream::default_stream(),
        }
    }

    /// Splits this table into contiguous partitions.
    ///
    /// Each partition is stored in its own contiguous device memory buffer,
    /// enabling efficient independent transfer or processing.
    ///
    /// # Arguments
    ///
    /// * `splits` -- Row indices at which to split. The indices are
    ///   **exclusive** upper bounds. For example, `&[2, 5]` on a 10-row
    ///   table produces three partitions: rows `[0..2)`, `[2..5)`, and
    ///   `[5..10)`. Values must be sorted and in range `[0, num_rows]`.
    ///
    /// # Returns
    ///
    /// A [`PackedTableVec`] with `splits.len() + 1` partitions. Use
    /// [`PackedTableVec::unpack`] to reconstruct individual partitions.
    ///
    /// # Errors
    ///
    /// Returns an error if split indices are out of bounds or not sorted.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::stream::GpuOp;
    ///
    /// // Split a 6-row table at rows 2 and 4
    /// let parts = table.contiguous_split(&[2, 4]).call()?;
    /// assert_eq!(parts.len(), 3); // [0..2), [2..4), [4..6)
    /// let first = parts.unpack(0)?;
    /// assert_eq!(first.len(), 2);
    /// ```
    ///
    /// Returns a [`ContiguousSplit`] builder. Use `.stream()` to set a
    /// custom CUDA stream, then `.call()` to execute.
    pub fn contiguous_split<'a>(&'a self, splits: &'a [i32]) -> ContiguousSplit<'a> {
        ContiguousSplit {
            table: self,
            splits,
            stream: Stream::default_stream(),
        }
    }
}
