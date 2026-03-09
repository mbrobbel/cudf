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
/// Created by [`Pack`](crate::table::Table::pack). Use [`unpack`](PackedColumns::unpack) to
/// reconstruct an owned [`Table`].
#[doc(alias = "packed_columns")]
pub struct PackedColumns(UniquePtr<cudf_sys::contiguous_split::ffi::PackedColumns>);

impl PackedColumns {
    /// Reconstructs an owned table from the packed representation.
    pub fn unpack(&self) -> Result<Table> {
        let t = cudf_sys::contiguous_split::ffi::unpack_packed(&self.0)?;
        Ok(Table(t))
    }

    /// Returns the host metadata as bytes.
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

/// A vector of table partitions from [`contiguous_split`](crate::table::Table::contiguous_split).
///
/// Each partition is stored contiguously in device memory. Use
/// [`unpack`](PackedTableVec::unpack) to reconstruct individual partitions.
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

    /// Reconstructs partition at `index` into an owned table.
    pub fn unpack(&self, index: usize) -> Result<Table> {
        let t = self.0.unpack_at(index)?;
        Ok(Table(t))
    }
}

/// Builder for [`Table::pack`].
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
        let p = cudf_sys::contiguous_split::ffi::pack_table(
            &self.table.0,
            self.stream.as_raw(),
        )?;
        Ok(PackedColumns(p))
    }
}

/// Builder for [`Table::packed_size`].
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
        let n = cudf_sys::contiguous_split::ffi::packed_size_of(
            &self.table.0,
            self.stream.as_raw(),
        )?;
        Ok(n)
    }
}

/// Builder for [`Table::contiguous_split`].
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
    /// `splits` contains the row indices at which to split. For example,
    /// `&[2, 5]` on a 10-row table produces three partitions: rows 0..2,
    /// 2..5, and 5..10.
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
