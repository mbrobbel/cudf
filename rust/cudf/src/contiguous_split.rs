// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Contiguous splitting, packing, and unpacking of tables.
//!
//! Packing serialises a [`Table`] into a contiguous device buffer plus host
//! metadata, enabling efficient IPC or device-to-device transfer.

use cxx::UniquePtr;
use rmm::device::{DeviceId, current_device};
use rmm::gpu_context::Allocator;
use rmm::gpu_context::ContextBound;
use rmm::gpu_context::Execution;

use crate::error::Result;
use crate::stream::GpuOp;
use crate::stream::GpuOpExt;
use crate::stream::Stream;
use crate::table::{BoundTable, RawTable, TableOwner, UnboundTable};

#[doc(hidden)]
pub trait UniquePtrOwner<T: cxx::memory::UniquePtrTarget> {
    fn as_unique_ptr(&self) -> &UniquePtr<T>;

    fn source_device(&self) -> Option<DeviceId> {
        None
    }
}

impl<T: cxx::memory::UniquePtrTarget> UniquePtrOwner<T> for UniquePtr<T> {
    fn as_unique_ptr(&self) -> &UniquePtr<T> {
        self
    }
}

impl<Brand, T: cxx::memory::UniquePtrTarget> UniquePtrOwner<T>
    for ContextBound<'_, Brand, UniquePtr<T>>
{
    fn as_unique_ptr(&self) -> &UniquePtr<T> {
        self
    }

    fn source_device(&self) -> Option<DeviceId> {
        self.device()
    }
}

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
pub struct PackedColumns<Raw = UniquePtr<cudf_sys::contiguous_split::ffi::PackedColumns>>(
    pub(crate) Raw,
);

impl PackedColumns {
    /// Reconstructs packed columns from host metadata and data bytes (H2D + unpack).
    pub fn from_host_parts(metadata: &[u8], gpu_data: &[u8]) -> Result<Self> {
        let inner = cudf_sys::contiguous_split::ffi::make_packed_from_host_parts(
            metadata,
            gpu_data,
            Stream::default_stream().as_raw(),
        )?;
        Ok(Self(inner))
    }
}

impl<Raw> PackedColumns<Raw>
where
    Raw: UniquePtrOwner<cudf_sys::contiguous_split::ffi::PackedColumns>,
{
    /// Reconstructs an owned [`Table`] from the packed representation.
    ///
    /// The returned table has the same schema and data as the original table
    /// that was packed.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails (e.g. corrupted metadata).
    pub fn unpack(&self) -> Result<UnboundTable> {
        self.unpack_builder().call()
    }

    /// Returns an explicit GPU operation builder that reconstructs this packed
    /// representation into an owned [`Table`].
    pub fn unpack_builder(&self) -> UnpackPacked<'_, Raw> {
        UnpackPacked {
            packed: self,
            stream: Stream::default_stream(),
            source_device: self.0.source_device(),
        }
    }

    /// Transfers this packed representation into `alloc`'s context.
    ///
    /// The returned builder lets the caller choose the destination execution
    /// stream via [`call_on`](crate::stream::AllocatedGpuOp::call_on) or
    /// [`submit_on`](crate::stream::AllocatedGpuOp::submit_on).
    pub fn transfer_to<'ctx, Brand>(
        &self,
        alloc: &Allocator<'ctx, Brand>,
    ) -> crate::stream::AllocatedGpuOp<'ctx, Brand, UnpackPacked<'_, Raw>> {
        self.unpack_builder().in_alloc(alloc)
    }

    /// Submits unpacking of this packed representation onto `stream`,
    /// retaining `self` until the destination work completes.
    pub fn submit_transfer_to<'ctx, Brand>(
        self,
        alloc: &Allocator<'ctx, Brand>,
        stream: &Execution<'ctx, Brand>,
    ) -> Result<PendingPackedTransfer<'ctx, Brand, Raw>> {
        let value = alloc.with_current(|| -> Result<UnboundTable> {
            let raw = stream.as_raw()?;
            self.unpack_builder()
                .stream(unsafe { Stream::from_raw(raw) })
                .call()
        })??;
        Ok(crate::stream::Pending::from_parts(
            crate::table::RawTable(alloc.bind(value.0)),
            *stream,
            Vec::new(),
            self,
        ))
    }

    /// Returns the host metadata as a byte vector.
    ///
    /// The metadata describes the table schema (column types, offsets, null
    /// masks) and is needed to reconstruct the table from the GPU data buffer.
    pub fn metadata(&self) -> Vec<u8> {
        self.0.as_unique_ptr().metadata_to_host()
    }

    /// Returns the size of the host metadata in bytes.
    pub fn metadata_size(&self) -> usize {
        self.0.as_unique_ptr().metadata_size()
    }

    /// Returns the size of the GPU data in bytes.
    pub fn gpu_data_size(&self) -> usize {
        self.0.as_unique_ptr().gpu_data_size()
    }

    /// Copies the packed GPU data to a host byte vector (D2H).
    pub fn gpu_data_to_host(&self) -> Vec<u8> {
        self.0.as_unique_ptr().gpu_data_to_host()
    }
}

/// A vector of table partitions from [`Table::contiguous_split`].
///
/// Each partition is stored contiguously in its own device memory buffer. Use
/// [`unpack`](PackedTableVec::unpack) to reconstruct individual partitions as
/// owned [`Table`] instances.
pub struct PackedTableVec<Raw = UniquePtr<cudf_sys::contiguous_split::ffi::PackedTableVec>>(
    pub(crate) Raw,
);

type BoundPacked<'ctx, Brand> = PackedColumns<
    ContextBound<'ctx, Brand, UniquePtr<cudf_sys::contiguous_split::ffi::PackedColumns>>,
>;
type PendingPackedTransfer<'ctx, Brand, Raw> =
    crate::stream::Pending<'ctx, Brand, BoundTable<'ctx, Brand>, PackedColumns<Raw>>;

#[doc(hidden)]
pub enum TransferKeepAlive<'src, 'dst, SrcBrand, DstBrand> {
    SourcePacked(BoundPacked<'src, SrcBrand>),
    DestinationPacked(BoundPacked<'dst, DstBrand>),
}

impl<SrcBrand, DstBrand> Drop for TransferKeepAlive<'_, '_, SrcBrand, DstBrand> {
    fn drop(&mut self) {
        match self {
            Self::SourcePacked(_packed) => {}
            Self::DestinationPacked(_packed) => {}
        }
    }
}

type PendingTableTransfer<'dst, 'src, DstBrand, SrcBrand> = crate::stream::Pending<
    'dst,
    DstBrand,
    BoundTable<'dst, DstBrand>,
    TransferKeepAlive<'src, 'dst, SrcBrand, DstBrand>,
>;

fn bind_packed_from_host_parts<'ctx, Brand>(
    alloc: &Allocator<'ctx, Brand>,
    stream: &Execution<'ctx, Brand>,
    metadata: &[u8],
    gpu_data: &[u8],
) -> Result<BoundPacked<'ctx, Brand>> {
    let packed = alloc.with_current(|| -> Result<PackedColumns> {
        let raw = stream.as_raw()?;
        let inner =
            cudf_sys::contiguous_split::ffi::make_packed_from_host_parts(metadata, gpu_data, raw)?;
        Ok(PackedColumns(inner))
    })??;
    Ok(PackedColumns(alloc.bind(packed.0)))
}

/// Builder for explicit cross-context table transfer.
///
/// This is layered on top of the packed-table representation:
///
/// 1. pack the source table in `src_alloc` on `src_exec`
/// 2. unpack it into `dst_alloc` on `dst_exec`
///
/// The initial implementation is intentionally blocking. Cross-context async
/// transfer needs stronger dependency retention and event plumbing before it
/// can be exposed honestly.
pub struct TransferTable<'a, 'src, 'dst, SrcBrand, DstBrand> {
    table: &'a BoundTable<'src, SrcBrand>,
    src_alloc: Allocator<'src, SrcBrand>,
    dst_alloc: Allocator<'dst, DstBrand>,
}

impl<'src, 'dst, SrcBrand, DstBrand> TransferTable<'_, 'src, 'dst, SrcBrand, DstBrand> {
    fn validate_endpoints(
        &self,
        src_exec: &Execution<'src, SrcBrand>,
        dst_exec: &Execution<'dst, DstBrand>,
    ) -> Result<(DeviceId, DeviceId)> {
        let src_device = self.src_alloc.device();
        let dst_device = self.dst_alloc.device();
        if src_exec.device() != src_device {
            return Err(crate::error::Error::InvalidArgument(format!(
                "source execution device {} does not match source allocator device {src_device}",
                src_exec.device(),
            )));
        }
        if dst_exec.device() != dst_device {
            return Err(crate::error::Error::InvalidArgument(format!(
                "destination execution device {} does not match destination allocator device {dst_device}",
                dst_exec.device(),
            )));
        }
        Ok((src_device, dst_device))
    }

    /// Packs the source table on `src_exec`, then submits the destination
    /// unpack on `dst_exec` without waiting.
    ///
    /// This is currently a staged async transfer:
    ///
    /// 1. the source pack step completes before this method returns
    /// 2. the destination unpack remains pending on `dst_exec`
    ///
    /// The returned pending value retains the packed source representation
    /// until the destination submission completes.
    pub fn submit_between(
        self,
        src_exec: &Execution<'src, SrcBrand>,
        dst_exec: &Execution<'dst, DstBrand>,
    ) -> Result<PendingTableTransfer<'dst, 'src, DstBrand, SrcBrand>> {
        let (src_device, dst_device) = self.validate_endpoints(src_exec, dst_exec)?;
        let packed = self
            .table
            .pack()
            .in_alloc(&self.src_alloc)
            .call_on(src_exec)?;
        if src_device == dst_device {
            let value = self.dst_alloc.with_current(|| -> Result<UnboundTable> {
                let raw = dst_exec.as_raw()?;
                packed
                    .unpack_builder()
                    .stream(unsafe { Stream::from_raw(raw) })
                    .call()
            })??;
            return Ok(crate::stream::Pending::from_parts(
                crate::table::RawTable(self.dst_alloc.bind(value.0)),
                *dst_exec,
                Vec::new(),
                TransferKeepAlive::SourcePacked(packed),
            ));
        }

        let metadata = packed.metadata();
        let gpu_data = packed.gpu_data_to_host();
        let dst_packed =
            bind_packed_from_host_parts(&self.dst_alloc, dst_exec, &metadata, &gpu_data)?;
        let value = self.dst_alloc.with_current(|| -> Result<UnboundTable> {
            let raw = dst_exec.as_raw()?;
            dst_packed
                .unpack_builder()
                .stream(unsafe { Stream::from_raw(raw) })
                .call()
        })??;
        Ok(crate::stream::Pending::from_parts(
            crate::table::RawTable(self.dst_alloc.bind(value.0)),
            *dst_exec,
            Vec::new(),
            TransferKeepAlive::DestinationPacked(dst_packed),
        ))
    }

    /// Packs the source table on `src_exec` and reconstructs it in the
    /// destination context on `dst_exec`.
    pub fn call_between(
        self,
        src_exec: &Execution<'src, SrcBrand>,
        dst_exec: &Execution<'dst, DstBrand>,
    ) -> Result<BoundTable<'dst, DstBrand>> {
        self.submit_between(src_exec, dst_exec)?.wait()
    }
}

impl<Raw> PackedTableVec<Raw>
where
    Raw: UniquePtrOwner<cudf_sys::contiguous_split::ffi::PackedTableVec>,
{
    /// Returns the number of partitions.
    pub fn len(&self) -> usize {
        self.0.as_unique_ptr().size()
    }

    /// Returns `true` if there are no partitions.
    pub fn is_empty(&self) -> bool {
        self.0.as_unique_ptr().size() == 0
    }

    /// Reconstructs partition at `index` into an owned [`Table`].
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds or the libcudf call fails.
    pub fn unpack(&self, index: usize) -> Result<UnboundTable> {
        self.unpack_builder(index).call()
    }

    /// Returns an explicit GPU operation builder that reconstructs partition
    /// `index` into an owned [`Table`].
    pub fn unpack_builder(&self, index: usize) -> UnpackPartition<'_, Raw> {
        UnpackPartition {
            packed: self,
            index,
            stream: Stream::default_stream(),
            source_device: self.0.source_device(),
        }
    }

    /// Transfers partition `index` into `alloc`'s context.
    pub fn transfer_partition_to<'ctx, Brand>(
        &self,
        index: usize,
        alloc: &Allocator<'ctx, Brand>,
    ) -> crate::stream::AllocatedGpuOp<'ctx, Brand, UnpackPartition<'_, Raw>> {
        self.unpack_builder(index).in_alloc(alloc)
    }
}

/// Builder for [`Table::pack`].
///
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct Pack<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for Pack<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = PackedColumns;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let p = cudf_sys::contiguous_split::ffi::pack_table(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(PackedColumns(p))
    }
}

/// Builder for [`Table::packed_size`].
///
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct PackedSize<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for PackedSize<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = usize;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let n = cudf_sys::contiguous_split::ffi::packed_size_of(
            self.table.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(n)
    }
}

/// Builder for [`Table::contiguous_split`].
///
/// Call [`.call()`](crate::stream::GpuOp::call) to execute.
/// Use [`.stream()`](crate::stream::GpuOp::stream) to set a custom CUDA stream.
pub struct ContiguousSplit<'a, Raw = UniquePtr<cudf_sys::ffi::Table>> {
    table: &'a RawTable<Raw>,
    splits: &'a [i32],
    stream: Stream,
}

impl<Raw> crate::stream::GpuOp for ContiguousSplit<'_, Raw>
where
    Raw: TableOwner,
{
    type Output = PackedTableVec;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        let v = cudf_sys::contiguous_split::ffi::contiguous_split_table(
            self.table.0.as_unique_ptr(),
            self.splits,
            self.stream.as_raw(),
        )?;
        Ok(PackedTableVec(v))
    }
}

/// Builder for [`PackedColumns::unpack_builder`].
pub struct UnpackPacked<'a, Raw = UniquePtr<cudf_sys::contiguous_split::ffi::PackedColumns>> {
    packed: &'a PackedColumns<Raw>,
    stream: Stream,
    source_device: Option<DeviceId>,
}

impl<Raw> crate::stream::GpuOp for UnpackPacked<'_, Raw>
where
    Raw: UniquePtrOwner<cudf_sys::contiguous_split::ffi::PackedColumns>,
{
    type Output = crate::table::UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        validate_transfer_device(self.source_device)?;
        let t = cudf_sys::contiguous_split::ffi::unpack_packed_on_stream(
            self.packed.0.as_unique_ptr(),
            self.stream.as_raw(),
        )?;
        Ok(crate::table::RawTable(t))
    }
}

/// Builder for [`PackedTableVec::unpack_builder`].
pub struct UnpackPartition<'a, Raw = UniquePtr<cudf_sys::contiguous_split::ffi::PackedTableVec>> {
    packed: &'a PackedTableVec<Raw>,
    index: usize,
    stream: Stream,
    source_device: Option<DeviceId>,
}

impl<Raw> crate::stream::GpuOp for UnpackPartition<'_, Raw>
where
    Raw: UniquePtrOwner<cudf_sys::contiguous_split::ffi::PackedTableVec>,
{
    type Output = crate::table::UnboundTable;

    fn stream(mut self, stream: Stream) -> Self {
        self.stream = stream;
        self
    }

    fn call(self) -> Result<Self::Output> {
        validate_transfer_device(self.source_device)?;
        let t = self
            .packed
            .0
            .as_unique_ptr()
            .unpack_at_on_stream(self.index, self.stream.as_raw())?;
        Ok(crate::table::RawTable(t))
    }
}

fn validate_transfer_device(source_device: Option<DeviceId>) -> Result<()> {
    if let Some(expected_device) = source_device {
        let current_device = current_device();
        if current_device != expected_device {
            return Err(crate::error::Error::InvalidArgument(format!(
                "cross-device packed transfer is not implemented yet (source device {expected_device}, current device {current_device})",
            )));
        }
    }
    Ok(())
}

impl<Raw> RawTable<Raw>
where
    Raw: TableOwner,
{
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
    pub fn pack(&self) -> Pack<'_, Raw> {
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
    pub fn packed_size(&self) -> PackedSize<'_, Raw> {
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
    pub fn contiguous_split<'a>(&'a self, splits: &'a [i32]) -> ContiguousSplit<'a, Raw> {
        ContiguousSplit {
            table: self,
            splits,
            stream: Stream::default_stream(),
        }
    }
}

impl<'src, SrcBrand> RawTable<ContextBound<'src, SrcBrand, UniquePtr<cudf_sys::ffi::Table>>> {
    /// Transfers this table into `dst_alloc` using the allocator provenance
    /// already carried by the bound source table.
    pub fn transfer_to<'a, 'dst, DstBrand>(
        &'a self,
        dst_alloc: &Allocator<'dst, DstBrand>,
    ) -> TransferTable<'a, 'src, 'dst, SrcBrand, DstBrand> {
        let src_alloc = self
            .0
            .allocator()
            .expect("bound tables must retain allocator provenance");
        TransferTable {
            table: self,
            src_alloc,
            dst_alloc: *dst_alloc,
        }
    }

    /// Transfers this table into `dst_alloc` using an explicit source staging
    /// allocator.
    ///
    /// This returns a builder that requires both a source execution handle and
    /// a destination execution handle via [`TransferTable::call_between`].
    ///
    /// The source staging allocator controls where the temporary packed
    /// representation is allocated. The destination allocator controls where
    /// the reconstructed destination table is allocated.
    pub fn transfer_via<'a, 'dst, DstBrand>(
        &'a self,
        src_alloc: &Allocator<'src, SrcBrand>,
        dst_alloc: &Allocator<'dst, DstBrand>,
    ) -> TransferTable<'a, 'src, 'dst, SrcBrand, DstBrand> {
        TransferTable {
            table: self,
            src_alloc: *src_alloc,
            dst_alloc: *dst_alloc,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::column::Column;
    use crate::stream::{GpuOp, GpuOpExt};
    use crate::table::Table;
    use rmm::gpu_context::GpuContext;

    struct FirstBrand;
    struct SrcBrand;
    struct DstBrand;

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn packed_columns_transfer_between_contexts() {
        let _guard = crate::test_lock();
        let src_ctx = GpuContext::<SrcBrand>::current().unwrap();
        let dst_ctx = GpuContext::<DstBrand>::current().unwrap();

        let src_alloc = src_ctx.create_cuda_allocator().unwrap();
        let src_exec = src_ctx.create_stream().unwrap();
        let dst_alloc = dst_ctx.create_cuda_allocator().unwrap();
        let dst_exec = dst_ctx.create_stream().unwrap();

        let src = Table::from_columns_in(
            &src_alloc,
            vec![
                Column::from_slice_i32_in(&src_alloc, &[1, 2, 3]).unwrap(),
                Column::from_slice_i32_in(&src_alloc, &[10, 20, 30]).unwrap(),
            ],
        )
        .unwrap();

        let packed = src.pack().in_alloc(&src_alloc).call_on(&src_exec).unwrap();
        let dst = packed.transfer_to(&dst_alloc).call_on(&dst_exec).unwrap();

        assert_eq!(dst.len(), 3);
        assert_eq!(dst.columns_len(), 2);
        assert_eq!(
            dst.column(0).unwrap().to_vec_i32().call().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            dst.column(1).unwrap().to_vec_i32().call().unwrap(),
            vec![10, 20, 30]
        );
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn table_transfer_between_contexts() {
        let _guard = crate::test_lock();
        let src_ctx = GpuContext::<SrcBrand>::current().unwrap();
        let dst_ctx = GpuContext::<DstBrand>::current().unwrap();

        let src_alloc = src_ctx.create_cuda_allocator().unwrap();
        let src_exec = src_ctx.create_stream().unwrap();
        let dst_alloc = dst_ctx.create_cuda_allocator().unwrap();
        let dst_exec = dst_ctx.create_stream().unwrap();

        let src = Table::from_columns_in(
            &src_alloc,
            vec![
                Column::from_slice_i32_in(&src_alloc, &[7, 8, 9]).unwrap(),
                Column::from_slice_i32_in(&src_alloc, &[70, 80, 90]).unwrap(),
            ],
        )
        .unwrap();

        let dst = src
            .transfer_to(&dst_alloc)
            .call_between(&src_exec, &dst_exec)
            .unwrap();

        assert_eq!(dst.len(), 3);
        assert_eq!(dst.columns_len(), 2);
        assert_eq!(
            dst.column(0).unwrap().to_vec_i32().call().unwrap(),
            vec![7, 8, 9]
        );
        assert_eq!(
            dst.column(1).unwrap().to_vec_i32().call().unwrap(),
            vec![70, 80, 90]
        );
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn table_transfer_between_contexts_submit_between() {
        let _guard = crate::test_lock();
        let src_ctx = GpuContext::<SrcBrand>::current().unwrap();
        let dst_ctx = GpuContext::<DstBrand>::current().unwrap();

        let src_alloc = src_ctx.create_cuda_allocator().unwrap();
        let src_exec = src_ctx.create_stream().unwrap();
        let dst_alloc = dst_ctx.create_cuda_allocator().unwrap();
        let dst_exec = dst_ctx.create_stream().unwrap();

        let src = Table::from_columns_in(
            &src_alloc,
            vec![
                Column::from_slice_i32_in(&src_alloc, &[4, 5, 6]).unwrap(),
                Column::from_slice_i32_in(&src_alloc, &[40, 50, 60]).unwrap(),
            ],
        )
        .unwrap();

        let pending = src
            .transfer_to(&dst_alloc)
            .submit_between(&src_exec, &dst_exec)
            .unwrap();
        let dst = pending.wait().unwrap();

        assert_eq!(dst.len(), 3);
        assert_eq!(dst.columns_len(), 2);
        assert_eq!(
            dst.column(0).unwrap().to_vec_i32().call().unwrap(),
            vec![4, 5, 6]
        );
        assert_eq!(
            dst.column(1).unwrap().to_vec_i32().call().unwrap(),
            vec![40, 50, 60]
        );
    }

    #[test]
    #[ignore = "requires multiple CUDA devices and mutates the current device resource"]
    fn table_transfer_between_devices_uses_host_staging() {
        let _guard = crate::test_lock();
        if rmm::device::num_devices() < 2 {
            return;
        }

        let src_ctx = GpuContext::<SrcBrand>::new(rmm::device::DeviceId::new(0)).unwrap();
        let dst_ctx = GpuContext::<DstBrand>::new(rmm::device::DeviceId::new(1)).unwrap();

        let src_alloc = src_ctx.default_device_allocator();
        let src_exec = src_ctx.default_stream();
        let dst_alloc = dst_ctx.default_device_allocator();
        let dst_exec = dst_ctx.default_stream();

        let src = Table::from_columns_in(
            &src_alloc,
            vec![Column::from_slice_i32_in(&src_alloc, &[1, 2, 3]).unwrap()],
        )
        .unwrap();

        let dst = src
            .transfer_via(&src_alloc, &dst_alloc)
            .submit_between(&src_exec, &dst_exec)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(dst.len(), 3);
        assert_eq!(
            dst.column(0).unwrap().to_vec_i32().call().unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn unpack_bound_packed_columns_rejects_cross_device_current_context() {
        let _guard = crate::test_lock();
        if rmm::device::num_devices() < 2 {
            return;
        }

        let src_ctx = GpuContext::<FirstBrand>::new(rmm::device::DeviceId::new(0)).unwrap();
        let src_alloc = src_ctx.default_device_allocator();
        let src_exec = src_ctx.default_stream();

        let src = Table::from_columns_in(
            &src_alloc,
            vec![Column::from_slice_i32_in(&src_alloc, &[1, 2, 3]).unwrap()],
        )
        .unwrap();
        let packed = src.pack().in_alloc(&src_alloc).call_on(&src_exec).unwrap();

        let _switch = rmm::device::ScopedDevice::new(rmm::device::DeviceId::new(1)).unwrap();
        match packed.unpack() {
            Ok(_) => panic!("cross-device unpack unexpectedly succeeded"),
            Err(err) => {
                assert!(matches!(err, crate::error::Error::InvalidArgument(_)));
                assert!(
                    err.to_string()
                        .contains("cross-device packed transfer is not implemented yet")
                );
            }
        }
    }
}
