// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! CUDA stream handle for GPU operations.

use rmm::gpu_context::{Allocator, ContextBound, ContextEvent, ContextStream};

use crate::column::{BoundColumn, OwnedColumn, UnboundColumn};
use crate::contiguous_split::{PackedColumns, PackedTableVec};
use crate::join::MarkJoin;
use crate::scalar::Scalar;
use crate::table::{BoundTable, OwnedTable, UnboundTable};

#[doc(alias = "cuda_stream_view")]
/// A CUDA stream handle.
///
/// `Stream` is a lightweight `Copy` type wrapping a raw `cudaStream_t` pointer
/// (represented as `usize`). Streams provide ordering guarantees for GPU
/// operations: operations submitted to the same stream execute in order, while
/// operations on different streams may execute concurrently.
///
/// Use [`Stream::default_stream()`] to obtain the cudf default stream, which
/// is used by all operations unless overridden via the
/// [`.stream()`](GpuOp::stream) builder method.
///
/// # Examples
///
/// ```no_run
/// use cudf::stream::Stream;
///
/// // Use the default stream (most common case)
/// let stream = Stream::default_stream();
///
/// // Wrap an existing CUDA stream handle
/// # let raw_cuda_stream = 0usize;
/// let custom = unsafe { Stream::from_raw(raw_cuda_stream) };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Stream(usize);

impl Default for Stream {
    fn default() -> Self {
        Self(cudf_sys::ffi::get_default_stream())
    }
}

impl Stream {
    /// Returns the default CUDA stream used by cudf.
    ///
    /// This is the stream that all cudf operations use unless a custom stream
    /// is set via [`.stream()`](GpuOp::stream). In a per-thread default
    /// stream build of cudf, each thread gets its own default stream; in the
    /// legacy default stream build, all threads share the same stream.
    pub fn default_stream() -> Self {
        Self::default()
    }

    /// Creates a `Stream` from a raw `cudaStream_t` handle.
    ///
    /// The `raw` value is the `cudaStream_t` pointer cast to `usize`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The stream handle remains valid until all GPU work submitted through
    ///   this handle has completed.
    /// - The stream belongs to the current CUDA device.
    pub unsafe fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    /// Returns the raw `cudaStream_t` handle as a `usize`.
    ///
    /// This is useful when interacting with other CUDA libraries that accept
    /// raw stream handles.
    pub fn as_raw(self) -> usize {
        self.0
    }

    fn from_rmm_borrowed(stream: &rmm::stream::Stream) -> Self {
        Self(stream.as_raw())
    }

    /// Creates a `Stream` from an owning [`rmm::stream::Stream`].
    ///
    /// The returned cudf `Stream` borrows the raw handle. The caller must
    /// ensure the owning `rmm::stream::Stream` outlives any operations
    /// submitted to this handle.
    ///
    /// # Safety
    ///
    /// The caller must keep `stream` alive until all GPU work submitted
    /// through the returned handle has completed.
    pub unsafe fn from_rmm(stream: &rmm::stream::Stream) -> Self {
        Self(stream.as_raw())
    }
}

/// A GPU operation builder tied to the lifetime of an owning RMM stream.
///
/// This wrapper keeps the source [`rmm::stream::Stream`] borrowed until
/// [`call`](GpuOp::call) is invoked, which makes the common
/// `builder.stream_from_rmm(&stream).call()` path safe without exposing a raw
/// cudf stream handle to user code.
pub struct BorrowedRmmStreamOp<'a, Op> {
    op: Op,
    stream_guard: &'a rmm::stream::Stream,
}

impl<Op: GpuOp> GpuOp for BorrowedRmmStreamOp<'_, Op> {
    type Output = Op::Output;

    fn stream(self, stream: Stream) -> Self {
        Self {
            op: self.op.stream(stream),
            stream_guard: self.stream_guard,
        }
    }

    fn call(self) -> crate::error::Result<Self::Output> {
        self.op.call()
    }
}

/// Trait for GPU operation builders that support stream selection and execution.
///
/// All builders returned by cudf operations implement this trait. The typical
/// usage pattern is:
///
/// 1. Call a method that returns a builder (e.g. `column.view().fill(...)`).
/// 2. Optionally call `.stream()` to override the CUDA stream.
/// 3. Call `.call()` to execute the operation and get the result.
///
/// # Examples
///
/// ```no_run
/// use cudf::column::Column;
/// use cudf::scalar::Scalar;
/// use cudf::stream::{GpuOp, Stream};
///
/// let col = Column::from_scalar(&Scalar::from_i32(0), 10)
///     .stream(Stream::default_stream())
///     .call()?;
///
/// let rmm_stream = rmm::stream::Stream::new().unwrap();
/// let other = Column::from_scalar(&Scalar::from_i32(1), 3)
///     .stream_from_rmm(&rmm_stream)
///     .call()?;
/// assert_eq!(other.len(), 3);
/// # let _ = col;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub trait GpuOp: Sized {
    /// The result type produced by [`call`](GpuOp::call).
    type Output;

    /// Sets the CUDA stream for this operation.
    ///
    /// If not called, the operation uses the cudf default stream
    /// ([`Stream::default_stream()`]).
    fn stream(self, stream: Stream) -> Self;

    /// Sets the CUDA stream from an owning [`rmm::stream::Stream`].
    ///
    /// This is the safe way to run a single cudf operation on an RMM-owned
    /// stream. The returned wrapper keeps the owning stream borrowed until
    /// the operation is submitted via [`call`](GpuOp::call).
    fn stream_from_rmm(self, stream: &rmm::stream::Stream) -> BorrowedRmmStreamOp<'_, Self> {
        BorrowedRmmStreamOp {
            op: self.stream(Stream::from_rmm_borrowed(stream)),
            stream_guard: stream,
        }
    }

    /// Executes the GPU operation and returns the result.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](crate::error::Error) if the underlying libcudf
    /// call fails.
    fn call(self) -> crate::error::Result<Self::Output>;
}

/// Result of submitting GPU work onto an explicit CUDA stream.
///
/// `Pending` keeps the execution handle borrowed until completion and retains
/// the produced value plus any recorded dependency events. Dropping an
/// unfinished `Pending` synchronizes its stream before releasing those
/// resources, which prevents in-flight work from outliving the buffers it
/// uses.
#[must_use = "GPU work is pending; call .wait(), .handoff_to(), or .join()"]
pub struct Pending<'ctx, Brand, T, KeepAlive = ()> {
    value: Option<T>,
    stream: ContextStream<'ctx, Brand>,
    dependencies: Vec<ContextEvent<'ctx, Brand>>,
    keepalive: Option<KeepAlive>,
}

type ChainedPending<'ctx, Brand, T, KeepAlive, NextKeepAlive> =
    Pending<'ctx, Brand, T, (KeepAlive, NextKeepAlive)>;
type JoinedPending<'ctx, Brand, T, U, KeepAlive, OtherKeepAlive> =
    Pending<'ctx, Brand, (T, U), (KeepAlive, OtherKeepAlive)>;

impl<'ctx, Brand, T, KeepAlive> Pending<'ctx, Brand, T, KeepAlive> {
    pub(crate) fn from_parts(
        value: T,
        stream: ContextStream<'ctx, Brand>,
        dependencies: Vec<ContextEvent<'ctx, Brand>>,
        keepalive: KeepAlive,
    ) -> Self {
        Self {
            value: Some(value),
            stream,
            dependencies,
            keepalive: Some(keepalive),
        }
    }

    /// Waits for the submitted stream work to complete and returns the value.
    pub fn wait(mut self) -> crate::error::Result<T> {
        self.stream.synchronize()?;
        Ok(self
            .value
            .take()
            .expect("pending value already taken before wait"))
    }

    /// Transfers this pending chain onto `stream`.
    ///
    /// This records an event on the current stream and makes `stream` wait on
    /// it, so later work enqueued on the returned `Pending` executes after the
    /// original chain completes.
    pub fn handoff_to(mut self, stream: &ContextStream<'ctx, Brand>) -> crate::error::Result<Self> {
        if self.stream.as_raw()? != stream.as_raw()? {
            let event = self.stream.record_event()?;
            stream.wait_event(&event)?;
            self.dependencies.push(event);
            self.stream = *stream;
        }
        Ok(self)
    }

    /// Chains a continuation onto this pending stream submission.
    ///
    /// The closure receives the completed value plus the current execution
    /// stream. It must submit any follow-on GPU work onto that same stream and
    /// return the resulting [`Pending`]. If you want to continue on another
    /// stream, call [`handoff_to`](Pending::handoff_to) first and then chain.
    ///
    /// This shape keeps stream ordering explicit while still allowing the
    /// continuation to borrow from `value` during submission.
    pub fn then<U, NextKeepAlive, F>(
        mut self,
        f: F,
    ) -> crate::error::Result<ChainedPending<'ctx, Brand, U, KeepAlive, NextKeepAlive>>
    where
        F: FnOnce(
            T,
            &ContextStream<'ctx, Brand>,
        ) -> crate::error::Result<Pending<'ctx, Brand, U, NextKeepAlive>>,
    {
        let stream = self.stream;
        let mut next = f(
            self.value
                .take()
                .expect("pending value already taken before then"),
            &stream,
        )?;
        if next.stream.as_raw()? != stream.as_raw()? {
            return Err(crate::error::Error::InvalidArgument(
                "then continuation must submit onto the provided stream; call handoff_to() first"
                    .to_string(),
            ));
        }
        let mut dependencies = std::mem::take(&mut self.dependencies);
        next.dependencies.append(&mut dependencies);
        Ok(Pending {
            value: next.value.take(),
            stream: next.stream,
            dependencies: std::mem::take(&mut next.dependencies),
            keepalive: Some((
                self.keepalive
                    .take()
                    .expect("pending keepalive already taken before then"),
                next.keepalive
                    .take()
                    .expect("next pending keepalive already taken before then"),
            )),
        })
    }

    /// Joins two pending chains into one pending pair.
    ///
    /// If both chains are already on the same stream, this is free. If they
    /// are on different streams, the returned chain continues on `self`'s
    /// stream after waiting for `other`.
    pub fn join<U, OtherKeepAlive>(
        mut self,
        mut other: Pending<'ctx, Brand, U, OtherKeepAlive>,
    ) -> crate::error::Result<JoinedPending<'ctx, Brand, T, U, KeepAlive, OtherKeepAlive>> {
        let stream = self.stream;
        let mut dependencies = std::mem::take(&mut self.dependencies);
        dependencies.extend(std::mem::take(&mut other.dependencies));
        if stream.as_raw()? != other.stream.as_raw()? {
            let event = other.stream.record_event()?;
            stream.wait_event(&event)?;
            dependencies.push(event);
        }
        Ok(Pending {
            value: Some((
                self.value
                    .take()
                    .expect("left pending value already taken before join"),
                other
                    .value
                    .take()
                    .expect("right pending value already taken before join"),
            )),
            stream,
            dependencies,
            keepalive: Some((
                self.keepalive
                    .take()
                    .expect("left pending keepalive already taken before join"),
                other
                    .keepalive
                    .take()
                    .expect("right pending keepalive already taken before join"),
            )),
        })
    }
}

impl<Brand, T, KeepAlive> Drop for Pending<'_, Brand, T, KeepAlive> {
    fn drop(&mut self) {
        if self.value.is_some() {
            let _ = self.stream.synchronize();
        }
    }
}

/// A GPU operation with an explicit allocator bound ahead of execution.
///
/// This separates allocation placement from stream selection:
///
/// ```no_run
/// use cudf::column::Column;
/// use cudf::stream::GpuOpExt;
/// use rmm::gpu_context::GpuContext;
///
/// let ctx = GpuContext::<()>::current()?;
/// let alloc = ctx.default_device_allocator();
/// let exec = ctx.default_stream();
///
/// let col = Column::from_slice_i32(&[1, 2, 3])
///     .in_alloc(&alloc)
///     .call_on(&exec)?;
/// # let _ = col;
/// # Ok::<(), cudf::error::Error>(())
/// ```
pub struct AllocatedGpuOp<'ctx, Brand, Op> {
    op: Op,
    alloc: Allocator<'ctx, Brand>,
}

impl<'ctx, Brand, Op> AllocatedGpuOp<'ctx, Brand, Op>
where
    Op: GpuOp,
    Op::Output: BindContext<'ctx, Brand>,
{
    /// Submits the operation onto `stream` without waiting.
    pub fn submit_on(
        self,
        stream: &ContextStream<'ctx, Brand>,
    ) -> crate::error::Result<Pending<'ctx, Brand, <Op::Output as BindContext<'ctx, Brand>>::Bound>>
    {
        self.submit_on_retaining(stream, ())
    }

    /// Submits the operation onto `stream` without waiting while retaining
    /// `keepalive` until the submission is complete.
    pub fn submit_on_retaining<KeepAlive>(
        self,
        stream: &ContextStream<'ctx, Brand>,
        keepalive: KeepAlive,
    ) -> crate::error::Result<
        Pending<'ctx, Brand, <Op::Output as BindContext<'ctx, Brand>>::Bound, KeepAlive>,
    > {
        let value = self
            .alloc
            .with_current(|| -> crate::error::Result<Op::Output> {
                let raw = stream.as_raw()?;
                self.op.stream(unsafe { Stream::from_raw(raw) }).call()
            })??;
        Ok(Pending {
            value: Some(value.bind(&self.alloc)),
            stream: *stream,
            dependencies: Vec::new(),
            keepalive: Some(keepalive),
        })
    }

    /// Submits the operation onto `stream` and blocks until completion.
    pub fn call_on(
        self,
        stream: &ContextStream<'ctx, Brand>,
    ) -> crate::error::Result<<Op::Output as BindContext<'ctx, Brand>>::Bound> {
        self.submit_on(stream)?.wait()
    }
}

/// Maps GPU-owning outputs to context-bound wrappers.
pub trait BindContext<'ctx, Brand>: Sized {
    /// The context-bound output type.
    type Bound;

    /// Binds `self` to `alloc`'s context.
    fn bind(self, alloc: &Allocator<'ctx, Brand>) -> Self::Bound;
}

impl<'ctx, Brand: 'ctx> BindContext<'ctx, Brand> for UnboundColumn {
    type Bound = BoundColumn<'ctx, Brand>;

    fn bind(self, alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        crate::column::RawColumn(alloc.bind(self.0))
    }
}

impl<'ctx, Brand: 'ctx> BindContext<'ctx, Brand> for UnboundTable {
    type Bound = BoundTable<'ctx, Brand>;

    fn bind(self, alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        crate::table::RawTable(alloc.bind(self.0))
    }
}

impl<'ctx, Brand: 'ctx> BindContext<'ctx, Brand> for MarkJoin {
    type Bound = MarkJoin<ContextBound<'ctx, Brand, cxx::UniquePtr<cudf_sys::join::ffi::MarkJoin>>>;

    fn bind(self, alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        MarkJoin(alloc.bind(self.0))
    }
}

impl<'ctx, Brand: 'ctx> BindContext<'ctx, Brand> for PackedColumns {
    type Bound = PackedColumns<
        ContextBound<'ctx, Brand, cxx::UniquePtr<cudf_sys::contiguous_split::ffi::PackedColumns>>,
    >;

    fn bind(self, alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        PackedColumns(alloc.bind(self.0))
    }
}

impl<'ctx, Brand: 'ctx> BindContext<'ctx, Brand> for PackedTableVec {
    type Bound = PackedTableVec<
        ContextBound<'ctx, Brand, cxx::UniquePtr<cudf_sys::contiguous_split::ffi::PackedTableVec>>,
    >;

    fn bind(self, alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        PackedTableVec(alloc.bind(self.0))
    }
}

impl<'ctx, Brand> BindContext<'ctx, Brand> for Scalar {
    type Bound = Scalar;

    fn bind(self, _alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        self
    }
}

impl<'ctx, Brand, T> BindContext<'ctx, Brand> for Vec<T> {
    type Bound = Vec<T>;

    fn bind(self, _alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        self
    }
}

impl<'ctx, Brand> BindContext<'ctx, Brand> for usize {
    type Bound = usize;

    fn bind(self, _alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        self
    }
}

impl<'ctx, Brand> BindContext<'ctx, Brand> for bool {
    type Bound = bool;

    fn bind(self, _alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        self
    }
}

impl<'ctx, Brand> BindContext<'ctx, Brand> for () {
    type Bound = ();

    fn bind(self, _alloc: &Allocator<'ctx, Brand>) -> Self::Bound {
        let () = self;
    }
}

impl<'ctx> BindContext<'ctx, ()> for OwnedColumn<'ctx> {
    type Bound = OwnedColumn<'ctx>;

    fn bind(self, _alloc: &Allocator<'ctx, ()>) -> Self::Bound {
        self
    }
}

impl<'ctx> BindContext<'ctx, ()> for OwnedTable<'ctx> {
    type Bound = OwnedTable<'ctx>;

    fn bind(self, _alloc: &Allocator<'ctx, ()>) -> Self::Bound {
        self
    }
}

/// Safe execution extensions for allocator-bound GPU operations.
pub trait GpuOpExt: GpuOp + Sized {
    /// Binds an explicit allocator to this GPU operation before execution.
    fn in_alloc<'ctx, Brand>(
        self,
        alloc: &Allocator<'ctx, Brand>,
    ) -> AllocatedGpuOp<'ctx, Brand, Self> {
        AllocatedGpuOp {
            op: self,
            alloc: *alloc,
        }
    }

    /// Executes this operation in `alloc` on an owned stream and waits for
    /// completion before returning the result.
    fn call_in<'ctx, Brand>(
        self,
        alloc: &Allocator<'ctx, Brand>,
    ) -> crate::error::Result<<Self::Output as BindContext<'ctx, Brand>>::Bound>
    where
        Self::Output: BindContext<'ctx, Brand>,
    {
        let stream = alloc.stream()?;
        self.in_alloc(alloc).call_on(&stream)
    }

    /// Submits this operation onto `stream` in `alloc` without waiting.
    fn submit_in<'ctx, Brand>(
        self,
        alloc: &Allocator<'ctx, Brand>,
        stream: &ContextStream<'ctx, Brand>,
    ) -> crate::error::Result<Pending<'ctx, Brand, <Self::Output as BindContext<'ctx, Brand>>::Bound>>
    where
        Self::Output: BindContext<'ctx, Brand>,
    {
        self.in_alloc(alloc).submit_on(stream)
    }
}

impl<T: GpuOp + Sized> GpuOpExt for T {}

#[cfg(test)]
mod tests {
    use super::{GpuOp, GpuOpExt};
    use crate::column::Column;
    use crate::scalar::Scalar;
    use crate::table::Table;
    use rmm::gpu_context::GpuContext;

    #[test]
    fn stream_from_rmm_runs_operation() {
        let stream = rmm::stream::Stream::new().unwrap();
        let col = Column::from_scalar(&Scalar::from_i32(7), 2)
            .stream_from_rmm(&stream)
            .call()
            .unwrap();
        assert_eq!(col.to_vec_i32().call().unwrap(), vec![7, 7]);
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn in_alloc_call_on_explicit_stream_runs_operation() {
        let _guard = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.create_cuda_allocator().unwrap();
        let exec = ctx.create_stream().unwrap();

        let col = Column::from_slice_i32(&[1, 2, 3])
            .in_alloc(&alloc)
            .call_on(&exec)
            .unwrap();
        assert_eq!(col.to_vec_i32().call().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn pending_handoff_to_other_stream_preserves_ordering() {
        let _guard = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.create_cuda_allocator().unwrap();
        let transfer = ctx.create_stream().unwrap();
        let compute = ctx.create_stream().unwrap();

        let col = Column::from_slice_i32(&[4, 5, 6])
            .in_alloc(&alloc)
            .submit_on(&transfer)
            .unwrap()
            .handoff_to(&compute)
            .unwrap()
            .wait()
            .unwrap();

        assert_eq!(col.to_vec_i32().call().unwrap(), vec![4, 5, 6]);
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn pending_then_chains_on_same_stream() {
        let _guard = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.create_cuda_allocator().unwrap();
        let exec = ctx.create_stream().unwrap();

        let col = Column::from_slice_i32(&[1, 2, 3])
            .in_alloc(&alloc)
            .submit_on(&exec)
            .unwrap()
            .then(|col, stream| col.view().reverse().in_alloc(&alloc).submit_on(stream))
            .unwrap()
            .wait()
            .unwrap();

        assert_eq!(col.to_vec_i32().call().unwrap(), vec![3, 2, 1]);
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn pending_join_waits_for_both_streams() {
        let _guard = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.create_cuda_allocator().unwrap();
        let left_stream = ctx.create_stream().unwrap();
        let right_stream = ctx.create_stream().unwrap();

        let left_pending = Column::from_slice_i32(&[1, 2])
            .in_alloc(&alloc)
            .submit_on(&left_stream)
            .unwrap();
        let right_pending = Column::from_slice_i32(&[3, 4])
            .in_alloc(&alloc)
            .submit_on(&right_stream)
            .unwrap();

        let (left_col, right_col) = left_pending.join(right_pending).unwrap().wait().unwrap();
        assert_eq!(left_col.to_vec_i32().call().unwrap(), vec![1, 2]);
        assert_eq!(right_col.to_vec_i32().call().unwrap(), vec![3, 4]);
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn bound_table_can_sort_on_explicit_stream() {
        let _guard = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.create_cuda_allocator().unwrap();
        let exec = ctx.create_stream().unwrap();

        let col = Column::from_slice_i32_in(&alloc, &[3, 1, 2]).unwrap();
        let table = Table::from_columns_in(&alloc, vec![col]).unwrap();

        let sorted = table
            .sort_ascending()
            .in_alloc(&alloc)
            .call_on(&exec)
            .unwrap();
        assert_eq!(
            sorted.column(0).unwrap().to_vec_i32().call().unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    #[ignore = "mutates the current device resource; run explicitly on CUDA hosts"]
    fn bound_tables_can_join_on_explicit_stream() {
        let _guard = crate::test_lock();
        let ctx = GpuContext::<()>::current().unwrap();
        let alloc = ctx.create_cuda_allocator().unwrap();
        let exec = ctx.create_stream().unwrap();

        let left = Table::from_columns_in(
            &alloc,
            vec![
                Column::from_slice_i32_in(&alloc, &[1, 2, 3]).unwrap(),
                Column::from_slice_i32_in(&alloc, &[10, 20, 30]).unwrap(),
            ],
        )
        .unwrap();
        let right = Table::from_columns_in(
            &alloc,
            vec![
                Column::from_slice_i32_in(&alloc, &[2, 3, 4]).unwrap(),
                Column::from_slice_i32_in(&alloc, &[200, 300, 400]).unwrap(),
            ],
        )
        .unwrap();

        let joined = left
            .inner_join(&right, &[0], &[0])
            .in_alloc(&alloc)
            .call_on(&exec)
            .unwrap();
        assert_eq!(joined.len(), 2);
        assert_eq!(joined.columns_len(), 4);
    }
}
