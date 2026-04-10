// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0

//! Single-device GPU execution contexts.
//!
//! A [`GpuContext`] owns device allocators, pinned-host allocators, and CUDA
//! execution handles for one CUDA device. Handles borrowed from the context
//! are cheap capabilities; the context itself owns the underlying resources.
//!
//! The current implementation still carries a generic `Brand` parameter as a
//! transition aid while the rest of the stack moves from wrapper-based context
//! binding to direct `<'ctx>`-bound owning GPU types.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Deref;

use cxx::UniquePtr;

use crate::device::{self, DeviceId, ScopedDevice};
use crate::error::{Error, Result};
use crate::memory_resource::{
    CudaAsyncMemoryResource, CudaMemoryResource, ManagedMemoryResource, MemoryResourceRef,
    PinnedHostBuffer, PinnedHostMemoryResource, PoolMemoryResource, set_current_device_resource,
};
use crate::stream::Stream as RmmStream;

#[derive(Debug)]
enum OwnedMemoryResource {
    Cuda(CudaMemoryResource),
    Async(CudaAsyncMemoryResource),
    Managed(ManagedMemoryResource),
    Pool(PoolMemoryResource),
}

impl OwnedMemoryResource {
    fn as_ref(&self) -> MemoryResourceRef<'_> {
        match self {
            Self::Cuda(resource) => resource.as_ref(),
            Self::Async(resource) => resource.as_ref(),
            Self::Managed(resource) => resource.as_ref(),
            Self::Pool(resource) => resource.as_ref(),
        }
    }
}

#[derive(Debug, Clone)]
enum DefaultDeviceAllocatorKind {
    Cuda,
    CudaAsync,
    CudaAsyncWithLimits {
        initial_pool_size: usize,
        release_threshold: usize,
    },
    Managed,
    Pool,
    PoolWithLimits {
        initial_size: usize,
        maximum_size: usize,
    },
}

impl DefaultDeviceAllocatorKind {
    fn build(&self) -> Result<OwnedMemoryResource> {
        match self {
            Self::Cuda => Ok(OwnedMemoryResource::Cuda(CudaMemoryResource::new()?)),
            Self::CudaAsync => Ok(OwnedMemoryResource::Async(CudaAsyncMemoryResource::new()?)),
            Self::CudaAsyncWithLimits {
                initial_pool_size,
                release_threshold,
            } => Ok(OwnedMemoryResource::Async(
                CudaAsyncMemoryResource::with_limits(*initial_pool_size, *release_threshold)?,
            )),
            Self::Managed => Ok(OwnedMemoryResource::Managed(ManagedMemoryResource::new()?)),
            Self::Pool => Ok(OwnedMemoryResource::Pool(PoolMemoryResource::new()?)),
            Self::PoolWithLimits {
                initial_size,
                maximum_size,
            } => Ok(OwnedMemoryResource::Pool(PoolMemoryResource::with_limits(
                *initial_size,
                *maximum_size,
            )?)),
        }
    }
}

/// Configuration for a [`GpuContext`].
///
/// This controls which allocator is installed as the context's explicit
/// default device allocator. Default execution and pinned-host resources are
/// always created automatically.
#[derive(Debug, Clone)]
pub struct GpuContextConfig {
    default_device_allocator: DefaultDeviceAllocatorKind,
}

impl GpuContextConfig {
    /// Creates a config that uses `cudaMalloc`/`cudaFree` as the default device
    /// allocator.
    pub fn with_cuda_allocator(mut self) -> Self {
        self.default_device_allocator = DefaultDeviceAllocatorKind::Cuda;
        self
    }

    /// Creates a config that uses `cudaMallocAsync` as the default device
    /// allocator.
    pub fn with_cuda_async_allocator(mut self) -> Self {
        self.default_device_allocator = DefaultDeviceAllocatorKind::CudaAsync;
        self
    }

    /// Creates a config that uses `cudaMallocAsync` with explicit limits as
    /// the default device allocator.
    pub fn with_cuda_async_allocator_limits(
        mut self,
        initial_pool_size: usize,
        release_threshold: usize,
    ) -> Self {
        self.default_device_allocator = DefaultDeviceAllocatorKind::CudaAsyncWithLimits {
            initial_pool_size,
            release_threshold,
        };
        self
    }

    /// Creates a config that uses managed memory as the default device
    /// allocator.
    pub fn with_managed_allocator(mut self) -> Self {
        self.default_device_allocator = DefaultDeviceAllocatorKind::Managed;
        self
    }

    /// Creates a config that uses the default pool allocator as the default
    /// device allocator.
    pub fn with_pool_allocator(mut self) -> Self {
        self.default_device_allocator = DefaultDeviceAllocatorKind::Pool;
        self
    }

    /// Creates a config that uses a pool allocator with explicit limits as
    /// the default device allocator.
    pub fn with_pool_allocator_limits(mut self, initial_size: usize, maximum_size: usize) -> Self {
        self.default_device_allocator = DefaultDeviceAllocatorKind::PoolWithLimits {
            initial_size,
            maximum_size,
        };
        self
    }
}

impl Default for GpuContextConfig {
    fn default() -> Self {
        Self {
            default_device_allocator: DefaultDeviceAllocatorKind::Cuda,
        }
    }
}

/// A single-device owner for allocators, pinned-host resources, and execution
/// streams.
///
/// `GpuContext` is the recommended entry point for the safe RMM/cuDF path:
/// values allocated through handles borrowed from the context carry the
/// context lifetime, which prevents mixing resources from unrelated contexts
/// in safe Rust.
///
/// # Examples
///
/// ```no_run
/// use rmm::buffer::DeviceBuffer;
/// use rmm::device::current_device;
/// use rmm::gpu_context::GpuContext;
///
/// let ctx = GpuContext::<()>::new(current_device())?;
/// let alloc = ctx.default_device_allocator();
/// let exec = ctx.default_stream();
///
/// let buf = DeviceBuffer::new_in(&alloc, 1024)?;
/// exec.synchronize()?;
/// assert_eq!(buf.size(), 1024);
/// # Ok::<(), rmm::error::Error>(())
/// ```
#[derive(Debug)]
pub struct GpuContext<Brand = ()> {
    device: DeviceId,
    resources: RefCell<Vec<OwnedMemoryResource>>,
    pinned_resources: RefCell<Vec<PinnedHostMemoryResource>>,
    streams: RefCell<Vec<RmmStream>>,
    default_device_allocator_id: usize,
    default_pinned_allocator_id: usize,
    _brand: PhantomData<fn() -> Brand>,
}

pub(crate) type ContextMarker<'ctx> = PhantomData<fn(&'ctx ()) -> &'ctx ()>;

impl<Brand> GpuContext<Brand> {
    /// Creates a new context bound to `device` with the provided `config`.
    pub fn with_config(device: DeviceId, config: &GpuContextConfig) -> Result<Self> {
        let _device_guard = ScopedDevice::new(device)?;

        let mut resources = Vec::new();
        let default_device_allocator_id = resources.len();
        resources.push(config.default_device_allocator.build()?);

        let mut pinned_resources = Vec::new();
        let default_pinned_allocator_id = pinned_resources.len();
        pinned_resources.push(PinnedHostMemoryResource::new()?);

        Ok(Self {
            device,
            resources: RefCell::new(resources),
            pinned_resources: RefCell::new(pinned_resources),
            streams: RefCell::new(Vec::new()),
            default_device_allocator_id,
            default_pinned_allocator_id,
            _brand: PhantomData,
        })
    }

    /// Creates a new context bound to `device` with the default config.
    pub fn new(device: DeviceId) -> Result<Self> {
        Self::with_config(device, &GpuContextConfig::default())
    }

    /// Creates a new context bound to the current CUDA device with the default config.
    pub fn current() -> Result<Self> {
        Self::new(device::current_device())
    }

    /// Creates a new context for the current CUDA device using `config`.
    pub fn current_with_config(config: &GpuContextConfig) -> Result<Self> {
        Self::with_config(device::current_device(), config)
    }

    /// Returns the CUDA device permanently associated with this context.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Returns the explicit default device allocator for this context.
    pub fn default_device_allocator(&self) -> DeviceAllocator<'_, Brand> {
        DeviceAllocator {
            context: self,
            resource_id: self.default_device_allocator_id,
            _context_marker: PhantomData,
            _brand: PhantomData,
        }
    }

    /// Returns the explicit default pinned-host allocator for this context.
    pub fn default_pinned_allocator(&self) -> PinnedAllocator<'_, Brand> {
        PinnedAllocator {
            context: self,
            resource_id: self.default_pinned_allocator_id,
            _context_marker: PhantomData,
            _brand: PhantomData,
        }
    }

    /// Returns the explicit default execution handle for this context.
    pub fn default_stream(&self) -> Execution<'_, Brand> {
        Execution {
            context: self,
            kind: ExecutionKind::Default,
            _context_marker: PhantomData,
            _brand: PhantomData,
        }
    }

    /// Creates an owned CUDA stream in this context.
    pub fn create_stream(&self) -> Result<Execution<'_, Brand>> {
        let _device_guard = ScopedDevice::new(self.device)?;
        let mut streams = self.streams.borrow_mut();
        let stream_id = streams.len();
        streams.push(RmmStream::new()?);
        Ok(Execution {
            context: self,
            kind: ExecutionKind::Owned(stream_id),
            _context_marker: PhantomData,
            _brand: PhantomData,
        })
    }

    /// Creates a `cudaMalloc`/`cudaFree` allocator owned by this context.
    pub fn create_cuda_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.push_resource(|| Ok(OwnedMemoryResource::Cuda(CudaMemoryResource::new()?)))
    }

    /// Creates a `cudaMallocAsync` allocator owned by this context.
    pub fn create_cuda_async_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.push_resource(|| Ok(OwnedMemoryResource::Async(CudaAsyncMemoryResource::new()?)))
    }

    /// Creates a `cudaMallocAsync` allocator with explicit limits.
    pub fn create_cuda_async_allocator_with_limits(
        &self,
        initial_pool_size: usize,
        release_threshold: usize,
    ) -> Result<DeviceAllocator<'_, Brand>> {
        self.push_resource(|| {
            Ok(OwnedMemoryResource::Async(
                CudaAsyncMemoryResource::with_limits(initial_pool_size, release_threshold)?,
            ))
        })
    }

    /// Creates a managed-memory allocator owned by this context.
    pub fn create_managed_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.push_resource(|| Ok(OwnedMemoryResource::Managed(ManagedMemoryResource::new()?)))
    }

    /// Creates a pool allocator owned by this context.
    pub fn create_pool_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.push_resource(|| Ok(OwnedMemoryResource::Pool(PoolMemoryResource::new()?)))
    }

    /// Creates a pool allocator with explicit limits.
    pub fn create_pool_allocator_with_limits(
        &self,
        initial_size: usize,
        maximum_size: usize,
    ) -> Result<DeviceAllocator<'_, Brand>> {
        self.push_resource(|| {
            Ok(OwnedMemoryResource::Pool(PoolMemoryResource::with_limits(
                initial_size,
                maximum_size,
            )?))
        })
    }

    /// Creates a pinned-host allocator owned by this context.
    pub fn create_pinned_allocator(&self) -> Result<PinnedAllocator<'_, Brand>> {
        let _device_guard = ScopedDevice::new(self.device)?;
        let mut resources = self.pinned_resources.borrow_mut();
        let resource_id = resources.len();
        resources.push(PinnedHostMemoryResource::new()?);
        Ok(PinnedAllocator {
            context: self,
            resource_id,
            _context_marker: PhantomData,
            _brand: PhantomData,
        })
    }

    // Transitional aliases while the higher layers migrate.

    /// Transitional alias for [`GpuContext::create_stream`].
    pub fn stream(&self) -> Result<Execution<'_, Brand>> {
        self.create_stream()
    }

    /// Transitional alias for [`GpuContext::create_cuda_allocator`].
    pub fn cuda_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.create_cuda_allocator()
    }

    /// Transitional alias for [`GpuContext::create_cuda_async_allocator`].
    pub fn cuda_async_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.create_cuda_async_allocator()
    }

    /// Transitional alias for [`GpuContext::create_cuda_async_allocator_with_limits`].
    pub fn cuda_async_allocator_with_limits(
        &self,
        initial_pool_size: usize,
        release_threshold: usize,
    ) -> Result<DeviceAllocator<'_, Brand>> {
        self.create_cuda_async_allocator_with_limits(initial_pool_size, release_threshold)
    }

    /// Transitional alias for [`GpuContext::create_managed_allocator`].
    pub fn managed_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.create_managed_allocator()
    }

    /// Transitional alias for [`GpuContext::create_pool_allocator`].
    pub fn pool_allocator(&self) -> Result<DeviceAllocator<'_, Brand>> {
        self.create_pool_allocator()
    }

    /// Transitional alias for [`GpuContext::create_pool_allocator_with_limits`].
    pub fn pool_allocator_with_limits(
        &self,
        initial_size: usize,
        maximum_size: usize,
    ) -> Result<DeviceAllocator<'_, Brand>> {
        self.create_pool_allocator_with_limits(initial_size, maximum_size)
    }

    fn push_resource(
        &self,
        build: impl FnOnce() -> Result<OwnedMemoryResource>,
    ) -> Result<DeviceAllocator<'_, Brand>> {
        let _device_guard = ScopedDevice::new(self.device)?;
        let mut resources = self.resources.borrow_mut();
        let resource_id = resources.len();
        resources.push(build()?);
        Ok(DeviceAllocator {
            context: self,
            resource_id,
            _context_marker: PhantomData,
            _brand: PhantomData,
        })
    }

    pub(crate) fn allocate_pinned_bytes(&self, resource_id: usize, size: usize) -> Result<usize> {
        let _device_guard = ScopedDevice::new(self.device)?;
        let mut resources = self.pinned_resources.borrow_mut();
        let resource = resources.get_mut(resource_id).ok_or_else(|| {
            Error::InvalidArgument("pinned allocator handle does not belong to this context".into())
        })?;
        resource.allocate(size)
    }

    pub(crate) fn deallocate_pinned_bytes(&self, resource_id: usize, ptr: usize, size: usize) {
        if let Ok(_device_guard) = ScopedDevice::new(self.device) {
            let mut resources = self.pinned_resources.borrow_mut();
            if let Some(resource) = resources.get_mut(resource_id) {
                resource.deallocate(ptr, size);
            } else {
                debug_assert!(
                    false,
                    "pinned allocator handle does not belong to this context"
                );
            }
        }
    }
}

/// A non-owning device allocator handle borrowed from a [`GpuContext`].
#[derive(Debug)]
pub struct DeviceAllocator<'ctx, Brand> {
    pub(crate) context: &'ctx GpuContext<Brand>,
    resource_id: usize,
    _context_marker: ContextMarker<'ctx>,
    _brand: PhantomData<fn() -> Brand>,
}

impl<Brand> Copy for DeviceAllocator<'_, Brand> {}

impl<Brand> Clone for DeviceAllocator<'_, Brand> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'ctx, Brand> DeviceAllocator<'ctx, Brand> {
    /// Returns the device associated with this allocator's context.
    pub fn device(&self) -> DeviceId {
        self.context.device
    }

    /// Creates a new explicit CUDA stream in the same context as this allocator.
    pub fn stream(&self) -> Result<Execution<'ctx, Brand>> {
        self.context.create_stream()
    }

    /// Brands `value` with this allocator's context lifetime and brand.
    pub fn bind<T>(&self, value: T) -> ContextBound<'ctx, Brand, T> {
        OwnedValue::bound(value, *self)
    }

    #[doc(hidden)]
    pub fn with_current<T>(&self, f: impl FnOnce() -> T) -> Result<T> {
        let device_guard = ScopedDevice::new(self.context.device)?;
        let resources = self.context.resources.borrow();
        let resource_guard = {
            let resource = resources.get(self.resource_id).ok_or_else(|| {
                Error::InvalidArgument("allocator handle does not belong to this context".into())
            })?;
            set_current_device_resource(resource.as_ref())
        };
        let _device_guard = device_guard;
        let _resource_guard = resource_guard;
        Ok(f())
    }
}

/// Backward-compatible alias during the transition from generic `Allocator`
/// naming to explicit ownership/execution naming.
pub type Allocator<'ctx, Brand = ()> = DeviceAllocator<'ctx, Brand>;

/// A non-owning pinned-host allocator handle borrowed from a [`GpuContext`].
#[derive(Debug)]
pub struct PinnedAllocator<'ctx, Brand> {
    context: &'ctx GpuContext<Brand>,
    resource_id: usize,
    _context_marker: ContextMarker<'ctx>,
    _brand: PhantomData<fn() -> Brand>,
}

impl<Brand> Copy for PinnedAllocator<'_, Brand> {}

impl<Brand> Clone for PinnedAllocator<'_, Brand> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'ctx, Brand> PinnedAllocator<'ctx, Brand> {
    /// Returns the device associated with this allocator's context.
    pub fn device(&self) -> DeviceId {
        self.context.device
    }

    /// Allocates a pinned-host byte buffer in this context.
    pub fn allocate_bytes(&self, size: usize) -> Result<PinnedHostBuffer<'ctx, Brand>> {
        let ptr = self.context.allocate_pinned_bytes(self.resource_id, size)?;
        Ok(PinnedHostBuffer::from_context(
            self.context,
            self.resource_id,
            ptr,
            size,
        ))
    }

    /// Allocates and initializes a pinned-host byte buffer from `src`.
    pub fn from_slice(&self, src: &[u8]) -> Result<PinnedHostBuffer<'ctx, Brand>> {
        let mut buffer = self.allocate_bytes(src.len())?;
        buffer.write(src)?;
        Ok(buffer)
    }
}

#[derive(Debug, Clone, Copy)]
enum ExecutionKind {
    Default,
    Owned(usize),
}

/// Internal CUDA event used to connect explicit execution chains.
#[doc(hidden)]
pub struct ContextEvent<'ctx, Brand> {
    raw: UniquePtr<rmm_sys::ffi::CudaEvent>,
    _context_marker: ContextMarker<'ctx>,
    _brand: PhantomData<fn() -> Brand>,
}

/// A non-owning execution handle borrowed from a [`GpuContext`].
#[derive(Debug)]
pub struct Execution<'ctx, Brand> {
    context: &'ctx GpuContext<Brand>,
    kind: ExecutionKind,
    _context_marker: ContextMarker<'ctx>,
    _brand: PhantomData<fn() -> Brand>,
}

impl<Brand> Copy for Execution<'_, Brand> {}

impl<Brand> Clone for Execution<'_, Brand> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'ctx, Brand> Execution<'ctx, Brand> {
    /// Returns the device associated with this execution handle.
    pub fn device(&self) -> DeviceId {
        self.context.device
    }

    /// Returns `true` if this handle represents the context's default stream.
    pub fn is_default(&self) -> bool {
        matches!(self.kind, ExecutionKind::Default)
    }

    /// Synchronizes the stream, blocking until all queued work completes.
    pub fn synchronize(&self) -> Result<()> {
        let _device_guard = ScopedDevice::new(self.context.device)?;
        match self.kind {
            ExecutionKind::Default => crate::memory_resource::stream_synchronize(self.as_raw()?),
            ExecutionKind::Owned(stream_id) => {
                let streams = self.context.streams.borrow();
                let stream = streams.get(stream_id).ok_or_else(|| {
                    Error::InvalidArgument(
                        "execution handle does not belong to this context".into(),
                    )
                })?;
                stream.synchronize()
            }
        }
    }

    #[doc(hidden)]
    pub fn as_raw(&self) -> Result<usize> {
        let _device_guard = ScopedDevice::new(self.context.device)?;
        match self.kind {
            ExecutionKind::Default => Ok(rmm_sys::ffi::cuda_stream_default_view()),
            ExecutionKind::Owned(stream_id) => {
                let streams = self.context.streams.borrow();
                let stream = streams.get(stream_id).ok_or_else(|| {
                    Error::InvalidArgument(
                        "execution handle does not belong to this context".into(),
                    )
                })?;
                Ok(stream.as_raw())
            }
        }
    }

    #[doc(hidden)]
    pub fn record_event(&self) -> Result<ContextEvent<'ctx, Brand>> {
        let _device_guard = ScopedDevice::new(self.context.device)?;
        let mut event = rmm_sys::ffi::cuda_event_new()?;
        rmm_sys::ffi::cuda_event_record(event.pin_mut(), self.as_raw()?)?;
        Ok(ContextEvent {
            raw: event,
            _context_marker: PhantomData,
            _brand: PhantomData,
        })
    }

    #[doc(hidden)]
    pub fn wait_event(&self, event: &ContextEvent<'ctx, Brand>) -> Result<()> {
        let _device_guard = ScopedDevice::new(self.context.device)?;
        rmm_sys::ffi::cuda_stream_wait_event_raw(self.as_raw()?, &event.raw)?;
        Ok(())
    }

    #[doc(hidden)]
    pub fn wait_for(&self, predecessor: &Self) -> Result<()> {
        if self.as_raw()? == predecessor.as_raw()? {
            return Ok(());
        }
        let event = predecessor.record_event()?;
        self.wait_event(&event)
    }
}

/// Backward-compatible alias during the transition from `ContextStream` to a
/// clearer execution-handle model.
pub type ContextStream<'ctx, Brand = ()> = Execution<'ctx, Brand>;

/// An owning GPU value that may either be unbound or tied to a specific
/// [`GpuContext`] for allocator-correct destruction.
pub struct OwnedValue<'ctx, Brand, T> {
    value: Option<T>,
    alloc: Option<DeviceAllocator<'ctx, Brand>>,
    _context_marker: ContextMarker<'ctx>,
    _brand: PhantomData<fn() -> Brand>,
}

impl<'ctx, Brand, T> OwnedValue<'ctx, Brand, T> {
    pub(crate) fn unbound(value: T) -> Self {
        Self {
            value: Some(value),
            alloc: None,
            _context_marker: PhantomData,
            _brand: PhantomData,
        }
    }

    pub(crate) fn bound(value: T, alloc: DeviceAllocator<'ctx, Brand>) -> Self {
        Self {
            value: Some(value),
            alloc: Some(alloc),
            _context_marker: PhantomData,
            _brand: PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn into_inner(mut self) -> T {
        self.value
            .take()
            .expect("context-bound value already taken")
    }

    #[doc(hidden)]
    pub fn device(&self) -> Option<DeviceId> {
        self.alloc.map(|alloc| alloc.device())
    }

    #[doc(hidden)]
    pub fn allocator(&self) -> Option<DeviceAllocator<'ctx, Brand>> {
        self.alloc
    }
}

impl<Brand, T: std::fmt::Debug> std::fmt::Debug for OwnedValue<'_, Brand, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value
            .as_ref()
            .expect("context-bound value already taken")
            .fmt(f)
    }
}

impl<Brand, T> Deref for OwnedValue<'_, Brand, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
            .as_ref()
            .expect("context-bound value already taken")
    }
}

impl<Brand, T> std::ops::DerefMut for OwnedValue<'_, Brand, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
            .as_mut()
            .expect("context-bound value already taken")
    }
}

impl<Brand, T> Drop for OwnedValue<'_, Brand, T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            if let Some(alloc) = self.alloc {
                let _ = alloc.with_current(|| drop(value));
            } else {
                drop(value);
            }
        }
    }
}

/// Backward-compatible alias while higher layers move from wrapper-bound
/// ownership to direct `<'ctx>`-carrying owners.
pub type ContextBound<'ctx, Brand, T> = OwnedValue<'ctx, Brand, T>;

#[cfg(test)]
mod tests {
    use super::{GpuContext, GpuContextConfig};
    use crate::device::{self, DeviceId};

    struct TestBrand;

    #[test]
    fn context_tracks_target_device() {
        let _test_lock = crate::test_lock();
        let device = device::current_device();
        let ctx = GpuContext::<TestBrand>::new(device).unwrap();
        assert_eq!(ctx.device(), device);
    }

    #[test]
    fn context_exposes_explicit_defaults() {
        let _test_lock = crate::test_lock();
        let ctx = GpuContext::<TestBrand>::current().unwrap();
        let alloc = ctx.default_device_allocator();
        let stream = ctx.default_stream();
        let pinned = ctx.default_pinned_allocator();
        assert_eq!(alloc.device(), ctx.device());
        assert_eq!(stream.device(), ctx.device());
        assert!(stream.is_default());
        assert_eq!(pinned.device(), ctx.device());
        stream.synchronize().unwrap();
    }

    #[test]
    fn context_creates_allocators_and_streams() {
        let _test_lock = crate::test_lock();
        let ctx = GpuContext::<TestBrand>::current().unwrap();
        let alloc = ctx
            .create_pool_allocator_with_limits(1 << 20, 1 << 26)
            .unwrap();
        let stream = ctx.create_stream().unwrap();
        assert_eq!(alloc.device(), ctx.device());
        assert_eq!(stream.device(), ctx.device());
        assert!(!stream.is_default());
        stream.synchronize().unwrap();
    }

    #[test]
    fn context_config_can_select_default_allocator() {
        let _test_lock = crate::test_lock();
        let cfg = GpuContextConfig::default().with_cuda_allocator();
        let ctx = GpuContext::<TestBrand>::current_with_config(&cfg).unwrap();
        let alloc = ctx.default_device_allocator();
        assert_eq!(alloc.device(), ctx.device());
    }

    #[test]
    fn pinned_allocator_allocates_host_buffer() {
        let _test_lock = crate::test_lock();
        let ctx = GpuContext::<TestBrand>::current().unwrap();
        let pinned = ctx.default_pinned_allocator();
        let buf = pinned.from_slice(&[1, 2, 3]).unwrap();
        assert_eq!(buf.to_vec().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn invalid_device_errors() {
        let _test_lock = crate::test_lock();
        let invalid = DeviceId::new(device::num_devices());
        assert!(GpuContext::<TestBrand>::new(invalid).is_err());
    }
}
