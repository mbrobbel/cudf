#![allow(clippy::wildcard_imports)]
use super::*;

impl ColumnView<'_> {
    #[doc(alias = "size")]
    /// Returns the number of elements in this view, including nulls.
    pub fn len(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_size(self.0))
    }

    /// Returns `true` if this view has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of null elements in this view.
    pub fn null_count(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_null_count(self.0))
    }

    /// Returns `true` if this view contains any null elements.
    pub fn has_nulls(&self) -> bool {
        cudf_sys::ffi::column_view_has_nulls(self.0)
    }

    /// Returns the offset of this view into the underlying data buffer.
    ///
    /// Sliced views may have a non-zero offset. For columns created
    /// directly, this is always `0`.
    pub fn offset(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_offset(self.0))
    }

    #[doc(alias = "type")]
    /// Returns the [`TypeId`] of this view's elements.
    pub fn type_id(&self) -> TypeId {
        let id = cudf_sys::ffi::column_view_type_id(self.0);
        // C++ column views always have a valid type_id.
        cudf_sys::type_id_from_i32(id).unwrap_or(TypeId::EMPTY)
    }

    // -- Child column access --

    /// Returns the number of child columns.
    ///
    /// For `STRUCT` columns this is the number of fields. For `LIST`
    /// columns this is 2 (offsets and child values). For `DICTIONARY32`
    /// columns this is 2 (indices and keys). For primitive types this
    /// is 0.
    pub fn num_children(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_num_children(self.0))
    }

    /// Deep-copies a child column by `index`, returning an owned
    /// [`Column`].
    ///
    /// The child index meaning depends on the column type:
    /// - **STRUCT**: `0..N-1` are the struct fields.
    /// - **LIST**: `0` is the offsets column, `1` is the child values.
    /// - **DICTIONARY32**: `0` is indices, `1` is keys.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range.
    pub fn child(&self, index: usize) -> Child<'_> {
        Child {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }

    // -- Device pointer access (for direct D2H/H2D) --

    /// Returns the raw device pointer to this column's data buffer.
    ///
    /// For fixed-width types, this points to the contiguous array of elements.
    /// For STRING columns, this points to the character data. For STRUCT/LIST
    /// columns, this returns 0 (data is in child columns).
    ///
    /// The returned `usize` is an opaque device pointer. Pass it to
    /// [`rmm::memory_resource::memcpy_d2h`] for host readback.
    pub fn data_ptr(&self) -> usize {
        cudf_sys::ffi::column_view_data_ptr(self.0)
    }

    /// Returns the raw device pointer to the null mask (validity bitmap).
    ///
    /// Returns 0 if this column has no null mask (i.e. `!has_nulls()`).
    pub fn null_mask_ptr(&self) -> usize {
        cudf_sys::ffi::column_view_null_mask_ptr(self.0)
    }

    /// Returns the size in bytes of one element of this column's type.
    ///
    /// For example, `INT64` returns 8, `FLOAT32` returns 4.
    /// Returns 0 for variable-width types (STRING, LIST, STRUCT).
    pub fn type_byte_size(&self) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_type_size(self.0))
    }

    /// Returns the size of the character data in bytes (STRING columns only).
    ///
    /// This is the total byte length of all strings in the column, not the
    /// number of elements. For non-STRING columns, this is meaningless.
    pub fn chars_size(&self, stream: Stream) -> usize {
        i32_to_usize(cudf_sys::ffi::column_view_chars_size(
            self.0,
            stream.as_raw(),
        ))
    }

    // -- Unary ops --

    /// Casts every element in this column to a different type.
    ///
    /// The `target` [`TypeId`] specifies the desired output type. For
    /// example, casting an `INT32` column to `FLOAT64` converts each integer
    /// to its floating-point equivalent.
    ///
    /// Returns a [`Cast`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the cast between the source and target types is
    /// not supported.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(42), 2).call()?;
    /// let f64_col = col.view().cast(TypeId::FLOAT64).call()?;
    /// assert_eq!(f64_col.type_id(), TypeId::FLOAT64);
    /// assert_eq!(f64_col.to_vec_f64().call()?, [42.0, 42.0]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn cast(&self, target: TypeId) -> Cast<'_> {
        Cast {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a null element.
    ///
    /// The output column has the same length as this column and no null mask
    /// of its own. Elements that are null in the source map to `true`;
    /// valid elements map to `false`.
    ///
    /// Returns an [`IsNull`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
    /// let nulls = col.view().is_null().call()?;
    /// assert_eq!(nulls.to_vec_bool().call()?, [false, false, false]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn is_null(&self) -> IsNull<'_> {
        IsNull {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a valid (non-null)
    /// element.
    ///
    /// This is the logical inverse of [`is_null`](Self::is_null).
    ///
    /// Returns an [`IsValid`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(1), 3).call()?;
    /// let valid = col.view().is_valid().call()?;
    /// assert_eq!(valid.to_vec_bool().call()?, [true, true, true]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn is_valid(&self) -> IsValid<'_> {
        IsValid {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns a `BOOL8` column where `true` indicates a NaN element.
    ///
    /// Only applicable to floating-point columns (`FLOAT32` / `FLOAT64`).
    /// Null elements produce `false` (not NaN). For the inverse check, see
    /// [`is_not_nan`](Self::is_not_nan).
    ///
    /// Returns an [`IsNan`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a floating-point type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0]).call()?;
    /// let nans = col.view().is_nan().call()?;
    /// assert_eq!(nans.to_vec_bool().call()?, [false, true, false]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn is_nan(&self) -> IsNan<'_> {
        IsNan {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Negates every element (unary minus).
    ///
    /// Applicable to numeric columns. The result column has the same type
    /// as the input.
    ///
    /// Returns a [`Negate`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(5), 2).call()?;
    /// let neg = col.view().negate().call()?;
    /// assert_eq!(neg.to_vec_i32().call()?, [-5, -5]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn negate(&self) -> Negate<'_> {
        Negate {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the absolute value of every element.
    ///
    /// Applicable to signed numeric columns. The result column has the same
    /// type as the input.
    ///
    /// Returns an [`Abs`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(-7), 2).call()?;
    /// let a = col.view().abs().call()?;
    /// assert_eq!(a.to_vec_i32().call()?, [7, 7]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn abs(&self) -> Abs<'_> {
        Abs {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying --

    /// Creates an empty (zero-length) column with the same type as this
    /// view. The result has no data and no null mask.
    pub fn empty_like(&self) -> crate::column::UnboundColumn {
        crate::column::RawColumn(cudf_sys::copying::ffi::empty_like_column(self.0))
    }

    /// Creates an owning deep copy of this view's data.
    ///
    /// This copies all device memory (element data, null mask, and child
    /// columns for nested types) into a new independently-owned
    /// [`Column`]. The original data is not modified.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let copy = col.view().to_owned_column().call()?;
    /// assert_eq!(copy.to_vec_i32().call()?, [1, 2, 3]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn to_owned_column(&self) -> ToOwnedColumn<'_> {
        ToOwnedColumn {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Host data extraction (operates directly on view, no deep copy) --

    /// Copies the view's data from GPU to host as `Vec<i8>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i8(&self) -> ToVecI8<'_> {
        ToVecI8 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<i16>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i16(&self) -> ToVecI16<'_> {
        ToVecI16 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<i32>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i32(&self) -> ToVecI32<'_> {
        ToVecI32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<i64>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_i64(&self) -> ToVecI64<'_> {
        ToVecI64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<f32>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_f32(&self) -> ToVecF32<'_> {
        ToVecF32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<f64>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_f64(&self) -> ToVecF64<'_> {
        ToVecF64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u8>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u8(&self) -> ToVecU8<'_> {
        ToVecU8 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u16>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u16(&self) -> ToVecU16<'_> {
        ToVecU16 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u32>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u32(&self) -> ToVecU32<'_> {
        ToVecU32 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<u64>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_u64(&self) -> ToVecU64<'_> {
        ToVecU64 {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the view's data from GPU to host as `Vec<bool>`.
    ///
    /// See [`Column::to_vec_i8`] for details on null handling.
    pub fn to_vec_bool(&self) -> ToVecBool<'_> {
        ToVecBool {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies the per-element null mask from GPU to host as `Vec<bool>`.
    ///
    /// See [`Column::null_mask_to_host`] for details.
    pub fn null_mask_to_host(&self) -> NullMaskToHost<'_> {
        NullMaskToHost {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }
    /// Copies string data from GPU to host as `Vec<String>`.
    ///
    /// See [`Column::to_vec_string`] for details.
    pub fn to_vec_string(&self) -> ToVecString<'_> {
        ToVecString {
            view: self.0,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying extras --

    /// Shifts column elements by `offset` positions, filling vacated
    /// positions with `fill_value`.
    ///
    /// Positive `offset` shifts elements to the right (later indices);
    /// negative shifts to the left. The result has the same length as
    /// the input.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3]).call()?;
    /// let fill = Scalar::from_i32(0);
    /// let shifted = col.view().shift(1, &fill).call()?;
    /// assert_eq!(shifted.to_vec_i32().call()?, [0, 1, 2]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn shift<'a>(&'a self, offset: i32, fill_value: &'a Scalar) -> Shift<'a> {
        Shift {
            view: self,
            offset,
            fill_value,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the element at `index` as a [`Scalar`].
    ///
    /// If the element is null, the returned scalar is null. This
    /// involves a GPU-to-host transfer of a single value.
    ///
    /// # Errors
    ///
    /// Returns an error if `index >= self.len()`.
    pub fn get_element(&self, index: usize) -> GetElement<'_> {
        GetElement {
            view: self,
            index,
            stream: Stream::default_stream(),
        }
    }

    /// Reverses the order of elements, returning a new column.
    pub fn reverse(&self) -> Reverse<'_> {
        Reverse {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Extracts the half-open range `[begin, end)` as a new owned
    /// column.
    ///
    /// The result is an independent deep copy of the specified range.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds.
    pub fn slice(&self, begin: usize, end: usize) -> Slice<'_> {
        Slice {
            view: self,
            begin,
            end,
            stream: Stream::default_stream(),
        }
    }

    /// Selects elements from `self` where `mask` is `true`, and from
    /// `rhs` where `mask` is `false`, returning a new column.
    ///
    /// All three columns (`self`, `rhs`, `mask`) must have the same
    /// length. `mask` must be a `BOOL8` column. `self` and `rhs` must
    /// have compatible types.
    pub fn copy_if_else<'a>(
        &'a self,
        rhs: &'a ColumnView<'_>,
        mask: &'a ColumnView<'_>,
    ) -> CopyIfElse<'a> {
        CopyIfElse {
            view: self,
            rhs,
            mask,
            stream: Stream::default_stream(),
        }
    }

    // -- Label bins --

    /// Assigns integer bin labels to each element based on bin edges.
    ///
    /// Bin *i* is defined by `[left_edges[i], right_edges[i]]` with the
    /// inclusivity of each edge controlled by `left_inclusive` and
    /// `right_inclusive`. Elements that fall outside all bins receive `-1`.
    ///
    /// # Arguments
    ///
    /// * `left_edges` -- Column of left boundaries, one per bin. Must be
    ///   sorted in ascending order.
    /// * `left_inclusive` -- Whether the left edge of each bin is inclusive
    ///   ([`Inclusive::YES`](crate::labeling::Inclusive::YES)) or exclusive.
    /// * `right_edges` -- Column of right boundaries, one per bin. Must
    ///   have the same length as `left_edges`.
    /// * `right_inclusive` -- Whether the right edge of each bin is inclusive
    ///   or exclusive.
    ///
    /// # Returns
    ///
    /// An `INT32` column with the same length as `self`, where each element
    /// is the index of the bin it falls into, or `-1` if it falls outside
    /// all bins.
    ///
    /// # Errors
    ///
    /// Returns an error if edge columns have mismatched lengths or the
    /// libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::labeling::Inclusive;
    /// use cudf::stream::GpuOp;
    ///
    /// let values = Column::from_slice_i32(&[1, 5, 15]).call()?;
    /// let left = Column::from_slice_i32(&[0, 10]).call()?;
    /// let right = Column::from_slice_i32(&[10, 20]).call()?;
    /// let labels = values.view().label_bins(
    ///     &left.view(), Inclusive::YES,
    ///     &right.view(), Inclusive::NO,
    /// ).call()?;
    /// // labels: [0, 0, 1] (1 and 5 in bin 0, 15 in bin 1)
    /// ```
    pub fn label_bins<'a>(
        &'a self,
        left_edges: &'a ColumnView<'_>,
        left_inclusive: crate::labeling::Inclusive,
        right_edges: &'a ColumnView<'_>,
        right_inclusive: crate::labeling::Inclusive,
    ) -> LabelBins<'a> {
        LabelBins {
            view: self,
            left_edges,
            left_inclusive,
            right_edges,
            right_inclusive,
            stream: Stream::default_stream(),
        }
    }

    // -- Search --

    /// Searches this column for a scalar value.
    ///
    /// Returns `true` if `needle` appears anywhere in the column, `false`
    /// otherwise. The column does not need to be sorted.
    ///
    /// Returns a [`ContainsScalar`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[10, 20, 30]).call()?;
    /// assert!(col.view().contains_scalar(&Scalar::from_i32(20)).call()?);
    /// assert!(!col.view().contains_scalar(&Scalar::from_i32(25)).call()?);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn contains_scalar<'a>(&'a self, needle: &'a Scalar) -> ContainsScalar<'a> {
        ContainsScalar {
            view: self,
            needle,
            stream: Stream::default_stream(),
        }
    }

    /// Checks which values from `needles` exist in this column.
    ///
    /// Returns a `BOOL8` column with the same length as `needles`. Each
    /// output element is `true` if the corresponding needle is found
    /// anywhere in `self`, and `false` otherwise. Neither column needs to
    /// be sorted.
    ///
    /// Returns a [`ContainsColumn`] builder. Use `.stream()` to set a custom
    /// CUDA stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let haystack = Column::from_slice_i32(&[10, 20, 30, 40, 50]).call()?;
    /// let needles = Column::from_slice_i32(&[20, 60]).call()?;
    /// let found = haystack.view()
    ///     .contains_column(&needles.view())
    ///     .call()?;
    /// assert_eq!(found.to_vec_bool().call()?, [true, false]);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn contains_column<'a>(&'a self, needles: &'a ColumnView<'_>) -> ContainsColumn<'a> {
        ContainsColumn {
            view: self,
            needles,
            stream: Stream::default_stream(),
        }
    }

    // -- Sorting (column-level) --

    /// Computes the rank of each element in this column.
    ///
    /// Ranks are 1-based by default. The ranking strategy is controlled by
    /// `method` (see [`RankMethod`](crate::sorting::RankMethod)):
    ///
    /// - `method` -- how to resolve ties among equal values.
    /// - `order` -- [`Order::ASCENDING`](crate::sorting::Order::ASCENDING)
    ///   ranks smallest values first;
    ///   [`Order::DESCENDING`](crate::sorting::Order::DESCENDING) ranks
    ///   largest values first.
    /// - `null_handling` -- whether null elements receive a rank
    ///   ([`NullPolicy::INCLUDE`](crate::compaction::NullPolicy::INCLUDE))
    ///   or are left as null
    ///   ([`NullPolicy::EXCLUDE`](crate::compaction::NullPolicy::EXCLUDE)).
    /// - `null_precedence` -- where nulls sort relative to non-null values.
    /// - `percentage` -- when `true`, ranks are normalized to the range
    ///   `[0.0, 1.0]` and the output type is `FLOAT64`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::compaction::NullPolicy;
    /// use cudf::sorting::{Order, NullOrder, RankMethod};
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[30, 10, 20, 10]).call()?;
    /// let view = col.view();
    ///
    /// // Dense rank ascending: [3, 1, 2, 1]
    /// let ranks = view.rank(
    ///     RankMethod::Dense,
    ///     Order::ASCENDING,
    ///     NullPolicy::EXCLUDE,
    ///     NullOrder::AFTER,
    ///     false,
    /// ).call()?;
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU error occurs.
    pub fn rank(
        &self,
        method: crate::sorting::RankMethod,
        order: crate::sorting::Order,
        null_handling: crate::compaction::NullPolicy,
        null_precedence: crate::sorting::NullOrder,
        percentage: bool,
    ) -> Rank<'_> {
        Rank {
            view: self,
            method,
            order,
            null_handling,
            null_precedence,
            percentage,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the `k` largest (or smallest) values as a new column.
    ///
    /// `order` controls the direction: `ASCENDING` returns the `k`
    /// smallest, `DESCENDING` returns the `k` largest.
    pub fn top_k(&self, k: usize, order: crate::sorting::Order) -> TopK<'_> {
        TopK {
            view: self,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the row indices of the `k` largest (or smallest)
    /// values as an `INT32` column.
    ///
    /// See [`top_k`](ColumnView::top_k) for the `order` semantics.
    pub fn top_k_order(&self, k: usize, order: crate::sorting::Order) -> TopKOrder<'_> {
        TopKOrder {
            view: self,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the `k` largest (or smallest) values within each segment.
    ///
    /// `segment_offsets` is an `INT32` column defining segment
    /// boundaries (same format as list offsets).
    pub fn segmented_top_k<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'_>,
        k: usize,
        order: crate::sorting::Order,
    ) -> SegmentedTopK<'a> {
        SegmentedTopK {
            view: self,
            segment_offsets,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    /// Returns the row indices of the `k` largest (or smallest) values
    /// within each segment.
    ///
    /// See [`segmented_top_k`](ColumnView::segmented_top_k) for details.
    pub fn segmented_top_k_order<'a>(
        &'a self,
        segment_offsets: &'a ColumnView<'_>,
        k: usize,
        order: crate::sorting::Order,
    ) -> SegmentedTopKOrder<'a> {
        SegmentedTopKOrder {
            view: self,
            segment_offsets,
            k,
            order,
            stream: Stream::default_stream(),
        }
    }

    // -- Copying (new) --

    /// Copies `self[source_begin..source_end]` into `target` starting
    /// at `target_begin`, returning a new column based on `target`.
    ///
    /// This is an out-of-place operation: neither `self` nor `target`
    /// is modified. The result is a copy of `target` with the specified
    /// range overwritten by elements from `self`.
    ///
    /// # Arguments
    ///
    /// * `target` -- The column to copy into (used as the base).
    /// * `source_begin` -- Start index in `self` (inclusive).
    /// * `source_end` -- End index in `self` (exclusive).
    /// * `target_begin` -- Start index in `target` where copied elements
    ///   are written.
    ///
    /// # Returns
    ///
    /// A new [`Column`] that is a copy of `target` with the range
    /// `[target_begin, target_begin + (source_end - source_begin))` replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if ranges are out of bounds, types are mismatched,
    /// or the libcudf call fails.
    pub fn copy_range_into<'a>(
        &'a self,
        target: &'a ColumnView<'_>,
        source_begin: usize,
        source_end: usize,
        target_begin: usize,
    ) -> CopyRangeInto<'a> {
        CopyRangeInto {
            view: self,
            target,
            source_begin,
            source_end,
            target_begin,
            stream: Stream::default_stream(),
        }
    }

    /// Creates an uninitialized column with the same type and size as
    /// this view. The data buffer is allocated but not initialized.
    pub fn allocate_like(&self) -> AllocateLike<'_> {
        AllocateLike {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Checks whether this column has null rows that contain non-empty
    /// data (relevant for variable-width types like `LIST` and
    /// `STRING`).
    ///
    /// This requires a GPU kernel. For a fast host-side check that may
    /// return false positives, use
    /// [`may_have_nonempty_nulls`](ColumnView::may_have_nonempty_nulls).
    pub fn has_nonempty_nulls(&self) -> HasNonemptyNulls<'_> {
        HasNonemptyNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Fast host-side check for whether this column might have non-empty
    /// data in null rows.
    ///
    /// May return `true` even when no non-empty nulls exist (false
    /// positive), but never returns `false` when they do exist. Use
    /// [`has_nonempty_nulls`](ColumnView::has_nonempty_nulls) for an
    /// exact check.
    pub fn may_have_nonempty_nulls(&self) -> bool {
        cudf_sys::copying::ffi::may_have_nonempty_nulls(self.0)
    }

    /// Returns a new column with non-empty null row data cleared.
    ///
    /// For variable-width types (`LIST`, `STRING`), null rows may
    /// still contain data. This method produces a column where null
    /// rows have zero-length content, which can be required for
    /// certain interop scenarios.
    pub fn purge_nonempty_nulls(&self) -> PurgeNonemptyNulls<'_> {
        PurgeNonemptyNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Replace (new) --

    /// Replaces null values using a fill policy, returning a new column.
    ///
    /// When `preceding` is `true`, each null is replaced by the last
    /// non-null value before it (forward fill). When `false`, each null
    /// is replaced by the next non-null value after it (backward fill).
    /// Leading/trailing nulls that have no fill source remain null.
    pub fn replace_nulls_policy(&self, preceding: bool) -> ReplaceNullsPolicy<'_> {
        ReplaceNullsPolicy {
            view: self,
            preceding,
            stream: Stream::default_stream(),
        }
    }

    /// Clamps column values to `[lo, hi]`, replacing out-of-range
    /// values with separate replacement scalars.
    ///
    /// Elements less than `lo` are replaced with `lo_replace`. Elements
    /// greater than `hi` are replaced with `hi_replace`. Elements
    /// within the range are unchanged.
    pub fn clamp_with_replace<'a>(
        &'a self,
        lo: &'a Scalar,
        lo_replace: &'a Scalar,
        hi: &'a Scalar,
        hi_replace: &'a Scalar,
    ) -> ClampWithReplace<'a> {
        ClampWithReplace {
            view: self,
            lo,
            lo_replace,
            hi,
            hi_replace,
            stream: Stream::default_stream(),
        }
    }

    /// Normalizes NaN and zero values in a floating-point column.
    ///
    /// Converts all negative NaN representations to the canonical
    /// positive NaN, and `-0.0` to `+0.0`. This is useful before
    /// operations that require consistent equality semantics.
    pub fn normalize_nans_and_zeros(&self) -> NormalizeNansAndZeros<'_> {
        NormalizeNansAndZeros {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Fill --

    /// Returns a new column with elements in `[begin, end)` replaced
    /// by `value`, leaving elements outside the range unchanged.
    ///
    /// This is an out-of-place version of
    /// [`Column::fill_in_place`]. `value` must have the same type as
    /// the column.
    ///
    /// # Arguments
    ///
    /// * `begin` -- Start index of the fill range (inclusive).
    /// * `end` -- End index of the fill range (exclusive). The range
    ///   must satisfy `begin <= end <= self.len()`.
    /// * `value` -- The scalar value to fill with. Its type must match
    ///   the column's [`TypeId`].
    ///
    /// # Returns
    ///
    /// A new [`Column`] with the range filled.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds or the scalar type
    /// does not match.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(1), 5).call()?;
    /// let result = col.view().fill(1, 3, &Scalar::from_i32(99)).call()?;
    /// // result contains [1, 99, 99, 1, 1]
    /// ```
    pub fn fill<'a>(&'a self, begin: usize, end: usize, value: &'a Scalar) -> Fill<'a> {
        Fill {
            view: self,
            begin,
            end,
            value,
            stream: Stream::default_stream(),
        }
    }
}
