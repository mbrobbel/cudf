#![allow(clippy::wildcard_imports)]
use super::*;

impl ColumnView<'_> {
    /// Computes the sum of all non-null elements, returning a [`Scalar`].
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned scalar (for example, summing `INT32` values into an
    /// `INT64` scalar to avoid overflow).
    ///
    /// Returns a [`Sum`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible with the column
    /// type.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(10), 4).call()?;
    /// let total = col.view().sum(TypeId::INT32).call()?;
    /// assert_eq!(total.as_i32(), Some(40));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn sum(&self, output_type: TypeId) -> Sum<'_> {
        Sum {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column to the minimum element.
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned [`Scalar`].
    ///
    /// Returns a [`Min`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[5, 2, 8]).call()?;
    /// let m = col.view().min(TypeId::INT32).call()?;
    /// assert_eq!(m.as_i32(), Some(2));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn min(&self, output_type: TypeId) -> Min<'_> {
        Min {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column to the maximum element.
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned [`Scalar`].
    ///
    /// Returns a [`Max`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[5, 2, 8]).call()?;
    /// let m = col.view().max(TypeId::INT32).call()?;
    /// assert_eq!(m.as_i32(), Some(8));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn max(&self, output_type: TypeId) -> Max<'_> {
        Max {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column to a single scalar by multiplying all elements.
    ///
    /// Null values are skipped. The `output_type` controls the [`TypeId`] of
    /// the returned [`Scalar`].
    ///
    /// Returns a [`Product`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type is incompatible.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::data_type::TypeId;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_i32(2), 3).call()?;
    /// let p = col.view().product(TypeId::INT32).call()?;
    /// assert_eq!(p.as_i32(), Some(8)); // 2 * 2 * 2
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn product(&self, output_type: TypeId) -> Product<'_> {
        Product {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces a `BOOL8` column, returning `true` if any element is true.
    ///
    /// Null values are skipped. The result is a [`Scalar`] of type `BOOL8`.
    /// An empty column (or one with all nulls) yields `false`.
    ///
    /// Returns an [`Any`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_bool(true), 3).call()?;
    /// let result = col.view().any().call()?;
    /// assert_eq!(result.as_bool(), Some(true));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn any(&self) -> Any<'_> {
        Any {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces a `BOOL8` column, returning `true` only if every element is
    /// true.
    ///
    /// Null values are skipped. The result is a [`Scalar`] of type `BOOL8`.
    /// An empty column (or one with all nulls) yields `true` (vacuous truth).
    ///
    /// Returns an [`All`] builder. Use `.stream()` to set a custom CUDA
    /// stream, then `.call()` to execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::scalar::Scalar;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_scalar(&Scalar::from_bool(false), 3).call()?;
    /// let result = col.view().all().call()?;
    /// assert_eq!(result.as_bool(), Some(false));
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn all(&self) -> All<'_> {
        All {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Quantile --

    /// Computes quantiles of this column using linear interpolation.
    ///
    /// `quantiles` is a slice of values in `[0.0, 1.0]`. Returns a `FLOAT64`
    /// column with one row per requested quantile.
    ///
    /// For custom interpolation, use
    /// [`quantile_with_interp`](ColumnView::quantile_with_interp).
    ///
    /// # Arguments
    ///
    /// * `quantiles` -- Slice of quantile values, each in `[0.0, 1.0]`.
    ///   For example, `&[0.25, 0.5, 0.75]` computes the quartiles.
    ///
    /// # Returns
    ///
    /// A `FLOAT64` [`Column`] with `quantiles.len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 4, 5]).call()?;
    /// let median = col.view().quantile(&[0.5]).call()?;
    /// assert_eq!(median.len(), 1);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn quantile<'a>(&'a self, quantiles: &'a [f64]) -> Quantile<'a> {
        Quantile {
            view: self,
            quantiles,
            stream: Stream::default_stream(),
        }
    }

    /// Computes quantiles with a specified
    /// [`Interpolation`](crate::quantile::Interpolation) method.
    ///
    /// See [`quantile`](ColumnView::quantile) for details.
    ///
    /// # Arguments
    ///
    /// * `quantiles` -- Slice of quantile values in `[0.0, 1.0]`.
    /// * `interp` -- Controls how values between data points are estimated
    ///   (e.g. `LINEAR`, `LOWER`, `HIGHER`, `MIDPOINT`, `NEAREST`).
    ///
    /// # Returns
    ///
    /// A `FLOAT64` [`Column`] with `quantiles.len()` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the libcudf call fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::quantile::Interpolation;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_i32(&[1, 2, 3, 4]).call()?;
    /// let q = col
    ///     .view()
    ///     .quantile_with_interp(&[0.5], Interpolation::LOWER)
    ///     .call()?;
    /// assert_eq!(q.len(), 1);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    pub fn quantile_with_interp<'a>(
        &'a self,
        quantiles: &'a [f64],
        interp: crate::quantile::Interpolation,
    ) -> QuantileWithInterp<'a> {
        QuantileWithInterp {
            view: self,
            quantiles,
            interp,
            stream: Stream::default_stream(),
        }
    }

    // -- Transform --

    /// Converts NaN values to null in a floating-point column.
    ///
    /// Returns a new [`Column`] where every NaN
    /// element has been replaced by a null. Non-NaN values (including
    /// existing nulls) are preserved. The column must have a floating-point
    /// type (`FLOAT32` or `FLOAT64`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::column::Column;
    /// use cudf::stream::GpuOp;
    ///
    /// let col = Column::from_slice_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]).call()?;
    /// let clean = col.view().nans_to_nulls().call()?;
    /// assert_eq!(clean.null_count(), 2);
    /// assert_eq!(clean.len(), 5);
    /// # Ok::<(), cudf::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a floating-point type or a GPU
    /// error occurs.
    pub fn nans_to_nulls(&self) -> NansToNulls<'_> {
        NansToNulls {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct values in this column.
    ///
    /// When `include_nulls` is `true`, null is counted as one distinct
    /// value (regardless of how many null rows exist). When
    /// `nan_is_null` is `true`, NaN values are treated as null for
    /// counting purposes.
    pub fn distinct_count(
        &self,
        include_nulls: bool,
        nan_is_null: bool,
    ) -> ColumnDistinctCount<'_> {
        ColumnDistinctCount {
            view: self,
            include_nulls,
            nan_is_null,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of consecutive groups of unique values.
    ///
    /// Unlike [`distinct_count`](ColumnView::distinct_count), this
    /// only counts transitions between consecutive distinct values,
    /// so the column should typically be sorted first.
    ///
    /// `include_nulls` and `nan_is_null` have the same meaning as in
    /// [`distinct_count`](ColumnView::distinct_count).
    pub fn unique_count(&self, include_nulls: bool, nan_is_null: bool) -> ColumnUniqueCount<'_> {
        ColumnUniqueCount {
            view: self,
            include_nulls,
            nan_is_null,
            stream: Stream::default_stream(),
        }
    }

    /// Computes approximate percentiles from a pre-built t-digest column.
    ///
    /// `self` must be a t-digest column (a STRUCT column produced by
    /// the t-digest aggregation). `percentiles` is a `FLOAT64` column
    /// of values in `[0.0, 1.0]`.
    pub fn percentile_approx<'a>(
        &'a self,
        percentiles: &'a ColumnView<'a>,
    ) -> PercentileApprox<'a> {
        PercentileApprox {
            view: self,
            percentiles,
            stream: Stream::default_stream(),
        }
    }

    // -- Reductions --

    /// Computes the arithmetic mean of all non-null elements, returning
    /// a [`Scalar`].
    ///
    /// `output_type` is typically `FLOAT64`.
    pub fn mean(&self, output_type: TypeId) -> Mean<'_> {
        Mean {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the standard deviation of all non-null elements.
    ///
    /// `ddof` is the delta degrees of freedom (0 for population, 1 for
    /// sample standard deviation). `output_type` is typically `FLOAT64`.
    pub fn std_dev(&self, output_type: TypeId, ddof: i32) -> StdDev<'_> {
        StdDev {
            view: self,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the variance of all non-null elements.
    ///
    /// `ddof` is the delta degrees of freedom (0 for population, 1 for
    /// sample variance). `output_type` is typically `FLOAT64`.
    pub fn variance(&self, output_type: TypeId, ddof: i32) -> Variance<'_> {
        Variance {
            view: self,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the median of all non-null elements, returning a
    /// [`Scalar`].
    ///
    /// `output_type` is typically `FLOAT64`.
    pub fn median(&self, output_type: TypeId) -> Median<'_> {
        Median {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }

    /// Counts the number of distinct non-null values, returning a
    /// [`Scalar`] of type `INT32`.
    ///
    /// Null values are excluded from the count.
    pub fn nunique(&self) -> Nunique<'_> {
        Nunique {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the minimum value using the minmax kernel, which finds
    /// both min and max in a single pass. Returns just the minimum.
    ///
    /// Prefer this over [`min`](ColumnView::min) when you also need
    /// [`minmax_max`](ColumnView::minmax_max), since the kernel
    /// computes both simultaneously.
    pub fn minmax_min(&self) -> MinmaxMin<'_> {
        MinmaxMin {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    /// Computes the maximum value using the minmax kernel. Returns just
    /// the maximum. See [`minmax_min`](ColumnView::minmax_min).
    pub fn minmax_max(&self) -> MinmaxMax<'_> {
        MinmaxMax {
            view: self,
            stream: Stream::default_stream(),
        }
    }

    // -- Generic reduce --

    /// Reduces the column using a generic
    /// [`AggregationKind`](crate::groupby::AggregationKind), returning
    /// a [`Scalar`].
    ///
    /// `output_type` specifies the result scalar type. `ddof` (delta
    /// degrees of freedom) is only used for `STD` and `VAR`
    /// aggregations; pass `0` for others.
    pub fn reduce(
        &self,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
    ) -> Reduce<'_> {
        Reduce {
            view: self,
            agg,
            output_type,
            ddof,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces the column with an explicit initial value.
    ///
    /// Like [`reduce`](ColumnView::reduce), but the reduction starts
    /// from `init` instead of the identity element. Supports `SUM`,
    /// `PRODUCT`, `MIN`, `MAX`, `ANY`, and `ALL` aggregations.
    pub fn reduce_with_init<'a>(
        &'a self,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        init: &'a Scalar,
    ) -> ReduceWithInit<'a> {
        ReduceWithInit {
            view: self,
            agg,
            output_type,
            ddof,
            init,
            stream: Stream::default_stream(),
        }
    }

    /// Reduces each segment independently, returning one result per
    /// segment in a new column.
    ///
    /// `offsets` is an `INT32` column of length `N+1` defining `N`
    /// segments, similar to list offsets. `exclude_nulls` controls
    /// whether null values are skipped during aggregation. `ddof` is
    /// used only for `STD` and `VAR` aggregations.
    pub fn segmented_reduce<'a>(
        &'a self,
        offsets: &'a ColumnView<'_>,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        exclude_nulls: bool,
    ) -> SegmentedReduce<'a> {
        SegmentedReduce {
            view: self,
            offsets,
            agg,
            output_type,
            ddof,
            exclude_nulls,
            stream: Stream::default_stream(),
        }
    }

    /// Segmented reduce with an explicit initial value.
    ///
    /// Like [`segmented_reduce`](ColumnView::segmented_reduce), but
    /// each segment's reduction starts from `init`. Only `SUM`,
    /// `PRODUCT`, `MIN`, `MAX`, `ANY`, and `ALL` aggregations are
    /// supported.
    #[doc(alias = "segmented_reduce")]
    pub fn segmented_reduce_with_init<'a>(
        &'a self,
        offsets: &'a ColumnView<'_>,
        agg: crate::groupby::AggregationKind,
        output_type: TypeId,
        ddof: i32,
        exclude_nulls: bool,
        init: &'a Scalar,
    ) -> SegmentedReduceWithInit<'a> {
        SegmentedReduceWithInit {
            view: self,
            offsets,
            agg,
            output_type,
            ddof,
            exclude_nulls,
            init,
            stream: Stream::default_stream(),
        }
    }

    // -- Scan --

    /// Computes a prefix scan (cumulative operation) on this column.
    ///
    /// `agg_kind` specifies the scan operation (e.g. `SUM` for a
    /// cumulative sum, `MIN` for a running minimum). When `inclusive`
    /// is `true`, element `i` includes itself; when `false`, element
    /// `i` is the result of the first `i` elements (exclusive scan).
    pub fn scan(&self, agg_kind: crate::groupby::AggregationKind, inclusive: bool) -> Scan<'_> {
        Scan {
            view: self,
            agg_kind,
            inclusive,
            stream: Stream::default_stream(),
        }
    }
}
