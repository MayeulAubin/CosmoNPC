# Density-split clustering: definitions and implementation steps

Reference: [Paillas et al., arXiv:2309.16541, section 2.2.2](https://arxiv.org/html/2309.16541#S2.SS2.SSS2).
That analysis uses smoothed environments sampled at random positions and
configuration-space quantile auto/cross correlations. CosmoCM uses equal-volume
mesh cells as its query sample. This is a gridded, volume-sampled version, not
an exact reproduction of the paper's random-query estimator. Bispectrum
measurements are an explicit extension using three specified field legs.

## Step 1: environment and quantiles

`quantiles.py` implements these operations independently of the estimators:

1. Smooth the environmental overdensity with a periodic Gaussian,
   `W(k) = exp(-k² R² / 2)`, using the existing pmesh FFT and chosen precision.
   The supplied tracer field remains unchanged.
2. For a binary coverage mask M, use `G(M delta) / G(M)` for ranking within M.
   Outside cells do not enter the ranking. This compensates smoothing near mask
   boundaries; it does not deconvolve the survey window in a spectrum.
3. Rank covered cells globally, lowest density first, into equal-population
   quantiles Q0, Q1, ... . Counts differ by at most one if division is uneven.
   Repeated densities are ordered by a seeded, bijective permutation of global
   cell IDs. This tie rule is deterministic across MPI decompositions. Boundary
   densities and tie populations are recorded. A constant environment raises an
   error rather than creating arbitrary environmental structure.
4. Define `delta_Q = M * (indicator_Q / f_Q - 1)`, where f_Q is its actual
   fraction of covered cells. This is a query-position overdensity, not the
   original density restricted to that bin. Consequently `sum(f_Q delta_Q)=0`
   at every cell, and each quantile field has zero mean over the covered volume.

The exact distributed order-statistic algorithm sorts local keys and reduces
only scalar counts. Neither the environmental mesh nor the query list is
collected on rank zero. Memory is proportional to local mesh size; boundary
selection requires multiple scalar collectives per quantile. Large ties can
make a split sensitive to the tie seed, which must be retained with results.

Default scientific choices for the integrated workflow are five quantiles and
Gaussian radius 10 in the same length units as `boxsize`. No quantile is silently
discarded. The linear dependence is exposed through the partition identity;
users may choose which quantiles to include in later inference.

Current scope is periodic boxes and zero-filled masked boxes. A survey workflow
requires selection-function/random-catalogue treatment and appropriate
normalization, and must not be inferred from masked-box measurements.
