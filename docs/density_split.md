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

## Step 2: tracer loading and power multipoles

Enable the independent workflow by adding `density_split` to a normal box
configuration. Standard runs without that key keep their existing behavior.
The usual input formats, precision, RSD handling, mesh size and output settings
continue to apply. Supply either `fields` or `catalogs` as before.

```python
from cosmonpc import run_stats

config = {
    "input_type": "density", "fields": {"data_a": delta},
    "statistic": "pk", "geometry": "box-like",
    "nmesh": [64, 64, 64], "boxsize": [500., 500., 500.],
    "k_min": .01, "k_max": .15, "k_bins": 10,
    "poles": [0, 2], "precision": "single", "output_dir": "./results",
    "density_split": {
        "source": "data_a", "nquantiles": 5,
        "smoothing_radius": 10., "tie_seed": 0,
        "measurements": [["Q", "data_a"], ["Q", "Q"]],
    },
}
result = run_stats(config)
```

`Q` expands into Q0 through Q4, with repeated Q tokens referring to the same
quantile. Explicit labels such as `["Q0", "Q3"]` select cross-quantile spectra.
`data_a`, `data_b`, and `data_c` refer to available original tracers. The
`source` selects which one defines the environment. Only referenced tracers
and the environmental source are loaded. The measurement legs determine
relationships; the ordinary `correlation_mode`/`tracer_type` flags do not select
density-split legs. Omitting `measurements` chooses Q–source and Q–Q spectra.
Quantile selection is volume weighted, even for weighted input catalogues.

Overdensity fields are preserved, including their supplied means. Count fields
and weighted catalogue meshes are converted to overdensity with a mean over
the common covered volume, or a supplied `mean_density` for a count field. No
further mean subtraction is applied. All leg measurements use the unsmoothed
tracers; smoothing affects only environmental ranking.

For catalogues, use `input_type="catalog"` and `catalogs` with the ordinary
column/RSD settings. Query points are mesh cells, not the catalogue galaxies:
splitting galaxies by their local density would be a different statistic.
In-memory catalogues retain the existing preprocessed/local-row convention.

`compensation` defaults to true for catalogues and false for density fields.
When enabled, the pure assignment window corresponding to `sampler` is
removed from original tracer meshes once, before environmental smoothing and
masking. Quantile indicators are not deconvolved and are not interlaced.
The same corrected tracer convention is used for environmental classification
and the measurement legs. Keep compensation settings consistent when comparing
a painted catalogue to an already-gridded field.

An optional `density_split["mask"]` accepts a global binary NumPy array or
Simbelmyne HDF5 mask descriptor. Effective coverage is its intersection with
coverage of all referenced incomplete tile inputs. Zeros in a supplied
*overdensity* are physical values and do not define a mask. Missing tile cells
never enter a quantile. Outside cells stay zero in every measurement leg.

The workflow writes `density_split_pk.npy`, containing a `measurements`
dictionary (keys such as `Q0__data_a`), quantile populations/fractions,
boundaries and tie information, smoothing parameters, source normalization,
precision, binning, and coverage metadata. It returns the same dictionary on
rank zero, and None on other ranks. Input arrays and the caller configuration
are not modified.

There is no Poisson subtraction in this workflow (`shotnoise_policy="none"`).
A density-selected query field is a nonlinear function of the tracer: ordinary
catalogue auto/cross shot-noise formulas do not describe its contractions.
This policy does **not** claim that particle noise is absent. Quantile auto
statistics also contain mesh-sampling/discretization effects. Masked-box
measurements retain full-box normalization and are window-convolved; they are
not survey estimators. Treat these choices consistently in modelling and mocks.
