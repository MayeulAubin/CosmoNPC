# Density-split clustering: definitions and implementation steps

Reference: [Paillas et al., arXiv:2309.16541, section 2.2.2](https://arxiv.org/html/2309.16541#S2.SS2.SSS2).
That analysis uses smoothed environments sampled at random positions and
configuration-space quantile auto/cross correlations. CosmoCM uses equal-volume
mesh cells for boxes and samples the random catalogue for surveys. The box
implementation is a gridded, volume-sampled version, not an exact reproduction
of the paper's random-query estimator. Bispectrum
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

The box workflow supports periodic and zero-filled masked boxes with all current
field and catalogue input formats. The survey workflow (Step 4) uses the current
catalogue/random interface. As in ordinary runs, gridded field inputs currently
require box-like geometry. A masked box is not a survey estimator.

## Step 2: box tracer loading and power multipoles

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

There is no Poisson subtraction in the box workflow (`shotnoise_policy="none"`).
A density-selected query field is a nonlinear function of the tracer: ordinary
catalogue auto/cross shot-noise formulas do not describe its contractions.
This policy does **not** claim that particle noise is absent. Quantile auto
statistics also contain mesh-sampling/discretization effects. Masked-box
measurements retain full-box normalization and are window-convolved; they are
not survey estimators. Treat these choices consistently in modelling and mocks.

## Step 3: ordered Sugiyama bispectrum legs

Use the same split settings and inputs, change `statistic` to `"bk_sugi"`, and
specify three legs per measurement:

```python
config["statistic"] = "bk_sugi"
config["angu_config"] = [0, 0, 0]  # e.g. [2, 0, 2] for an angular multipole
config["data_vector_mode"] = "full"  # or "diagonal"
config["density_split"]["measurements"] = [
    ["Q", "data_a", "data_a"],  # Qgg
    ["Q", "Q", "data_a"],       # QQg
    ["Q", "Q", "Q"],            # QQQ
    ["Q0", "Q4", "data_a"],     # a particular ordered cross-quantile triple
]
result = run_stats(config)
```

These three repeated-Q templates are the defaults for bispectra. Leg order is
preserved: `["Q0", "data_a", "data_a"]`, `["data_a", "Q0", "data_a"]`, and
`["data_a", "data_a", "Q0"]` need not produce the same angular statistic.
Require `2*k_max` below Nyquist, as in the existing Sugiyama workflow.

The calculation calls the existing `calculate_bk_sugi_box` general three-field
path, with independent copies even for repeated tokens. This avoids compounding
the estimator's in-place rescaling or applying inappropriate repeated-catalogue
shot-noise contractions to quantile indicator meshes. The existing binning,
block size, angular coupling, normalization, and diagonal/full output convention
are retained. Output is `density_split_bk_sugi.npy`, with `B_sugi`, `Bk_raw`,
mode counts, and shot-noise metadata for each ordered combination.

Tests compare six ordered combinations in diagonal/full modes against direct
NumPy Fourier contractions for angular configurations `[0,0,0]` and `[2,0,2]`.
They also check catalogue/count-field parity, both precisions, and repeat-call
invariance. These are numerical estimator checks, not a validation of a
cosmological model for density-selected bispectra.


## Step 4: survey catalogues and existing estimator paths

Survey runs use the same catalogue inputs as ordinary CosmoNPC survey runs:
FITS files with the existing redshift/column/completeness configuration, or
preprocessed structured NumPy catalogues with local rows on each MPI rank.
Each referenced `data_a`, `data_b`, or `data_c` requires its corresponding
`randoms_a`, `randoms_b`, or `randoms_c`. Survey field inputs remain unsupported,
matching the ordinary pipeline; a zero-filled density field uses the masked-box
path instead.

```python
from copy import deepcopy
from cosmonpc.config import pk_survey
from cosmonpc import run_stats

config = deepcopy(pk_survey.CONFIG)
config["catalogs"] = {"data_a": "galaxies.fits", "randoms_a": "randoms.fits"}
config["output_dir"] = "./density_split_results"
# Required for cross measurements, just as in the ordinary survey estimators.
config["normalization_scheme"] = "mixed-mesh"
config["density_split"] = {
    "source": "data_a", "nquantiles": 5,
    "smoothing_radius": 10., "query_ratio": 5., "tie_seed": 0,
    "measurements": [["Q", "data_a"], ["Q", "Q"]],
}
result = run_stats(config)
```

Change to `statistic="bk_sugi"`, choose `angu_config` and diagonal/full mode,
and use three-leg measurements as in Step 3. Existing `block_size`,
`shotnoise_mode` (`ana`, `fft`, `both`), and `alpha_scheme` settings are passed
through. The supported normalization restrictions remain those of the existing
estimators: auto measurements may use their supported particle/mesh/mixed-mesh
schemes, while cross measurements require mixed-mesh normalization. The fast
and comparison high-order P modes remain available for auto-only selections
with the same supported pole lists as ordinary runs.

### Environmental field and query positions

The environmental catalogue and its full random catalogue are painted with
completeness weights, excluding FKP weights. Using
`alpha_env = sum(W_data) / sum(W_random)`, the environmental mesh is
`G / (alpha_env R) - 1` wherever R is positive. Gaussian smoothing is
coverage-normalized as in Step 1. Environmental painting is non-interlaced to
keep the random-count coverage nonnegative; the **measurement** meshes retain
the user's interlacing and sampler settings.

Queries are sampled from the source random catalogue, not from the galaxies.
The smoothed environmental field is read at query positions with CiC
interpolation. Interpolation is normalized by the interpolated binary
coverage to avoid dilution at masked edges. Query points outside coverage are
excluded. Quantiles have equal numbers of accepted queries (within one),
ranked by the sampled environment. This follows the survey-query construction,
rather than treating every box cell as an observed survey location.

`query_ratio=5` requests five queries per source galaxy after the existing input
cuts. The actual count is capped by available query candidates; both requested
and accepted counts are recorded. Queries retain the random catalogue's weight
and FKP convention. Their NZ is scaled from the parent random NZ by the
quantile's total completeness weight divided by the source galaxies' total
completeness weight. The same scale is applied to that quantile's background
random NZ. This assumes the parent selection profile is appropriate for the
quantile, as in standard shared-selection density-split modelling; it does not
fit a new radial selection function for each density environment.

### Independent random pools and normalization

The full source random catalogue defines the environmental field. For measured
spectra, random positions are deterministically partitioned into four pools:
query candidates (pool 0) and three independent background pools (1–3), one per
distinct ordered leg. Identical coordinates always get the same pool, including
across reordered catalogues. Repeated legs reuse their own pool; distinct legs
use different pools. This avoids shared-random self-pairs and self-triples that
the existing cross estimators do not model. It reduces the available random
population relative to using every random for every leg; provide enough
randoms and check the recorded pool/query counts.

The pool assignment is coordinate-based and independent of MPI decomposition.
Exact query-count capping and tied-query ranking use rank-concatenated input-row
IDs. Reproducibility of those two operations requires preserving global row
order when repartitioning preprocessed arrays (as contiguous file partitions do).

Each generated Q catalogue and its background are passed to the **existing**
`get_mesh_pk_survey` / `get_mesh_bk_survey` and corresponding clustering
estimators. Ordered triples map to the correct `aaa`, `aab`, `aba`, `abb`, or
`abc` branch, keeping the existing self-term formulas. A shared boxcenter is
computed from the full environmental random catalogue, preventing per-pool
bounding boxes from shifting the embedding between measurements. The only
mesh-API addition is an optional fixed `boxcenter`, which defaults to the
previous behavior in ordinary runs. Estimator formulas are unchanged except
for fixing the previously crashing zero `SN3_FFT` export in survey `abc` +
`shotnoise_mode="both"` runs.

Survey output records
`shotnoise_policy="existing_catalogue_self_terms_disjoint_random_pools"`.
It subtracts the existing catalogue self terms; it does not claim to remove all
noise induced by estimating an environment from a finite galaxy sample.
Survey window effects and the need for appropriate modelling/mocks remain as
in ordinary survey measurements.

## Validation and execution

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src mpiexec -n 2 python -m unittest discover -s tests -v
```

Box checks include independent Gaussian FFT smoothing, exact global-rank
references, quantile identities, P0/P2 and B000/B202 Fourier oracles, ordered
legs, catalogue/field parity, masks, HDF5 tiles, both precisions, and unchanged
input arrays. Survey checks compare generated catalogue measurements to the
ordinary public pipeline, covering all five bispectrum tracer patterns, FITS
versus preprocessed input, normalization modes, alpha schemes, interlacing,
high-order power modes, and both shot-noise branches.

The development sandbox permits single-rank MPI but blocks the sockets needed
for multi-rank execution. Multi-rank behavior is implemented using pmesh and MPI
collectives but has not been validated in that sandbox; run the second command
on the target cluster before production use.
