# Wavelet scattering transform

Implementation is split into separately validated commits: distributed mathematical
core; input and survey adapters; optional directional statistics; optimized filter
generation; performance tools and user documentation.

The baseline follows [Valogiannis & Dvorkin (2022)](https://arxiv.org/abs/2204.13717),
equations 6–7 and Appendix A. Write
`U(j,l) = sqrt(sum_m |FFT^-1(FFT(I) psi_hat(j,l,m))|^2)`.
Then `S0(q)=mean(|I|^q)`, `S1(j,l,q)=mean(U(j,l)^q)` and
`S2(j1,j2,l,q)=mean(U(j2,l)[U(j1,l)]^q)`, with `j2>j1`.
The power q is applied when averaging, never before the next convolution.
The `l=0` member is a Gaussian with nonzero DC, as in Kymatio; do not silently
subtract a field mean or clear its DC coefficient.

Fourier multipliers use the normalization in
[Kymatio's solid-harmonic filter bank](https://github.com/kymatio/kymatio/blob/main/kymatio/scattering3d/filter_bank.py).
The radius is `sigma * 2**j * |k * cellsize|`; sigma is in grid cells.
CosmoCM reports spatial **means**, whereas Kymatio's integral backend uses sums.
Multiply by the number of covered cells to compare the latter convention.

Every wavelet component is evaluated on local Fourier slabs, transformed through
pmesh, accumulated, and discarded. The input FFT is reused. Only one first-layer
modulus spectrum is kept while traversing its second-layer descendants. No field
is gathered to a root, and no filter bank is stored. Coefficients use float64
MPI reductions, even when mesh storage and transforms use complex64.

For an incomplete region, a binary mask zeros the input and every layer's modulus;
moments average over covered cells. Fractional survey selection weights multiply
the **input** only; they are not repeatedly multiplied into each layer. No
mask-division correction is made. This is a window-dependent statistic requiring
consistently masked mocks. FFTs implement circular convolutions on the embedding
box: enlarge the box and test padding convergence if opposite boundaries could
interact. The Gaussian tails have no compact support.

The full complex Fourier grid is intentional: solid harmonics have complex
responses. Keeping just a real inverse transform would produce incorrect moments.
All m components are evaluated, including negative m, so no continuum conjugacy
assumption is made on discrete Nyquist planes.

Wavelet multipliers are evaluated with a normalized Cartesian regular-solid-
harmonic recurrence. This avoids per-cell polar/azimuthal angles and repeated
special-function evaluation. Only scalar normalization/recurrence constants are
cached; all grid-dependent arrays remain slab-sized. Checks against spherical
harmonics cover every m through l=8, including poles and the origin.

## Inputs and survey conventions

Use `statistic='wst'` through the existing `run_stats(config)` entry point.
Overdensity fields are preserved, including their mean. Explicit `kind='counts'`
fields and periodic catalogues are divided by their covered-cell mean and have
one subtracted. `mean_density` can override the counts reference density.
Catalogue RSD, assignment, and interlacing use the existing box reader/painter.
`compensation=False` is the WST default for all inputs, matching measurement of
the painted density; explicit assignment compensation is applied once before WST.
Only the real physical density is retained after painting/deconvolution.

Survey fields accept the same `survey_weight` (array or HDF5 Field, [0,1]) or
`survey_randoms` as the power-spectrum field adapter. They measure
`I = W*delta/sqrt(I2)`, with `I2=dV*sum(W**2)` over covered cells. Random-derived
W includes completeness and FKP weights and is rasterized at field cell centers.
Its overall normalization cancels against sqrt(I2). An overdensity field remains
uncentered. Supplied counts use the covered-volume mean unless `mean_density`
is specified; counts alone cannot separate unknown radial selection from density.

Survey catalogues reuse `get_mesh_pk_survey`, with its completeness/FKP weights,
alpha conventions, bounds checks and fixed/automatic box center. WST uses
`I = (n_data_weighted-alpha*n_randoms_weighted)/sqrt(I2)` where the existing
`particle` normalization (default) gives I2=I_rand, or `mixed-mesh` gives I2=I_mesh.
This field is not a selection-divided estimate of delta.

`wst.mask` supplies the binary footprint, as a global array or HDF5 Field.
Missing density tiles are automatically excluded. For survey fields, coverage
is additionally intersected with positive selection. For survey catalogues,
an omitted mask is inferred from noninterlaced positive random coverage: finite
random density can leave holes, so supply the actual footprint when available.
An explicit catalogue mask replaces this inferred footprint. Catalogue grid
indices follow existing painting coordinates `Position-boxcenter` modulo the box;
field grids use `field_origin+(index+0.5)*cellsize`. Align a mask to the appropriate
convention. WST is translation invariant on a periodic box, but relative alignment
of mask and field matters.

`wst.sources=['data_a','data_b']` computes independent WST vectors sequentially,
with independent coverage masks and constant working mesh count. Cross-WST
conventions are not defined by this estimator; `correlation_mode='cross'` raises
an error instead of borrowing the power-spectrum meaning. WST and density_split
are separate estimators. No Poisson shot-noise subtraction is applied to these
nonlinear coefficients; noise and selection must be represented in mocks.

Root receives the output dictionary and saves `output_dir/wst.npy`. Each entry
in `measurements` contains S0 `(nq,)`, S1 `(nfirst,nq)`, S2 `(nsecond,nq)`, and
`index1=(j1,l)` / `index2=(j1,j2,l)` rows identifying the coefficients. Powers,
wavelet convention, coverage, input normalization and FFT plan are recorded.

## Running and scaling

```python
from copy import deepcopy
from cosmonpc import run_stats
from cosmonpc.config.wst_fields import CONFIG

config = deepcopy(CONFIG)  # edit paths and grid geometry first
result = run_stats(config)  # dictionary on root, None on other ranks
```

Every rank must call `run_stats` with the same numerical settings. As with the
existing estimators, in-memory catalogues contain **rank-local rows**; global
density arrays and array masks are replicated caller-owned inputs. Prefer HDF5
fields/tiles for large meshes: those read only the local real-space rectangle.
All FFT work uses pmesh's distributed decomposition, and only small coefficient
vectors / scalar metadata are reduced. Files are written by root only.

At second order the core retains at most five complex local mesh arrays
(including the input) plus one real accumulator. With s>1 this is approximately
`44*Nlocal` bytes in single or `88*Nlocal` in double precision; first order uses
four complex arrays plus the accumulator. Add masks, slab temporaries, pmesh FFT
workspace/plans, source catalogue/array storage, and interpreter/library memory.
These figures are explicit array counts, **not** a total RSS prediction. The
array count is independent of J/L and of the number of output coefficients.
The optional directional traversal reuses those buffers. Processing additional
sources sequentially also keeps the same resident working mesh count.

Useful runtime controls, without changing any selected coefficient:

- Use `precision='single'` after checking accuracy for your fields.
- Select `wst.scales=[0,2,4]` and/or `wst.ells=[0,1,2]`. Retained paths are
  computed at the original mesh resolution, with their actual j labels.
- Use `max_order=1` when only first order is needed.
- Compute several q values in one run; they share all convolutions.
- Keep `m_resolved=False` unless directional coefficients are needed.

Full-resolution second order is considerably more expensive than a power
spectrum: the paper's baseline needs 396 FFTs per source before input preparation.
No downsampling is used after the nonlinear modulus. Even a Gaussian wavelet
does not make a nonlinear modulus strictly band limited; coarse-grid acceleration
would require a separately measured alias/error budget. Gaussian wavelets also
have finite-grid discretization and Nyquist sensitivity at the finest scales.
Test mesh and sigma convergence rather than interpreting sigma as a hard k cut.

Measure your MPI layouts with the included benchmark (run from the repository):

```bash
PYTHONPATH=src mpiexec -n 4 python benchmarks/benchmark_wst.py \
    --nmesh 256 --J 3 --L 3 --precision single --repeat 2
PYTHONPATH=src mpiexec -n 4 python benchmarks/benchmark_wst.py \
    --nmesh 256 --J 3 --L 3 --precision single --survey
```

Run each rank/size setting in a fresh process. The script generates its input
locally with rank-independent physical values and reports FFT counts, coefficient
checksum, maximum-rank elapsed times and peak RSS (Linux KiB). It performs no
global mesh gather. It times the core, excluding reads, painting and output.
Library/MPI diagnostics can accompany the JSON record on some installations.
Use equal grid sizes for strong scaling; compare checksums as well as timings.
For weak scaling grow the number of cells with ranks; global coefficients need
not remain equal because the physical input also changes with grid size.

Run small-grid independent-oracle tests across decompositions:

```bash
PYTHONPATH=src mpiexec -n 2 python -m unittest discover -s tests -p 'test_wavelet*.py' -v
PYTHONPATH=src mpiexec -n 4 python -m unittest discover -s tests -p 'test_wavelet*.py' -v
```

These test fixtures deliberately use tiny replicated NumPy reference fields;
the production estimator never constructs such a reference or gathers a mesh.
Actual strong/weak scaling remains to be measured on an MPI-enabled machine.
The development sandbox rejects PMIx sockets before workers start, so a
single-rank MPI run cannot validate those claims.

## Validation recorded during implementation

The existing 31 tests and 12 WST tests cover independent NumPy FFT calculations,
low-degree Cartesian filters, degree-four plane-wave addition-theorem checks,
spherical-harmonic checks through degree eight, amplitude and translation
identities, both precisions, masked propagation through both layers, optional
directional moments/ratios, input immutability, counts/catalogue equivalence,
HDF5 full fields and incomplete tiles, survey masks/randoms, FKP normalization,
multiple independent sources, invalid settings and collective output errors.

A separate local check used the real `t2_b32_final_density_sCOLA.h5` Simbelmynë
Field (256 cubed, float32, explicit counts, box 250, origin -125). For J=L=0,
q=2, single precision, S0=33.7228683644 matched a direct counts-overdensity
moment. S1=10.8271396409 matched an independent SciPy real-FFT Parseval calculation
10.8271423767 to relative error 2.6e-7. This validates a real full Field;
postprocessed tile WST tests use synthetic schema-compatible files, not a real
Silmarilli survey tile collection. Installed Kymatio was unavailable locally;
normalization checks use the documented filter formulas rather than claiming
an executed package-to-package comparison.

Representative **single-rank** core benchmarks, complex64, q=.8:

| Grid | J, L | Footprint | Elapsed seconds | Peak process RSS |
| --- | --- | --- | --- | --- |
| 64 cubed | 2, 2 | whole box | 4.21, 3.90 | 184.1 MiB |
| 128 cubed | 2, 2 | whole box | 33.25, 37.22 | 262.6 MiB |
| 64 cubed | 4, 4 | spherical mask | 43.97 | 184.8 MiB |

RSS includes interpreter/library and pmesh allocations. The extra scales/degrees
increase runtime substantially while the working array count stays fixed.
These development-process timings are indicative, not dedicated-machine scaling
measurements. The earlier spherical-coordinate filter implementation took
6.54/6.87 seconds on the 64-cubed J=L=2 case; the Cartesian recurrence preserved
the coefficient checksum exactly in these runs.

For scale planning, a 4096-cubed single-precision second-order core has about
2.75 TiB of explicit global working arrays, distributed across ranks; on 1024
balanced ranks that is about 2.75 GiB per rank **before** FFT workspaces and input
storage. This is an array-size calculation, not a measured run or a memory limit.
Measure rank-local peak RSS, filesystem behavior and communication costs on the
target cluster before selecting a production configuration.

For s selected scales and e angular degrees, second order costs
`1 + e*(s-1)` forward FFTs and
`(s+s*(s-1)/2)*sum_l(2*l+1)` inverse FFTs. At J=L=4 that means 21 forward and
375 inverse transforms for 76 coefficients per q. Additional q values only add
moment reductions. Full resolution is retained at both layers: there is no
unvalidated subsampling or nonlinear alias-removal approximation.

## Optional directional coefficients

Set `wst.m_resolved=True` to additionally evaluate the individual-m moments
and adjacent-mode ratios introduced by
[Jiang et al. (2025)](https://arxiv.org/abs/2505.14400), equations 4–7.
`M1(j,l,m,q)=mean(|I*psi(j,l,m)|**q)` and
`M2(j1,j2,l,m,q)=mean(||I*psi(j1,l,m)|*psi(j2,l,m)|**q)` use
nonnegative m and the same m and l at both layers. The baseline S1/S2 are also
returned. These cannot be reconstructed from the rotation norm when q differs
from 2. R1/R2 average adjacent m ratios for each l>0; undefined ratios are NaN
with false `ratio_valid1/2` entries. Rows are specified by `index_m1/2` and
`index_r1/2`. Individual modes and ratios depend on the Cartesian axis orientation;
they do not implement a spatially varying radial line of sight.

This opt-in traversal adds `sum_l(l+1)*(s+s*(s-1)/2)` inverse FFTs and
`sum_l(l+1)*(s-1)` forward FFTs at second order, with no additional resident
mesh. The ratios cancel a global linear amplitude rescaling; this is not a
guarantee against nonlinear tracer or survey bias. The newer paper's high-density
apodization is a separate preprocessing choice and is not applied here.
