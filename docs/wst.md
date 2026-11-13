# Wavelet scattering transform

Implementation proceeds in separately validated commits: distributed mathematical
core; input and survey adapters; performance tools and user documentation.

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
