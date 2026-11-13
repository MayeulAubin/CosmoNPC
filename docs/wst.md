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

For s selected scales and e angular degrees, second order costs
`1 + e*(s-1)` forward FFTs and
`(s+s*(s-1)/2)*sum_l(2*l+1)` inverse FFTs. At J=L=4 that means 21 forward and
375 inverse transforms for 76 coefficients per q. Additional q values only add
moment reductions. Full resolution is retained at both layers: there is no
unvalidated subsampling or nonlinear alias-removal approximation.

Recent work includes [Jiang et al. (2025)](https://arxiv.org/abs/2505.14400),
which retains individual m responses and introduces mode ratios and high-density
apodization. Those define different statistics from the baseline rotation norm;
neither directional ratios nor density remapping is silently applied here.
