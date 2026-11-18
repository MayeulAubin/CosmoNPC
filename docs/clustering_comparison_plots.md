# Standard clustering comparison plots

Use this layout when asked for a standard CosmoCM clustering comparison. The
figure set covers density slices, power multipoles, density-split spectra,
equilateral bispectrum multipoles, and wavelet scattering. Use the sample,
geometry, line of sight, mesh size, and estimator settings requested for the
comparison.

## Figure conventions

- Put the measured quantities for both catalogues above their ratio. The
  comparison catalogue is the numerator and the reference catalogue is the
  denominator. Label the ratio with both names. Use a horizontal unity guide
  and add accuracy guides to zoomed ratio panels.
- Use the Silmarilli catalogue-comparison Matplotlib settings with serif type,
  15-point base text, LaTeX rendering with `amsmath`, `upgreek`, and `bm`,
  and a 2-point default linewidth. Export PNGs at **400 dpi**. Use dashed
  lines for the comparison and solid lines for the reference. When color
  encodes quintile or harmonic degree, use that same color for both catalogues
  and for the single ratio curve.
- Keep signed powers, bispectra, and ratios. A zero reference leaves the ratio
  undefined. Flag bins where a near-zero denominator makes a ratio unstable.
  Keep the full measurements in the saved data even if the displayed ratio
  panel is zoomed. Report the mesh assignment and shot-noise convention
  because they can differ by statistic.
- Use common bins, modes, and units for each pair. Include the epoch and
  line of sight where applicable. Show and check one figure at a time when
  the user wants progressive results.

## Standard figure set

| Figure | Layout and measurement |
| --- | --- |
| **01 Density slice** | Use the standard `sbmy_control` slice and map tools. Show the same unsmoothed plane for both catalogues, side by side on a shared color scale. For a fixed Cartesian line of sight, use the transverse midplane by default. Plot $\log_{10}(1+\delta_h)$ above and the pixelwise ratio of $1+\delta_h$ below, with each field normalized by its own mean. Mask pixels with zero reference density. |
| **02 Power multipoles** | Put $\ell=0$ and $\ell=2$ in separate columns. Show signed $kP_\ell(k)$ above and the ratio below, using linear $k$ axes. For a periodic number-weighted catalogue auto-spectrum, record the Poisson subtraction from $P_0$. |
| **03 Density-split spectra** | Put quintile–tracer cross-spectra $Q_i\times h$ on the left and quintile autos $Q_i\times Q_i$ on the right, with $P_0$ in the upper row and $P_2$ in the lower row. Each cell contains signed $kP_\ell$ above its ratio. Overlay the five quintiles in distinct, consistent colors, from $Q_0$ (lowest density) to $Q_4$ (highest). For periodic boxes, rank equal-volume cells independently in each catalogue with a periodic Gaussian environmental radius of $10\,h^{-1}\mathrm{Mpc}$ by default. Measure the tracer leg on the unsmoothed field. Record the quintile boundaries and check for ties when changing the smoothing scale. |
| **04 Equilateral bispectrum** | Put $B_0$ and $B_2$ in separate columns and show signed $k^2B_\ell(k,k,k)$ above the ratio. Restrict all three legs to the same $k$ shell before triangle closure. Save the effective $k$, triangle counts, and shot-noise terms. Use a wider-shell run as a separate diagnostic when bin-to-bin swings warrant checking. |
| **05 Wavelet scattering** | Keep a narrow $S_0$ column and wider $S_1$ and $S_2$ columns, each with its ratio below. Keep **both** versions described below. Report $J,L,\sigma,q$ and the input field convention. |

### The two WST layouts

- **`05_wst.png`, physical scales.** Use color for harmonic degree $\ell$.
  For order 1, plot the Gaussian envelope width $R_j=\sigma\,2^j\,\Delta x$
  on a log axis. For order 2, group points by the first width $R_{j_1}$
  and label the horizontal ticks within each group by the second width
  $R_{j_2}$. Within a group, larger widths lie to the right. The second
  axis restarts in each new group. Larger filter widths probe larger
  real-space scales, whereas increasing Fourier $k$ corresponds to
  shorter wavelengths.
- **`05_wst_by_degree.png`, harmonic degrees.** Group coefficients by
  $\ell$ on the horizontal axis. Within each group, order $S_1$ by $j$
  and $S_2$ by $(j_1,j_2)$, with $j_2>j_1$. Use the same saved
  coefficients and ratios as the physical-scale version.

## Worked example and ratio windows

The example used remapped sCOLA and Gadget4 Rockstar hosts at $z=0.8$,
with $M>10^{13}\,M_\odot$ after remapping, a $512\,h^{-1}\mathrm{Mpc}$
periodic box, a $256^3$ mesh, and fixed $z$-axis RSD from `corevel`.
The ratios were sCOLA/Gadget4. The WST used the unsmoothed CIC overdensity
with $J=L=4$, $\sigma=0.8$ cells, $q=0.8$, and orders 0–2.
The resulting filter widths were $1.6,\ 3.2,\ 6.4,\ 12.8,\ 25.6\,
h^{-1}\mathrm{Mpc}$.

| Ratio panel | Linear display limits used |
| --- | --- |
| Power $P_0$ | 0.98–1.06 |
| Power $P_2$ | 0.70–1.10 |
| Density-split $Q_i\times h$, $P_0$ | 0.80–1.20 |
| Density-split $Q_i\times Q_i$, $P_0$ | 0.90–1.10 |
| Density-split $P_2$, both columns | 0.50–1.50 |

Use these windows as starting points for similar data. For density-split
plots, use guides at unity and at ±5% and ±10% for $P_0$, and ±10% and
±25% for $P_2$. Choose bispectrum and WST ratio limits from their measured
ranges and note any values clipped by the plot window.
