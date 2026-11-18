# WST density slices

`wst_density_slices.png` and `.pdf` contain the original overdensity plus 15 selected WST responses, all at the central z plane (index 128). Coordinates follow the HDF5 box (length 256 per axis); the file does not specify a physical unit.

The calculation uses the full 256³ grid, periodic 3D FFT convolutions, sigma=0.8 grid cells and q=0.8. First-order paths use j in {0,2,4} and ell in {0,1,2}; second-order paths use (j1,j2) in {(0,2),(0,4),(2,4)} and ell in {1,2}. Each modulus is the rotation norm sqrt(sum_m |convolution_m|²), and both layers use the same ell. The ell=0 filter is Gaussian. Powers are applied only when averaging, never before the second convolution. S0 and all coefficients below panels are full-volume means, not slice means. Input values are already overdensities and are left unchanged.

Responses use separate 99.5th-percentile upper limits and square-root color stretches to make spatial structure visible across scales. Color bars show raw amplitudes. The original uses a symmetric logarithmic color scale. These display transforms and clipping do not enter coefficients.

The script loads the two pure filter functions directly from `src/cosmonpc/wavelet.py` using Python AST, without importing its unavailable MPI dependency. It uses SciPy full complex FFTs and float32/complex64 working arrays, with float64 moment reductions. This is a serial visualization adapter, not a run of the distributed estimator.

Run from any directory:

```bash
OPENBLAS_NUM_THREADS=1 python /home/aubin/Projects/CosmoCM/visualizations/wst/plot_wst_slices.py
```

Use `--input` and `--output` to override paths. `wst_coefficients.json` records the paths, conventions and numerical coefficients; `wst_slices.npz` stores the 16 raw slices in figure order. All j labels refer to the original grid, without downsampling. Gaussian radii sigma*2**j are 0.8, 3.2 and 12.8 cells.
