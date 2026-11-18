![cnpc](./resources/cnpc.jpg)

# CosmoCM

CosmoCM is a fork of [CosmoNPC](https://github.com/YunchenXie/CosmoNPC),
adding selectable mesh/FFT precision, density-field inputs, density-split
clustering, and a distributed wavelet scattering transform.
The Python import remains `cosmonpc` for compatibility with existing scripts.
The original MIT license and authorship are retained.

## Wavelet scattering transform

Set `statistic="wst"` with the existing catalogue or density inputs and
`wst={"J": 4, "L": 4, "sigma": .8, "q": [.8], "max_order": 2}`.
This measures solid-harmonic S0, S1 and S2 with distributed pmesh FFTs,
streaming wavelet components instead of storing a filter bank. Both precisions,
incomplete tiles and survey selection are supported. Optional `m_resolved=True`
adds directional moments and mode ratios. Selected `scales` and `ells` reduce
runtime. WST does not need `k_min`, `k_max`, `k_bins` or `poles`.

[Conventions, survey treatment, memory/runtime costs and validation](docs/wst.md)
and an [editable field configuration](src/cosmonpc/config/wst_fields.py) describe
how to run it. Survey coefficients retain window and noise effects; use matching
mocks. Multi-rank cluster scaling has not been measured in the development sandbox.

[Local benchmarks](docs/benchmark.md) record measured CPU seconds and peak RSS
for Pl, Bl and WST from 16³ to 512³, with explicit workload and thread settings.
The scripts in `benchmarks/` reproduce the measurements and export scientific
coefficient plots and CPU/memory scaling figures.

## Standard comparison plots

The [clustering comparison plotting guide](docs/clustering_comparison_plots.md)
records the standard figure set, ratio panels, and both WST layouts.

## Precision

Set `config["precision"] = "single"` for `complex64` meshes, FFTs, and
bispectrum working arrays, or `"double"` (the default) for `complex128`.
This applies to catalogue and density inputs, including survey auxiliary
shot-noise meshes. Coordinates, normalization metadata, mode counts and some
scalar/bin reductions retain double precision. Single precision reduces field
storage by half; accuracy depends on the signal and cancellation between terms.

## Density inputs

Density inputs support periodic boxes and gridded surveys (`pk` and `bk_sugi`,
auto and cross, including all supported bispectrum tracer combinations).
For `geometry="survey-like"`, supply `survey_weight` (a same-grid field in [0,1])
or `survey_randoms`, and use mesh normalization. [Survey-field examples and
normalization](docs/density_split.md#step-5-gridded-survey-fields-tile-coverage-and-survey-weights)
cover incomplete sCOLA tiles and physical cell coordinates.
Use `input_type="density"` and provide `fields` instead of `catalogs`:

```python
import numpy as np
from cosmonpc import run_stats

config = {
    "input_type": "density",
    "fields": {"data_a": np.zeros((32, 32, 32), dtype=np.float32)},
    "precision": "single",
    "statistic": "pk",
    "geometry": "box-like",
    "correlation_mode": "auto",
    "nmesh": [32, 32, 32],
    "boxsize": [500., 500., 500.],
    "k_min": 0.01, "k_max": 0.15, "k_bins": 10,
    "poles": [0, 2], "rsd": [0, 0, 1],
    "output_dir": "./results",
}
run_stats(config)
```

Each `fields["data_a"]` (or `data_b`, `data_c`) accepts:

```python
# Global NumPy overdensity array: delta = rho / mean(rho) - 1.
source = delta
source = {"array": delta, "kind": "overdensity", "name": "matter"}

# Simbelmyne scalar Field HDF5, containing overdensity by default.
source = "density_field.h5"
source = {"path": "density_field.h5", "kind": "overdensity"}

# Raw density or cell counts: explicitly request conversion to overdensity.
source = {"path": "density_field.h5", "kind": "counts"}
source = {"array": rho, "kind": "counts", "mean_density": 2.5}

# Tiles default to overdensity, just like arrays and full HDF5 fields.
source = {"tiles": ["tile1_density_field.h5", "tile2_density_field.h5"]}
# If the tile postprocessor wrote raw counts, declare them explicitly.
source = {"tiles": ["tile1_density_field.h5", "tile2_density_field.h5"], "kind": "counts"}
```

Every field defaults to `kind="overdensity"`: its values are used unchanged
apart from the selected precision. No normalization or mean subtraction is
performed, even for incomplete tiles or a nonzero input mean. To convert raw
cell counts (or raw density), explicitly set `kind="counts"`: covered cells
become `counts / mean_counts - 1`. `mean_counts` is computed across all covered
cells, or supplied using `mean_density`. No additional centering is performed,
so an explicit reference mean is respected. For compatibility, `"delta"` is an
alias for `"overdensity"` and `"density"` is an alias for `"counts"`.
`mean_density` is only valid with counts inputs.

HDF5 reads use `/scalars/field` and root attributes `/info/scalars/N0..N2`,
`L0..L2`, `corner0..corner2`, `rank` and optional `time`. Scalar ranks 0 and 1
are accepted. Override the data path with `"dataset": "/another/path"`.
Arrays must be real, finite, three-dimensional, and match `nmesh`.
The HDF5 axis order must match `(N0, N1, N2)`; no implicit transpose is applied.
Raw densities must be nonnegative and have a positive global mean.

Tile order and filenames do not affect placement. Tile metadata must describe
a complete, nonoverlapping, grid-aligned covering of the configured box, with
the same cell spacing and cosmic time. Missing/overlapping tiles are errors
by default. Buffered overlapping tiles must
be cropped before use. `field_origin` defaults to `[0., 0., 0.]`; set it to the
corner of a box with another origin. No regridding or periodic wrapping of
out-of-box tiles is performed.

For an incomplete tile set, explicitly enable zero filling:

```python
config["fields"]["data_a"] = {
    "tiles": ["tile1_density_field.h5", "tile3_density_field.h5"],
    "kind": "overdensity",  # default; use "counts" for raw cell counts
    "missing_tiles": "zero",  # default: "error"
}
```

Uncovered cells and nonexistent listed files are treated as outside the mask.
Missing cells remain exactly zero in the **final overdensity mesh**. Supplied
overdensity values are preserved without mean subtraction. Counts inputs are
converted using the mean over all covered cells, or an explicit positive
`mean_density`; uncovered cells remain zero rather than becoming delta = -1.
Present zero-count cells still count as covered.
An empty tile list, or a list whose files are all missing, produces a zero
field. Overlap, corrupt files, incompatible metadata, and invalid values still
raise errors. Output `field_metadata` records `covered_cells`,
`coverage_fraction`, and the missing-tile policy for each tracer.

These are masked box-estimator statistics; zero filling does not implement a
survey selection function, window deconvolution, or survey normalization.

For MPI, provide the same global NumPy array on every rank, or identical HDF5
path descriptors on every rank. HDF5 files must be visible to all ranks. Each
rank reads only the HDF5 regions intersecting its local pmesh domain; the full
field is not assembled on rank zero. Counts normalization uses the
whole box (covered cells only for incomplete tiles), never an independent
mean for each tile. Overdensities are not normalized. Their supplied DC mode
is retained, including in power-spectrum measurements whose bins contain k=0.

Continuous fields have zero Poisson shot-noise subtraction. Catalogue RSD and
interlacing do not apply: set `apply_rsd=False` if adapting a catalogue config;
interlacing is disabled for fields. Supply an already distorted density field
for redshift-space measurements. Assignment-window compensation defaults to
`False`; set `compensation=True` and the correct `sampler` only when the supplied
field has that assignment window. Simbelmyne-specific higher-order assignment
kernels are not inferred from the file.

For bispectra, replace `statistic` with `"bk_sugi"`, add
`"tracer_type": "aaa"`, `"angu_config": [0, 0, 0]`, and choose
`"data_vector_mode": "diagonal"` or `"full"`. Ensure `2*k_max` is below the
Nyquist frequency. For cross statistics, set `correlation_mode="cross"` and
provide `data_b` (also `data_c` for `abc`). Catalogue/field mixing in one
measurement is not currently supported.

## Development checks

With the dependencies installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
# Run the same numerical checks over multiple MPI domains:
PYTHONPATH=src mpiexec -n 2 python -m unittest discover -s tests -v
```

The tests compare periodic power to an independent NumPy FFT, check precision,
HDF5/tile equivalence, normalization, invalid inputs, and bispectrum tracer and
output modes.

## Density-split clustering

Add `density_split` to a normal configuration to measure quantile auto/cross
power multipoles or ordered Sugiyama bispectra. Box catalogue and density-field
inputs share uniform mesh queries. Survey catalogues use random-catalogue
queries; gridded surveys use selection-weighted cells with an explicit weight
field or random-derived window. Both reuse the existing survey estimators.
Defaults are five quantiles and
Gaussian smoothing of radius 10 in `boxsize` units.

```python
config["density_split"] = {
    "nquantiles": 5, "smoothing_radius": 10.,
    "measurements": [["Q", "data_a"], ["Q", "Q"]],  # pk
}
# For bk_sugi, specify three ordered legs instead:
# config["density_split"]["measurements"] = [
#     ["Q", "data_a", "data_a"], ["Q", "Q", "data_a"], ["Q", "Q", "Q"]
# ]
result = run_stats(config)
```

Catalogue survey cross measurements require `normalization_scheme="mixed-mesh"`;
gridded surveys use `normalization_scheme="mesh"`.
[Definitions, step-by-step implementation, survey random-pool treatment,
examples, and validation](docs/density_split.md) explain the estimator choices
and the distinction between box and survey queries.

## Upstream functionality

CosmoNPC is a Python/MPI code for mesh-based measurements of large-scale structure statistics in box-like and survey-like geometries.

Current focus:

- power spectrum multipoles: `pk`
- Sugiyama bispectrum multipoles: `bk_sugi`
- auto and cross correlations
- box and survey catalogs
- MPI-distributed FFT-based estimators

## Current Scope

- `pk`
  - box-like
  - survey-like
  - auto / cross
  - user-specified `poles`

- `bk_sugi`
  - box-like
  - survey-like
  - auto (`aaa`) and cross (`aab`, `aba`, `abb`, `abc`)
  - `data_vector_mode` = `diagonal` or `full`
  - `shotnoise_mode` = `ana`, `fft`, or `both`

`bk_sco` is not part of the stable workflow yet.

## Package Layout

Main modules:

- `task_executor.py`
- `catalog_processor.py`
- `mesh_generator.py`
- `math_evaluator.py`
- `clustering_estimator.py`
- `param_helper.py`

Representative configs:

- `src/cosmonpc/config/pk_box.py`
- `src/cosmonpc/config/pk_survey.py`
- `src/cosmonpc/config/bk_sugi_box.py`
- `src/cosmonpc/config/bk_sugi_survey.py`

Example:

- `example.py`

## Install

```bash
pip install -e .
```

Core dependencies are listed in `pyproject.toml`. In practice you need at least:

- `numpy`
- `mpi4py`
- `pmesh`
- `sympy`
- `numexpr`
- `astropy`
- `fitsio`
- `h5py`
- `scipy`

## Minimal Usage

The code is config-driven.

```python
from copy import deepcopy
from cosmonpc import run_stats
from cosmonpc.config import bk_sugi_survey

config = deepcopy(bk_sugi_survey.CONFIG)
config["output_dir"] = "./results"
run_stats(config)
```

Run with MPI, for example:

```bash
mpirun -n 4 python example.py
```

## Key Configuration Fields

- `statistic`: `pk` or `bk_sugi`
- `geometry`: `box-like` or `survey-like`
- `correlation_mode`: `auto` or `cross`
- `tracer_type`: `aaa`, `aab`, `aba`, `abb`, or `abc` (the order identifies the three bispectrum legs)
- `catalogs`: input file paths
- `column_names`: catalog column mapping
- `nmesh`: mesh size `[nx, ny, nz]`
- `boxsize`: box size in `Mpc/h`
- `sampler`: `ngp`, `cic`, `tsc`, or `pcs`
- `interlaced`: whether to use interlaced painting
- `k_min`, `k_max`, `k_bins`: k-bin settings
- `poles`: power-spectrum multipoles
- `angu_config`: bispectrum angular configuration `[ell_1, ell_2, L]`
- `data_vector_mode`: `diagonal` or `full` for `bk_sugi`
- `shotnoise_mode`: `ana`, `fft`, or `both`
- `normalization_scheme`: normalization mode
- `alpha_scheme`: survey alpha convention
- `high_order_mode`: `default`, `fast`, or `compare` for survey `pk` high-order multipoles
- `output_dir`: output directory

## Input Formats

Box-like catalogs:

- `.npy`, `.fits`, `.h5`, `.hdf5`
- Cartesian positions are expected
- optional velocity columns can be used for RSD

Survey-like catalogs:

- currently `.fits`
- typical fields include `RA`, `DEC`, `Z`, `WEIGHT_FKP`, completeness weights, and `NZ`

Preprocessed catalogs can also be passed directly through `catalogs` as structured NumPy arrays, without writing FITS or HDF5 files. They should contain `Position` and `WEIGHT`; survey catalogs additionally require `WEIGHT_FKP` and `NZ`. In MPI runs, each rank should pass its local rows.

## Output

Results are written to `output_dir` as `.npy` dictionaries.

Typical outputs include:

- final measured statistic
- raw signal term
- shot-noise terms
- normalization metadata
- effective k values and mode counts

## Technical Features

- Symmetry-aware bispectrum loops reduce redundant calculations.
- The box and survey Sugiyama estimators include tracer-dependent shot-noise corrections.
- In `bk_sugi` full mode, `block_size` accelerates the calculation by trading memory for reuse: larger blocks keep more intermediate shell/binned fields in memory, so the code can reuse them across many `(k1, k2)` pairs instead of rebuilding them repeatedly.

## Notes

- `full` bispectrum mode supports block-based evaluation through `block_size`.
- Survey and box pipelines do not share exactly the same normalization and shot-noise structure.
- Small smoke tests can be run with reduced `nmesh`, smaller catalogs, and fewer bins before launching production jobs.
