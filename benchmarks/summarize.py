"""Render measured CPU/RSS tables and scaling figures from benchmark JSON."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('benchmarks/results/local'))
    p.add_argument('--document',type=Path,default=Path('docs/benchmark.md'))
    p.add_argument('--plots',type=Path)
    a=p.parse_args()
    rows={}
    for path in a.results.glob('*.json'):
        d=json.loads(path.read_text())
        rows[(d['estimator'],d['nmesh'])]=d
    text='''# Local estimator benchmarks

Measurements made on 2026-11-13 with the production CosmoCM kernels, an Intel
Core Ultra 7 165H workstation (30 GiB RAM), Python 3.11.18, NumPy 1.26.4,
single-precision complex64 meshes and one MPI rank. This is **not** an MPI scaling
test. Each case runs in a fresh process, once, with cold estimator setup.

## CPU and memory conventions

**CPU seconds** are measured using `time.process_time()` around the estimator
call: user + system CPU time summed across every thread in that process. Under
MPI the runner sums these times across ranks. These numbers are not wall seconds
and are not wall time multiplied by a requested thread count. Input generation,
imports, initial pmesh creation/planning and result serialization are excluded;
estimator-specific allocations, symbolic angular-function compilation, FFTs and
coefficient reductions are included.

**Peak RSS** is Linux `ru_maxrss`, converted from KiB to MiB (1024 KiB). It is the
process-lifetime high-water mark, including libraries, input mesh, pmesh plans
and estimator workspace. Thus its measurement scope is broader than CPU time.
Under MPI this is the maximum rank RSS, not the sum across ranks. RSS is not
virtual address space, allocated-array byte counts or an incremental baseline.

NumExpr, OpenMP and BLAS thread limits are explicitly set to **8** before imports.
NumExpr reports eight active threads. The installed pmesh/PFFT extension has
statically linked FFT code and exposes no FFT threading controls: the FFTs here
remain single-threaded per rank. WST does not use NumExpr for its wavelet kernels.
Independent cases below 256 cubed run concurrently (two processes), so the suite
uses multiple cores without changing the production FFT backend. Large cases run
alone to bound RAM usage. Parallel sections do not imply eight-core utilization
throughout; timings include competition with other local activity. FFTW threading
requires explicit initialization and threaded plans; setting `OMP_NUM_THREADS`
alone does not enable it ([FFTW documentation](https://fftw.org/doc/Usage-of-Multi_002dthreaded-FFTW.html)).

Each worker has a **12 GiB virtual-address-space ceiling** to avoid exhausting
the interactive workstation; this is independent of the measured peak RSS.
A failed/limited case is explicitly labeled and is never filled with an
extrapolated timing. Partial CPU/RSS measurements, when available, remain in
the raw JSON, not in the successful timing columns.

## Fixed workloads

All cases use the same continuous non-Gaussian test field sampled at cell centers
in a periodic box of side 1024 h^-1 Mpc:

```
g = .65 cos(2 pi (x+2y)) + .35 sin(2 pi (y+z)) + .25 cos(2 pi (2x-z))
delta = g + .25 (g*g - (.65**2+.35**2+.25**2)/2)
```

Coordinates x,y,z here are in units of box length. Input generation uses local
planes and does not construct a global field on root. There is no assignment
compensation, particle shot-noise subtraction, survey window or density split in
these baseline workloads. They measure the underlying estimators; input readers,
catalogue painting and quantile construction have additional costs. No hard
Nyquist boundary is approached: even at 16 cubed, 2*kmax=0.04 is below Nyquist.

- **Pl:** auto power multipoles 0, 2, 4, fixed z LOS, six linear bins from
  k=0.005 to 0.02 h/Mpc.
- **Bl:** Sugiyama B000, full 6 x 6 k1,k2 vector over the same bins, auto continuous
  field, block_size=1. Other angular configurations and block sizes cost differently.
- **WST:** J=L=2, sigma=0.8 cells, q=0.8, max_order=2, all scales/degrees,
  m_resolved=False. This is 19 coefficients, seven forward and 54 inverse FFTs.
  The paper's J=L=4 profile is substantially more expensive (76 coefficients,
  21 forward + 375 inverse FFTs); the table does not represent that workload.
  Fixed sigma in cells corresponds to different physical smoothing at each grid.

'''
    for estimator,title in [('pk','Pl'),('bk','Bl'),('wst','WST')]:
        text+=f'## {title}\n\n| Grid | CPU seconds | Peak RSS (MiB) | Status |\n| --- | ---: | ---: | --- |\n'
        for n in [16,32,64,128,256,512]:
            d=rows.get((estimator,n),{})
            ok=d.get('status')=='ok'
            cpu=f"{d['cpu_seconds']:.3f}" if ok else '—'
            rss=f"{d['peak_rss_kib']/1024:.1f}" if ok else '—'
            status='Completed' if ok else d.get('status','Not run')
            if not ok and 'error' in d:
                status+=': '+d['error'].replace('|','/')[:180]
            text+=f'| {n}³ | {cpu} | {rss} | {status} |\n'
        text+='\n'
    text+='''## Reproducing and extending

Run from the repository with the NPC Python environment (or an equivalent
installation of the project dependencies):

```bash
PYTHONPATH=src python benchmarks/run_suite.py --threads 8 --jobs 2
python benchmarks/summarize.py --plots "$HOME/Téléchargements/CosmoCM"
```

The sweep uses all six requested grids, stores raw metrics and small coefficient
vectors in `benchmarks/results/local/*.json`, and keeps diagnostic logs alongside
them (logs are gitignored). Completed records are reused; use `--force` or a new
`--output-dir` for a new measurement campaign. `--memory-gib` and `--timeout`
configure the worker ceiling and per-case wall timeout. Wall seconds are retained
as raw diagnostics, but the tables and CPU plots use measured CPU seconds only.

For a single case, including future multi-rank measurements:

```bash
PYTHONPATH=src mpiexec -n 8 python benchmarks/benchmark_estimator.py wst \\
    --nmesh 1024 --threads 8 --memory-gib 64 --output /path/to/wst_1024.json
```

Request enough CPUs per rank for the selected thread limit, bind ranks/threads
appropriately, and first run the independent numerical tests across MPI layouts.
Keep physical box, bins, WST paths, precision and thread settings fixed when
comparing strong scaling. MPI timing/RSS aggregation must be labeled separately
from these local single-rank tables.

## Scientific plots

`benchmarks/plot_statistics.py` computes separate WST and B000/B202 figures from
the available real Simbelmynë final-density Field. Its native 256-cubed counts
are rebinned to 128 cubed by volume averaging, then converted explicitly to
overdensity. The box remains 250 h^-1 Mpc. These figures are **not** the synthetic
benchmark field and their plotting/measurement time is not included above.
They use J=L=4 with q=0.8,1,2 and the full bispectrum over 12 bins from
0.025 to 0.35 h/Mpc. Arrays, provenance JSON and PDF/PNG exports accompany them.

```bash
PYTHONPATH=src python benchmarks/plot_statistics.py /path/to/field.h5 \\
    --nmesh 128 --threads 8 --output-dir "$HOME/Téléchargements/CosmoCM"
```

Benchmark CPU/RSS scaling plots and the scientific coefficient plots are saved
under `Téléchargements/CosmoCM`. Future MPI/large-grid measurements should use
their own results directory; do not overwrite this local measurement campaign.
'''
    a.document.parent.mkdir(parents=True,exist_ok=True)
    a.document.write_text(text)
    if a.plots:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a.plots.mkdir(parents=True,exist_ok=True)
        for estimator,title in [('pk','Pl'),('bk','Bl'),('wst','WST')]:
            data=[rows[(estimator,n)] for n in [16,32,64,128,256,512] if rows.get((estimator,n),{}).get('status')=='ok']
            if not data:
                continue
            n=np.array([d['nmesh'] for d in data])
            fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
            for ax,values,label in [(axes[0],[d['cpu_seconds'] for d in data],'CPU seconds (all process threads)'),
                                    (axes[1],[d['peak_rss_kib']/1024 for d in data],'Peak process RSS (MiB)')]:
                ax.loglog(n,values,'o-',base=2)
                ax.set_xticks([16,32,64,128,256,512],labels=['16³','32³','64³','128³','256³','512³'])
                ax.set(xlabel='Grid',ylabel=label)
                ax.grid(alpha=.2,which='both')
            fig.suptitle(f'CosmoCM {title}: measured local benchmark\nSingle MPI rank; 8 NumExpr threads; production PFFT; complex64')
            for suffix in ['png','pdf']:
                fig.savefig(a.plots/f'{title}_benchmark.{suffix}',dpi=180)
            plt.close(fig)


if __name__=='__main__':
    main()
