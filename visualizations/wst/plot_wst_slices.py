"""Full-resolution 3D WST slices using CosmoCM's actual filter functions.
Run: python plot_wst_slices.py [--input FIELD.h5] [--output DIR]
Only pure filter definitions are loaded via AST, avoiding the MPI dependency.
"""
import argparse
import ast
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import h5py
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, SymLogNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/home/aubin/Téléchargements/CosmoCM_tiling_test/results/final_density_sim.h5'))
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2] / 'src/cosmonpc/wavelet.py'
    tree = ast.parse(source.read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_harmonic_constants', 'solid_harmonic')], type_ignores=[])
    namespace = dict(np=np, math=math, lru_cache=lru_cache)
    exec(compile(functions, str(source), 'exec'), namespace)
    harmonic = namespace['solid_harmonic']
    with h5py.File(args.input) as f:
        delta = f['scalars/field'][:].astype(np.float32)
        box = [float(f.attrs[f'/info/scalars/L{i}']) for i in range(3)]
        origin = [float(f.attrs[f'/info/scalars/corner{i}']) for i in range(3)]
    if not np.isfinite(delta).all():
        raise ValueError('Input contains nonfinite values')
    shape = delta.shape
    cell = np.asarray(box) / shape
    k = [2*np.pi*fft.fftfreq(n, d=d).reshape(tuple(n if j == i else 1 for j in range(3))) for i, (n,d) in enumerate(zip(shape,cell))]
    q, sigma = .8, .8
    iz = shape[2]//2
    spectrum = fft.fftn(delta, workers=4)
    records, slices = [], [delta[:,:,iz].copy()]
    def moment(a):
        return float(sum(np.sum(np.abs(plane.astype(np.float64))**q) for plane in a)/a.size)
    s0 = moment(delta)
    def modulus(fourier, j, ell):
        acc = np.zeros(shape, dtype=np.float32)
        for m in range(-ell,ell+1):
            # Filter function and normalization are taken verbatim from CosmoCM.
            filt = harmonic(k, cell, sigma*2**j, ell, m).astype(np.complex64)
            response = fft.ifftn(fourier*filt, workers=4)
            np.hypot(acc, np.abs(response), out=acc)
        return acc
    first = [(j,ell) for ell in (0,1,2) for j in (0,2,4)]
    second = [(j1,j2,ell) for ell in (1,2) for j1,j2 in ((0,2),(0,4),(2,4))]
    second_slices = {}
    second_records = {}
    for j,ell in first:
        u = modulus(spectrum,j,ell)
        val = moment(u)
        print(f'S1 j={j} ell={ell}: {val:.8g}', flush=True)
        slices.append(u[:,:,iz].copy())
        records.append(dict(order=1,j1=j,ell=ell,coefficient=val))
        children = [p for p in second if p[0]==j and p[2]==ell]
        if children:
            fu = fft.fftn(u, workers=4)
            for path in children:
                v = modulus(fu,path[1],ell)
                val2 = moment(v)
                print(f'S2 {path}: {val2:.8g}', flush=True)
                second_slices[path] = v[:,:,iz].copy()
                second_records[path] = dict(order=2,j1=j,j2=path[1],ell=ell,coefficient=val2)
            del fu
    for path in second:
        slices.append(second_slices[path]); records.append(second_records[path])
    meta = dict(input=str(args.input),filter_source=str(source),shape=list(shape),boxsize=box,origin=origin,slice_axis=2,slice_index=iz,slice_coordinate=origin[2]+(iz+.5)*cell[2],q=q,sigma_cells=sigma,S0=s0,coefficients=records,convention='Periodic 3D convolutions, U=sqrt(sum_m |response_m|^2); S=mean_volume(U^q); same ell at both layers; no downsampling.',display='Original: symmetric log; responses: independent 99.5th percentile limits and power norm gamma=0.5. Color bars show raw field amplitudes; coefficients use untransformed full-volume arrays.')
    (args.output/'wst_coefficients.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez_compressed(args.output/'wst_slices.npz',slices=np.asarray(slices),**{'q':q,'sigma':sigma})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#0b101b','axes.facecolor':'#0b101b','text.color':'#e8edf5','axes.labelcolor':'#b8c5d6','xtick.color':'#9cabbf','ytick.color':'#9cabbf','axes.edgecolor':'#364359'})
    fig,axes = plt.subplots(4,4,figsize=(18,19))
    fig.subplots_adjust(left=.055,right=.955,bottom=.085,top=.88,wspace=.42,hspace=.55)
    fig.suptitle('Dark matter through the wavelet scattering transform', fontsize=23,x=.055,ha='left',y=.975)
    fig.text(.055,.939,r'$256^3$ cells · periodic 3D convolutions · central $xy$ slice · $q=0.8$, $\sigma=0.8$ cells',fontsize=12,color='#acbdd1')
    fig.text(.055,.911,r'$U_1=\left(\sum_m|\delta\ast\psi_{j_1\ell m}|^2\right)^{1/2}$     $U_2=\left(\sum_m|U_1\ast\psi_{j_2\ell m}|^2\right)^{1/2}$     $S_n=\langle U_n^{0.8}\rangle_{\rm 3D}$',fontsize=13)
    extent=[origin[0],origin[0]+box[0],origin[1],origin[1]+box[1]]
    for idx,(ax,a) in enumerate(zip(axes.flat,slices)):
        if idx==0:
            limit=float(np.percentile(np.abs(a),99.5))
            norm=SymLogNorm(linthresh=.3,vmin=-limit,vmax=limit)
            title=r'Original overdensity $\delta$'; label=rf'$S_0=\langle|\delta|^{{0.8}}\rangle={s0:.5f}$'; cmap='RdBu_r'
        else:
            r=records[idx-1]; norm=PowerNorm(.5,vmin=0,vmax=float(np.percentile(a,99.5)))
            if r['order']==1:
                title=rf"$U_1$: $j={r['j1']},\ \ell={r['ell']}$"
            else:
                title=rf"$U_2$: $j_1={r['j1']}\to j_2={r['j2']},\ \ell={r['ell']}$"
            label=rf"$S_{r['order']}={r['coefficient']:.5f}$"; cmap='magma'
        im=ax.imshow(a.T,origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest')
        ax.set_title(title,loc='left',fontsize=12,pad=9)
        ax.set_xticks([0,128,256]); ax.set_yticks([0,128,256])
        ax.set_xlabel(label,fontsize=12,labelpad=7)
        cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.025,extend='both' if idx==0 else 'max')
        cb.ax.tick_params(labelsize=8)
    fig.text(.055,.045,'9 first-order + 6 second-order responses. Each response combines every m mode; coefficients average the entire 3D box.',fontsize=11,color='#acbdd1')
    fig.text(.055,.025,'Response panels use independent color limits (99.5th percentile) and a square-root color stretch. Color bars give raw amplitudes.',fontsize=10,color='#acbdd1')
    fig.savefig(args.output/'wst_density_slices.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(args.output/'wst_density_slices.pdf',facecolor=fig.get_facecolor())
    print('Saved figures and data to',args.output,flush=True)

if __name__=='__main__':
    main()
