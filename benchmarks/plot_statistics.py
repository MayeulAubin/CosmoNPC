"""Compute and plot WST and Bl of a real Simbelmyne counts Field.

Rebin by volume averaging; arrays, PDF/PNG plots and provenance go to output-dir.
Plotting is separate from the estimator benchmark timings.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('field',type=Path)
    p.add_argument('--nmesh',type=int,default=128)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reuse',action='store_true',help='Redraw existing coefficient files with matching provenance')
    a=p.parse_args()
    for key in ['OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]=str(a.threads)
    import logging
    import h5py
    import numpy as np
    from mpi4py import MPI
    from cosmonpc.density_processor import get_density_meshes
    from cosmonpc.wavelet import calculate_scattering
    from cosmonpc.clustering_estimator import calculate_bk_sugi_box
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import NullLocator, FuncFormatter
    logging.disable(logging.CRITICAL)
    if MPI.COMM_WORLD.size!=1:
        raise ValueError('This plotting fixture is intentionally serial')
    out=a.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if a.reuse:
        provenance=json.loads((out/'provenance.json').read_text())
        if provenance['source']!=str(a.field.resolve()) or provenance['nmesh']!=a.nmesh:
            raise ValueError('Existing coefficient provenance does not match source/grid')
    with h5py.File(a.field) as f:
        rho=f['/scalars/field'][...]
        box=[float(f.attrs[f'/info/scalars/L{i}']) for i in range(3)]
    original_shape=rho.shape
    n=a.nmesh
    if any(s%n for s in rho.shape):
        raise ValueError('nmesh must divide all input dimensions')
    factors=[s//n for s in rho.shape]
    rho=rho.reshape(n,factors[0],n,factors[1],n,factors[2]).mean(axis=(1,3,5),dtype=np.float64).astype('f4')
    attrs,field,_,_=get_density_meshes({'data_a':{'array':rho,'kind':'counts'}},'auto',[n]*3,box,
        MPI.COMM_WORLD,'bk_sugi',tracer_type='aaa',precision='single')
    del rho
    options=dict(J=4,L=4,q=[.8,1.,2.],sigma=.8,max_order=2)
    if a.reuse:
        with np.load(out/'WST_coefficients.npz') as data:
            wst={key:data[key] for key in data.files}
    else:
        wst=calculate_scattering(field,options)
        np.savez_compressed(out/'WST_coefficients.npz',**{k:wst[k] for k in ['S0','S1','S2','index1','index2','q']})
    print('WST coefficients saved',flush=True)
    # One panel per q; q-th roots compare response amplitudes in delta units.
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for iq,q in enumerate(wst['q']):
        ax=axes[0,iq]
        for ell in range(5):
            select=wst['index1'][:,1]==ell
            radius=.8*2.**wst['index1'][select,0]*box[0]/n
            ax.loglog(radius,wst['S1'][select,iq]**(1/q),'o-',label=fr'$\ell={ell}$')
        ax.set(xlabel=r'$\sigma 2^j\,\Delta x\ [h^{-1}\mathrm{Mpc}]$',ylabel=r'$S_1^{1/q}$',title=f'First order, q={q:g}')
        ax.set_xticks(radius,labels=[f'{r:.3g}' for r in radius])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2,fontsize=8)
    radii=.8*2.**np.arange(5)*box[0]/n
    for ax,ell in zip(axes[1],[1,2,4]):
        matrix=np.full((5,5),np.nan)
        for (j1,j2,l),value in zip(wst['index2'],wst['S2'][:,0]):
            if l==ell:
                denom=wst['S1'][np.all(wst['index1']==[j1,ell],axis=1),0][0]
                matrix[j2,j1]=value/denom
        image=ax.imshow(matrix,origin='lower',vmin=0,cmap='viridis')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=[f'{r:.3g}' for r in radii],yticklabels=[f'{r:.3g}' for r in radii],
               xlabel=r'First-layer radius $[h^{-1}\mathrm{Mpc}]$',ylabel=r'Second-layer radius $[h^{-1}\mathrm{Mpc}]$',
               title=fr'$S_2/S_1$, $q=0.8$, $\ell={ell}$')
        fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle(f'CosmoCM WST — Simbelmynë matter field, {n}³, box {box[0]:g} h⁻¹ Mpc\nMass-preserving rebin from {original_shape[0]}³; single precision')
    for suffix in ['png','pdf']:
        fig.savefig(out/f'WST_coefficients.{suffix}',dpi=180)
    plt.close(fig)
    attrs.update(input_type='density',boxsize=box,nmesh=[n]*3,rsd=[0,0,1],sampler='cic',interlaced=False,
                 compensation=False,tracer_type='aaa',k_min=.025,k_max=.35,k_bins=12,
                 data_vector_mode='full',block_size=1,shotnoise_mode='ana')
    if a.reuse:
        with np.load(out/'Bl_coefficients.npz') as data:
            k,b0,b2=data['k'],data['B000'],data['B202']
    else:
        results={}
        for angular in [(0,0,0),(2,0,2)]:
            results[angular]=calculate_bk_sugi_box(field.copy(),None,None,'auto',{**attrs,'angu_config':angular},MPI.COMM_WORLD)
            print('Bl',angular,'computed',flush=True)
        k=results[(0,0,0)]['k_eff']
        b0=np.real(results[(0,0,0)]['B_sugi'])
        b2=np.real(results[(2,0,2)]['B_sugi'])
        np.savez_compressed(out/'Bl_coefficients.npz',k=k,B000=b0,B202=b2)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    edges=np.linspace(.025,.35,13)
    for ax,b,label in zip(axes[:2],[b0,b2],['000','202']):
        vmax=float(np.max(abs(b)))
        im=ax.pcolormesh(edges,edges,b.T,norm=SymLogNorm(max(vmax*1e-3,1e-12),vmin=-vmax,vmax=vmax),cmap='RdBu_r',shading='flat')
        ax.set(xlabel=r'$k_1\ [h\,\mathrm{Mpc}^{-1}]$',ylabel=r'$k_2\ [h\,\mathrm{Mpc}^{-1}]$',title=fr'$B_{{{label}}}(k_1,k_2)$')
        bar=fig.colorbar(im,ax=ax,shrink=.8,label=r'$[h^{-6}\mathrm{Mpc}^{6}]$',ticks=[-vmax,-vmax*.01,0,vmax*.01,vmax])
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:'0' if value==0 else f'{value:.1e}'))
    for b,label in [(b0,'000'),(b2,'202')]:
        axes[2].plot(k,np.diag(b),'o-',label=fr'$B_{{{label}}}(k,k)$')
    axes[2].set(xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',ylabel=r'$B(k,k)\ [h^{-6}\mathrm{Mpc}^{6}]$',title='Diagonal configurations',yscale='symlog')
    threshold=max(float(np.max(abs(b0))),float(np.max(abs(b2))))*1e-3
    axes[2].set_yscale('symlog',linthresh=threshold)
    ymax=max(float(np.max(abs(b0))),float(np.max(abs(b2))))
    ticks=10.**np.arange(np.ceil(np.log10(threshold)),np.floor(np.log10(ymax))+1)
    axes[2].set_yticks(np.r_[-ticks[::-1],0,ticks])
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value,_:'0' if value==0 else f'{value:.0e}'))
    axes[2].legend()
    axes[2].grid(alpha=.2)
    fig.suptitle(f'CosmoCM Sugiyama bispectrum — same {n}³ matter overdensity; fixed Cartesian z LOS\nContinuous density: no Poisson subtraction')
    for suffix in ['png','pdf']:
        fig.savefig(out/f'Bl_coefficients.{suffix}',dpi=180)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(source=str(a.field.resolve()),original_shape=original_shape,
        nmesh=n,boxsize=box,kind='counts',rebin='volume_average',precision='single',wst=options,
        bk={key:attrs[key] for key in ['k_min','k_max','k_bins','data_vector_mode','block_size']},
        angular_configs=[[0,0,0],[2,0,2]],thread_limit=a.threads),indent=2)+'\n')
    print('Plots saved in',out,flush=True)


if __name__=='__main__':
    main()
