"""One fresh-process benchmark of production Pl, Sugiyama Bl or WST.

CPU time excludes imports, input generation and output. Peak RSS includes them.
Use run_suite.py to run a grid sweep and keep structured failure records.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import resource
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('estimator',choices=['pk','bk','wst'])
    p.add_argument('--nmesh',type=int,required=True)
    p.add_argument('--threads',type=int,default=8,help='NumExpr/OpenMP/BLAS thread limit; PFFT is not configured by this')
    p.add_argument('--precision',choices=['single','double'],default='single')
    p.add_argument('--memory-gib',type=float,default=12.)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--J',type=int,default=2)
    p.add_argument('--L',type=int,default=2)
    p.add_argument('--bk-angles',type=int,nargs=3,default=[0,0,0])
    a=p.parse_args()
    if a.nmesh<16 or a.threads<1 or a.memory_gib<=0:
        p.error('Require nmesh>=16, threads>=1, memory-gib>0')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS']:
        os.environ[key]=str(a.threads)
    # A large failed case must not exhaust the interactive workstation's RAM.
    limit=int(a.memory_gib*1024**3)
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    import logging
    import numpy as np
    import numexpr
    from mpi4py import MPI
    from pmesh import ParticleMesh
    from cosmonpc.clustering_estimator import calculate_power_spectrum_box, calculate_bk_sugi_box
    from cosmonpc.wavelet import calculate_scattering, scattering_options
    logging.disable(logging.CRITICAL)
    comm=MPI.COMM_WORLD
    report=dict(estimator=a.estimator,nmesh=a.nmesh,ranks=comm.size,precision=a.precision,
                thread_limit=a.threads,numexpr_threads=numexpr.get_num_threads(),
                fft_backend='production pmesh/PFFT; no FFT threading control',
                memory_limit_gib=a.memory_gib,python=platform.python_version(),numpy=np.__version__,
                status='starting',boxsize=1024.,input='analytic_nonGaussian_periodic_v1',
                cpu_scope='estimator calls only; all process threads, summed over ranks',
                rss_scope='process lifetime; maximum over ranks; Linux KiB')
    cpu_start=None
    started=None
    try:
        n=a.nmesh
        pm=ParticleMesh(Nmesh=[n]*3,BoxSize=[1024.]*3,comm=comm,
                        dtype='complex64' if a.precision=='single' else 'complex128')
        field=pm.create(type='real')
        # Identical continuous field sampled at cell centers at every resolution.
        y=(np.arange(field.shape[1])+field.start[1]+.5)[None,: ,None]/n
        z=(np.arange(field.shape[2])+field.start[2]+.5)[None,None,:]/n
        variance=(.65**2+.35**2+.25**2)/2
        for i in range(field.shape[0]):
            x=(i+field.start[0]+.5)/n
            g=.65*np.cos(2*np.pi*(x+2*y))+.35*np.sin(2*np.pi*(y+z))+.25*np.cos(2*np.pi*(2*x-z))
            field.value[i]=g[0]+.25*(g[0]**2-variance)
        attrs=dict(input_type='density',geometry='box-like',nmesh=[n]*3,boxsize=[1024.]*3,
                   poles=[0,2,4],rsd=[0,0,1],k_min=.005,k_max=.02,k_bins=6,
                   sampler='cic',interlaced=False,compensation=False,
                   NZ_a=n**3/1024.**3,N_gal_a=n**3,tracer_type='aaa',
                   data_vector_mode='full',block_size=1,shotnoise_mode='ana',angu_config=a.bk_angles)
        options=scattering_options(dict(J=a.J,L=a.L,sigma=.8,q=[.8],max_order=2),[n]*3)
        report['settings']=(options if a.estimator=='wst' else
                            {k:attrs[k] for k in ['poles','k_min','k_max','k_bins','angu_config','data_vector_mode','block_size']})
        comm.Barrier()
        started=time.perf_counter()
        cpu_start=time.process_time()
        if a.estimator=='pk':
            result=calculate_power_spectrum_box(field,None,'auto',attrs,comm)
            keys=['k_eff','k_num','P0','P2','P4']
        elif a.estimator=='bk':
            result=calculate_bk_sugi_box(field,None,None,'auto',attrs,comm)
            keys=['k_eff','nmodes','B_sugi']
        else:
            result=calculate_scattering(field,options)
            keys=['S0','S1','S2','index1','index2','q']
        cpu=time.process_time()-cpu_start
        wall=time.perf_counter()-started
        report['cpu_seconds']=comm.allreduce(cpu,op=MPI.SUM)
        report['wall_seconds_max_rank']=comm.allreduce(wall,op=MPI.MAX)
        if comm.rank==0:
            report['coefficients']={key:np.real(np.asarray(result[key])).tolist() for key in keys}
            report['max_abs_imaginary']=max(float(np.max(np.abs(np.imag(np.asarray(result[key]))))) for key in keys)
            if any(not np.all(np.isfinite(result[key])) for key in keys):
                raise ValueError('Nonfinite benchmark coefficients')
        report['status']='ok'
    except Exception as exc:
        report['status']='failed'
        report['error']=f'{type(exc).__name__}: {exc}'
        if cpu_start is not None:
            report['partial_cpu_seconds']=time.process_time()-cpu_start
            report['partial_wall_seconds']=time.perf_counter()-started
    report['peak_rss_kib']=comm.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,op=MPI.MAX)
    if comm.rank==0:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='coefficients'}),flush=True)
    if report['status']!='ok':
        sys.exit(1)


if __name__=='__main__':
    main()
