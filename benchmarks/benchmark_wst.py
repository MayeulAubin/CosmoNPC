"""MPI WST benchmark with rank-independent local input and no global mesh.

Example: mpiexec -n 4 python benchmarks/benchmark_wst.py --nmesh 256 --J 3 --L 3
Use separate processes for each setting: ru_maxrss is a process lifetime peak.
"""
import argparse
import json
import resource
import numpy as np
from mpi4py import MPI
from pmesh import ParticleMesh
from cosmonpc.wavelet import calculate_scattering, scattering_options, scattering_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nmesh',type=int,default=64)
    parser.add_argument('--J',type=int,default=2)
    parser.add_argument('--L',type=int,default=2)
    parser.add_argument('--precision',choices=['single','double'],default='single')
    parser.add_argument('--max-order',type=int,choices=[1,2],default=2)
    parser.add_argument('--q',type=float,nargs='+',default=[.8])
    parser.add_argument('--scales',type=int,nargs='+')
    parser.add_argument('--ells',type=int,nargs='+')
    parser.add_argument('--m-resolved',action='store_true')
    parser.add_argument('--survey',action='store_true')
    parser.add_argument('--repeat',type=int,default=1)
    args=parser.parse_args()
    if args.repeat<1:
        parser.error('--repeat must be positive')
    comm=MPI.COMM_WORLD
    options=dict(J=args.J,L=args.L,max_order=args.max_order,q=args.q,m_resolved=args.m_resolved)
    for key in ['scales','ells']:
        if getattr(args,key) is not None:
            options[key]=getattr(args,key)
    options=scattering_options(options,[args.nmesh]*3)
    pm=ParticleMesh(Nmesh=[args.nmesh]*3,BoxSize=[float(args.nmesh)]*3,
                    dtype='complex64' if args.precision=='single' else 'complex128',comm=comm)
    field=pm.create(type='real')
    axes=[(np.arange(field.shape[i])+field.start[i]+.5)/args.nmesh for i in range(3)]
    x,y,z=axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:]
    field[...] = (.2+np.sin(2*np.pi*7*x)+.4*np.cos(2*np.pi*(3*y+2*z))+
                  .1*np.sin(2*np.pi*(5*x-4*y+z)))
    mask=((x-.5)**2+(y-.5)**2+(z-.5)**2 < .4**2) if args.survey else None
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    times=[]
    for _ in range(args.repeat):
        comm.Barrier()
        result=calculate_scattering(field,options,mask)
        times.append(result['elapsed_seconds_max_rank'])
    peak=comm.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,op=MPI.MAX)
    baseline=comm.allreduce(baseline,op=MPI.MAX)
    complex_bytes=field.value.dtype.itemsize
    cells=comm.allreduce(field.size,op=MPI.SUM)
    rank_cells=comm.allreduce(field.size,op=MPI.MAX)
    has_second=options['max_order']==2 and len(options['scales'])>1
    resident_arrays=5 if has_second else 4
    # Count only explicitly resident core arrays incl input; pmesh FFT workspaces,
    # source arrays, metadata, masks, imports, and filter slab temporaries are extra.
    explicit_bytes=rank_cells*(resident_arrays*complex_bytes+complex_bytes//2)
    report=dict(nmesh=args.nmesh,ranks=comm.size,precision=args.precision,
                options=options,plan=scattering_plan(options),local_cells_max=rank_cells,
                global_cells=cells,explicit_core_array_bytes_max_rank=explicit_bytes,
                peak_rss_kib_max_rank=peak,baseline_rss_kib_max_rank=baseline,
                seconds_max_rank=times,covered_cells=result['covered_cells'],
                coefficient_checksum=float(sum(np.sum(result[key]) for key in ['S0','S1','S2'])))
    if comm.rank==0:
        print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
