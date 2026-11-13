"""Run isolated benchmark cases and retain reports, logs and process failures.

Small cases may run concurrently; >=256^3 cases run alone to bound total RAM.
Completed JSON files are reused unless --force is given.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',type=int,nargs='+',default=[16,32,64,128,256,512])
    p.add_argument('--estimators',nargs='+',choices=['pk','bk','wst'],default=['pk','bk','wst'])
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--timeout',type=float,default=3600.)
    p.add_argument('--memory-gib',type=float,default=12.)
    p.add_argument('--output-dir',type=Path,default=Path('benchmarks/results/local'))
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=a.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(root/'src'))
    def run(case):
        estimator,n=case
        path=out/f'{estimator}_{n}.json'
        if path.exists() and not a.force:
            print('reuse',path,flush=True)
            return
        cmd=[sys.executable,str(root/'benchmarks/benchmark_estimator.py'),estimator,'--nmesh',str(n),
             '--threads',str(a.threads),'--memory-gib',str(a.memory_gib),'--output',str(path)]
        started=time.monotonic()
        print('start',estimator,n,flush=True)
        with (out/f'{estimator}_{n}.log').open('w') as log:
            try:
                result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                status='ok' if result.returncode==0 else f'process_exit_{result.returncode}'
            except subprocess.TimeoutExpired:
                status='timeout'
        if not path.exists():
            path.write_text(json.dumps(dict(estimator=estimator,nmesh=n,status=status,
                error='Worker did not produce metrics; see adjacent log. No successful timing inferred.',
                timeout_seconds=a.timeout,thread_limit=a.threads,memory_limit_gib=a.memory_gib),indent=2)+'\n')
        print('finish',estimator,n,status,round(time.monotonic()-started,2),flush=True)
    small=[(e,n) for n in a.grids if n<256 for e in a.estimators]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run,small))
    for n in a.grids:
        if n>=256:
            for estimator in a.estimators:
                run((estimator,n))


if __name__=='__main__':
    main()
