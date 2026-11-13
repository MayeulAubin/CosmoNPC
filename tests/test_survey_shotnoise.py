"""Regression for the standard survey abc + both shot-noise output."""
import logging
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mpi4py import MPI
from cosmonpc import run_stats

COMM = MPI.COMM_WORLD
logging.disable(logging.CRITICAL)


class SurveyShotnoiseTests(unittest.TestCase):
    def test_distinct_tracers_export_zero_fft_term(self):
        rng=np.random.default_rng(331)
        dtype=[('Position','f8',(3,)),('WEIGHT','f8'),('WEIGHT_FKP','f8'),('NZ','f8')]
        catalogs={}
        for label in 'abc':
            for kind,n in [('data',100),('randoms',2500)]:
                array=np.ones(n,dtype=dtype)
                array['Position']=rng.uniform(500,570,(n,3))
                array['NZ']=.001
                catalogs[f'{kind}_{label}']=np.array_split(array,COMM.size)[COMM.rank]
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ['diagonal','full']:
                c=dict(statistic='bk_sugi',geometry='survey-like',input_type='catalog',
                       correlation_mode='cross',tracer_type='abc',catalogs=catalogs,
                       nmesh=[8]*3,boxsize=[100.]*3,sampler='cic',interlaced=False,
                       column_names=[],z_range=None,comp_weight_plan={'scheme':None,'name_alias':None},
                       cosmology=None,normalization_scheme='mixed-mesh',precision='single',
                       angu_config=[0,0,0],data_vector_mode=mode,shotnoise_mode='both',
                       k_min=.05,k_max=.10,k_bins=1,output_dir=str(Path(tmp)/mode))
                run_stats(c)
                if COMM.rank==0:
                    result=np.load(next((Path(tmp)/mode).glob('*.npy')),allow_pickle=True).item()
                    np.testing.assert_array_equal(result['SN_terms']['SN3_FFT'],0)
                    np.testing.assert_array_equal(result['Shot_noise'],0)
                    self.assertTrue(np.all(np.isfinite(result['B_sugi'])))
