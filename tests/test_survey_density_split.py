"""Survey adapter parity with unchanged standard catalogue estimators."""
import tempfile
import unittest
from pathlib import Path
import logging
import numpy as np
import fitsio
from mpi4py import MPI
from cosmonpc import run_stats
from cosmonpc.survey_density_split import (prepare_survey_density_split, measurement_catalogues, _position_pools)
from cosmonpc.catalog_processor import catalog_reader

COMM = MPI.COMM_WORLD
logging.disable(logging.CRITICAL)
DTYPE=[('Position','f8',(3,)),('WEIGHT','f8'),('WEIGHT_FKP','f8'),('NZ','f8')]


def inputs():
    rng=np.random.default_rng(237)
    def sample(n,lo,hi):
        data=np.ones(n,dtype=DTYPE)
        data['Position']=rng.uniform(lo,hi,(n,3))
        data['WEIGHT']=rng.uniform(.9,1.1,n)
        data['WEIGHT_FKP']=rng.uniform(.8,1.,n)
        data['NZ']=.001
        return data
    d,r=sample(500,970,1030),sample(12000,960,1040)
    # Contiguous rank partitions preserve the same global query-row convention.
    return np.array_split(d,COMM.size)[COMM.rank],np.array_split(r,COMM.size)[COMM.rank]


def config(tmp,**extra):
    d,r=inputs()
    c=dict(statistic='pk',input_type='catalog',geometry='survey-like',
           catalogs={'data_a':d,'randoms_a':r},nmesh=[8]*3,boxsize=[120.]*3,
           k_min=.04,k_max=.09,k_bins=1,poles=[0,2],precision='double',
           output_dir=str(tmp),density_split={'nquantiles':3,'smoothing_radius':10.,
           'measurements':[['Q0','data_a'],['Q0','Q0']]})
    c.update(extra)
    return c


class SurveySplitTests(unittest.TestCase):
    def test_query_populations_and_independent_random_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared=prepare_survey_density_split(config(tmp))
            registry,randoms,roles,backgrounds,info,base,options,measurements=prepared
            self.assertEqual(info['query_sampling'],'source_random_catalogue')
            self.assertEqual(info['query_count'],min(info['requested_query_count'],info['available_query_count']))
            self.assertLessEqual(np.ptp(info['counts']),1)
            original_positions = randoms['data_a']['Position']
            np.testing.assert_array_equal(_position_pools(original_positions[::-1],0)[::-1], roles['data_a'])
            parents=COMM.allgather(randoms['data_a']['Position'])
            pools=COMM.allgather(roles['data_a'])
            positions=np.concatenate(parents)
            allroles=np.concatenate(pools)
            for q in range(3):
                selected=np.concatenate(COMM.allgather(registry[f'Q{q}']['Position']))
                self.assertTrue(set(map(tuple,selected)) <= set(map(tuple,positions[allroles==0])))
            catalogs,mode,pattern=measurement_catalogues(('Q0','Q1','data_a'),registry,randoms,roles,backgrounds)
            self.assertEqual((mode,pattern),('cross','abc'))
            backgrounds_sets=[]
            for label in 'abc':
                points=np.concatenate(COMM.allgather(catalogs[f'randoms_{label}']['Position']))
                backgrounds_sets.append(set(map(tuple,points)))
            for i in range(3):
                for j in range(i):
                    self.assertFalse(backgrounds_sets[i] & backgrounds_sets[j])
            self.assertFalse(backgrounds_sets[0] & set(map(tuple,positions[allroles==0])))
            # NZ scales with the selected query density, not the original galaxy density.
            self.assertGreater(backgrounds['Q0']['nz_scale'],0.)

    def test_existing_estimator_parity_for_power_and_ordered_bispectra(self):
        with tempfile.TemporaryDirectory() as tmp:
            for statistic in ['pk','bk_sugi']:
                legs=([['Q0','data_a'],['Q0','Q0']] if statistic=='pk' else
                    [['Q0','data_a','data_a'],['data_a','Q0','data_a'],['data_a','data_a','Q0'],
                     ['Q0','Q0','data_a'],['Q0','Q0','Q0'],['Q0','Q1','data_a']])
                for precision in ['double','single']:
                    c=config(Path(tmp)/'split',statistic=statistic,precision=precision,
                        angu_config=[0,0,0],shotnoise_mode='both',data_vector_mode='full',
                        interlaced=True,sampler='tsc',alpha_scheme='nbodykit',
                        density_split={'nquantiles':3,'smoothing_radius':10.,'measurements':legs})
                    prepared=prepare_survey_density_split(c)
                    result=run_stats(c)
                    registry,randoms,roles,backgrounds,info,base,_,_=prepared
                    for i, tokens in enumerate(legs):
                        catalogs,mode,pattern=measurement_catalogues(tokens,registry,randoms,roles,backgrounds)
                        folder=Path(tmp)/f'{statistic}_{precision}_{i}'
                        ordinary={**base,'catalogs':catalogs,'correlation_mode':mode,
                                  'tracer_type':pattern,'boxcenter':info['boxcenter'],'output_dir':str(folder)}
                        run_stats(ordinary)
                        if COMM.rank==0:
                            reference=np.load(next(folder.glob('*.npy')),allow_pickle=True).item()
                            selected=result['measurements']['__'.join(tokens)]
                            keys=['P0','P2','I_norm'] if statistic=='pk' else ['B_sugi','Bk_raw','Shot_noise','I_norm']
                            for key in keys:
                                np.testing.assert_allclose(selected[key],reference[key],rtol=1e-12,atol=1e-8)
                            self.assertEqual(selected['tracer_type'],pattern)

    def test_auto_normalizations_and_high_order_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for normalization in ['particle','mesh','mixed-mesh']:
                result=run_stats(config(tmp,statistic='bk_sugi',angu_config=[0,0,0],
                    normalization_scheme=normalization,shotnoise_mode='fft',
                    density_split={'nquantiles':3,'measurements':[['Q0','Q0','Q0']]}))
                if COMM.rank==0:
                    self.assertTrue(np.all(np.isfinite(result['measurements']['Q0__Q0__Q0']['B_sugi'])))
            result=run_stats(config(tmp,poles=[0,2,4],high_order_mode='compare',
                normalization_scheme='particle',density_split={'nquantiles':3,'measurements':[['Q0','Q0']]}))
            if COMM.rank==0:
                self.assertIn('P4b',result['measurements']['Q0__Q0'])
            with self.assertRaisesRegex(ValueError,'mixed-mesh'):
                run_stats(config(tmp,normalization_scheme='particle'))
            with self.assertRaisesRegex(ValueError,'auto measurements'):
                run_stats(config(tmp,poles=[0,2,4],high_order_mode='fast'))

    def test_fits_inputs_equal_preprocessed_catalogues(self):
        with tempfile.TemporaryDirectory() as local_tmp:
            tmp=Path(COMM.bcast(local_tmp if COMM.rank==0 else None,root=0))
            columns=['RA','DEC','Z','WEIGHT','WEIGHT_FKP','NZ']
            if COMM.rank==0:
                rng=np.random.default_rng(82)
                for name,n in [('data',300),('randoms',8000)]:
                    table=np.ones(n,dtype=[(key,'f8') for key in columns])
                    table['RA']=rng.uniform(40,50,n)
                    table['DEC']=rng.uniform(-5,5,n)
                    table['Z']=rng.uniform(.45,.50,n)
                    table['NZ']=.001
                    fitsio.write(str(tmp/f'{name}.fits'),table,clobber=True)
            COMM.Barrier()
            cosmology={'h':.67,'Omega0':.31}
            c=config(tmp/'fits',catalogs={'data_a':str(tmp/'data.fits'),'randoms_a':str(tmp/'randoms.fits')},
                     column_names=columns,cosmology=cosmology,z_range=[.44,.51],
                     boxsize=[800.]*3,k_min=.005,k_max=.012,precision='single')
            processed_r,center=catalog_reader(c['catalogs']['randoms_a'],'survey-like',columns,c['z_range'],
                {'scheme':None,'name_alias':None},COMM,para_cosmo=cosmology,boxsize=c['boxsize'],catalog_type='randoms')
            processed_d=catalog_reader(c['catalogs']['data_a'],'survey-like',columns,c['z_range'],
                {'scheme':None,'name_alias':None},COMM,para_cosmo=cosmology,boxsize=c['boxsize'],boxcenter=center,catalog_type='data')
            for statistic in ['pk','bk_sugi']:
                settings={'nquantiles':3,'measurements':[['Q0','data_a']] if statistic=='pk' else [['Q0','data_a','data_a']]}
                disk=run_stats({**c,'statistic':statistic,'angu_config':[0,0,0],'density_split':settings})
                memory=run_stats({**c,'statistic':statistic,'angu_config':[0,0,0],'density_split':settings,
                    'output_dir':str(tmp/'memory'),'catalogs':{'data_a':processed_d,'randoms_a':processed_r}})
                if COMM.rank==0:
                    key='P0' if statistic=='pk' else 'B_sugi'
                    name=next(iter(disk['measurements']))
                    np.testing.assert_array_equal(disk['measurements'][name][key],memory['measurements'][name][key])
