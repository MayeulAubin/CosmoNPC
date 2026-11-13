"""Public WST entry point, survey masks, HDF5 tiles and catalogue equivalence."""
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from mpi4py import MPI
from cosmonpc import run_stats
from cosmonpc.wavelet import scattering_options
from test_wavelet import oracle

COMM = MPI.COMM_WORLD
N,L = 8,80.
GRID = np.indices((N,)*3)
DELTA = .2+.13*np.cos(2*np.pi*GRID[0]/N)+.09*np.sin(2*np.pi*GRID[1]/N)
WINDOW = (GRID[0]+1)/N*(GRID[2]%3!=0)
OPTIONS = dict(J=1,L=2,q=[.8,1.,2.],sigma=.8)


def config(tmp, **extra):
    c = dict(statistic='wst',input_type='density',geometry='box-like',
             fields={'data_a':DELTA},nmesh=[N]*3,boxsize=[L]*3,
             wst=OPTIONS,output_dir=str(tmp),sampler='ngp')
    c.update(extra)
    return c


def catalogue(weights, centers=False):
    positions = np.moveaxis(GRID,0,-1).reshape(-1,3)*L/N
    if centers:
        positions += .5*L/N
    rows = np.arange(len(positions))[COMM.rank::COMM.size]
    a = np.empty(len(rows),dtype=[('Position','f8',3),('WEIGHT','f8'),('WEIGHT_FKP','f8'),('NZ','f8')])
    a['Position'] = positions[rows]
    a['WEIGHT'] = np.asarray(weights).ravel()[rows]
    a['WEIGHT_FKP'] = 1.
    a['NZ'] = .001
    return a


def write(path, values, corner):
    with h5py.File(path,'w') as f:
        f.create_dataset('/scalars/field',data=values)
        for i in range(3):
            f.attrs[f'/info/scalars/N{i}'] = values.shape[i]
            f.attrs[f'/info/scalars/L{i}'] = values.shape[i]*L/N
            f.attrs[f'/info/scalars/corner{i}'] = corner[i]
        f.attrs['/info/scalars/rank'] = 1


class WaveletInputTests(unittest.TestCase):
    def assert_coefficients(self, result, expected, tol=2e-12):
        if COMM.rank==0:
            measured = result['measurements']['data_a']
            for key,value in zip(['S0','S1','S2'],expected):
                np.testing.assert_allclose(measured[key],value,rtol=tol,atol=1e-13)

    def test_array_counts_catalogue_and_independent_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=scattering_options(OPTIONS,[N]*3)
            self.assert_coefficients(run_stats(config(tmp)),oracle(DELTA,settings))
            counts=1+DELTA
            normalized=counts/counts.mean()-1
            expected=oracle(normalized,settings)
            self.assert_coefficients(run_stats(config(tmp,fields={'data_a':{'array':counts,'kind':'counts'}})),expected)
            for precision,tol in [('double',2e-12),('single',4e-6)]:
                c=config(tmp,input_type='catalog',catalogs={'data_a':catalogue(counts)},precision=precision)
                self.assert_coefficients(run_stats(c),expected,tol)
                # All existing assignment/interlacing options also traverse WST.
                r=run_stats({**c,'sampler':'tsc','interlaced':True,'compensation':True})
                if COMM.rank==0:
                    self.assertTrue(np.isfinite(r['measurements']['data_a']['S2']).all())
            r=run_stats(config(tmp,fields={'data_a':DELTA,'data_b':2*DELTA},wst={**OPTIONS,'sources':['data_a','data_b']}))
            if COMM.rank==0:
                for key in ['S0','S1','S2']:
                    np.testing.assert_allclose(r['measurements']['data_b'][key],r['measurements']['data_a'][key]*2**np.array(OPTIONS['q']),rtol=1e-12)
                self.assertTrue((Path(tmp)/'wst.npy').exists())

    def test_hdf5_full_and_missing_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            full,tile=tmp/'full.h5',tmp/'tile.h5'
            write(full,DELTA,[0.,0.,0.])
            write(tile,DELTA[:4],[0.,0.,0.])
            settings=scattering_options(OPTIONS,[N]*3)
            self.assert_coefficients(run_stats(config(tmp,fields={'data_a':str(full)})),oracle(DELTA,settings))
            mask=GRID[0]<4
            descriptor={'tiles':[str(tile),str(tmp/'absent.h5')],'missing_tiles':'zero'}
            for geometry in ['box-like','survey-like']:
                extra={} if geometry=='box-like' else {'survey_weight':WINDOW}
                result=run_stats(config(tmp,fields={'data_a':descriptor},geometry=geometry,**extra))
                active=mask if geometry=='box-like' else mask&(WINDOW>0)
                field=DELTA if geometry=='box-like' else DELTA*WINDOW/np.sqrt((L/N)**3*np.sum((WINDOW*active)**2))
                self.assert_coefficients(result,oracle(field,settings,active))
                if COMM.rank==0:
                    self.assertEqual(result['measurements']['data_a']['covered_cells'],int(active.sum()))

    def test_survey_array_hdf5_and_random_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            maskfile=Path(tmp)/'mask.h5'
            write(maskfile,WINDOW,[0.,0.,0.])
            active=WINDOW>0
            norm=(L/N)**3*np.sum(WINDOW**2)
            settings=scattering_options(OPTIONS,[N]*3)
            expected=oracle(DELTA*WINDOW/np.sqrt(norm),settings,active)
            for selection in [dict(survey_weight=WINDOW),dict(survey_weight=str(maskfile)),
                              dict(survey_randoms=catalogue(WINDOW,centers=True)),dict(survey_weight=WINDOW*.3)]:
                c=config(tmp,geometry='survey-like',**selection)
                self.assert_coefficients(run_stats(c),expected)
            mask=active&(GRID[0]<6)
            c=config(tmp,geometry='survey-like',survey_weight=WINDOW,wst={**OPTIONS,'mask':mask})
            norm=(L/N)**3*np.sum((WINDOW*mask)**2)
            self.assert_coefficients(run_stats(c),oracle(DELTA*WINDOW/np.sqrt(norm),settings,mask))

    def test_survey_catalogue_fkp_and_footprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Lattice NGP catalogue: exact FKP normalization known independently.
            mask=GRID[0]<6
            random_weights=mask.astype(float)
            data_weights=(1+DELTA)*mask
            alpha=data_weights.sum()/random_weights.sum()
            i2=alpha*.001*random_weights.sum()
            field=(data_weights-alpha*random_weights)/(L/N)**3/np.sqrt(i2)
            data,randoms=catalogue(data_weights),catalogue(random_weights)
            for cat in [data,randoms]:
                cat['Position'][cat['Position']>=L/2] -= L
            c=config(tmp,input_type='catalog',geometry='survey-like',boxcenter=[0.,0.,0.],
                     catalogs={'data_a':data,'randoms_a':randoms})
            expected=oracle(field,scattering_options(OPTIONS,[N]*3),mask)
            self.assert_coefficients(run_stats(c),expected)
            self.assert_coefficients(run_stats({**c,'wst':{**OPTIONS,'mask':mask}}),expected)
            for scheme in ['particle','mixed-mesh']:
                r=run_stats({**c,'normalization_scheme':scheme,'precision':'single','sampler':'tsc','interlaced':True,'compensation':True,'alpha_scheme':'nbodykit'})
                if COMM.rank==0:
                    self.assertTrue(np.isfinite(r['measurements']['data_a']['S1']).all())

    def test_bad_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in [dict(correlation_mode='cross'),dict(wst={'sources':[]}),
                          dict(density_split={}),dict(geometry='unknown'),
                          dict(geometry='survey-like'),dict(wst={'mask':np.ones((4,)*3)}),
                          dict(wst={'mask':np.ones((N,)*3)*.5})]:
                with self.assertRaises((ValueError,AssertionError)):
                    run_stats(config(tmp,**extra))

    def test_root_output_failure_is_collective(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'file_not_directory'
            file.write_text('existing file')
            with self.assertRaisesRegex(ValueError,'Cannot save WST output'):
                run_stats(config(tmp,output_dir=str(file),wst={'J':0,'L':0}))


if __name__=='__main__':
    unittest.main()
