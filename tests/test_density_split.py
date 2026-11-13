"""Independent Fourier and catalogue/field parity checks for density splits."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mpi4py import MPI
from cosmonpc import run_stats
from cosmonpc.density_split import prepare_density_split
from test_density import write_field

COMM = MPI.COMM_WORLD
N, L = 16, 100.
XYZ = np.indices((N,)*3)
SIGNAL = .2*(np.cos(2*np.pi*XYZ[0]/N)+np.cos(2*np.pi*XYZ[1]/N)
             +np.cos(2*np.pi*(XYZ[0]+XYZ[1])/N))
SIGNAL += np.random.default_rng(81).normal(0,.01,SIGNAL.shape)
SIGNAL -= SIGNAL.mean()


def configuration(output, **extra):
    c = dict(statistic='pk', geometry='box-like', input_type='density',
             nmesh=[N]*3, boxsize=[L]*3, k_min=.04, k_max=.19, k_bins=2,
             poles=[0,2], rsd=[0,0,1], precision='double', compensation=False,
             fields={'data_a':SIGNAL}, output_dir=str(output),
             density_split={'nquantiles':5, 'smoothing_radius':7.})
    c.update(extra)
    return c


def power_oracle(a, b, c):
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    grid = np.meshgrid(k,k,k,indexing='ij')
    norm = np.sqrt(sum(x*x for x in grid))
    mu = np.divide(grid[2], norm, out=np.zeros_like(norm), where=norm!=0)
    product = L**3*np.real(np.fft.fftn(a)*np.conj(np.fft.fftn(b)))/N**6
    edges = np.linspace(c['k_min'],c['k_max'],c['k_bins']+1)
    result = {}
    for pole in c['poles']:
        weight = np.polynomial.legendre.legval(mu,[0]*pole+[1])*(2*pole+1)
        result[f'P{pole}'] = np.array([(product*weight)[(norm>=lo)&(norm<hi)].mean()
                                     for lo,hi in zip(edges[:-1],edges[1:])])
    return result


def reference_quantiles(signal, mask, count=5, radius=7.):
    k = 2*np.pi*np.fft.fftfreq(N,d=L/N)
    kernel = np.exp(-.5*radius**2*sum(x*x for x in np.meshgrid(k,k,k,indexing='ij')))
    environment = np.fft.ifftn(np.fft.fftn(signal*mask)*kernel).real
    if not np.all(mask):
        environment /= np.fft.ifftn(np.fft.fftn(mask)*kernel).real
    indices = np.flatnonzero(mask)
    ranked = indices[np.argsort(environment.ravel()[indices], kind='stable')]
    fields = []
    for q in range(count):
        chosen = ranked[len(ranked)*q//count:len(ranked)*(q+1)//count]
        indicator = np.zeros_like(signal)
        indicator.ravel()[chosen] = len(ranked)/len(chosen)
        indicator[mask] -= 1
        fields.append(indicator)
    return fields


class DensitySplitTests(unittest.TestCase):
    def test_power_oracle_and_quantile_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = configuration(tmp)
            original = SIGNAL.copy()
            output = run_stats(c)
            np.testing.assert_array_equal(SIGNAL, original)
            self.assertNotIn('correlation_mode', c)  # caller config is unchanged
            if COMM.rank == 0:
                quantiles = reference_quantiles(SIGNAL, np.ones(SIGNAL.shape,dtype=bool))
                self.assertEqual(len(output['measurements']),10)
                weighted = np.zeros(2)
                for q, fraction in enumerate(output['quantiles']['fractions']):
                    for legs, arrays in [(f'Q{q}__data_a',(quantiles[q],SIGNAL)),
                                          (f'Q{q}__Q{q}',(quantiles[q],quantiles[q]))]:
                        oracle = power_oracle(*arrays,c)
                        for key,value in oracle.items():
                            np.testing.assert_allclose(output['measurements'][legs][key],value,rtol=1e-11,atol=1e-9)
                    weighted += fraction*output['measurements'][f'Q{q}__data_a']['P0']
                np.testing.assert_allclose(weighted,0,atol=1e-9)
                self.assertEqual(output['shotnoise_policy'],'none')
                self.assertTrue((Path(tmp)/'density_split_pk.npy').is_file())
                self.assertNotIn('fields',output)

    def test_catalogue_and_count_field_parity(self):
        ids = np.arange(N**3)
        data = np.empty(N**3,dtype=[('Position','f8',(3,)),('WEIGHT','f8')])
        data['Position'] = np.moveaxis(XYZ,0,-1).reshape(-1,3)*L/N
        data['WEIGHT'] = (1+SIGNAL).ravel()
        local = data[ids%COMM.size == COMM.rank]
        with tempfile.TemporaryDirectory() as tmp:
            for precision in ['single','double']:
                settings = {'nquantiles':3,'smoothing_radius':7.,'measurements':[['Q','data_a']]}
                cat = run_stats(configuration(Path(tmp)/'cat', input_type='catalog',
                    catalogs={'data_a':local}, sampler='ngp', precision=precision,density_split=settings))
                field = run_stats(configuration(Path(tmp)/'field',
                    fields={'data_a':{'array':1+SIGNAL,'kind':'counts'}},precision=precision,density_split=settings))
                if COMM.rank == 0:
                    for name in cat['measurements']:
                        for pole in ['P0','P2']:
                            np.testing.assert_allclose(cat['measurements'][name][pole],field['measurements'][name][pole],rtol=1e-6,atol=1e-8)

    def test_incomplete_tiles_and_explicit_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tile.h5'
            write_field(path,SIGNAL[:8]+.3)
            c = configuration(Path(tmp)/'tile',fields={'data_a':{'tiles':[path],'missing_tiles':'zero'}})
            tracers, labels, info, _, _, _, _ = prepare_density_split(c)
            mesh = tracers['data_a']
            fullmask = np.zeros(SIGNAL.shape,dtype=bool)
            fullmask[:8] = True
            mask = fullmask[mesh.slices]
            np.testing.assert_array_equal(labels[~mask],-1)
            np.testing.assert_array_equal(mesh.value.real[mask],(SIGNAL+.3)[mesh.slices][mask])
            self.assertEqual(info['covered_cells'],N**3//2)
            tile = run_stats(c)
            array = run_stats(configuration(Path(tmp)/'array',fields={'data_a':SIGNAL+.3},
                density_split={'mask':fullmask,'nquantiles':5,'smoothing_radius':7.}))
            if COMM.rank == 0:
                qs = reference_quantiles(SIGNAL+.3,fullmask)
                for name in tile['measurements']:
                    np.testing.assert_allclose(tile['measurements'][name]['P0'],array['measurements'][name]['P0'],rtol=1e-12,atol=1e-9)
                expected = power_oracle(qs[0],(SIGNAL+.3)*fullmask,c)
                np.testing.assert_allclose(tile['measurements']['Q0__data_a']['P0'],expected['P0'],atol=1e-9)

    def test_bispectrum_ordered_legs_and_multipoles(self):
        with tempfile.TemporaryDirectory() as tmp:
            quantiles = reference_quantiles(SIGNAL, np.ones(SIGNAL.shape,dtype=bool),count=3)
            registry = {f'Q{i}':q for i,q in enumerate(quantiles)}
            registry['data_a'] = SIGNAL
            pairs = [['Q0','data_a','data_a'],['data_a','Q0','data_a'],
                     ['data_a','data_a','Q0'],['Q1','Q1','data_a'],
                     ['Q2','Q2','Q2'],['Q0','Q1','data_a']]
            k = 2*np.pi*np.fft.fftfreq(N,d=L/N)
            grid = np.meshgrid(k,k,k,indexing='ij')
            norm = np.sqrt(sum(x*x for x in grid))
            mu = np.divide(grid[2],norm,out=np.zeros_like(norm),where=norm!=0)
            edges = np.linspace(.04,.19,3)
            masks = [(norm>=lo)&(norm<hi) for lo,hi in zip(edges[:-1],edges[1:])]
            for angular in [[0,0,0],[2,0,2]]:
                weight = np.ones_like(norm) if angular == [0,0,0] else .5*(3*mu**2-1)
                amplitude = 1 if angular == [0,0,0] else 5
                for mode in ['full','diagonal']:
                    c = configuration(tmp,statistic='bk_sugi',angu_config=angular,
                        data_vector_mode=mode,density_split={'nquantiles':3,'smoothing_radius':7.,'measurements':pairs})
                    result = run_stats(c)
                    if COMM.rank != 0:
                        continue
                    for legs in pairs:
                        a,b,d = [registry[token] for token in legs]
                        first = [np.fft.ifftn(np.fft.fftn(a)*mask*weight).real for mask in masks]
                        second = [np.fft.ifftn(np.fft.fftn(b)*mask).real for mask in masks]
                        expected = np.array([[amplitude*L**6*np.mean(x*y*d)/(masks[i].sum()*masks[j].sum())
                                               for j,y in enumerate(second)] for i,x in enumerate(first)])
                        if mode == 'diagonal':
                            expected = expected.diagonal()
                        measured = result['measurements']['__'.join(legs)]
                        np.testing.assert_allclose(measured['B_sugi'],expected,rtol=2e-10,atol=2e-6)
                        self.assertEqual(measured['Shot_noise'],0.)
                    self.assertTrue((Path(tmp)/'density_split_bk_sugi.npy').is_file())
            # Repeated calls cannot compound the estimator's in-place scaling.
            first = run_stats(c)
            second = run_stats(c)
            if COMM.rank == 0:
                for key in first['measurements']:
                    np.testing.assert_array_equal(first['measurements'][key]['B_sugi'],second['measurements'][key]['B_sugi'])

    def test_bispectrum_catalogue_parity_and_precision(self):
        ids = np.arange(N**3)
        data = np.empty(N**3,dtype=[('Position','f8',(3,)),('WEIGHT','f8')])
        data['Position'] = np.moveaxis(XYZ,0,-1).reshape(-1,3)*L/N
        data['WEIGHT'] = (1+SIGNAL).ravel()
        with tempfile.TemporaryDirectory() as tmp:
            settings = {'nquantiles':3,'smoothing_radius':7.,
                        'measurements':[['Q','data_a','data_a'],['Q','Q','data_a'],['Q','Q','Q']]}
            outputs = []
            for precision in ['double','single']:
                c = configuration(Path(tmp)/precision,statistic='bk_sugi',angu_config=[0,0,0],
                                  data_vector_mode='full',precision=precision,density_split=settings)
                field = run_stats(c)
                cat = run_stats({**c,'input_type':'catalog','sampler':'ngp',
                                 'catalogs':{'data_a':data[ids%COMM.size == COMM.rank]}})
                if COMM.rank == 0:
                    outputs.append(field)
                    self.assertEqual(len(field['measurements']),9)
                    for key in field['measurements']:
                        np.testing.assert_allclose(cat['measurements'][key]['B_sugi'],field['measurements'][key]['B_sugi'],rtol=2e-5,atol=3.)
                        dtype = np.complex64 if precision == 'single' else np.complex128
                        self.assertEqual(field['measurements'][key]['Bk_raw'].dtype,dtype)
            if COMM.rank == 0:
                for key in outputs[0]['measurements']:
                    expected = outputs[0]['measurements'][key]['B_sugi']
                    np.testing.assert_allclose(outputs[1]['measurements'][key]['B_sugi'],expected,
                                               rtol=2e-5,atol=np.max(abs(expected))*2e-6)

    def test_selected_source_cross_quantiles_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = configuration(tmp,fields={'data_a':SIGNAL,'data_b':-SIGNAL},density_split={
                'source':'data_b','smoothing_radius':0.,'nquantiles':3,
                'measurements':[['Q0','Q2'],['Q','data_a']]})
            result = run_stats(c)
            if COMM.rank == 0:
                self.assertEqual(result['quantiles']['source'],'data_b')
                self.assertEqual(len(result['measurements']),4)
            for options in [{'nquantiles':True},{'smoothing_radius':-1}, {'source':'data_c'},
                            {'measurements':[['Q9','data_a']]},{'measurements':[['data_a','data_a']]},
                            {'unknown':1},{'mask':np.full(SIGNAL.shape,.5)}]:
                with self.assertRaises((ValueError,TypeError)):
                    run_stats(configuration(tmp,density_split=options))
            with self.assertRaisesRegex(ValueError,'[Ss]urvey'):
                run_stats(configuration(tmp,geometry='survey-like'))
