"""Numerical regression tests; run with unittest, optionally under mpiexec."""
import tempfile
import unittest
from pathlib import Path
import logging
import numpy as np
import h5py
from mpi4py import MPI
from cosmonpc.density_processor import read_local_field, get_density_meshes
from cosmonpc.mesh_generator import pm_painter
from cosmonpc.clustering_estimator import calculate_power_spectrum_box, calculate_bk_sugi_box
from cosmonpc.param_helper import validate_config
from cosmonpc import run_stats

COMM = MPI.COMM_WORLD
logging.disable(logging.CRITICAL)
N, L = 16, 100.
GRID = np.indices((N,)*3)
DELTA = .2 * (np.cos(2*np.pi*GRID[0]/N) + np.cos(2*np.pi*GRID[1]/N)
               + np.cos(2*np.pi*(GRID[0]+GRID[1])/N))


def write_field(path, data, corner=(0., 0., 0.), lengths=None, time=1.):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('/scalars/field', data=data)
        for i in range(3):
            hf.attrs[f'/info/scalars/N{i}'] = data.shape[i]
            hf.attrs[f'/info/scalars/L{i}'] = lengths[i] if lengths else data.shape[i]*L/N
            hf.attrs[f'/info/scalars/corner{i}'] = corner[i]
        hf.attrs['/info/scalars/rank'] = 1
        hf.attrs['/info/scalars/time'] = time


def config(**extra):
    c = dict(input_type='density', statistic='pk', geometry='box-like',
             correlation_mode='auto', nmesh=[N]*3, boxsize=[L]*3,
             fields={'data_a': DELTA}, poles=[0, 2], k_min=.04, k_max=.19,
             k_bins=2, rsd=[0, 0, 1], precision='double')
    c.update(extra)
    validate_config(c)
    return c


def measure(c):
    meshes = get_density_meshes(c['fields'], c['correlation_mode'], c['nmesh'], c['boxsize'],
                               COMM, c['statistic'], tracer_type=c.get('tracer_type','ab'),
                               precision=c['precision'])
    attrs = {**meshes[0], **{k:v for k,v in c.items() if k != 'fields'}}
    if c['statistic'] == 'pk':
        return calculate_power_spectrum_box(*meshes[1:], c['correlation_mode'], attrs, COMM)
    return calculate_bk_sugi_box(*meshes[1:], c['correlation_mode'], attrs, COMM)


class DensityTests(unittest.TestCase):
    def test_fourier_power_and_cross(self):
        c = config(poles=[0])
        result = measure(c)
        if COMM.rank == 0:
            k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
            norm = np.sqrt(sum(x*x for x in np.meshgrid(k,k,k,indexing='ij')))
            edges = np.linspace(c['k_min'], c['k_max'], c['k_bins']+1)
            power = L**3 * abs(np.fft.fftn(DELTA)/N**3)**2
            expected = [power[(norm>=a)&(norm<b)].mean() for a,b in zip(edges[:-1],edges[1:])]
            np.testing.assert_allclose(result['P0'], expected, atol=1e-10, rtol=1e-12)
            self.assertEqual(result['P_shot'], 0.)
        cross = measure(config(poles=[0], correlation_mode='cross', fields={'data_a': DELTA, 'data_b': 2*DELTA}))
        if COMM.rank == 0:
            np.testing.assert_allclose(cross['P0'], 2*result['P0'], atol=1e-10)

    def test_precision_and_catalog_painting(self):
        for precision, dtype in [('single', np.complex64), ('double', np.complex128)]:
            mesh = get_density_meshes({'data_a': DELTA}, 'auto', [N]*3, [L]*3,
                                     COMM, 'pk', precision=precision)[1]
            self.assertEqual(mesh.dtype, dtype)
            self.assertEqual(mesh.r2c().dtype, dtype)
            rng = np.random.default_rng(11 + COMM.rank)
            for interlaced in [False, True]:
                painted = pm_painter(rng.uniform(0,L,(30,3)), np.ones(30), [N]*3,
                                     [L]*3, 'cic', interlaced, COMM, precision=precision)
                self.assertEqual(painted.dtype, dtype)
        single, double = measure(config(precision='single')), measure(config())
        if COMM.rank == 0:
            np.testing.assert_allclose(single['P0'], double['P0'], rtol=2e-6, atol=1e-7)

    def test_hdf5_tiles_and_normalization(self):
        # Each rank owns its temporary files, exercising independent serial HDF5 reads.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rho = 3*(1+DELTA)
            write_field(tmp/'full.h5', rho)
            paths = []
            for i in range(2):
                path = tmp/f'tile_{i}.h5'
                write_field(path, rho[i*8:(i+1)*8], corner=(i*L/2,0,0))
                paths.append(path)
            reference = get_density_meshes({'data_a': DELTA}, 'auto', [N]*3, [L]*3, COMM, 'pk')[1]
            for source in [ {'path': tmp/'full.h5', 'kind': 'density'}, {'tiles': paths[::-1]}, {'array':rho,'kind':'density'} ]:
                mesh = get_density_meshes({'data_a':source}, 'auto', [N]*3, [L]*3, COMM, 'pk')[1]
                np.testing.assert_allclose(mesh.value, reference.value, atol=1e-14)
            # Ensure a tile subset cannot masquerade as a full periodic box.
            with self.assertRaisesRegex(ValueError, 'cover'):
                get_density_meshes({'data_a': {'tiles':paths[:1]}}, 'auto', [N]*3, [L]*3, COMM, 'pk')
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                get_density_meshes({'data_a': {'tiles':paths+paths}}, 'auto', [N]*3, [L]*3, COMM, 'pk')

    def test_incomplete_tiles_preserve_zero_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tile.h5'
            absent = Path(tmp)/'missing.h5'
            for kind in ['density', 'delta']:
                data = 4*(1+DELTA[:8]) if kind == 'density' else DELTA[:8]
                write_field(path, data)
                expected = np.zeros_like(DELTA)
                present = data/data.mean()-1 if kind == 'density' else data.copy()
                present -= present.mean()
                expected[:8] = present
                for precision, tol in [('double', 1e-14), ('single', 2e-7)]:
                    source = {'tiles':[path, absent], 'kind':kind, 'missing_tiles':'zero'}
                    attrs, mesh, _ = get_density_meshes({'data_a':source}, 'auto', [N]*3,
                                                       [L]*3, COMM, 'pk', precision=precision)
                    np.testing.assert_allclose(mesh.value.real, expected[mesh.slices], atol=tol)
                    np.testing.assert_array_equal(mesh.value.imag, 0)
                    missing = np.indices(mesh.shape)[0] + mesh.start[0] >= 8
                    np.testing.assert_array_equal(mesh.value[missing], 0)
                    self.assertEqual(attrs['field_metadata']['a']['coverage_fraction'], .5)
                    # The actual estimator must agree with an explicitly masked array.
                    measured = measure(config(fields={'data_a':source}, precision=precision))
                    oracle = measure(config(fields={'data_a':expected}, precision=precision))
                    if COMM.rank == 0:
                        np.testing.assert_allclose(measured['P0'], oracle['P0'], rtol=2e-6, atol=1e-9)
                # Explicit reference density controls scaling without filling the mask.
                if kind == 'density':
                    source = {'tiles':[path], 'missing_tiles':'zero', 'mean_density':8.}
                    attrs, mesh, _ = get_density_meshes({'data_a':source}, 'auto', [N]*3, [L]*3, COMM, 'pk')
                    expected[:8] = (data-data.mean())/8.
                    np.testing.assert_allclose(mesh.value.real, expected[mesh.slices], atol=1e-14)
                    self.assertEqual(attrs['field_metadata']['a']['mean_density'], 8.)
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                get_density_meshes({'data_a':{'tiles':[path,path], 'missing_tiles':'zero'}},
                                   'auto', [N]*3, [L]*3, COMM, 'pk')
            # Empty selections, including ranks wholly outside the mask, stay zero.
            for paths in [[], [absent]]:
                attrs, mesh, _ = get_density_meshes({'data_a':{'tiles':paths, 'missing_tiles':'zero'}},
                                                   'auto', [N]*3, [L]*3, COMM, 'pk')
                np.testing.assert_array_equal(mesh.value, 0)
                self.assertEqual(attrs['field_metadata']['a']['coverage_fraction'], 0.)
            for bad in [{'tiles':[path], 'missing_tiles':'ignore'},
                        {'array':DELTA, 'missing_tiles':'zero'},
                        {'tiles':[absent]}]:
                with self.assertRaises(ValueError):
                    get_density_meshes({'data_a':bad}, 'auto', [N]*3, [L]*3, COMM, 'pk')

    def test_invalid_inputs(self):
        slices = (slice(0,N),)*3
        for data in [np.full((N,)*3, np.nan), np.ones((N,)*3, dtype=complex), np.ones((3,3,3))]:
            with self.assertRaises((ValueError, TypeError)):
                read_local_field(data, [N]*3, [L]*3, [0]*3, slices, np.float64)
        with self.assertRaises(ValueError):
            config(precision='half')
        with self.assertRaises(ValueError):
            config(geometry='survey-like')
        with self.assertRaises(ValueError):
            config(apply_rsd=True)

    def test_bispectrum_modes_and_tracers(self):
        for mode in ['diagonal', 'full']:
            baseline = None
            for tracer in ['aaa', 'aab', 'aba', 'abb', 'abc']:
                c = config(statistic='bk_sugi', data_vector_mode=mode, tracer_type=tracer,
                           correlation_mode='auto' if tracer=='aaa' else 'cross',
                           angu_config=[0,0,0], k_max=.19, shotnoise_mode='both',
                           fields={'data_a': DELTA, 'data_b':DELTA, 'data_c':DELTA})
                result = measure(c)
                self.assertEqual(result['Shot_noise'], 0.)
                self.assertEqual(result['SN_terms']['SN3_FFT'], 0.)
                if baseline is None:
                    baseline = result['B_sugi']
                np.testing.assert_allclose(result['B_sugi'], baseline, atol=1e-9)
                c['precision'] = 'single'
                single = measure(c)
                self.assertEqual(single['Bk_raw'].dtype, np.complex64)
                # Near-zero bins contain FFT cancellation residuals; compare
                # against the signal scale, not their tiny double-precision value.
                np.testing.assert_allclose(single['B_sugi'], baseline, rtol=5e-6,
                                           atol=np.max(abs(baseline))*1e-7)
            zero = measure(config(statistic='bk_sugi', fields={'data_a': np.zeros((N,)*3)},
                                  angu_config=[0,0,0], tracer_type='aaa', data_vector_mode=mode))
            np.testing.assert_allclose(zero['B_sugi'], 0, atol=1e-20)

    def test_direct_bispectrum_and_small_amplitude(self):
        c = config(statistic='bk_sugi', angu_config=[0,0,0], tracer_type='aaa', data_vector_mode='full')
        result = measure(c)
        k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
        norm = np.sqrt(sum(x*x for x in np.meshgrid(k,k,k,indexing='ij')))
        edges = np.linspace(c['k_min'], c['k_max'], c['k_bins']+1)
        masks = [(norm>=a)&(norm<b) for a,b in zip(edges[:-1],edges[1:])]
        filtered = [np.fft.ifftn(np.fft.fftn(DELTA)*mask).real for mask in masks]
        expected = np.array([[L**6*np.mean(a*b*DELTA)/(masks[i].sum()*masks[j].sum())
                              for j,b in enumerate(filtered)] for i,a in enumerate(filtered)])
        np.testing.assert_allclose(result['B_sugi'], expected, rtol=1e-12, atol=1e-8)
        # Linear initial-condition delta may be much smaller than float32 epsilon.
        double = measure(config(poles=[0], fields={'data_a': DELTA*1e-8}))
        single = measure(config(poles=[0], fields={'data_a': DELTA*1e-8}, precision='single'))
        if COMM.rank == 0:
            np.testing.assert_allclose(single['P0'], double['P0'], rtol=1e-6, atol=1e-26)

    def test_catalogue_entrypoints_in_both_precisions(self):
        from copy import deepcopy
        from cosmonpc.config import pk_box, pk_survey, bk_sugi_box, bk_sugi_survey
        rng = np.random.default_rng(123 + COMM.rank)
        dtype = [('Position','f8',(3,)), ('WEIGHT','f8'), ('WEIGHT_FKP','f8'), ('NZ','f8')]
        data, randoms = np.ones(150,dtype=dtype), np.ones(500,dtype=dtype)
        data['Position'] = rng.uniform(20,80,(150,3))
        randoms['Position'] = rng.uniform(0,100,(500,3))
        data['NZ'] = randoms['NZ'] = .001
        with tempfile.TemporaryDirectory() as tmp:
            for name, module in [('pk_box',pk_box), ('bk_box',bk_sugi_box),
                                 ('pk_survey',pk_survey), ('bk_survey',bk_sugi_survey)]:
                results = []
                for precision in ['double', 'single']:
                    c = deepcopy(module.CONFIG)
                    output = Path(tmp)/name/precision
                    c.update(nmesh=[N]*3, boxsize=[L]*3, k_min=.04, k_max=.19, k_bins=2,
                             catalogs={'data_a':data, 'randoms_a':randoms}, apply_rsd=False,
                             output_dir=str(output), precision=precision, poles=[0,2],
                             angu_config=[0,0,0], interlaced=False, normalization_scheme='particle')
                    run_stats(c)
                    if COMM.rank == 0:
                        results.append(np.load(next(output.glob('*.npy')), allow_pickle=True).item())
                if COMM.rank == 0:
                    key = 'P0' if name.startswith('pk') else 'B_sugi'
                    np.testing.assert_allclose(results[1][key], results[0][key], rtol=1e-5)
                    if key == 'B_sugi':
                        self.assertEqual(results[1][key].dtype, np.complex64)

    def test_public_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            for statistic in ['pk', 'bk_sugi']:
                c = config(output_dir=tmp, statistic=statistic, angu_config=[0,0,0], tracer_type='aaa')
                run_stats(c)
                if COMM.rank == 0:
                    files = list(Path(tmp).glob(f'{statistic}*.npy'))
                    self.assertEqual(len(files), 1)
                    result = np.load(files[0], allow_pickle=True).item()
                    self.assertEqual(result['input_type'], 'density')
                    self.assertNotIn('fields', result)


if __name__ == '__main__':
    unittest.main()
