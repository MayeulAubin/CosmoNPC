import unittest
import numpy as np
from mpi4py import MPI
from pmesh import ParticleMesh
from cosmonpc.quantiles import (assign_quantiles, global_cell_ids, gaussian_smooth,
                                quantile_field, _permuted_ids)

COMM = MPI.COMM_WORLD


class QuantileTests(unittest.TestCase):
    def mesh(self, precision='complex128'):
        return ParticleMesh(Nmesh=[8]*3, BoxSize=[80.]*3, dtype=precision, comm=COMM).create(type='real')

    def test_gaussian_against_numpy_and_input_preservation(self):
        for precision, tol in [('complex128', 1e-14), ('complex64', 2e-6)]:
            mesh = self.mesh(precision)
            ids = global_cell_ids(mesh.shape, mesh.start, mesh.pm.Nmesh)
            full = np.cos(2*np.pi*np.indices((8,)*3)[0]/8) + .3
            mesh[...] = full[mesh.slices]
            original = mesh.value.copy()
            result = gaussian_smooth(mesh, 7.)
            expected = .3 + (full-.3)*np.exp(-.5*(2*np.pi/80*7)**2)
            np.testing.assert_allclose(result, expected[mesh.slices], atol=tol)
            np.testing.assert_array_equal(mesh.value, original)
            mask = ids % 3 != 0
            smooth = gaussian_smooth(mesh, 7., mask)
            fullmask = np.arange(512).reshape((8,)*3) % 3 != 0
            k = 2*np.pi*np.fft.fftfreq(8, d=10.)
            kernel = np.exp(-.5*49*sum(x*x for x in np.meshgrid(k,k,k,indexing='ij')))
            expected = np.fft.ifftn(np.fft.fftn(full*fullmask)*kernel).real / np.fft.ifftn(np.fft.fftn(fullmask)*kernel).real
            expected[~fullmask] = 0
            np.testing.assert_allclose(smooth, expected[mesh.slices], atol=tol)

    def test_exact_ranks_ties_mask_and_partition_identity(self):
        mesh = self.mesh()
        ids = global_cell_ids(mesh.shape, mesh.start, mesh.pm.Nmesh)
        values = (ids % 7).astype(float)
        mask = ids % 5 != 0
        labels, meta = assign_quantiles(values, mask, ids, 5, COMM, tie_seed=42)
        allids = np.arange(512,dtype=np.uint64)
        chosen = allids[allids%5 != 0]
        order = np.lexsort((_permuted_ids(chosen, 42), chosen%7))
        expected = np.full(512, -1)
        for q in range(5):
            expected[chosen[order[len(chosen)*q//5:len(chosen)*(q+1)//5]].astype(int)] = q
        np.testing.assert_array_equal(labels, expected.reshape((8,)*3)[mesh.slices])
        self.assertLessEqual(np.ptp(meta['counts']), 1)
        accumulator = np.zeros(mesh.shape)
        for q, fraction in enumerate(meta['fractions']):
            field = quantile_field(labels, q, fraction, mesh)
            total = COMM.allreduce(float(field.value.real.sum()),op=MPI.SUM)
            self.assertAlmostEqual(total, 0., places=10)
            np.testing.assert_array_equal(field.value[~mask], 0)
            accumulator += fraction * field.value.real
        np.testing.assert_allclose(accumulator, 0, atol=3e-16)
        again, _ = assign_quantiles(values, mask, ids, 5, COMM, tie_seed=42)
        np.testing.assert_array_equal(labels, again)

    def test_decomposition_independent_ids(self):
        full = global_cell_ids((8,8,8),(0,0,0),(8,8,8))
        split = np.concatenate([global_cell_ids((4,8,8),(i,0,0),(8,8,8)) for i in [0,4]])
        np.testing.assert_array_equal(full, split)
        np.testing.assert_array_equal(_permuted_ids(full,0),_permuted_ids(split,0))
        self.assertEqual(len(np.unique(_permuted_ids(full,0))),512)

    def test_invalid_and_degenerate_environments(self):
        mesh=self.mesh()
        ids=global_cell_ids(mesh.shape,mesh.start,mesh.pm.Nmesh)
        mask=np.ones(mesh.shape,dtype=bool)
        for values, selected, count in [(np.ones(mesh.shape),mask,5),
                                         (ids.astype(float),np.zeros(mesh.shape,dtype=bool),5),
                                         (ids.astype(float),mask,1),
                                         (np.full(mesh.shape,np.nan),mask,5)]:
            with self.assertRaises(ValueError):
                assign_quantiles(values,selected,ids,count,COMM)
