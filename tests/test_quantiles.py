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

    def test_weighted_quantile_cdf_and_identity(self):
        mesh=self.mesh()
        ids=global_cell_ids(mesh.shape,mesh.start,[8]*3)
        values=(ids%11).astype(float)
        weights=(.2+.8*(ids%17)/16).astype(float)
        mask=ids%5!=0
        labels,info=assign_quantiles(values,mask,ids,3,COMM,tie_seed=12,weights=weights)
        chosen=np.arange(8**3,dtype=np.uint64)
        chosen=chosen[chosen%5!=0]
        order=np.lexsort((_permuted_ids(chosen,12),chosen%11))
        ranked=chosen[order]
        w=.2+.8*(ranked%17)/16
        cumulative=np.cumsum(w)
        cuts=[0]+[np.searchsorted(cumulative,cumulative[-1]*q/3,side='left')+1 for q in [1,2]]+[len(ranked)]
        expected=np.full(8**3,-1)
        for q in range(3):
            expected[ranked[cuts[q]:cuts[q+1]].astype(int)]=q
        np.testing.assert_array_equal(labels,expected.reshape((8,)*3)[mesh.slices])
        identity=np.zeros(mesh.shape)
        for q,fraction in enumerate(info['fractions']):
            delta=np.where(mask,(labels==q)/fraction-1.,0.)
            self.assertAlmostEqual(COMM.allreduce(float(np.sum(weights*delta)),op=MPI.SUM),0.,places=10)
            identity+=fraction*delta
        np.testing.assert_allclose(identity,0,atol=3e-16)
        self.assertLessEqual(np.ptp(info['query_weight_sums']),2*info['maximum_cell_weight'])

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
