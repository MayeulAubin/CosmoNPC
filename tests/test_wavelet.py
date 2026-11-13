"""Independent NumPy FFT and low-degree Cartesian harmonic WST oracles."""
import unittest
import numpy as np
from pmesh import ParticleMesh
from mpi4py import MPI
from cosmonpc.wavelet import calculate_scattering, scattering_options, scattering_plan

COMM = MPI.COMM_WORLD


def filters(n, sigma, ell):
    k = 2*np.pi*np.fft.fftfreq(n)*sigma
    x,y,z = np.meshgrid(k,k,k,indexing='ij')
    r2 = x*x+y*y+z*z
    g = np.exp(-r2/2)
    if ell == 0:
        return [g]
    if ell == 1:
        return [-1j*np.sqrt(np.pi)/4*(x-1j*y)*g,
                -1j*np.sqrt(np.pi/8)*z*g,
                1j*np.sqrt(np.pi)/4*(x+1j*y)*g]
    return [-(x-1j*y)**2/np.sqrt(24)*g,
            -z*(x-1j*y)/np.sqrt(6)*g,
            -(3*z*z-r2)/6*g,
            z*(x+1j*y)/np.sqrt(6)*g,
            -(x+1j*y)**2/np.sqrt(24)*g]


def oracle(a, settings, mask=None):
    n = a.shape[0]
    if mask is None:
        mask = np.ones_like(a,dtype=bool)
    q = np.atleast_1d(settings['q'])
    mean = lambda u: np.array([(abs(u[mask])**p).mean() for p in q])
    a = a*mask
    s1,s2=[],[]
    def transform(a,j,ell):
        responses=[np.fft.ifftn(np.fft.fftn(a)*f) for f in filters(n,settings['sigma']*2**j,ell)]
        return np.sqrt(sum(abs(r)**2 for r in responses))*mask
    for ell in settings['ells']:
        for ij,j1 in enumerate(settings['scales']):
            u=transform(a,j1,ell)
            s1.append(mean(u))
            if settings['max_order']==2:
                for j2 in settings['scales'][ij+1:]:
                    s2.append(mean(transform(u,j2,ell)))
    return mean(a),np.array(s1),np.array(s2).reshape(-1,len(q))


def mesh(a, precision='double'):
    pm=ParticleMesh(Nmesh=list(a.shape),BoxSize=[80.]*3,
                    dtype='complex128' if precision=='double' else 'complex64',comm=COMM)
    field=pm.create(type='real')
    field[...] = a[field.slices]
    return field


class WaveletCoreTests(unittest.TestCase):
    def test_numpy_cartesian_oracle_both_precisions_and_mask(self):
        a=np.random.default_rng(123).normal(size=(8,)*3)+.2
        options=scattering_options(dict(J=2,L=2,q=[.5,.8,1.,2.],sigma=.8),a.shape)
        mask=np.indices(a.shape)[0]%3!=0
        for precision,tol in [('double',2e-13),('single',3e-6)]:
            f=mesh(a,precision)
            before=f.value.copy()
            for m in [None,mask]:
                result=calculate_scattering(f,options,None if m is None else m[f.slices])
                expected=oracle(a,options,m)
                for key,value in zip(['S0','S1','S2'],expected):
                    np.testing.assert_allclose(result[key],value,rtol=tol,atol=1e-13)
            np.testing.assert_array_equal(f.value,before)

    def test_constant_plane_wave_translation_and_homogeneity(self):
        settings=dict(J=1,L=2,q=[1.,2.],sigma=1.2)
        a=np.ones((8,)*3)*.7
        result=calculate_scattering(mesh(a),settings)
        np.testing.assert_allclose(result['S0'],[.7,.49])
        np.testing.assert_allclose(result['S1'][:2],[result['S0']]*2,atol=1e-15)
        np.testing.assert_allclose(result['S1'][2:],0,atol=1e-15)
        a=np.cos(2*np.pi*np.indices(a.shape)[0]/8)
        base=calculate_scattering(mesh(a),settings)
        moved=calculate_scattering(mesh(np.roll(a,3,axis=0)),settings)
        scaled=calculate_scattering(mesh(-3*a),settings)
        for key in ['S0','S1','S2']:
            np.testing.assert_allclose(base[key],moved[key],atol=1e-14)
            np.testing.assert_allclose(scaled[key],base[key]*[3,9],atol=1e-14)
        # l=0 Gaussian q=2 has an analytic plane-wave response.
        self.assertAlmostEqual(base['S1'][0,1],.5*np.exp(-(1.2*2*np.pi/8)**2),places=13)

    def test_options_and_plan(self):
        opts=scattering_options(dict(J=4,L=4),[64]*3)
        plan=scattering_plan(opts)
        self.assertEqual(plan['coefficients_per_q'],76)
        self.assertEqual(plan['forward_ffts'],21)
        self.assertEqual(plan['inverse_ffts'],375)
        for bad in [dict(J=7),dict(L=-1),dict(q=[0]),dict(q=[True]),dict(q=[1,1]),dict(scales=[1,0]),dict(sigma=0),dict(max_order=3),dict(foo=1)]:
            with self.assertRaises(ValueError):
                scattering_options(bad,[8]*3)
        with self.assertRaises(ValueError):
            calculate_scattering(mesh(np.ones((8,)*3)),dict(J=0,L=0),np.zeros((8,)*3))


if __name__=='__main__':
    unittest.main()
