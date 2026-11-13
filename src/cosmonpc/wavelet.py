"""Streaming distributed solid-harmonic scattering (no global mesh gathers).

Conventions: Kymatio solid-harmonic Fourier normalization; spatial *means*
instead of sums; same ell at both layers, j2 > j1. Full complex FFTs retain
the complex wavelet responses, including the Nyquist planes.
"""
import time
import math
from functools import lru_cache
import numpy as np
from mpi4py import MPI


def scattering_options(options, nmesh):
    options = dict(options)
    unknown = options.keys() - {'J', 'L', 'sigma', 'q', 'max_order', 'scales', 'ells', 'mask', 'm_resolved'}
    if unknown:
        raise ValueError(f'Unknown wst options: {sorted(unknown)}')
    for key, default in [('J', min(4, int(np.log2(min(nmesh))))), ('L', 4), ('max_order', 2)]:
        value = options.setdefault(key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f'wst.{key} must be a nonnegative integer')
    if options['max_order'] not in (1, 2):
        raise ValueError('wst.max_order must be 1 or 2')
    if options['J'] > int(np.log2(min(nmesh))):
        raise ValueError('wst.J must not exceed floor(log2(min(nmesh)))')
    for key, upper in [('scales', options['J']), ('ells', options['L'])]:
        values = options.setdefault(key, list(range(upper + 1)))
        if (not isinstance(values, (list, tuple)) or not values or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v <= upper for v in values) or
            list(values) != sorted(set(values))):
            raise ValueError(f'wst.{key} must be sorted distinct integers between 0 and {upper}')
        options[key] = list(values)
    sigma = options.setdefault('sigma', .8)
    if isinstance(sigma, (bool, np.bool_)) or not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('wst.sigma must be finite and positive (in grid cells)')
    raw = options.setdefault('q', [.8])
    q = np.atleast_1d(raw)
    if q.ndim != 1 or q.size == 0 or q.dtype.kind not in 'ifu' or np.any(~np.isfinite(q)) or np.any(q <= 0) or len(set(q)) != len(q):
        raise ValueError('wst.q must contain distinct finite positive powers')
    options['q'] = q.astype(float).tolist()
    if not isinstance(options.setdefault('m_resolved', False), (bool, np.bool_)):
        raise ValueError('wst.m_resolved must be boolean')
    return options


@lru_cache(maxsize=256)
def _harmonic_constants(ell, m):
    # Regular solid harmonics by normalized Cartesian recurrence. Cache scalars
    # only, never arrays depending on box size, decomposition, or scale.
    a = abs(m)
    base = (-1)**a * math.exp(.5*(math.log((2*a+1)/(4*np.pi))+math.lgamma(2*a+1))
                              - a*math.log(2)-math.lgamma(a+1))
    steps = [(degree, math.sqrt((4*degree**2-1)/(degree**2-a*a)),
              math.sqrt((2*degree+1)*((degree-1)**2-a*a)/((2*degree-3)*(degree**2-a*a))))
             for degree in range(a+2,ell+1)]
    if ell % 2 == 0:
        norm = 1/(2*np.pi*math.sqrt(ell+.5)*math.prod(range(ell+1,0,-2)))
    else:
        norm = 1/(2**((ell+3)/2)*math.sqrt(np.pi*(2*ell+1))*math.factorial((ell+1)//2))
    return base, steps, (2*np.pi)**1.5 * (-1j)**ell * norm


def solid_harmonic(k, cellsize, sigma, ell, m):
    """One slab of the Fourier multiplier, evaluated in float64/complex128.

    No grid-sized filter bank or radius/angle cache. The caller casts into the
    chosen mesh precision. sigma is in cells, k is in inverse physical units.
    """
    x, y, z = [np.asarray(axis.real, dtype=np.float64) * cellsize[i] * sigma for i, axis in enumerate(k)]
    r2 = x*x + y*y + z*z
    gaussian = np.exp(-.5*r2)
    if ell == 0:
        return gaussian
    base, steps, norm = _harmonic_constants(ell,m)
    a = abs(m)
    previous = base*(x+1j*y)**a
    if ell == a:
        harmonic = previous
    else:
        harmonic = math.sqrt(2*a+3)*z*previous
        for _, c1, c2 in steps:
            previous, harmonic = harmonic, c1*z*harmonic-c2*r2*previous
    if m < 0:
        harmonic = (-1)**a*np.conj(harmonic)
    return harmonic * gaussian * norm


def scattering_plan(options):
    """Exact FFT counts for the streaming traversal, excluding input painting."""
    s, e = len(options['scales']), len(options['ells'])
    pairs = s*(s-1)//2 if options['max_order'] == 2 else 0
    modes = sum(l+1 for l in options['ells']) if options['m_resolved'] else 0
    return {'forward_ffts': 1 + ((e+modes)*(s-1) if pairs else 0),
            'inverse_ffts': (sum(2*l+1 for l in options['ells'])+modes)*(s+pairs),
            'coefficients_per_q': 1 + e*(s+pairs),
            'm_coefficients_per_q': modes*(s+pairs),
            'filter_cache': 'none', 'mesh_gathers': 0,
            'algorithm': 'full_resolution_streaming', 'decimation': False}


def calculate_scattering(field, options, mask=None):
    """Return coefficients on all ranks; mask is local, binary, reapplied per layer.

    The input is never modified. Only scalar/small-vector reductions communicate
    coefficients; all transforms use the supplied pmesh communicator/decomposition.
    """
    comm = field.pm.comm
    options = scattering_options(options, field.pm.Nmesh)
    if np.dtype(field.dtype).kind != 'c':
        raise ValueError('WST requires a full complex pmesh (complex64 or complex128)')
    error = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != field.shape or np.any((mask != 0) & (mask != 1)):
            error = 'WST mask must be a local binary array with the mesh shape'
        else:
            mask = mask.astype(bool)
    if np.any(~np.isfinite(field.value)):
        error = 'WST input contains nonfinite values'
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError(next(e for e in errors if e))
    count = comm.allreduce(field.size if mask is None else int(mask.sum()), op=MPI.SUM)
    if not count:
        raise ValueError('WST mask has no covered cells')
    started = time.perf_counter()
    q = np.asarray(options['q'])

    def moments(values):
        # Loop through slabs: temporary float64 powers occupy at most a plane.
        local = np.zeros(len(q), dtype=np.float64)
        for i in range(values.shape[0]):
            a = np.abs(values[i]) if mask is None else np.abs(values[i][mask[i]])
            a = a.astype(np.float64, copy=False)
            for iq, power in enumerate(q):
                local[iq] += np.sum(a**power, dtype=np.float64)
        total = np.empty_like(local)
        comm.Allreduce(local, total, op=MPI.SUM)
        if np.any(~np.isfinite(total)):
            raise ValueError('Nonfinite WST moments: reduce input amplitude or q')
        return total/count

    s0 = moments(field.value)
    work = field.copy()
    if mask is not None:
        work.value[~mask] = 0
    original = work.r2c()
    del work
    buffer = original.copy()
    response = field.pm.create(type='real')
    cellsize = field.pm.BoxSize/field.pm.Nmesh
    accumulator = np.empty(field.shape, dtype=field.value.real.dtype)

    def component(fourier, j, ell, m):
        for k, source, target in zip(fourier.slabs.x, fourier.slabs, buffer.slabs):
            np.multiply(source, solid_harmonic(k, cellsize, options['sigma']*2**j, ell, m), out=target)
        buffer.c2r(out=response)

    def modulus(fourier, j, ell):
        accumulator.fill(0)
        for m in range(-ell, ell+1):
            component(fourier, j, ell, m)
            for a, r in zip(accumulator, response.value):
                np.hypot(a, np.abs(r), out=a)
        if mask is not None:
            accumulator[~mask] = 0
        return accumulator

    first, second, index1, index2 = [], [], [], []
    for ell in options['ells']:
        for ij, j1 in enumerate(options['scales']):
            u1 = modulus(original, j1, ell)
            first.append(moments(u1))
            index1.append((j1, ell))
            later = options['scales'][ij+1:] if options['max_order'] == 2 else []
            if later:
                # Reuse the response real buffer to FFT U1. Only one U1 spectrum
                # survives at a time; none are cached across first-order paths.
                response[...] = u1
                u1_fourier = response.r2c()
                for j2 in later:
                    second.append(moments(modulus(u1_fourier, j2, ell)))
                    index2.append((j1, j2, ell))
                del u1_fourier
    extras = {}
    if options['m_resolved']:
        m1, m2, mi1, mi2 = [], [], [], []
        for ell in options['ells']:
            for ij, j1 in enumerate(options['scales']):
                for m in range(ell+1):
                    component(original, j1, ell, m)
                    np.abs(response.value, out=accumulator)
                    if mask is not None:
                        accumulator[~mask] = 0
                    m1.append(moments(accumulator))
                    mi1.append((j1, ell, m))
                    later = options['scales'][ij+1:] if options['max_order']==2 else []
                    if later:
                        response[...] = accumulator
                        u1_fourier = response.r2c()
                        for j2 in later:
                            component(u1_fourier,j2,ell,m)
                            np.abs(response.value,out=accumulator)
                            if mask is not None:
                                accumulator[~mask] = 0
                            m2.append(moments(accumulator))
                            mi2.append((j1,j2,ell,m))
                        del u1_fourier
        def ratios(indices, values):
            groups = {}
            for index, value in zip(indices, values):
                if index[-2] > 0:
                    groups.setdefault(index[:-1], []).append(value)
            rows, ratios, valid = [], [], []
            for index, modes in groups.items():
                modes = np.asarray(modes)
                good = np.all(modes[:-1] > 0, axis=0)
                ratio = np.full(len(q), np.nan)
                ratio[good] = np.mean(modes[1:,good]/modes[:-1,good], axis=0)
                rows.append(index)
                ratios.append(ratio)
                valid.append(good)
            return rows, np.asarray(ratios).reshape(-1,len(q)), np.asarray(valid,dtype=bool).reshape(-1,len(q))
        ri1,r1,valid1 = ratios(mi1,m1)
        ri2,r2,valid2 = ratios(mi2,m2)
        extras = {'M1':np.asarray(m1).reshape(-1,len(q)), 'M2':np.asarray(m2).reshape(-1,len(q)),
                  'index_m1':np.asarray(mi1,dtype=int).reshape(-1,3),
                  'index_m2':np.asarray(mi2,dtype=int).reshape(-1,4),
                  'R1':r1, 'R2':r2, 'ratio_valid1':valid1, 'ratio_valid2':valid2,
                  'index_r1':np.asarray(ri1,dtype=int).reshape(-1,2),
                  'index_r2':np.asarray(ri2,dtype=int).reshape(-1,3),
                  'm_convention':'nonnegative_m_same_m_and_ell_at_both_layers'}
    elapsed = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
    return {'S0': s0, 'S1': np.asarray(first).reshape(-1, len(q)),
            'S2': np.asarray(second).reshape(-1, len(q)),
            'index1': np.asarray(index1, dtype=int).reshape(-1, 2),
            'index2': np.asarray(index2, dtype=int).reshape(-1, 3),
            'q': q, 'averaging': 'mean_over_covered_cells', 'covered_cells': count,
            'wavelet_normalization': 'kymatio_solid_harmonic',
            'mask_policy': 'zero_input_and_each_layer',
            'plan': scattering_plan(options), 'elapsed_seconds_max_rank': elapsed, **extras}
