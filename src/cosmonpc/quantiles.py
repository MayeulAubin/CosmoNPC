"""Volume-sampled density quantiles on distributed pmesh domains.

Only small order-statistic counters are reduced: field values are never gathered
onto rank zero. Ties use a deterministic permutation of global cell IDs.
"""
import numpy as np
from mpi4py import MPI


def global_cell_ids(shape, start, nmesh):
    """C-order cell IDs, independent of MPI decomposition."""
    i, j, k = np.ogrid[tuple(slice(int(a), int(a+b)) for a, b in zip(start, shape))]
    return ((i * int(nmesh[1]) + j) * int(nmesh[2]) + k).astype(np.uint64)


def _permuted_ids(ids, seed):
    # SplitMix64 is a bijection on uint64: unlike random floating variates,
    # this gives unique tie keys for unique global cells.
    x = np.asarray(ids, dtype=np.uint64) ^ np.uint64(seed)
    x = x + np.uint64(0x9e3779b97f4a7c15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


def _ordered_keys(values):
    values = np.array(values, dtype=np.float64, copy=True)
    values[values == 0] = 0.  # Treat signed zeros as equal densities.
    bits = values.view(np.uint64)
    sign = np.uint64(1 << 63)
    return np.where(bits & sign, ~bits, bits ^ sign)


def _key_value(key):
    bits = np.uint64(key)
    sign = np.uint64(1 << 63)
    return float(np.array(bits ^ sign if bits & sign else ~bits).view(np.float64))


def _global_kth(sorted_keys, rank, comm):
    """Exact distributed order statistic on uint64 keys, using <=64 reductions."""
    lo = comm.allreduce(int(sorted_keys[0]) if len(sorted_keys) else (1 << 64)-1, op=MPI.MIN)
    hi = comm.allreduce(int(sorted_keys[-1]) if len(sorted_keys) else 0, op=MPI.MAX)
    while lo < hi:
        mid = lo + (hi-lo)//2
        count = comm.allreduce(int(np.searchsorted(sorted_keys, np.uint64(mid), side='right')), op=MPI.SUM)
        if count <= rank:
            lo = mid+1
        else:
            hi = mid
    return np.uint64(lo)


def _global_weighted_cut(sorted_keys, prefix_weights, target, comm):
    """First distributed key whose cumulative positive weight reaches target."""
    lo = comm.allreduce(int(sorted_keys[0]) if len(sorted_keys) else (1 << 64)-1, op=MPI.MIN)
    hi = comm.allreduce(int(sorted_keys[-1]) if len(sorted_keys) else 0, op=MPI.MAX)
    while lo < hi:
        mid = lo + (hi-lo)//2
        end = int(np.searchsorted(sorted_keys,np.uint64(mid),side='right'))
        weight = comm.allreduce(float(prefix_weights[end-1]) if end else 0.,op=MPI.SUM)
        if weight < target:
            lo = mid+1
        else:
            hi = mid
    return np.uint64(lo)


def gaussian_smooth(mesh, radius, mask=None):
    """Periodic Gaussian smoothing, normalized by smoothed coverage when masked.

    The Gaussian is exp(-k^2 R^2 / 2), with R in the BoxSize length units.
    Only the returned environment changes; the supplied tracer mesh is intact.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('smoothing_radius must be finite and nonnegative')
    if mask is None:
        mask = np.ones(mesh.shape, dtype=bool)
    if radius == 0:
        return np.where(mask, mesh.value.real, 0).copy()

    def convolve(values):
        work = mesh.pm.create(type='real')
        work[...] = values
        fourier = work.r2c()
        def kernel(k, value):
            return value * np.exp(-.5 * radius**2 * sum(ki**2 for ki in k))
        fourier.apply(kernel, kind='wavenumber', out=Ellipsis)
        return fourier.c2r().value.real.copy()

    smoothed = convolve(np.where(mask, mesh.value.real, 0))
    if not mesh.pm.comm.allreduce(bool(np.all(mask)), op=MPI.LAND):
        weight = convolve(mask)
        bad = np.any((weight <= 0) & mask)
        if mesh.pm.comm.allreduce(bool(bad), op=MPI.LOR):
            raise ValueError('Smoothed mask has nonpositive weight in covered cells')
        np.divide(smoothed, weight, out=smoothed, where=mask)
    smoothed[~mask] = 0
    return smoothed


def assign_quantiles(values, mask, cell_ids, nquantiles, comm, tie_seed=0, weights=None):
    """Equal-population ranks over covered cells; Q0 is lowest density.

    Unweighted populations differ by at most one cell. Optional positive
    selection weights define weighted-CDF quantiles (up to one-cell resolution).
    A constant environment is rejected
    because its arbitrary partition would contain no density information.
    """
    error = None
    if isinstance(nquantiles, (bool, np.bool_)) or not isinstance(nquantiles, (int, np.integer)) or nquantiles < 2:
        error = 'nquantiles must be an integer >= 2'
    elif not isinstance(tie_seed, (int, np.integer)) or isinstance(tie_seed, (bool, np.bool_)) or not 0 <= tie_seed < 1 << 64:
        error = 'tie_seed must be a uint64-compatible nonnegative integer'
    elif values.shape != mask.shape or values.shape != cell_ids.shape:
        error = 'values, mask, and cell_ids must have identical local shapes'
    elif mask.dtype != np.bool_ or np.any(~np.isfinite(values[mask])):
        error = 'mask must be boolean and covered environmental values must be finite'
    if error is None and weights is not None:
        if weights.shape != mask.shape or np.any(~np.isfinite(weights[mask])) or np.any(weights[mask] <= 0):
            error = 'Query weights must have matching shape and be finite and positive in covered cells'
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError(next(e for e in errors if e))
    local_values = values[mask]
    population = comm.allreduce(int(local_values.size), op=MPI.SUM)
    if population < nquantiles:
        raise ValueError('Not enough covered cells for nonempty quantiles')
    keys = _ordered_keys(local_values)
    local_weights = None if weights is None else np.asarray(weights[mask], dtype=np.float64)
    weighted = False
    if local_weights is not None:
        wmin = comm.allreduce(float(local_weights.min()) if len(local_weights) else np.inf,op=MPI.MIN)
        wmax = comm.allreduce(float(local_weights.max()) if len(local_weights) else -np.inf,op=MPI.MAX)
        weighted = wmin != wmax
        weight_total = comm.allreduce(float(np.sum(local_weights,dtype=np.float64)),op=MPI.SUM)
    if weighted:
        order = np.argsort(keys)
        sorted_keys = keys[order]
        prefix_weights = np.cumsum(local_weights[order],dtype=np.float64)
    else:
        sorted_keys = np.sort(keys)
    lo = comm.allreduce(float(local_values.min()) if len(local_values) else np.inf, op=MPI.MIN)
    hi = comm.allreduce(float(local_values.max()) if len(local_values) else -np.inf, op=MPI.MAX)
    if lo == hi:
        raise ValueError('A constant environmental density cannot define density quantiles')
    hashes = _permuted_ids(cell_ids[mask], tie_seed)
    local_labels = np.zeros(keys.size, dtype=np.int32)
    boundaries = []
    for q in range(1, nquantiles):
        cut = population*q//nquantiles
        target = weight_total*q/nquantiles if weighted else None
        edge = (_global_weighted_cut(sorted_keys,prefix_weights,target,comm) if weighted
                else _global_kth(sorted_keys, cut-1, comm))
        below = comm.allreduce(int(np.count_nonzero(keys < edge)), op=MPI.SUM)
        tied = keys == edge
        tie_count = comm.allreduce(int(np.count_nonzero(tied)), op=MPI.SUM)
        if weighted:
            below_weight = comm.allreduce(float(np.sum(local_weights[keys < edge],dtype=np.float64)),op=MPI.SUM)
            order = np.argsort(hashes[tied])
            last_hash = _global_weighted_cut(hashes[tied][order],
                np.cumsum(local_weights[tied][order],dtype=np.float64),max(target-below_weight,0.),comm)
        else:
            last_hash = _global_kth(np.sort(hashes[tied]), cut-below-1, comm)
        local_labels += (keys > edge) | (tied & (hashes > last_hash))
        boundaries.append({'value': _key_value(edge), 'tie_count': tie_count,
                           'last_tie_hash': int(last_hash), 'rank_cut': cut})
    labels = np.full(mask.shape, -1, dtype=np.int32)
    labels[mask] = local_labels
    counts = np.bincount(local_labels, minlength=nquantiles).astype(np.int64)
    total_counts = np.empty_like(counts)
    comm.Allreduce(counts, total_counts, op=MPI.SUM)
    if np.any(total_counts == 0):
        raise ValueError('Selection weights produce an empty quantile; use a finer grid or fewer quantiles')
    for q,boundary in enumerate(boundaries,1):
        boundary['rank_cut'] = int(np.sum(total_counts[:q]))
    fractions = total_counts/population
    extra = {}
    if local_weights is not None:
        local_sums = np.bincount(local_labels,weights=local_weights,minlength=nquantiles)
        weight_sums = np.empty_like(local_sums)
        comm.Allreduce(local_sums,weight_sums,op=MPI.SUM)
        fractions = weight_sums/weight_sums.sum()
        for q,boundary in enumerate(boundaries,1):
            boundary['query_weight_cut'] = float(np.sum(weight_sums[:q]))
            boundary['target_query_weight'] = float(weight_sums.sum()*q/nquantiles)
        extra = {'query_weight_sums':weight_sums, 'total_query_weight':float(weight_sums.sum()),
                 'maximum_cell_weight':wmax, 'query_sampling':'selection_weighted_mesh_cells'}
    return labels, {'nquantiles': int(nquantiles), 'counts': total_counts,
                    'fractions': fractions, 'covered_cells': population,
                    'boundaries': boundaries, 'tie_seed': int(tie_seed),
                    'query_sampling': 'uniform_mesh_cells', 'density_range': [lo, hi], **extra}


def quantile_field(labels, q, fraction, template):
    """delta_Q = mask * (1_Q / f_Q - 1), not a density-weighted selection."""
    result = template.pm.create(type='real')
    result[...] = np.where(labels >= 0, (labels == q)/fraction - 1., 0.)
    return result
