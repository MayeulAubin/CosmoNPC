"""Survey density-split preparation, reusing existing catalogue estimators.

Random-query data and each distinct leg's background randoms occupy disjoint
seeded pools. This avoids introducing shared-object Poisson contractions that
the ordinary cross-catalogue estimators do not subtract.
"""
import logging
import os
import numpy as np
from mpi4py import MPI
from .catalog_processor import catalog_reader, check_bounds_in_box, get_position_bounds
from .mesh_generator import pm_painter, get_mesh_pk_survey, get_mesh_bk_survey
from .clustering_estimator import calculate_power_spectrum_survey, calculate_bk_sugi_survey
from .density_processor import read_local_field
from .quantiles import gaussian_smooth, assign_quantiles, _permuted_ids, _global_kth
from .density_split import _configuration, _collective_error


def _row_ids(length, comm):
    offset = comm.exscan(int(length), op=MPI.SUM)
    return np.arange(length, dtype=np.uint64) + np.uint64(offset or 0)


def _position_pools(positions, seed):
    """Identical random positions share a pool, including reordered catalogues."""
    coordinates = np.array(positions, dtype=np.float64, copy=True)
    coordinates[coordinates == 0] = 0.
    bits = coordinates.view(np.uint64)
    combined = _permuted_ids(bits[:,0], seed)
    combined ^= _permuted_ids(bits[:,1], seed ^ 0x9e3779b97f4a7c15)
    combined ^= _permuted_ids(bits[:,2], seed ^ 0xbf58476d1ce4e5b9)
    return (_permuted_ids(combined, seed) % np.uint64(4)).astype(np.uint8)


def _catalogue_call(source, base, comm, kind, boxcenter=None):
    return catalog_reader(source, 'survey-like', base['column_names'], base['z_range'],
        base['comp_weight_plan'], comm, para_cosmo=base['cosmology'], boxsize=base['boxsize'],
        catalog_type=kind, normalization_scheme=base['normalization_scheme'], boxcenter=boxcenter)


def _mask_values(source, base, template):
    if isinstance(source, np.ndarray):
        if source.shape != tuple(base['nmesh']) or source.dtype.kind not in 'bifu':
            raise ValueError('mask must be a real global array with shape nmesh')
        values = source[template.slices]
    else:
        values, _ = read_local_field(source, base['nmesh'], base['boxsize'],
            base.get('field_origin',[0.,0.,0.]), template.slices, np.float64)
    if not np.all(np.isfinite(values)) or np.any((values != 0) & (values != 1)):
        raise ValueError('mask must contain only boolean or 0/1 values')
    return values.astype(bool)


def prepare_survey_density_split(config, comm=MPI.COMM_WORLD):
    base, options, sources, measurements = _configuration(config)
    if base['input_type'] != 'catalog' or base['geometry'] != 'survey-like':
        raise ValueError('Survey preparation requires survey-like catalogue inputs')
    source_name = options['source']
    random_source = base['catalogs'][source_name.replace('data_','randoms_')]
    center = base.get('boxcenter')
    preview = _catalogue_call(random_source, base, comm, 'randoms', center)
    if center is None:
        randoms_source, center = preview
    else:
        randoms_source = preview
    del preview
    center = np.asarray(center, dtype=float)
    data, randoms, roles, ids = {}, {}, {}, {}
    for name, source in sources.items():
        data[name] = _catalogue_call(source, base, comm, 'data', center)
        randoms[name] = (randoms_source if name == source_name else
            _catalogue_call(base['catalogs'][name.replace('data_','randoms_')], base, comm, 'randoms', center))
        for kind, array in [('data',data[name]),('randoms',randoms[name])]:
            check_bounds_in_box(get_position_bounds(array['Position'],comm), center,
                                np.asarray(base['boxsize']), f'{kind}_{name}', comm)
            error = None
            if np.any(~np.isfinite(array['Position'])) or np.any(~np.isfinite(array['WEIGHT'])) or np.any(array['WEIGHT'] < 0):
                error = f'{kind}_{name} requires finite nonnegative environmental weights'
            _collective_error(error,comm)
        ids[name] = _row_ids(len(randoms[name]),comm)
        roles[name] = _position_pools(randoms[name]['Position'],options['tie_seed'])
    source_data, source_randoms = data[source_name], randoms[source_name]
    weights_d = comm.allreduce(float(np.sum(source_data['WEIGHT'],dtype=np.float64)),op=MPI.SUM)
    weights_r = comm.allreduce(float(np.sum(source_randoms['WEIGHT'],dtype=np.float64)),op=MPI.SUM)
    if not np.isfinite(weights_d) or not np.isfinite(weights_r) or weights_d <= 0 or weights_r <= 0:
        raise ValueError('Environmental data and randoms require positive total weights')
    painter = dict(Nmesh=base['nmesh'], boxsize=base['boxsize'], sampler=base['sampler'],
                   interlaced=False,comm=comm,boxcenter=center,precision=base['precision'])
    galaxies = pm_painter(source_data['Position'],source_data['WEIGHT'],**painter)
    background = pm_painter(source_randoms['Position'],source_randoms['WEIGHT'],**painter)
    mask = background.value.real > 0
    if options.get('mask') is not None:
        error = None
        try:
            mask &= _mask_values(options['mask'],base,galaxies)
        except Exception as exc:
            error = f'Invalid density-split mask: {exc}'
        _collective_error(error,comm)
    # Completeness weights define the environment, without FKP weighting.
    ratio = galaxies.pm.create(type='real')
    ratio[...] = 0
    np.divide(galaxies.value.real, (weights_d/weights_r)*background.value.real,
              out=ratio.value.real, where=mask)
    ratio.value.real[mask] -= 1
    environment = gaussian_smooth(ratio,options['smoothing_radius'],mask)
    ratio[...] = environment
    coverage = galaxies.pm.create(type='real')
    coverage[...] = mask

    candidates = roles[source_name] == 0
    query_hashes = _permuted_ids(ids[source_name],options['tie_seed'])
    positions = source_randoms['Position'] - center
    # A CiC readout must exchange every particle's complete support.
    layout = galaxies.pm.decompose(positions, smoothing=2.)
    values = ratio.readout(positions,resampler='cic',layout=layout).real
    support = coverage.readout(positions,resampler='cic',layout=layout).real
    inside = coverage.readout(positions,resampler='nearest',layout=layout).real > .5
    candidates &= inside & (support > 0)
    np.divide(values,support,out=values,where=support>0)
    available = comm.allreduce(int(candidates.sum()),op=MPI.SUM)
    ngal = comm.allreduce(int(len(source_data)),op=MPI.SUM)
    requested = max(1,int(options['query_ratio']*ngal))
    if available > requested:
        last = _global_kth(np.sort(query_hashes[candidates]),requested-1,comm)
        candidates &= query_hashes <= last
    queries = source_randoms[candidates].copy()
    query_ids = ids[source_name][candidates]
    labels, info = assign_quantiles(values[candidates],np.ones(len(queries),dtype=bool),query_ids,
                                    options['nquantiles'],comm,options['tie_seed'])
    registry = dict(data)
    backgrounds = {name:{'source':name,'nz_scale':1.} for name in data}
    for q in range(options['nquantiles']):
        name = f'Q{q}'
        selected = queries[labels == q].copy()
        scale = comm.allreduce(float(np.sum(selected['WEIGHT'],dtype=np.float64)),op=MPI.SUM)/weights_d
        selected['NZ'] *= scale
        registry[name] = selected
        backgrounds[name] = {'source':source_name,'nz_scale':scale}
    pool_counts = {}
    for name in randoms:
        counts = np.bincount(roles[name], minlength=4).astype(np.int64)
        total = np.empty_like(counts)
        comm.Allreduce(counts,total,op=MPI.SUM)
        pool_counts[name] = total
    info['query_count'] = info.pop('covered_cells')
    info.update(source=source_name,smoothing_radius=options['smoothing_radius'],smoothing_filter='gaussian',
                query_sampling='source_random_catalogue',requested_query_count=requested,
                available_query_count=available,query_ratio=options['query_ratio'],
                query_interpolation='cic',random_partition='four_seeded_disjoint_coordinate_pools',random_pool_counts=pool_counts,
                quantile_nz_scales={f'Q{q}':backgrounds[f'Q{q}']['nz_scale'] for q in range(options['nquantiles'])},
                query_id_convention='rank_concatenated_input_row_order',geometry='survey-like',
                environmental_alpha=weights_d/weights_r,environment_interlaced=False,
                environment_sampler=base['sampler'],boxcenter=center,
                covered_mesh_cells=comm.allreduce(int(mask.sum()),op=MPI.SUM))
    return registry, randoms, roles, backgrounds, info, base, options, measurements


def measurement_catalogues(legs, registry, randoms, roles, backgrounds, comm=MPI.COMM_WORLD):
    """Map ordered symbolic legs onto existing aaa/aab/aba/abb/abc inputs."""
    unique = list(dict.fromkeys(legs))
    lookup = {token:'abc'[i] for i,token in enumerate(unique)}
    pattern = ''.join(lookup[token] for token in legs)
    catalogs = {}
    for i, token in enumerate(unique):
        label = 'abc'[i]
        descriptor = backgrounds[token]
        name = descriptor['source']
        subset = randoms[name][roles[name] == i+1].copy()
        if not comm.allreduce(int(len(subset)),op=MPI.SUM):
            raise ValueError(f'No background randoms in pool {i+1} for {token}; provide more randoms')
        subset['NZ'] *= descriptor['nz_scale']
        catalogs[f'data_{label}'] = registry[token]
        catalogs[f'randoms_{label}'] = subset
    return catalogs, 'auto' if len(unique) == 1 else 'cross', pattern


def measure_survey_legs(legs, prepared, comm=MPI.COMM_WORLD):
    registry, randoms, roles, backgrounds, info, base, options, _ = prepared
    catalogs, mode, pattern = measurement_catalogues(legs,registry,randoms,roles,backgrounds,comm)
    common = dict(nmesh=base['nmesh'],geometry='survey-like',column_names=base['column_names'],
        boxsize=base['boxsize'],sampler=base['sampler'],interlaced=base['interlaced'],
        z_range=base['z_range'],comp_weight_plan=base['comp_weight_plan'],para_cosmo=base['cosmology'],
        comm=comm,normalization_scheme=base['normalization_scheme'],alpha_scheme=base['alpha_scheme'],
        precision=base['precision'],boxcenter=info['boxcenter'])
    kwargs = {k:v for k,v in base.items() if k not in ('catalogs','correlation_mode','comm')}
    if base['statistic'] == 'pk':
        attrs,a,b = get_mesh_pk_survey(catalogs,mode,**common)
        attrs.update(base)
        result = calculate_power_spectrum_survey(attrs,a,b,mode,comm,**kwargs)
    else:
        attrs,a,b,c = get_mesh_bk_survey(catalogs,mode,tracer_type=pattern,angu_config=base['angu_config'],**common)
        attrs.update(base)
        attrs['tracer_type'] = pattern
        result = calculate_bk_sugi_survey(a,b,c,mode,attrs,comm,catalogs=catalogs,**kwargs)
    if comm.rank == 0:
        result['legs'] = legs
        result['correlation_mode'] = mode
        result['tracer_type'] = pattern
        result['normalization_scheme'] = base['normalization_scheme']
        result['boxcenter'] = info['boxcenter']
    return result


def run_survey_density_split(config,comm=MPI.COMM_WORLD):
    prepared = prepare_survey_density_split(config,comm)
    _,_,_,_,info,base,_,measurements = prepared
    output = {'statistic':base['statistic'],'geometry':'survey-like','precision':base['precision'],
              'input_type':'catalog','quantiles':info,'nmesh':base['nmesh'],'boxsize':base['boxsize'],
              'poles':base['poles'],'angu_config':base['angu_config'],
              'data_vector_mode':base.get('data_vector_mode','diagonal'),
              'normalization_scheme':base['normalization_scheme'],'alpha_scheme':base['alpha_scheme'],
              'k_min':base['k_min'],'k_max':base['k_max'],'k_bins':base['k_bins'],
              'shotnoise_policy':'existing_catalogue_self_terms_disjoint_random_pools',
              'measurements':{}}
    for legs in measurements:
        result = measure_survey_legs(legs,prepared,comm)
        if comm.rank == 0:
            output['measurements']['__'.join(legs)] = result
    if comm.rank == 0:
        os.makedirs(base['output_dir'],exist_ok=True)
        path = os.path.join(base['output_dir'],f'density_split_{base["statistic"]}.npy')
        np.save(path,output)
        logging.info('Survey density-split results saved to %s',path)
    comm.Barrier()
    return output if comm.rank == 0 else None
