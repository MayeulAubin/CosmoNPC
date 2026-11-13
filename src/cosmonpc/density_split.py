"""Density-split orchestration using the existing box and survey estimators."""
import logging
import os
import numpy as np
from pmesh import ParticleMesh
from mpi4py import MPI
from .precision import complex_dtype
from .param_helper import validate_config, catalog_check, validate_poles, validate_sugi_poles
from .density_processor import read_local_field
from .mesh_generator import get_mesh_box
from .math_evaluator import get_compensation_bk_sugi
from .quantiles import global_cell_ids, gaussian_smooth, assign_quantiles, quantile_field
from .clustering_estimator import calculate_power_spectrum_box, calculate_bk_sugi_box


def _configuration(config, require_quantile=True):
    base = dict(config)
    options = base.pop('density_split')
    if not isinstance(options, dict):
        raise TypeError('density_split must be a configuration dictionary')
    options = dict(options)
    allowed = {'source', 'nquantiles', 'smoothing_radius', 'tie_seed', 'mask', 'measurements', 'query_ratio'}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown density_split options: {sorted(unknown)}')
    if base.get('geometry', 'box-like') not in ('box-like','survey-like'):
        raise ValueError('geometry must be box-like or survey-like')
    if base.get('statistic') not in ('pk', 'bk_sugi'):
        raise ValueError('Density-split statistic must be pk or bk_sugi')
    base.setdefault('geometry', 'box-like')
    base.setdefault('input_type', 'catalog')
    base.setdefault('sampler', 'cic')
    base.setdefault('interlaced', False)
    base.setdefault('apply_rsd', False)
    base.setdefault('column_names', [])
    base.setdefault('cosmology', None)
    base.setdefault('redshift_box', None)
    base.setdefault('rsd', [0, 0, 1])
    base.setdefault('poles', [0, 2])
    base.setdefault('angu_config', [0, 0, 0])
    base.setdefault('compensation', base['input_type'] == 'catalog')
    if base['geometry'] == 'survey-like':
        base.setdefault('z_range', None)
        base.setdefault('comp_weight_plan', {'scheme':None, 'name_alias':None})
        base.setdefault('normalization_scheme', 'mesh' if base['input_type'] == 'density' else 'mixed-mesh')
        base.setdefault('alpha_scheme', 'pypower')
        if base['alpha_scheme'] not in ('pypower','nbodykit'):
            raise ValueError('alpha_scheme must be pypower or nbodykit')
        supported = (('mesh',) if base['input_type'] == 'density' else
                     ('particle','mixed-mesh') if base['statistic'] == 'pk' else ('particle','mesh','mixed-mesh'))
        if base['normalization_scheme'] not in supported:
            raise ValueError(f'normalization_scheme must be one of {supported}')
    # In density-split mode the measurement legs determine auto/cross and
    # bispectrum ordering; these ordinary task flags do not select input legs.
    base['correlation_mode'] = 'auto'
    base['tracer_type'] = 'aaa'
    validate_config(base)
    if base['statistic'] == 'pk':
        validate_poles(base['poles'])
        if not base['poles'] or (base['geometry'] == 'box-like' and all(p % 2 for p in base['poles'])):
            raise ValueError('At least one even power-spectrum pole is required for boxes')
    else:
        validate_sugi_poles(base['angu_config'], base['geometry'])
        if base.get('data_vector_mode', 'diagonal') not in ('diagonal','full'):
            raise ValueError('data_vector_mode must be diagonal or full')
    kmin, kmax, bins = base['k_min'], base['k_max'], base['k_bins']
    if not np.isfinite(kmin) or not np.isfinite(kmax) or not 0 <= kmin < kmax:
        raise ValueError('Require finite 0 <= k_min < k_max')
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('k_bins must be a positive integer')
    nyquist = np.pi * base['nmesh'][0] / base['boxsize'][0]
    factor = 2 if base['statistic'] == 'bk_sugi' else 1
    if factor*kmax >= nyquist:
        raise ValueError(f'{factor} * k_max must be below Nyquist for {base["statistic"]}')
    options.setdefault('source', 'data_a')
    options.setdefault('nquantiles', 5)
    options.setdefault('smoothing_radius', 10.)
    options.setdefault('tie_seed', 0)
    radius = options['smoothing_radius']
    if isinstance(radius, (bool, np.bool_)) or not np.isscalar(radius) or not np.isfinite(radius) or radius < 0:
        raise ValueError('smoothing_radius must be finite and nonnegative')
    count = options['nquantiles']
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError('nquantiles must be an integer >= 2')
    source_inputs = base['fields'] if base['input_type'] == 'density' else base['catalogs']
    sources = {k: v for k, v in source_inputs.items() if k in ('data_a','data_b','data_c') and v is not None}
    if options['source'] not in sources:
        raise ValueError('density_split.source must name an available data_a, data_b, or data_c')
    if base['input_type'] == 'catalog':
        catalog_check(base['catalogs'], base['geometry'], 'auto', base['statistic'])
    defaults = ([['Q', options['source']], ['Q', 'Q']] if base['statistic'] == 'pk' else
                [['Q', options['source'], options['source']], ['Q', 'Q', options['source']], ['Q', 'Q', 'Q']])
    templates = options.get('measurements', defaults)
    measurements = _expand_measurements(templates, count, sources, 2 if base['statistic'] == 'pk' else 3,require_quantile)
    used = {options['source']} | {token for legs in measurements for token in legs if not token.startswith('Q')}
    sources = {name: source for name, source in sources.items() if name in used}
    if base['geometry'] == 'survey-like' and base['input_type'] == 'catalog':
        for name in sources:
            if base['catalogs'].get(name.replace('data_', 'randoms_')) is None:
                raise ValueError(f'Survey density-split requires randoms for {name}')
        cross = any(len(set(legs)) > 1 for legs in measurements)
        if cross and base['normalization_scheme'] != 'mixed-mesh':
            raise ValueError('Cross survey measurements require normalization_scheme=mixed-mesh, as in the standard estimators')
        if base['statistic'] == 'pk' and cross and base['high_order_mode'] != 'default':
            raise ValueError('Survey high_order_mode fast/compare requires auto measurements only')
        ratio = options.setdefault('query_ratio', 5.)
        if isinstance(ratio,(bool,np.bool_)) or not np.isscalar(ratio) or not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('query_ratio must be finite and positive')
    elif 'query_ratio' in options:
        raise ValueError('query_ratio applies only to survey random queries')
    seed = options['tie_seed']
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 1 << 64:
        raise ValueError('tie_seed must be a uint64-compatible nonnegative integer')
    return base, options, sources, measurements


def _expand_measurements(templates, count, sources, legs, require_quantile=True):
    if not isinstance(templates, (list, tuple)) or not templates:
        raise ValueError('measurements must be a nonempty list of leg lists')
    allowed = set(sources) | {f'Q{i}' for i in range(count)} | {'Q'}
    expanded = []
    for template in templates:
        if not isinstance(template, (list, tuple)) or len(template) != legs:
            raise ValueError(f'Each measurement requires exactly {legs} legs')
        if any(not isinstance(token, str) or token not in allowed for token in template):
            raise ValueError(f'Invalid measurement leg: {template}; use Q, Q0..Q{count-1}, or input names')
        if require_quantile and not any(token.startswith('Q') for token in template):
            raise ValueError('A density-split measurement must contain a quantile leg')
        for q in range(count) if 'Q' in template else [None]:
            entry = tuple(f'Q{q}' if token == 'Q' else token for token in template)
            if entry not in expanded:
                expanded.append(entry)
    return expanded


def _collective_error(error, comm):
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError(next(e for e in errors if e))


def _read_mesh_tracers(base, options, sources, comm):
    pm = ParticleMesh(Nmesh=base['nmesh'], BoxSize=base['boxsize'],
                      dtype=complex_dtype(base['precision']), comm=comm)
    template = pm.create(type='real')
    mask = np.ones(template.shape, dtype=bool)
    tracers, specs = {}, {}
    for name, source in sources.items():
        error = None
        if base['input_type'] == 'density':
            try:
                values, spec = read_local_field(source, base['nmesh'], base['boxsize'],
                                                base.get('field_origin', [0.,0.,0.]),
                                                template.slices, template.value.real.dtype)
                field = pm.create(type='real')
                field[...] = values
                if spec['_coverage'] is not None:
                    mask &= spec['_coverage']
                specs[name] = spec
            except Exception as exc:
                error = f'Cannot load {name}: {type(exc).__name__}: {exc}'
            _collective_error(error, comm)
        else:
            attrs, field, _ = get_mesh_box({'data_a':source}, 'auto', nmesh=base['nmesh'],
                geometry='box-like', boxsize=base['boxsize'], sampler=base['sampler'],
                interlaced=base['interlaced'], column_names=base['column_names'], comm=comm,
                statistic='pk', apply_rsd=base['apply_rsd'], para_cosmo=base['cosmology'],
                redshift_box=base['redshift_box'], los=base['rsd'], precision=base['precision'])
            specs[name] = {'kind':'counts'}
        tracers[name] = field

    if options.get('mask') is not None:
        error = None
        try:
            source = options['mask']
            if isinstance(source, np.ndarray):
                if source.shape != tuple(base['nmesh']) or source.dtype.kind not in 'bifu':
                    raise ValueError('mask must be a real global array with shape nmesh')
                local = source[template.slices]
            else:
                local, _ = read_local_field(source, base['nmesh'], base['boxsize'],
                    base.get('field_origin',[0.,0.,0.]), template.slices, np.float64)
            if not np.all(np.isfinite(local)) or np.any((local != 0) & (local != 1)):
                raise ValueError('mask must contain only boolean or 0/1 values')
            mask &= local.astype(bool)
        except Exception as exc:
            error = f'Invalid density-split mask: {exc}'
        _collective_error(error, comm)
    return tracers, specs, mask, next(iter(tracers.values()))


def _normalize_mesh_tracers(tracers, specs, mask, base, comm):
    metadata = {}
    covered = comm.allreduce(int(mask.sum()), op=MPI.SUM)
    if not covered:
        raise ValueError('Density-split mask has no covered cells')

    # Use one explicit assignment-window treatment on input tracers only.
    # The newly constructed quantile indicator has no particle-painting window.
    for name, field in tracers.items():
        values, spec = field.value.real, specs[name]
        error = None
        if np.any(~np.isfinite(values)):
            error = f'{name} contains nonfinite mesh values'
        _collective_error(error, comm)
        mean = comm.allreduce(float(np.sum(values[mask], dtype=np.float64)), op=MPI.SUM)/covered
        reference = None
        if spec['kind'] == 'counts':
            reference = float(spec.get('mean_density',mean))
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError(f'{name}: positive mean_density is required for counts')
            field[...] = field.value / reference - 1.
        if base['compensation']:
            compensation = get_compensation_bk_sugi(base['sampler'])
            fourier = field.r2c()
            fourier.apply(compensation[0][1], kind=compensation[0][2], out=Ellipsis)
            field[...] = fourier.c2r().value
        field.value[~mask] = 0
        metadata[name] = {'kind':spec['kind'], 'input_mean_covered':mean,
                          'mean_density':reference, 'assignment_compensation':base['compensation']}
    return metadata


def prepare_density_split(config, comm=MPI.COMM_WORLD):
    """Load tracers and build quantile labels without running any statistic.

    Returns local pmesh tracers, labels, quantile metadata, source metadata,
    validated ordinary settings, split settings, and ordered measurement legs.
    No user-supplied overdensity is centered or otherwise renormalized.
    """
    base, options, sources, measurements = _configuration(config)
    if base['geometry'] != 'box-like':
        raise ValueError('Use prepare_survey_density_split for survey inputs')
    tracers, specs, mask, template = _read_mesh_tracers(base, options, sources, comm)
    metadata = _normalize_mesh_tracers(tracers, specs, mask, base, comm)
    covered = comm.allreduce(int(mask.sum()), op=MPI.SUM)
    environment = gaussian_smooth(tracers[options['source']], options['smoothing_radius'], mask)
    ids = global_cell_ids(template.shape, template.start, base['nmesh'])
    labels, info = assign_quantiles(environment, mask, ids, options['nquantiles'], comm, options['tie_seed'])
    info.update(source=options['source'], smoothing_radius=options['smoothing_radius'],
                smoothing_filter='gaussian', mask_smoothing='normalized_convolution',
                coverage_fraction=covered/np.prod(base['nmesh']), geometry='box-like')
    return tracers, labels, info, metadata, base, options, measurements


def run_density_split(config, comm=MPI.COMM_WORLD):
    if config.get('geometry') == 'survey-like' and config.get('input_type') == 'density':
        from .survey_field import run_survey_field_statistics
        return run_survey_field_statistics(config,comm)
    if config.get('geometry') == 'survey-like':
        from .survey_density_split import run_survey_density_split
        return run_survey_density_split(config, comm)
    tracers, labels, info, metadata, base, options, measurements = prepare_density_split(config, comm)
    template = next(iter(tracers.values()))
    cells, volume = int(np.prod(base['nmesh'])), float(np.prod(base['boxsize']))
    output = {'statistic':base['statistic'], 'geometry':'box-like', 'precision':base['precision'],
              'input_type':base['input_type'], 'quantiles':info, 'source_metadata':metadata,
              'nmesh':base['nmesh'], 'boxsize':base['boxsize'], 'rsd':base['rsd'],
              'k_min':base['k_min'], 'k_max':base['k_max'], 'k_bins':base['k_bins'],
              'shotnoise_policy':'none',
              'shotnoise_note':'No Poisson correction for nonlinear quantile selection; discreteness noise remains.',
              'poles':base['poles'], 'angu_config':base['angu_config'],
              'data_vector_mode':base.get('data_vector_mode','diagonal'),
              'field_origin':base.get('field_origin',[0.,0.,0.]),
              'measurements':{}}
    for legs in measurements:
        fields = []
        for token in legs:
            if token.startswith('Q'):
                q = int(token[1:])
                fields.append(quantile_field(labels, q, info['fractions'][q], template))
            else:
                # Bispectrum routines modify their input meshes in place.
                fields.append(tracers[token].copy())
        attrs = {**base, 'input_type':'density', 'compensation':False, 'interlaced':False}
        for label in ('a', 'b', 'c'):
            attrs[f'NZ_{label}'], attrs[f'N_gal_{label}'] = cells/volume, cells
        if base['statistic'] == 'pk':
            mode = 'auto' if legs[0] == legs[1] else 'cross'
            result = calculate_power_spectrum_box(fields[0], fields[1] if mode == 'cross' else None,
                                                   mode, attrs, comm)
        else:
            # Always use the general ordered three-field path. Even repeated
            # tokens get independent meshes, since the estimator rescales inputs.
            attrs['tracer_type'] = 'abc'
            result = calculate_bk_sugi_box(*fields, 'cross', attrs, comm)
        if comm.rank == 0:
            result['legs'] = legs
            output['measurements']['__'.join(legs)] = result
        del fields
    if comm.rank == 0:
        os.makedirs(base['output_dir'], exist_ok=True)
        path = os.path.join(base['output_dir'], f'density_split_{base["statistic"]}.npy')
        np.save(path, output)
        logging.info('Density-split results saved to %s', path)
    comm.Barrier()
    return output if comm.rank == 0 else None
