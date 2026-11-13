"""WST input adapters; reuse catalogue painting and distributed field readers."""
import logging
import os
import numpy as np
from mpi4py import MPI
from .wavelet import scattering_options, calculate_scattering, scattering_plan
from .param_helper import validate_config, catalog_check
from .density_processor import read_local_field
from .density_split import _read_mesh_tracers, _normalize_mesh_tracers, _collective_error
from .mesh_generator import get_mesh_pk_survey, pm_painter
from .catalog_processor import catalog_reader
from .math_evaluator import get_compensation_bk_sugi
from .survey_field import _selection


def _configuration(config):
    base = dict(config)
    if 'density_split' in base:
        raise ValueError('wst and density_split are separate estimators')
    if base.get('correlation_mode', 'auto') != 'auto':
        raise ValueError('WST has no power-spectrum cross convention; select independent wst.sources instead')
    base.setdefault('geometry', 'box-like')
    base.setdefault('input_type', 'catalog')
    if base['geometry'] not in ('box-like','survey-like'):
        raise ValueError('geometry must be box-like or survey-like')
    defaults = dict(statistic='wst', correlation_mode='auto', sampler='tsc',
                    interlaced=False, compensation=False, apply_rsd=False,
                    column_names=[], cosmology=None, redshift_box=None, rsd=[0,0,1],
                    z_range=None, comp_weight_plan={'scheme':None,'name_alias':None},
                    alpha_scheme='pypower', output_dir='output')
    for key, value in defaults.items():
        base.setdefault(key, value)
    base.setdefault('normalization_scheme','mesh' if base['input_type']=='density' else 'particle')
    raw = base.pop('wst', {})
    if not isinstance(raw, dict):
        raise TypeError('wst must be a configuration dictionary')
    raw = dict(raw)
    sources = raw.pop('sources', ['data_a'])
    if (not isinstance(sources,(list,tuple)) or not sources or
        any(s not in ('data_a','data_b','data_c') for s in sources) or len(set(sources)) != len(sources)):
        raise ValueError('wst.sources must be distinct data_a, data_b, or data_c names')
    validate_config(base)
    options = scattering_options(raw, base['nmesh'])
    inputs = base.get('fields' if base['input_type']=='density' else 'catalogs', {})
    for name in sources:
        if inputs.get(name) is None:
            raise ValueError(f'Missing WST input {name}')
        if base['input_type']=='catalog':
            # Existing readers validate preprocessed catalogue columns too.
            catalog_check({'data_a':inputs[name], 'randoms_a':inputs.get(name.replace('data_','randoms_'))},
                          base['geometry'], 'auto', 'pk')
    if base['geometry']=='survey-like':
        supported = ('mesh',) if base['input_type']=='density' else ('particle','mixed-mesh')
        if base['normalization_scheme'] not in supported:
            raise ValueError(f'WST survey normalization_scheme must be one of {supported}')
        if base['alpha_scheme'] not in ('pypower','nbodykit'):
            raise ValueError('alpha_scheme must be pypower or nbodykit')
        if base['apply_rsd']:
            raise ValueError('Survey positions must already include radial RSD, as for survey Pl')
    return base, options, sources, inputs


def _mask(source, template, base, origin, comm):
    if source is None:
        return np.ones(template.shape, dtype=bool)
    error = None
    try:
        if isinstance(source,np.ndarray):
            if source.shape != tuple(base['nmesh']) or source.dtype.kind not in 'bifu':
                raise ValueError('wst.mask must be a real global array with shape nmesh')
            values = source[template.slices]
        else:
            values,_ = read_local_field(source,base['nmesh'],base['boxsize'],origin,
                                        template.slices,np.float64)
        if np.any(~np.isfinite(values)) or np.any((values != 0)&(values != 1)):
            raise ValueError('wst.mask must be binary')
        result = values.astype(bool)
    except Exception as exc:
        error = str(exc)
    _collective_error(error,comm)
    return result


def _survey_catalogue(base, options, name, inputs, comm):
    catalogs = {'data_a':inputs[name], 'randoms_a':inputs[name.replace('data_','randoms_')]}
    attrs, field, duplicate = get_mesh_pk_survey(catalogs,'auto',nmesh=base['nmesh'],
        geometry='survey-like',column_names=base['column_names'],boxsize=base['boxsize'],
        sampler=base['sampler'],interlaced=base['interlaced'],z_range=base['z_range'],
        comp_weight_plan=base['comp_weight_plan'],para_cosmo=base['cosmology'],comm=comm,
        normalization_scheme=base['normalization_scheme'],alpha_scheme=base['alpha_scheme'],
        precision=base['precision'],boxcenter=base.get('boxcenter'))
    del duplicate
    # Catalogue painting nodes are at boxcenter + i*cellsize modulo boxsize.
    # A supplied grid mask uses that same index ordering (not file cell centers).
    mask = _mask(options.get('mask'),field,base,attrs['boxcenter'],comm)
    if options.get('mask') is None:
        randoms = catalog_reader(catalogs['randoms_a'],'survey-like',base['column_names'],
            base['z_range'],base['comp_weight_plan'],comm,para_cosmo=base['cosmology'],
            boxsize=base['boxsize'],boxcenter=attrs['boxcenter'],catalog_type='randoms',
            normalization_scheme=base['normalization_scheme'])
        coverage = pm_painter(randoms['Position'],randoms['WEIGHT'],base['nmesh'],base['boxsize'],
            base['sampler'],False,comm,boxcenter=attrs['boxcenter'],precision=base['precision'])
        mask &= coverage.value.real > 0
        del randoms, coverage
    norm = attrs['I_rand'] if base['normalization_scheme']=='particle' else attrs['I_mesh']
    norm = float(np.real(norm))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('WST survey FKP normalization must be finite and positive')
    field[...] /= np.sqrt(norm)
    if base['compensation']:
        compensation = get_compensation_bk_sugi(base['sampler'])
        fft = field.r2c()
        fft.apply(compensation[0][1],kind=compensation[0][2],out=Ellipsis)
        field[...] = fft.c2r().value
        del fft
    field.value.imag = 0
    field.value[~mask] = 0
    metadata = {'input_field':'FKP_over_sqrt_I2','I2':norm,'alpha':attrs['alpha_a'],
                'boxcenter':attrs['boxcenter'], 'assignment_compensation':base['compensation'],
                'footprint':'explicit_mask' if options.get('mask') is not None else 'noninterlaced_random_coverage'}
    return field, mask, metadata


def prepare_wavelet_source(base, options, name, inputs, comm):
    """Read one source at a time. Never retain meshes for unrelated tracers."""
    if base['geometry']=='survey-like' and base['input_type']=='catalog':
        return _survey_catalogue(base,options,name,inputs,comm)
    tracers,specs,mask,field = _read_mesh_tracers(base,options,{name:inputs[name]},comm)
    selection = None
    if base['geometry']=='survey-like':
        window,query,selection = _selection(base,field,comm)
        mask &= (window>0)&(query>0)
        window[~mask] = 0
        del query
    metadata = _normalize_mesh_tracers(tracers,specs,mask,base,comm)[name]
    if selection is not None:
        dvolume = float(np.prod(base['boxsize'])/np.prod(base['nmesh']))
        i2 = dvolume*comm.allreduce(float(np.sum(window.astype(np.float64)**2)),op=MPI.SUM)
        if not np.isfinite(i2) or i2 <= 0:
            raise ValueError('WST survey selection must have a finite positive I2')
        field[...] = field.value*window/np.sqrt(i2)
        metadata.update(input_field='window_times_delta_over_sqrt_I2',I2=i2,selection=selection)
    else:
        metadata['input_field'] = 'overdensity'
    # These are physical real fields; interlacing can leave an imaginary Nyquist
    # residual in full-complex pmesh storage, which is not part of the density.
    field.value.imag = 0
    return field,mask,metadata


def run_wavelet(config, comm=MPI.COMM_WORLD):
    base,options,sources,inputs = _configuration(config)
    if comm.rank==0:
        logging.info('WST per-source FFT plan: %s',scattering_plan(options))
    output = {'statistic':'wst','geometry':base['geometry'],'input_type':base['input_type'],
              'precision':base['precision'],'nmesh':base['nmesh'],'boxsize':base['boxsize'],
              'field_origin':base.get('field_origin',[0.,0.,0.]),
              'wst':{k:v for k,v in options.items() if k!='mask'},
              'sampler':base['sampler'],'interlaced':base['interlaced'],
              'shotnoise_policy':'none_nonlinear_statistic_requires_matched_noise_mocks',
              'measurements':{}}
    for name in sources:
        field,mask,metadata = prepare_wavelet_source(base,options,name,inputs,comm)
        result = calculate_scattering(field,options,mask)
        result['source_metadata'] = metadata
        if comm.rank==0:
            output['measurements'][name] = result
        del field,mask,result
    if comm.rank==0:
        os.makedirs(base['output_dir'],exist_ok=True)
        path = os.path.join(base['output_dir'],'wst.npy')
        np.save(path,output)
        logging.info('WST saved to %s',path)
    comm.Barrier()
    return output if comm.rank==0 else None
