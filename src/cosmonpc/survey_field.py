"""Gridded survey fields and density splits with explicit survey selection.

The continuous weighted fields are fed into the existing survey angular
estimators. No particle catalogue, integral-constraint centering, or Poisson
contractions are invented for a supplied DM field.
"""
import logging
import os
import numpy as np
from mpi4py import MPI
from .density_split import (_configuration, _read_mesh_tracers, _normalize_mesh_tracers,
                            _collective_error)
from .density_processor import read_local_field
from .catalog_processor import catalog_reader
from .mesh_generator import pm_painter
from .quantiles import assign_quantiles, gaussian_smooth, global_cell_ids, quantile_field
from .clustering_estimator import calculate_power_spectrum_survey, calculate_bk_sugi_survey


def _selection(base, template, comm):
    origin = np.asarray(base.get('field_origin',[0.,0.,0.]),dtype=float)
    box = np.asarray(base['boxsize'],dtype=float)
    error = None
    if base.get('survey_weight') is not None:
        try:
            source = base['survey_weight']
            if isinstance(source,np.ndarray):
                if source.shape != tuple(base['nmesh']) or source.dtype.kind not in 'bifu':
                    raise ValueError('survey_weight must be a real array with shape nmesh')
                weights = np.array(source[template.slices],dtype=np.float64,copy=True)
            else:
                weights,_ = read_local_field(source,base['nmesh'],base['boxsize'],origin,
                                            template.slices,np.float64)
            if np.any(~np.isfinite(weights)) or np.any((weights<0)|(weights>1)):
                raise ValueError('survey_weight must be finite and between 0 and 1')
        except Exception as exc:
            error = str(exc)
        _collective_error(error,comm)
        return weights,weights.copy(),{'source':'survey_weight','query_weighting':'survey_weight'}

    source = base['survey_randoms']
    raw = source.get('array') if isinstance(source,dict) and 'array' in source else source
    if isinstance(raw,np.ndarray):
        if raw.dtype.names is None:
            positions = raw
            completeness = np.ones(len(raw))
            fkp = np.ones(len(raw))
        else:
            positions = raw['Position']
            completeness = raw['WEIGHT'] if 'WEIGHT' in raw.dtype.names else np.ones(len(raw))
            fkp = raw['WEIGHT_FKP'] if 'WEIGHT_FKP' in raw.dtype.names else np.ones(len(raw))
    else:
        data = catalog_reader(source,'survey-like',base['column_names'],base['z_range'],
            base['comp_weight_plan'],comm,para_cosmo=base['cosmology'],boxsize=box,
            boxcenter=origin+.5*box,catalog_type='randoms',normalization_scheme='mesh')
        positions,completeness,fkp = data['Position'],data['WEIGHT'],data['WEIGHT_FKP']
    if positions.ndim != 2 or positions.shape[1] != 3 or np.any(~np.isfinite(positions)):
        error = 'survey_randoms positions must have shape (N,3) and be finite'
    elif np.any(~np.isfinite(completeness)) or np.any(completeness<0) or np.any(~np.isfinite(fkp)) or np.any(fkp<0):
        error = 'survey_randoms weights must be finite and nonnegative'
    _collective_error(error,comm)
    inside = np.all((positions>=origin)&(positions<origin+box),axis=1)
    # File values live at cell centers. Shift the painting nodes accordingly.
    params = dict(Nmesh=base['nmesh'],boxsize=box,sampler=base['sampler'],interlaced=False,
                  comm=comm,boxcenter=origin+.5*box/np.asarray(base['nmesh']),precision=base['precision'])
    query = pm_painter(positions[inside],completeness[inside],**params).value.real.copy()
    weights = pm_painter(positions[inside],(completeness*fkp)[inside],**params).value.real.copy()
    qmax = comm.allreduce(float(query.max()) if query.size else 0.,op=MPI.MAX)
    wmax = comm.allreduce(float(weights.max()) if weights.size else 0.,op=MPI.MAX)
    if qmax <= 0 or wmax <= 0:
        raise ValueError('survey_randoms have no positive selection weight inside the field grid')
    query /= qmax
    weights /= wmax
    return weights,query,{'source':'survey_randoms','painting_sampler':base['sampler'],
                         'normalization':'divide_global_max','weight_peak':wmax,'query_peak':qmax,
                         'query_weighting':'completeness_without_FKP',
                         'random_count_in_grid':comm.allreduce(int(inside.sum()),op=MPI.SUM)}


def _ordinary_legs(config):
    if config['statistic']=='pk':
        return ['data_a','data_a'] if config.get('correlation_mode','auto')=='auto' else ['data_a','data_b']
    pattern = config.get('tracer_type','aaa')
    if pattern not in ('aaa','aab','aba','abb','abc'):
        raise ValueError('Invalid tracer_type for survey field bispectrum')
    return ['data_'+letter for letter in pattern]


def prepare_survey_fields(config, comm=MPI.COMM_WORLD):
    split = 'density_split' in config
    settings = dict(config)
    if not split:
        settings['density_split'] = {'measurements':[_ordinary_legs(settings)]}
    base,options,sources,measurements = _configuration(settings,require_quantile=split)
    if base['input_type']!='density' or base['geometry']!='survey-like':
        raise ValueError('Survey field preparation requires density inputs and survey-like geometry')
    if base['statistic']=='pk' and base['high_order_mode']!='default' and any(len(set(legs))>1 for legs in measurements):
        raise ValueError('Survey high_order_mode fast/compare requires auto measurements only')
    tracers,specs,mask,template = _read_mesh_tracers(base,options,sources,comm)
    window,queries,selection_info = _selection(base,template,comm)
    mask &= (window>0)&(queries>0)
    window[~mask] = 0
    queries[~mask] = 0
    metadata = _normalize_mesh_tracers(tracers,specs,mask,base,comm)
    window64 = window.astype(np.float64)
    dvolume = float(np.prod(base['boxsize'])/np.prod(base['nmesh']))
    integrals = {f'I{power}':dvolume*comm.allreduce(float(np.sum(window64**power)),op=MPI.SUM)
                 for power in (1,2,3)}
    if not np.isfinite(integrals['I2']) or integrals['I2']<=0 or not np.isfinite(integrals['I3']) or integrals['I3']<=0:
        raise ValueError('Survey selection has no finite positive effective normalization')
    selection_info.update(integrals)
    selection_info['covered_cells'] = comm.allreduce(int(mask.sum()),op=MPI.SUM)
    labels,info = None,None
    if split:
        environment = gaussian_smooth(tracers[options['source']],options['smoothing_radius'],mask)
        ids = global_cell_ids(template.shape,template.start,base['nmesh'])
        labels,info = assign_quantiles(environment,mask,ids,options['nquantiles'],comm,
                                      options['tie_seed'],weights=queries)
        info.update(source=options['source'],smoothing_radius=options['smoothing_radius'],
                    smoothing_filter='gaussian',mask_smoothing='normalized_convolution',
                    geometry='survey-like',query_weighting=selection_info['query_weighting'])
    return tracers,labels,info,window,selection_info,metadata,base,measurements


def run_survey_field_statistics(config,comm=MPI.COMM_WORLD):
    tracers,labels,info,window,selection,metadata,base,measurements = prepare_survey_fields(config,comm)
    template = next(iter(tracers.values()))
    origin = np.asarray(base.get('field_origin',[0.,0.,0.]),dtype=float)
    output = {'statistic':base['statistic'],'geometry':'survey-like','input_type':'density',
              'precision':base['precision'],'selection':selection,'quantiles':info,'source_metadata':metadata,
              'field_origin':origin,'boxsize':base['boxsize'],'nmesh':base['nmesh'],
              'poles':base['poles'],'angu_config':base['angu_config'],
              'data_vector_mode':base.get('data_vector_mode','diagonal'),
              'k_min':base['k_min'],'k_max':base['k_max'],'k_bins':base['k_bins'],
              'shotnoise_policy':'continuous_fields_no_Poisson_subtraction','measurements':{}}
    for legs in measurements:
        fields = []
        for token in legs:
            if token.startswith('Q'):
                q=int(token[1:])
                field=quantile_field(labels,q,info['fractions'][q],template)
            else:
                field=tracers[token].copy()
            field[...] = field.value*window
            fields.append(field)
        attrs={**base,'input_type':'density','compensation':False,'interlaced':False,
               'boxcenter':origin+.5*np.asarray(base['boxsize']),'field_origin':origin,
               'normalization_scheme':'mesh','I_mesh':selection['I2'],'N0':0.,'I_norm':selection['I3']}
        if base['statistic']=='pk':
            mode='auto' if legs[0]==legs[1] else 'cross'
            result=calculate_power_spectrum_survey(attrs,fields[0],fields[1] if mode=='cross' else fields[0],mode,comm)
        else:
            # All three legs are continuous fields, even when names repeat.
            attrs['tracer_type']='abc'
            result=calculate_bk_sugi_survey(*fields,'cross',attrs,comm,catalogs={})
        if comm.rank==0:
            result['legs']=legs
            output['measurements']['__'.join(legs)]=result
    if comm.rank==0:
        os.makedirs(base['output_dir'],exist_ok=True)
        prefix='density_split' if info is not None else 'survey_field'
        path=os.path.join(base['output_dir'],f'{prefix}_{base["statistic"]}.npy')
        np.save(path,output)
        logging.info('Gridded survey statistics saved to %s',path)
    comm.Barrier()
    return output if comm.rank==0 else None
