"""Survey field window normalization, physical LOS, and weighted quantiles."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import h5py
from mpi4py import MPI
from cosmonpc import run_stats
from cosmonpc.survey_field import prepare_survey_fields

COMM=MPI.COMM_WORLD
N,L=8,80.
ORIGIN=np.array([10.,-40.,100.])
GRID=np.indices((N,)*3)
DELTA=.3+.2*np.cos(2*np.pi*GRID[0]/N)+.15*np.sin(2*np.pi*GRID[2]/N)
WEIGHT=(GRID[0]+1)/N * (GRID[1]%4 != 0)


def config(tmp,**extra):
    c=dict(input_type='density',geometry='survey-like',statistic='pk',fields={'data_a':DELTA},
           survey_weight=WEIGHT,nmesh=[N]*3,boxsize=[L]*3,field_origin=ORIGIN.tolist(),
           k_min=.04,k_max=.14,k_bins=1,poles=[0,2],angu_config=[0,0,0],
           sampler='ngp',output_dir=str(tmp),precision='double')
    c.update(extra)
    return c


def write(path,data,corner):
    with h5py.File(path,'w') as hf:
        hf.create_dataset('/scalars/field',data=data)
        for i in range(3):
            hf.attrs[f'/info/scalars/N{i}']=data.shape[i]
            hf.attrs[f'/info/scalars/L{i}']=data.shape[i]*L/N
            hf.attrs[f'/info/scalars/corner{i}']=corner[i]
        hf.attrs['/info/scalars/rank']=1
        hf.attrs['/info/scalars/time']=1.


def power_oracle(a,b,window,c):
    coords=np.moveaxis(ORIGIN[:,None,None,None]+(GRID+.5)*L/N,0,-1).reshape(-1,3)
    xhat=coords/np.linalg.norm(coords,axis=1)[:,None]
    indices=np.moveaxis(GRID,0,-1).reshape(-1,3)*L/N
    k=2*np.pi*np.fft.fftfreq(N,d=L/N)
    grid=np.stack(np.meshgrid(k,k,k,indexing='ij'),axis=-1)
    norm=np.linalg.norm(grid,axis=-1)
    selected=(norm>=c['k_min'])&(norm<c['k_max'])
    vectors=grid[selected]
    hats=vectors/np.linalg.norm(vectors,axis=1)[:,None]
    phases=np.exp(-1j*(vectors@indices.T))
    f0=(L/N)**3*(phases@(a*window).ravel())
    normalization=(L/N)**3*np.sum(window**2)
    result={}
    for ell in c['poles']:
        mu=hats@xhat.T
        angular=np.polynomial.legendre.legval(mu,[0]*ell+[1])
        gl=(L/N)**3*np.sum(phases*angular*(b*window).ravel()[None,:],axis=1)
        result[f'P{ell}']=(2*ell+1)*np.real(f0*np.conj(gl)).mean()/normalization
    return result


class SurveyFieldTests(unittest.TestCase):
    def test_plain_survey_field_power_wide_angle_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            for precision,tol in [('double',1e-10),('single',3e-6)]:
                c=config(tmp,precision=precision)
                result=run_stats(c)
                if COMM.rank==0:
                    oracle=power_oracle(DELTA,DELTA,WEIGHT,c)
                    measured=result['measurements']['data_a__data_a']
                    for key,value in oracle.items():
                        np.testing.assert_allclose(measured[key],value,rtol=tol)
                    self.assertEqual(result['shotnoise_policy'],'continuous_fields_no_Poisson_subtraction')
                    self.assertAlmostEqual(result['selection']['I2'],(L/N)**3*np.sum(WEIGHT**2),places=8)
                    self.assertAlmostEqual(result['source_metadata']['data_a']['input_mean_covered'],DELTA[WEIGHT>0].mean())

    def test_split_weighted_power_and_selection_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=config(tmp,density_split={'nquantiles':3,'smoothing_radius':5.,'measurements':[['Q0','data_a'],['Q0','Q0']]})
            tracers,labels,info,window,selection,_,_,_=prepare_survey_fields(c)
            gathered=COMM.gather((tracers['data_a'].slices,labels.copy()),root=0)
            result=run_stats(c)
            scaled=run_stats({**c,'survey_weight':WEIGHT*.5})
            if COMM.rank==0:
                full_labels=np.full((N,)*3,-1)
                for slices,local in gathered:
                    full_labels[slices]=local
                q=np.where(WEIGHT>0,(full_labels==0)/info['fractions'][0]-1,0.)
                for legs,b in [('Q0__data_a',DELTA),('Q0__Q0',q)]:
                    oracle=power_oracle(q,b,WEIGHT,c)
                    for key,value in oracle.items():
                        np.testing.assert_allclose(result['measurements'][legs][key],value,rtol=1e-10)
                        np.testing.assert_allclose(result['measurements'][legs][key],scaled['measurements'][legs][key],rtol=1e-12)

    def test_incomplete_hdf5_tiles_weight_mask_and_random_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tile=Path(tmp)/'tile.h5'
            maskfile=Path(tmp)/'mask.h5'
            write(tile,DELTA[:4],ORIGIN)
            write(maskfile,WEIGHT,ORIGIN)
            coverage=np.zeros_like(WEIGHT)
            coverage[:4]=1
            positions=np.moveaxis(ORIGIN[:,None,None,None]+(GRID+.5)*L/N,0,-1).reshape(-1,3)
            points=np.ones(N**3,dtype=[('Position','f8',(3,)),('WEIGHT','f8'),('WEIGHT_FKP','f8')])
            points['Position']=positions
            points['WEIGHT']=WEIGHT.ravel()
            local=np.array_split(points,COMM.size)[COMM.rank]
            for statistic in ['pk','bk_sugi']:
                settings={'nquantiles':3,'smoothing_radius':5.,'measurements':[['Q0','data_a']] if statistic=='pk' else [['Q0','data_a','data_a']]}
                c=config(tmp,statistic=statistic,density_split=settings,shotnoise_mode='both',data_vector_mode='full')
                tiled=run_stats({**c,'fields':{'data_a':{'tiles':[tile],'missing_tiles':'zero'}},'survey_weight':str(maskfile)})
                reference=run_stats({**c,'survey_weight':WEIGHT*coverage})
                random=run_stats({**c,'survey_weight':None,'survey_randoms':local,
                                 'fields':{'data_a':{'tiles':[tile],'missing_tiles':'zero'}}})
                if COMM.rank==0:
                    key='P0' if statistic=='pk' else 'B_sugi'
                    name=next(iter(tiled['measurements']))
                    np.testing.assert_allclose(tiled['measurements'][name][key],reference['measurements'][name][key],rtol=1e-12)
                    np.testing.assert_allclose(random['measurements'][name][key],reference['measurements'][name][key],rtol=1e-12)
                    if statistic=='bk_sugi':
                        np.testing.assert_array_equal(tiled['measurements'][name]['Shot_noise'],0.)
                        np.testing.assert_array_equal(tiled['measurements'][name]['SN_terms']['SN3_FFT'],0.)

    def test_bispectrum_wide_angle_oracle(self):
        k=2*np.pi*np.fft.fftfreq(N,d=L/N)
        kgrid=np.stack(np.meshgrid(k,k,k,indexing='ij'))
        knorm=np.sqrt(np.sum(kgrid*kgrid,axis=0))
        khat=np.divide(kgrid,knorm[None,...],out=np.zeros_like(kgrid),where=knorm[None,...]!=0)
        positions=ORIGIN[:,None,None,None]+(GRID+.5)*L/N
        xhat=positions/np.sqrt(np.sum(positions**2,axis=0))[None,...]
        shell=(knorm>=.04)&(knorm<.14)
        signal=WEIGHT*DELTA
        transformed=np.fft.fftn(signal)
        second=np.fft.ifftn(transformed*shell)
        def y2(m,direction):
            x,y,z=direction
            if m==0:
                return .5*(3*z*z-1)
            if m<0:
                return (-1)**m*np.conj(y2(-m,direction))
            return (-np.sqrt(1.5)*z*(x+1j*y) if m==1 else np.sqrt(3/8)*(x+1j*y)**2)
        normalization=(L/N)**3*np.sum(WEIGHT**3)
        with tempfile.TemporaryDirectory() as tmp:
            for angular in [[0,0,0],[2,0,2]]:
                if angular==[0,0,0]:
                    contraction=np.mean(second*second*signal)
                else:
                    contraction=0j
                    for m in range(-2,3):
                        first=np.fft.ifftn(transformed*shell*y2(m,khat))
                        contraction+=5*(-1)**m*np.mean(first*second*signal*y2(-m,xhat))
                expected=L**9*contraction/(normalization*shell.sum()**2)
                for precision,tolerance in [('double',2e-11),('single',3e-6)]:
                    result=run_stats(config(tmp,statistic='bk_sugi',angu_config=angular,
                                           data_vector_mode='full',precision=precision,shotnoise_mode='both'))
                    if COMM.rank==0:
                        measured=result['measurements']['data_a__data_a__data_a']
                        np.testing.assert_allclose(measured['B_sugi'],expected,rtol=tolerance,atol=1e-6)
                        np.testing.assert_array_equal(measured['Shot_noise'],0.)

    def test_invalid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for source in [np.full((N,)*3,-.1),np.full((N,)*3,1.1),np.full((N,)*3,np.nan),np.zeros((N,)*3),np.ones((2,2,2))]:
                with self.assertRaises(ValueError):
                    run_stats(config(tmp,survey_weight=source))
            with self.assertRaisesRegex(ValueError,'exactly one'):
                run_stats(config(tmp,survey_randoms=np.ones((3,3))))
            with self.assertRaisesRegex(ValueError,'exactly one'):
                run_stats(config(tmp,survey_weight=None))
