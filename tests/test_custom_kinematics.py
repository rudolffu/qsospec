import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from qsospec import Spectrum
from qsospec.custom_kinematics import *
from qsospec.fitting.complexes import GenericComplexContext,fit_generic_complex


def recipe(shift=-650,width=1800,tied=False):
 cs=[]
 for i,(v,w) in enumerate([(0,300),(shift,width)]):
  for lid,tag in [('oiii_5008','5008'),('oiii_4960','4960')]:
   d=dict(id=f'OIII{tag}_{i}',line_ids=[lid],role='narrow',velocity_group=f'v{i}',width_group='w' if tied else f'w{i}',velocity_bounds_kms=[-2000,2000],fwhm_bands_kms=[[70,3500]])
   if tag=='4960':d.update(fixed_ratio_to=f'OIII5008_{i}',fixed_ratio=2.98)
   cs.append(d)
 return dict(id='test_custom',window=[4800,5150],components=cs,continuum_mode='residual_linear')


def synthetic(sp,shift,width,tied=False):
 r=custom_kinematics(sp);ctx=GenericComplexContext(r,[c.id for c in r.components],50)
 theta=ctx.initial.copy()
 for i in (0,1):
  theta[ctx.index[f'OIII5008_{i}.flux']]=80 if i==0 else 120
  theta[ctx.index[f'v{i}.velocity_kms']]=0 if i==0 else shift
  theta[ctx.index['w.fwhm_kms' if tied else f'w{i}.fwhm_kms']]=width if tied or i else 300
 wave=np.linspace(4800,5150,650);err=np.full_like(wave,.03)
 s=Spectrum.from_arrays(wave,ctx.model(theta,wave),err=err,wave_frame='rest',flux_unit='relative')
 cont=SimpleNamespace(model=np.zeros_like(wave),param_values={},component_models={},wave_rest=wave)
 return s,cont,ctx,theta

@pytest.mark.parametrize('shift,width,tied',[(-650,1800,False),(650,1800,False),(30,2000,False),(-900,700,True)])
def test_profiles(shift,width,tied):
 sp=recipe(shift,width,tied);s,c,ctx,theta=synthetic(sp,shift,width,tied)
 r=fit_custom_kinematics(s,c,sp,n_starts=4,initial_values=dict(zip(ctx.names,theta)))
 assert r.success and r.chi2<1e-8
 rows=component_measurements(r,sp)
 for i in (0,1):
  a=next(x for x in rows if x['component']==f'OIII5008_{i}')
  b=next(x for x in rows if x['component']==f'OIII4960_{i}')
  assert a['flux']/b['flux']==pytest.approx(2.98)
 assert len(r.metadata['multistart']['starts'])==4


def test_quantiles_independent_integration():
 rows=[dict(line_id='oiii_5008',flux=2,velocity_kms=-500,fwhm_kms=1500),dict(line_id='oiii_5008',flux=1,velocity_kms=200,fwhm_kms=300)]
 q=mixture_kinematics(rows);wave=np.linspace(.95,1.05,200001);pdf=np.zeros_like(wave)
 for row in rows:
  mu=np.exp(row['velocity_kms']/C);sig=mu*row['fwhm_kms']/C/2.354820045
  pdf+=row['flux']*np.exp(-.5*((wave-mu)/sig)**2)/sig
 cdf=cumulative_trapezoid(pdf,wave,initial=0);cdf/=cdf[-1]
 for p in PROBABILITIES:assert q[f'v{int(p*100):02d}_kms']==pytest.approx(C*np.log(np.interp(p,cdf,wave)),abs=.02)
 assert mixture_kinematics(rows+[dict(line_id='oiii_4960',flux=1e9,velocity_kms=9999,fwhm_kms=9)])==q
 assert mixture_kinematics(rows[::-1])==q
 shifted=[dict(r,velocity_kms=r['velocity_kms']+400) for r in rows]
 assert mixture_kinematics(shifted)['w80_kms']==pytest.approx(q['w80_kms'],abs=1e-6)
 assert mixture_kinematics([dict(rows[0],flux=-1)])['status']=='invalid'


def test_roundtrip_and_defaults():
 sp=recipe();assert asdict(custom_kinematics(sp))==asdict(custom_kinematics(json.loads(json.dumps(sp))))
 s,c,ctx,t=synthetic(sp,-650,1800)
 a=fit_generic_complex(s,c,custom_kinematics(sp));b=fit_generic_complex(s,c,custom_kinematics(sp),n_starts=1)
 assert a.chi2==b.chi2
 with pytest.raises(ValueError):fit_custom_kinematics(s,c,sp,initial_values={'no_such_parameter':4})


def test_blended_oii():
 cs=[dict(id=k,line_ids=[l],role='narrow',kinematic_group='oii',velocity_bounds_kms=[-1000,1000],fwhm_bands_kms=[[70,1200]]) for k,l in [('a','oii_3727'),('b','oii_3730')]]
 sp=dict(id='oii',window=[3700,3760],components=cs,continuum_mode='linear');r=custom_kinematics(sp);ctx=GenericComplexContext(r,['a','b'],20)
 theta=ctx.initial.copy()
 for key,val in {'a.flux':15,'b.flux':25,'oii.velocity_kms':160,'oii.fwhm_kms':350}.items():theta[ctx.index[key]]=val
 w=np.linspace(3700,3760,400);s=Spectrum.from_arrays(w,ctx.model(theta,w),err=np.ones_like(w)*.02,wave_frame='rest',flux_unit='relative')
 c=SimpleNamespace(model=np.zeros_like(w),param_values={},component_models={},wave_rest=w)
 fit=fit_custom_kinematics(s,c,sp,n_starts=4)
 assert fit.param_values['oii.velocity_kms']==pytest.approx(160,abs=.1)

@pytest.mark.parametrize('line,velocity',[('halpha',-1300),('hbeta',950),('mgii_2796',-450)])
def test_independent_broad_velocities(line,velocity):
 from qsospec import lines
 center=lines.get(line).vacuum_wavelength
 sp=dict(id='broad_independent',window=[center*.95,center*1.05],continuum_mode='residual_linear',components=[dict(id='b',line_ids=[line],role='broad',velocity_bounds_kms=[-6000,6000],fwhm_bands_kms=[[900,20000]])])
 recipe=custom_kinematics(sp);ctx=GenericComplexContext(recipe,['b'],200);theta=ctx.initial.copy()
 for k,v in {'b.flux':200,'b.velocity_kms':velocity,'b.fwhm_kms':2800}.items():theta[ctx.index[k]]=v
 w=np.linspace(*sp['window'],400);s=Spectrum.from_arrays(w,ctx.model(theta,w),err=np.ones_like(w)*.02,wave_frame='rest',flux_unit='relative')
 c=SimpleNamespace(model=np.zeros_like(w),param_values={},component_models={},wave_rest=w)
 r=fit_custom_kinematics(s,c,sp,n_starts=4)
 assert r.param_values['b.velocity_kms']==pytest.approx(velocity,abs=.1)


def test_peer_selection_requires_width_release_evidence():
 from qsospec.custom_kinematics import select_nested_candidates
 def item(n,tied,bic):
  sp=recipe(tied=tied)
  if n==1:sp['components']=sp['components'][:2]
  ctx=GenericComplexContext(custom_kinematics(sp),[c['id'] for c in sp['components']],100)
  params=dict(zip(ctx.names,ctx.initial));params.update({k:100. for k in params if k.endswith('.flux')})
  return sp,SimpleNamespace(param_values=params,covariance=np.eye(len(params)),success=True,bic=bic)
 c={'x_one':item(1,False,200),'x_tied':item(2,True,100),'x_free':item(2,False,90)}
 choice,audit=select_nested_candidates(c,'x',peer_widths=True)
 assert choice=='x_tied'
 c['x_free'][1].bic=70
 assert select_nested_candidates(c,'x',peer_widths=True)[0]=='x_free'


def test_summed_doublet_significance():
 from qsospec.custom_kinematics import select_nested_candidates
 def comp(k,line,g):return dict(id=k,line_ids=[line],velocity_group=g,width_group=g,role='broad',velocity_bounds_kms=[-6000,6000],fwhm_bands_kms=[[900,20000]])
 a={'components':[comp('a','mgii_2796','g')]};b={'components':a['components']+[comp('b','mgii_2796','h'),comp('c','mgii_2804','h')]}
 ar=SimpleNamespace(param_values={'a.flux':100},covariance=np.eye(1),success=True,bic=200)
 br=SimpleNamespace(param_values={'a.flux':100,'b.flux':10,'c.flux':10},covariance=np.array([[1,0,0],[0,100,-99],[0,-99,100]]),success=True,bic=100)
 assert select_nested_candidates({'m1':(a,ar),'m2':(b,br)},'m')[0]=='m2'


def test_warm_continuum_reoptimizes_noise():
 from qsospec import GlobalContinuumConfig,PolynomialContinuumConfig,BalmerPseudoContinuumConfig,RegionalIronConfig
 from qsospec.fitting.global_fit import fit_global_continuum
 from qsospec.custom_kinematics import refit_custom_continuum
 from dataclasses import replace
 w=np.linspace(2200,7000,800);flux=3*(w/3000)**-1.2;err=np.ones_like(w)*.05
 s=Spectrum.from_arrays(w,flux,err=err,wave_frame='rest',flux_unit='relative')
 cfg=GlobalContinuumConfig(uv_iron=None,optical_iron=None,regional_iron=RegionalIronConfig(enabled=False),polynomial=PolynomialContinuumConfig(mode="off"),balmer_pseudocontinuum=BalmerPseudoContinuumConfig(enabled=False))
 base=fit_global_continuum(s,cfg,compute_covariance=False)
 refit=refit_custom_continuum(replace(s,flux=flux+.1),base,cfg)
 assert refit.success
 assert np.mean(refit.model-base.model)>.05
