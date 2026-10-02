from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from qsospec import Spectrum,HbetaComplexConfig,fit_hbeta_complex
from qsospec.resolution import SpectralResolution
from qsospec.fitting.adaptive_oiii import adaptive_recipe,residual_diagnostics,canonicalize,C
from qsospec.fitting.complexes import GenericComplexContext,_profile
from qsospec.fitting.line_lsf import build_line_lsf,pixel_edges
from qsospec.oiii_measurements import profile_distribution,propagated_errors


def synthetic(components, *, error=.03, noise=True, resolution=None, z=0., mask=None):
    cfg=HbetaComplexConfig(broad_fwhm_bands_kms=((900.,6000.),))
    recipe=adaptive_recipe(cfg,len(components));ctx=GenericComplexContext(recipe,[c.id for c in recipe.components],100)
    p=dict(zip(ctx.names,ctx.initial))
    p.update({'Hb_broad1.flux':50.,'Hb_broad1.fwhm_kms':2300.,'Hb_narrow.flux':15.,
              'Hb_narrow.velocity_kms':350.,'Hb_narrow.fwhm_kms':200.,'continuum.slope':.0003})
    for (flux,velocity,width),label,group in zip(components,('core','wing','wing2'),('narrow','wing','wing2')):
        p.update({f'OIII5007_{label}.flux':flux,f'{group}.velocity_kms':velocity,f'{group}.fwhm_kms':width})
    w=np.linspace(4500,5300,1100);e=np.full_like(w,error)
    s=Spectrum.from_arrays(w,np.zeros_like(w),err=e,wave_frame='rest',flux_unit='relative',z=z,mask=mask,resolution=resolution)
    if resolution:ctx.line_lsf,_=build_line_lsf(s,cfg.window)
    flux=2+ctx.model(np.array([p[k] for k in ctx.names]),w)
    if noise:flux+=np.random.default_rng(130).normal(0,e)
    s=replace(s,flux=flux);cont=SimpleNamespace(model=np.full_like(w,2.),wave_rest=w,param_values={})
    return s,cont,cfg


@pytest.mark.parametrize('components',[
    [(70,0,320),(60,0,1300)], # centered base
    [(70,0,850),(65,-350,1450)], # width contrast below two
    [(70,0,300),(65,600,1300)], # red component
])
def test_supported_profiles(components):
    s,c,cfg=synthetic(components)
    fit=fit_hbeta_complex(s,c,cfg)
    assert fit.success and fit.selected_model=='wing'
    audit=fit.metadata['candidate_selection']
    assert audit[1]['accepted'] and audit[1]['added_component_flux_snr']>=5
    assert set(audit[1]['checks'])=={'convergence','bic_improvement','added_component_snr'}
    assert fit.param_values['Hb_narrow.velocity_kms']==pytest.approx(350,abs=8)
    assert fit.param_values['narrow.velocity_kms']==pytest.approx(0,abs=20)
    assert fit.metadata['input_redshift']==s.z
    assert len(audit[0]['multistart']['starts'])==24
    assert audit[0]['model_definition']['components'][:2]==audit[1]['model_definition']['components'][:2]
    assert fit.metrics['oiii_doublet_full_observed_core_relative_fraction_beyond_500_kms']==fit.metrics['oiii_5008_full_observed_core_relative_fraction_beyond_500_kms']


def test_three_component_trigger_and_restriction():
    s,c,cfg=synthetic([(65,-100,220),(90,-800,2300),(45,550,450)],error=.02)
    fit=fit_hbeta_complex(s,c,cfg)
    assert fit.selected_model=='three_component'
    audit=fit.metadata['candidate_selection']
    assert len(audit[1]['multistart']['starts'])==24
    assert audit[1]['residuals']['coherent'] and audit[2]['checks']['improved_both_lines']
    assert fit.param_values['wing.fwhm_kms']<fit.param_values['wing2.fwhm_kms']
    restricted=fit_hbeta_complex(s,c,replace(cfg,fit_oiii_wings=False))
    assert restricted.selected_model=='core' and len(restricted.metadata['candidate_selection'])==1
    assert not restricted.metadata['profile_adequate']


@pytest.mark.parametrize('amplitude',[0.,.03])
def test_noise_and_weak_components(amplitude):
    s,c,cfg=synthetic([(amplitude,0,300)],error=.2)
    fit=fit_hbeta_complex(s,c,cfg)
    assert fit.selected_model=='core'
    assert not fit.metadata['oiii_core_reference']['reliable']
    assert np.isnan(fit.metrics['oiii_5008_full_observed_core_relative_v50_kms'])
    assert 'oiii_5008_full_observed_input_frame_v50_kms' in fit.metrics


def test_residual_gaps_and_counterpart():
    w=np.linspace(4900,5050,1501);r=np.zeros_like(w);m=np.ones_like(w,bool)
    a=np.searchsorted(w,5008.24);b=int(np.argmin(abs(w-w[a]*4960.30/5008.24)))
    r[a:a+3]=4;r[b:b+3]=2.5
    assert residual_diagnostics(w,r,m,0)['coherent']
    m[a+1]=False
    assert not residual_diagnostics(w,r,m,0)['coherent']
    m[:]=True;r[b:b+3]=-2.5
    assert not residual_diagnostics(w,r,m,0)['coherent']


@pytest.mark.parametrize('resolution',[
    None,SpectralResolution(),SpectralResolution('banded_matrix',banded_matrix=np.eye(2)),
    SpectralResolution('sigma_lambda',values=np.array([-1.])),
    SpectralResolution('sigma_lambda',values=np.array([1.,1.]),wavelength=np.array([4950.,5030.])),
])
def test_resolution_fallback(resolution):
    s,c,cfg=synthetic([(70,0,300)],resolution=resolution,noise=False)
    op,status=build_line_lsf(s,cfg.window)
    assert op is None and status['status']=='observed_profile' and status['reason']


@pytest.mark.parametrize('mode,value,z',[('sigma_lambda',2.,.7),('resolving_power',1800.,0.),('sigma_kms',90.,.3)])
def test_lsf_conservation_derivative_and_frame(mode,value,z):
    res=SpectralResolution(mode,values=np.array([value]))
    s,c,cfg=synthetic([(70,0,300)],resolution=res,z=z,noise=False)
    op,status=build_line_lsf(s,cfg.window);assert op is not None
    w=s.wave_rest;edges=pixel_edges(w);center=5008.24
    basis,dv,dw=op.profile(w,center,80.,300.,'gaussian',integrated=True)
    assert np.sum(basis*np.diff(edges))==pytest.approx(1.,abs=2e-10)
    assert np.allclose(dv,(op.profile(w,center,80.01,300.,'gaussian',integrated=True)[0]-op.profile(w,center,79.99,300.,'gaussian',integrated=True)[0])/.02,atol=1e-8)
    assert np.allclose(dw,(op.profile(w,center,80.,300.01,'gaussian',integrated=True)[0]-op.profile(w,center,80.,299.99,'gaussian',integrated=True)[0])/.02,atol=1e-8)
    if mode=='sigma_lambda':
        from scipy.special import ndtr
        mu=center*np.exp(80/C);sig=np.hypot(mu*300/C/2.354820045,value/(1+z))
        analytic=np.diff(ndtr((edges-mu)/sig))/np.diff(edges)
        assert np.allclose(basis,analytic,atol=1e-8)
    subset=np.arange(0,len(w),3)
    assert np.allclose(op.profile(w[subset],center,80.,300.,'gaussian',integrated=True)[0],basis[subset])


def test_varying_lsf_and_observed_intrinsic_fit():
    res=SpectralResolution('sigma_lambda',values=np.array([1.,3.]),wavelength=np.array([4400.,5400.]))
    s,c,cfg=synthetic([(90,60,150)],resolution=res,noise=True)
    fit=fit_hbeta_complex(s,c,replace(cfg,fit_oiii_wings=False))
    assert fit.param_values['narrow.fwhm_kms']==pytest.approx(150,abs=8)
    assert fit.metadata['line_lsf']['status']=='forward_modeled'
    assert fit.metrics['oiii_5008_full_observed_input_frame_w80_kms']>fit.metrics['oiii_5008_full_intrinsic_input_frame_w80_kms']
    assert fit.metadata['peak_model']['components'][0]['line_lsf']['z']==s.z
    descriptor = fit.metadata['line_lsf']['descriptor']
    reference = fit.param_values['narrow.velocity_kms']
    a = profile_distribution(fit.param_values, 1, reference=reference, lsf=descriptor)
    b = profile_distribution(fit.param_values, 1, reference=reference, lsf=descriptor, line_id='oiii_4960')
    assert a['fraction_beyond_500_kms'] != b['fraction_beyond_500_kms']
    assert fit.metrics['oiii_doublet_full_observed_core_relative_fraction_beyond_500_kms'] == (2.98*a['fraction_beyond_500_kms']+b['fraction_beyond_500_kms'])/3.98


def test_covariance_parameter_draws():
    from copy import deepcopy
    s,c,cfg=synthetic([(70,0,320),(60,0,1300)])
    fit=fit_hbeta_complex(s,c,cfg)
    names=[k for k in fit.param_values if k.startswith(('OIII5007','narrow.','wing.'))]
    fn=lambda p:profile_distribution(p,2,reference=p['narrow.velocity_kms'])
    errors=propagated_errors(fit,fn,names)
    order=list(fit.param_values);ix=[order.index(k) for k in names]
    cov=fit.covariance[np.ix_(ix,ix)];mean=[fit.param_values[k] for k in names]
    values=[]
    for draw in np.random.default_rng(17).multivariate_normal(mean,cov,size=1200):
        values.append(fn({**fit.param_values,**dict(zip(names,draw))}))
    for metric in ['v02_kms','v50_kms','w80_kms','fraction_beyond_500_kms']:
        assert errors[metric]==pytest.approx(np.std([v[metric] for v in values],ddof=1),rel=.1)
    broken=deepcopy(fit);broken.param_errors[names[0]]=np.nan
    assert all(np.isnan(x) for x in propagated_errors(broken,fn,names).values())


def test_canonicalization_reorders_covariance_and_arrays():
    from qsospec.global_result import EmissionComplexResult
    from qsospec.line_peaks import profile_definitions
    cfg=HbetaComplexConfig(broad_fwhm_bands_kms=())
    recipe=adaptive_recipe(cfg,2);ctx=GenericComplexContext(recipe,[c.id for c in recipe.components],100)
    p=dict(zip(ctx.names,ctx.initial));p.update({'narrow.fwhm_kms':800.,'wing.fwhm_kms':300.,'wing.velocity_kms':-500.})
    names=list(p);cov=np.arange(len(names)**2).reshape(len(names),len(names));cov=cov@cov.T+np.eye(len(names))
    wave=np.linspace(4700,5150,500);theta=np.array(list(p.values()))
    arrays=ctx.components(theta,wave)
    fit=EmissionComplexResult(True,1,'ok','wing',p.copy(),dict(zip(names,np.sqrt(np.diag(cov)))),cov.copy(),{}, {},0,1,0,0,wave,np.zeros_like(wave),np.ones_like(wave),ctx.model(theta,wave),arrays.copy(),np.ones_like(wave,bool),metadata={'peak_model':{'components':profile_definitions(ctx)}})
    canonicalize(fit,recipe)
    assert fit.param_values['narrow.fwhm_kms']==300.
    np.testing.assert_array_equal(fit.component_models['OIII5007_core'],arrays['OIII5007_wing'])
    i,j=names.index('narrow.fwhm_kms'),names.index('wing.fwhm_kms')
    assert fit.covariance[i,i]==cov[j,j]
    assert fit.covariance[i,names.index('Hb_narrow.flux')]==cov[j,names.index('Hb_narrow.flux')]


def test_failed_starts_are_archived(monkeypatch):
    from qsospec.fitting import global_fit
    original=global_fit._solve_once_with_fallback
    calls=[0]
    def fail_once(*args,**kwargs):
        calls[0]+=1
        if calls[0]==1:raise RuntimeError('injected failed start')
        return original(*args,**kwargs)
    monkeypatch.setattr(global_fit,'_solve_once_with_fallback',fail_once)
    s,c,cfg=synthetic([(70,0,300)])
    fit=fit_hbeta_complex(s,c,replace(cfg,fit_oiii_wings=False))
    starts=fit.metadata['candidate_selection'][0]['multistart']['starts']
    assert not starts[0]['success'] and starts[0]['message']=='injected failed start'
    assert fit.success


def test_missing_flux_error_cannot_accept(monkeypatch):
    from qsospec.fitting import global_fit
    original=global_fit._covariance_from_jacobian
    def unavailable(jac,reduced,names):
        cov,errors,warnings=original(jac,reduced,names)
        for n in names:
            if n.startswith('OIII5007_wing'):errors[n]=np.nan
        return cov,errors,warnings
    monkeypatch.setattr(global_fit,'_covariance_from_jacobian',unavailable)
    s,c,cfg=synthetic([(70,0,300),(60,0,1300)])
    fit=fit_hbeta_complex(s,c,cfg)
    assert fit.selected_model=='core'
    assert 'added_component_snr' in fit.metadata['candidate_selection'][1]['rejection_reasons']


def test_lsf_run_store_roundtrip_and_peak_recovery(tmp_path,monkeypatch):
    from qsospec.global_result import GlobalContinuumResult,WorkflowResult
    from qsospec.io.run_store import RunStore,workflow_payload,load_model
    from qsospec.line_peaks import recover_line_peaks
    from qsospec.uncertainties import apply_bootstrap_errors
    resolution=SpectralResolution('sigma_lambda',values=np.array([2.]),is_approximate=True)
    s,c,cfg=synthetic([(90,60,90)],resolution=resolution,z=.5)
    fit=fit_hbeta_complex(s,c,replace(cfg,fit_oiii_wings=False))
    cont=GlobalContinuumResult(True,1,'known',{}, {},None,0,1,0,s.wave_rest,c.model,{'power_law':c.model},s.valid_mask,s.valid_mask)
    workflow=WorkflowResult(s,cont,cont,hbeta=fit)
    store=RunStore.create(str(tmp_path/'run'),configuration={})
    store.write_payload(workflow_payload(workflow,run_id=store.run_id,object_key='test',object_id='test',
        input_record={'source':'memory','row_index':0,'reader':'memory','metadata':{}}))
    loaded=load_model(store,'test')
    assert loaded.spectrum.z==s.z and loaded.spectrum.resolution.is_approximate
    np.testing.assert_allclose(loaded.spectrum.resolution.values,resolution.values)
    np.testing.assert_allclose(loaded.hbeta.covariance,fit.covariance)
    import json
    assert json.dumps(loaded.hbeta.metadata['candidate_selection'],sort_keys=True)==json.dumps(fit.metadata['candidate_selection'],sort_keys=True)
    for key in fit.metrics:assert loaded.hbeta.metrics[key]==pytest.approx(fit.metrics[key],nan_ok=True)
    from qsospec.fitting import global_fit
    def forbidden(*args,**kwargs):raise AssertionError('post-processing launched an optimizer')
    monkeypatch.setattr(global_fit,'_solve_once_with_fallback',forbidden)
    del loaded.hbeta.metadata['peak_model'];del loaded.hbeta.metadata['line_peaks']
    assert recover_line_peaks(loaded)['hbeta_oiii']=='available'
    assert loaded.hbeta.metrics['oiii_5008_full_peak_rest_angstrom']==pytest.approx(fit.metrics['oiii_5008_full_peak_rest_angstrom'],abs=1e-6)
    workflow.monte_carlo={'errors':{'line:hbeta_oiii:oiii_5008_full_w80_kms':17.},'percentiles':{},'valid_trial_counts':{},'n_requested':2}
    apply_bootstrap_errors(workflow)
    assert workflow.hbeta.metric_errors['oiii_5008_full_w80_kms']==17.


def test_masked_coverage_archives_no_fit(monkeypatch):
    s,c,cfg=synthetic([(70,0,300)],mask=np.zeros(1100,bool))
    from qsospec.fitting import global_fit
    def forbidden(*args,**kwargs):raise AssertionError('insufficient coverage launched optimizer')
    monkeypatch.setattr(global_fit,'_solve_once_with_fallback',forbidden)
    fit=fit_hbeta_complex(s,c,cfg)
    assert not fit.success
    assert fit.metadata['candidate_selection'][0]['rejection_reasons']==['insufficient_valid_pixels']
    assert fit.metadata['adopted_redshift_unchanged']


def test_unresolved_intrinsic_width_and_doublet_equivalence():
    resolution=SpectralResolution('resolving_power',values=np.array([900.]))
    s,c,cfg=synthetic([(90,0,25)],resolution=resolution,error=.15)
    fit=fit_hbeta_complex(s,c,replace(cfg,fit_oiii_wings=False))
    assert fit.metadata['oiii_components'][0]['unresolved']
    d=fit.metadata['line_lsf']['descriptor']
    a=profile_distribution(fit.param_values,1,lsf=d,line_id='oiii_5008')
    b=profile_distribution(fit.param_values,1,lsf=d,line_id='oiii_4960')
    assert a['fraction_beyond_500_kms']==pytest.approx(b['fraction_beyond_500_kms'],abs=1e-8)
    assert fit.metrics['oiii_doublet_full_observed_core_relative_fraction_beyond_500_kms'] == fit.metrics['oiii_5008_full_observed_core_relative_fraction_beyond_500_kms']
