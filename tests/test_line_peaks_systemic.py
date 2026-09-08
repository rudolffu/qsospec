from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
import qsospec
from qsospec.line_peaks import (C_KMS, refined_peak, measure_peak, record_fit_peaks,
    profile_definitions, evaluate_profile, recover_line_peaks)
from qsospec.systemic_redshift import (WS22_CALIBRATION, velocity_correction,
    combine_redshifts, estimate_systemic_redshift)


def fixture(feature='oiii_5008', reference=5008.24, velocity=0., role='narrow'):
    names = ['line.flux', 'line.velocity_kms', 'line.fwhm_kms']
    p = dict(zip(names, [100., velocity, 300.]))
    rows = [dict(component_id='line', feature=feature, role=role, reference_wave=reference,
        profile='gaussian', flux_parameter=names[0], velocity_parameter=names[1],
        width_parameter=names[2], flux_divisor=1.)]
    wave = np.linspace(reference-50, reference+50, 1001)
    fit = SimpleNamespace(success=True, param_values=p, param_errors=dict(zip(names,[1., 10., 5.])),
        covariance=np.diag([1.,100.,25.]), metadata={}, metrics={}, metric_errors={},
        wave_rest=wave, fit_mask=np.ones(len(wave),bool), component_models={'line':evaluate_profile(wave,p,rows)})
    record_fit_peaks(fit,.7,definitions=rows,bounds=(wave[0],wave[-1]))
    return fit, rows


def workflow(fit):
    return SimpleNamespace(spectrum=SimpleNamespace(z=.7, wave_rest=fit.wave_rest,
        valid_mask=fit.fit_mask, flux_density_scale_to_cgs=None),
        continuum=SimpleNamespace(), line_complexes={'test':fit}, metadata={}, monte_carlo={})


def test_analytic_peak_covariance_and_frames():
    fit, rows = fixture(velocity=123.)
    peak = fit.metadata['line_peaks']['measurements']['oiii_5008_full']
    center = 5008.24*np.exp(123./C_KMS)
    assert peak['peak_rest_angstrom'] == pytest.approx(center, abs=1.e-6)
    assert peak['peak_observed_angstrom'] == pytest.approx(center*1.7)
    assert peak['peak_velocity_kms'] == pytest.approx(123., abs=1.e-4)
    assert peak['peak_error_rest_angstrom'] == pytest.approx(center*10/C_KMS, rel=.002)
    rng=np.random.default_rng(22)
    draws = rng.normal(123.,10.,20000)
    assert np.std(5008.24*np.exp(draws/C_KMS)) == pytest.approx(peak['peak_error_rest_angstrom'],rel=.02)


def test_asymmetric_sum_and_unrelated_neighbor():
    fit, rows = fixture()
    row = dict(rows[0], component_id='wing', flux_parameter='wing.flux', velocity_parameter='wing.velocity_kms', width_parameter='wing.fwhm_kms', role='wing')
    fit.param_values.update({'wing.flux':100.,'wing.velocity_kms':-350.,'wing.fwhm_kms':900.})
    fit.covariance=None
    record_fit_peaks(fit,.7,definitions=rows+[row],bounds=(4950.,5050.))
    full=fit.metadata['line_peaks']['measurements']['oiii_5008_full']['peak_rest_angstrom']
    assert 5008.24*np.exp(-350/C_KMS)<full<5008.24
    assert fit.metadata['line_peaks']['measurements']['oiii_5008_narrow']['peak_rest_angstrom']==pytest.approx(5008.24,abs=1.e-6)


@pytest.mark.parametrize('function,status',[(lambda x:np.zeros_like(x),'absent_line'),(lambda x:x,'boundary_peak'),
    (lambda x:np.exp(-100*(x-.25)**2)+np.exp(-100*(x-.75)**2),'ambiguous_peak')])
def test_peak_flags(function,status):
    assert refined_peak(function,(0.,1.))[1]==status


def test_missing_and_nonidentifiable_covariance():
    fit,rows=fixture()
    fit.covariance=None
    assert np.isnan(measure_peak(fit,rows,.7,(4950,5050))['peak_error_rest_angstrom'])
    fit.covariance=np.zeros((3,3))
    assert np.isnan(measure_peak(fit,rows,.7,(4950,5050))['peak_error_rest_angstrom'])
    fit.covariance=np.diag([1.,100.,25.]);fit.param_errors['line.velocity_kms']=np.nan
    assert np.isnan(measure_peak(fit,rows,.7,(4950,5050))['peak_error_rest_angstrom'])


def test_weak_component_filter_preserves_ordinary_peak():
    fit,rows=fixture('mgii_blend',2798.75)
    weak=dict(rows[0],component_id='weak',flux_parameter='weak.flux',velocity_parameter='weak.velocity_kms',width_parameter='weak.fwhm_kms')
    fit.param_values.update({'weak.flux':4.,'weak.velocity_kms':500.,'weak.fwhm_kms':5.})
    fit.covariance=None
    record_fit_peaks(fit,.7,definitions=rows+[weak],bounds=(2700,2900))
    peaks=fit.metadata['line_peaks']['measurements']
    assert peaks['mgii_blend_full_ws22']['component_ids']==['line']
    assert peaks['mgii_blend_full']['peak_rest_angstrom']>peaks['mgii_blend_full_ws22']['peak_rest_angstrom']


def test_calibration_constants_and_luminosity_floor():
    assert velocity_correction('mgii')==(-57.,0.)
    assert velocity_correction('civ',45.)==(-269.,-438.)
    assert velocity_correction('siiv',45.)==(-151.,-345.)
    assert velocity_correction('civ',44.49)==(0.,0.)
    assert np.isnan(velocity_correction('civ')[0])
    assert WS22_CALIBRATION['lines']['ciii']['offset']==-151+8
    assert WS22_CALIBRATION['lines']['siiv']['wavelength']==1399.41


def test_single_line_correction_and_input_unchanged():
    fit,_=fixture(velocity=C_KMS*np.log(1-48/C_KMS))
    result=workflow(fit)
    before=deepcopy(result.spectrum.__dict__)
    output=estimate_systemic_redshift(result)
    assert output['z_sys']==pytest.approx(.7,abs=1.e-8)
    assert output['z_sys_error']>56*1.7/C_KMS
    assert result.spectrum.z==before['z']
    np.testing.assert_array_equal(result.spectrum.wave_rest,before['wave_rest'])
    assert result.metadata['systemic_redshift']==output
    with pytest.raises(ValueError):estimate_systemic_redshift(result,'unknown')


@pytest.mark.parametrize('change,reason',[
    (lambda f:f.metadata.update(peak_pixel_coverage_fraction=.49),'insufficient_coverage'),
    (lambda f:f.metadata['line_peaks']['measurements']['oiii_5008_full'].update(flux_error=100.),'line_detection_below_2sigma'),
    (lambda f:f.metadata['line_peaks']['measurements']['oiii_5008_full'].update(peak_error_rest_angstrom=np.nan),'peak_uncertainty_unavailable')])
def test_no_eligible_line(change,reason):
    fit,_=fixture();change(fit)
    output=estimate_systemic_redshift(workflow(fit))
    assert output['z_sys'] is None
    assert output['lines']['oiii']['reason']==reason


def test_clipping_and_correlated_mean():
    z=np.array([1.,1.001,1.002,1.003,1.008])
    mean,error,keep,weights=combine_redshifts(z,np.eye(5)*1.e-6)
    assert not keep[-1]
    assert mean==pytest.approx(1.0015)
    assert error==pytest.approx(.0005)
    _,error,keep,_=combine_redshifts([1.,1.001],np.array([[1.,.5],[.5,1.]]))
    assert keep.all() and error==pytest.approx(np.sqrt(.75))
    assert combine_redshifts([1.,1.,2.],np.eye(3))[2].all()


def test_existing_bootstrap_reused():
    fit,_=fixture();result=workflow(fit)
    name='test:oiii_5008_full_peak_rest_angstrom'
    result.monte_carlo={'draws':[{'trial_id':i,'values':{name:5008.24+delta}} for i,delta in enumerate([-.1,.1,-.2,.2])]}
    output=estimate_systemic_redshift(result)
    assert output['uncertainty_method']=='matched_existing_bootstrap'
    assert output['valid_matched_trials']==4


def test_recovery_is_explicit_and_preserves_models():
    fit,_=fixture();result=workflow(fit)
    del fit.metadata['line_peaks'];fit.metrics={};fit.metric_errors={}
    assert recover_line_peaks(result)=={'test':'available'}
    assert 'oiii_5008_full_peak_rest_angstrom' in fit.metrics
    fit.metadata={}
    assert recover_line_peaks(result)=={'test':'unsupported_profile_reconstruction'}


def test_adapter_roundtrip_and_legacy_recovery(tmp_path):
    from qsospec.fitting.global_fit import _HbetaContext
    from qsospec.global_result import GlobalContinuumResult, WorkflowResult
    from qsospec.io.run_store import RunStore, workflow_payload
    config=qsospec.HbetaComplexConfig(fit_oiii_wings=False)
    ctx=_HbetaContext(config,False,100.)
    wave=np.linspace(4500.,5200.,1200)
    rng=np.random.default_rng(3)
    spectrum=qsospec.Spectrum.from_arrays(wave,2.+ctx.model(ctx.initial,wave)+rng.normal(0,.01,len(wave)),
        err=np.full(len(wave),.01),z=.7,wave_frame='rest',flux_unit='relative')
    continuum=GlobalContinuumResult(True,0,'known',{}, {},None,0.,1,0.,wave,
        np.full(len(wave),2.),{'power_law':np.full(len(wave),2.)},spectrum.valid_mask,spectrum.valid_mask)
    fit=qsospec.fit_hbeta_complex(spectrum,continuum,config)
    result=WorkflowResult(spectrum,continuum,continuum,hbeta=fit,line_complexes={'hbeta_oiii':fit})
    output=estimate_systemic_redshift(result)
    assert output['status']=='available'
    store=RunStore.create(str(tmp_path/'peaks'),configuration={})
    store.write_payload(workflow_payload(result,run_id=store.run_id,object_key='test',object_id='test',
        input_record={'source':'memory','row_index':0,'reader':'memory','metadata':{}}))
    loaded=qsospec.load_model(store,'test')
    assert loaded.metadata['systemic_redshift']['z_sys']==output['z_sys']
    assert estimate_systemic_redshift(loaded)['z_sys']==pytest.approx(output['z_sys'],abs=1.e-12)
    assert loaded.spectrum.z==.7
    archived=loaded.line_complexes['hbeta_oiii']
    del archived.metadata['peak_model'];del archived.metadata['line_peaks']
    assert recover_line_peaks(loaded)['hbeta_oiii']=='available'
    assert archived.metadata['peak_recovery_provenance']=='validated_against_archived_component_arrays'


def test_continuum_luminosity_covariance():
    from qsospec.systemic_redshift import luminosity_1700
    fit,_=fixture('civ_blend',1549.06,role='broad')
    result=workflow(fit)
    wave=np.linspace(1400,1800,401)
    result.spectrum=qsospec.Spectrum.from_arrays(wave,np.ones(len(wave)),err=np.ones(len(wave)),
        z=2.,wave_frame='rest',flux_unit='cgs',flux_scale=1.e-17)
    result.continuum=SimpleNamespace(wave_rest=wave,
        component_models={'power_law':2*(wave/1700)**-1.5},param_values={'power_law.norm':2.,'power_law.slope':-1.5},
        param_errors={'power_law.norm':.1,'power_law.slope':.2},covariance=np.diag([.01,.04]),
        metadata={'power_law_pivot':1700.})
    luminosity=luminosity_1700(result)
    assert luminosity['error_dex']==pytest.approx(.1/2/np.log(10),rel=1.e-5)
    assert luminosity['status']=='available'
    result.continuum.covariance=None
    assert np.isnan(luminosity_1700(result)['error_dex'])


def test_generic_ties_and_lorentzian_profiles():
    from dataclasses import replace
    from qsospec import complex_recipes
    from qsospec.fitting.complexes import GenericComplexContext
    recipe=complex_recipes.get('civ')
    recipe=replace(recipe,components=tuple(replace(c,profile='lorentzian',line_ids=('civ_1548','civ_1551')) for c in recipe.components))
    context=GenericComplexContext(recipe,[c.id for c in recipe.components],100.)
    rows=profile_definitions(context)
    p=dict(zip(context.names,context.initial))
    wave=np.linspace(1450.,1700.,1000)
    np.testing.assert_allclose(evaluate_profile(wave,p,rows),context.model(context.initial,wave))
    assert len(rows)==6
    assert len({r['velocity_parameter'] for r in rows})==3


def test_asymmetric_peak_error_matches_correlated_parameter_draws():
    from qsospec.line_peaks import _peak
    fit,rows=fixture()
    row=dict(rows[0],component_id='wing',flux_parameter='wing.flux',velocity_parameter='wing.velocity_kms',width_parameter='wing.fwhm_kms')
    fit.param_values.update({'wing.flux':80.,'wing.velocity_kms':-220.,'wing.fwhm_kms':700.})
    fit.param_errors.update({'wing.flux':2.,'wing.velocity_kms':12.,'wing.fwhm_kms':10.})
    fit.covariance=np.diag([1.,100.,25.,4.,144.,100.])
    fit.covariance[1,4]=fit.covariance[4,1]=60.
    rows=rows+[row]
    peak=measure_peak(fit,rows,.7,(4950.,5050.))
    names=list(fit.param_values)
    rng=np.random.default_rng(15)
    draws=rng.multivariate_normal(list(fit.param_values.values()),fit.covariance,size=400)
    samples=[_peak(dict(zip(names,t)),rows,(4950.,5050.))[0] for t in draws]
    assert np.std(samples,ddof=1)==pytest.approx(peak['peak_error_rest_angstrom'],rel=.12)


def test_full_doublet_has_explicit_effective_reference():
    fit,rows=fixture('oii_3727',3727.09)
    second=dict(rows[0],feature='oii_3730',reference_wave=3729.88)
    record_fit_peaks(fit,.7,definitions=rows+[second],bounds=(3680.,3780.))
    peak=fit.metadata['line_peaks']['measurements']['oii_doublet_full']
    assert peak['reference_wave']==3728.48
    assert peak['peak_velocity_kms']==pytest.approx(C_KMS*np.log(peak['peak_rest_angstrom']/3728.48))
    assert estimate_systemic_redshift(workflow(fit))['lines']['oii']['reason']=='compatible_peak_unavailable'
