"""Full [O III] profile measurements conditional on the fitted decomposition."""
from dataclasses import replace
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from . import lines
from .line_peaks import record_fit_peaks, _covariance_for
from .fitting.adaptive_oiii import C, GROUPS, LABELS
from .fitting.line_lsf import saved_lsf

PROBABILITIES=(.02,.05,.10,.50,.90,.95,.98)


def profile_distribution(parameters, count, *, reference=0., lsf=None, line_id='oiii_5008'):
    """Wavelength-Gaussian CDF transformed to log velocity (not a Gaussian in v)."""
    rest=lines.get(line_id).vacuum_wavelength
    flux=np.array([parameters[f'OIII5007_{LABELS[i]}.flux'] for i in range(count)])
    velocity=np.array([parameters[GROUPS[i]+'.velocity_kms'] for i in range(count)])
    width=np.array([parameters[GROUPS[i]+'.fwhm_kms'] for i in range(count)])
    if np.any(flux<0) or flux.sum()<=0 or np.any(width<=0) or not np.all(np.isfinite(np.r_[flux,velocity,width])):
        return {**{f'v{int(p*100):02d}_kms':np.nan for p in PROBABILITIES},
                'w80_kms':np.nan,'w90_kms':np.nan,'fraction_beyond_500_kms':np.nan}
    weights=flux/flux.sum();centers=rest*np.exp(velocity/C);sigma=centers*width/C/2.354820045
    lo=max(.01,np.min(centers-12*sigma));hi=np.max(centers+12*sigma)
    if lsf:
        operator=saved_lsf(lsf)
        x=operator.source_wave
        support=(x>=lo)&(x<=hi);x=x[support];sig=operator.sigma[support]
        mass=np.sum(weights[:,None]*np.exp(-.5*((x-centers[:,None])/sigma[:,None])**2)/
                    (np.sqrt(2*np.pi)*sigma[:,None]),axis=0)*operator.step
        # Withhold truncated-profile measurements rather than normalize lost flux.
        if abs(mass.sum()-1.)>1.e-4:
            return profile_distribution({**parameters, f'OIII5007_{LABELS[0]}.flux':np.nan},count)
        cdf=lambda w: float(mass @ ndtr((w-x)/sig))
        lo=max(.01,lo-8*sig.max());hi=hi+8*sig.max()
    else:
        cdf=lambda w: float(weights @ ndtr((w-centers)/sigma))
    try:
        quantiles=[C*np.log(brentq(lambda w:cdf(w)-p,lo,hi,xtol=1.e-9)/rest)-reference for p in PROBABILITIES]
    except ValueError:
        quantiles=[np.nan]*len(PROBABILITIES)
    out={f'v{int(p*100):02d}_kms':float(q) for p,q in zip(PROBABILITIES,quantiles)}
    out.update(w80_kms=quantiles[4]-quantiles[2],w90_kms=quantiles[5]-quantiles[1],
        fraction_beyond_500_kms=float(cdf(rest*np.exp((reference-500)/C))+1-cdf(rest*np.exp((reference+500)/C))))
    return out


def propagated_errors(fit, function, names):
    """Propagate all selected correlations, withholding unidentified directions."""
    base=function(fit.param_values);keys=list(base)
    covariance=_covariance_for(fit,names)
    if covariance is None:return {k:np.nan for k in keys}
    jac=np.empty((len(keys),len(names)))
    for j,name in enumerate(names):
        step=max(abs(fit.param_values[name])*1.e-4,1.e-3 if name.endswith('kms') else 1.e-6)
        plus,minus=dict(fit.param_values),dict(fit.param_values)
        plus[name]+=step;minus[name]-=step
        a,b=function(plus),function(minus)
        jac[:,j]=[(a[k]-b[k])/(2*step) for k in keys]
    variance=np.einsum('ij,jk,ik->i',jac,covariance,jac)
    return {k:float(np.sqrt(v)) if np.isfinite(v) and v>0 and np.isfinite(base[k]) else np.nan for k,v in zip(keys,variance)}


def record_oiii_measurements(fit,spectrum,continuum,recipe,*,compute_covariance=True,defer_peaks=False):
    from .fitting.complexes import GenericComplexContext, generic_complex_metrics
    from .fitting.global_fit import _metric_errors
    from .fitting.line_lsf import GaussianLineLSF
    count=sum(c.line_ids==('oiii_5008',) for c in recipe.components)
    context=GenericComplexContext(recipe,[c.id for c in recipe.components],1.)
    if fit.metadata.get('line_lsf',{}).get('status')=='forward_modeled':
        context.line_lsf=GaussianLineLSF(fit.metadata['line_lsf']['descriptor'], spectrum.wave_rest)
    theta=np.array([fit.param_values[k] for k in context.names])
    # Retain legacy centroid/width/flux outputs and the Hbeta synchronization API.
    def metrics(vector):
        return generic_complex_metrics(context,vector,continuum,spectrum)
    fit.metrics.update(metrics(theta))
    if compute_covariance:
        order=list(fit.param_values);ix=[order.index(k) for k in context.names]
        cov=None if fit.covariance is None else fit.covariance[np.ix_(ix,ix)]
        fit.metric_errors.update(_metric_errors(theta,cov,metrics))
    for suffix in ('flux_input','flux_cgs','centroid','sigma_kms','fwhm_kms','ew_rest'):
        key='hbeta_broad_'+suffix;alias='Hb_broad_'+suffix
        fit.metrics[alias]=fit.metrics.get(key,np.nan)
        fit.metric_errors[alias]=fit.metric_errors.get(key,np.nan)
    center=fit.metrics['Hb_broad_centroid']
    fit.metrics['Hb_broad_velocity_kms']=float(C*np.log(center/lines.get('hbeta').vacuum_wavelength)) if center>0 else np.nan
    fit.metric_errors['Hb_broad_velocity_kms']=C*fit.metric_errors['Hb_broad_centroid']/center if center>0 else np.nan
    rows=[]
    descriptor=fit.metadata.get('line_lsf',{}).get('descriptor')
    resolution = descriptor.get('resolution', {}) if descriptor else {}
    resolution_values = np.asarray(resolution.get('values', []))
    equal_velocity_lsf = (not descriptor or (resolution.get('mode') in ('resolving_power', 'sigma_kms')
        and resolution_values.size > 0 and np.all(resolution_values == resolution_values.flat[0])))
    for i in range(count):
        group=GROUPS[i];label=LABELS[i];flux=fit.param_values[f'OIII5007_{label}.flux']
        err=fit.param_errors.get(f'OIII5007_{label}.flux',np.nan)
        width=fit.param_values[group+'.fwhm_kms'];width_error=fit.param_errors.get(group+'.fwhm_kms',np.nan)
        bounds=next(c.fwhm_bands_kms[0] for c in recipe.components if c.id=='OIII5007_'+label)
        rows.append(dict(component_id='OIII5007_'+label,doublet_members=['OIII4959_'+label,'OIII5007_'+label],
            flux_5008=flux,flux_error_5008=err,flux_snr=flux/err if err>0 else np.nan,
            velocity_kms=fit.param_values[group+'.velocity_kms'],velocity_error_kms=fit.param_errors.get(group+'.velocity_kms',np.nan),
            fwhm_kms=width,fwhm_error_kms=width_error,width_frame='intrinsic' if descriptor else 'observed',
            unresolved=bool(descriptor and (not np.isfinite(width_error) or width_error<=0 or width/width_error<3 or width-bounds[0]<=1.e-4*(bounds[1]-bounds[0]))),
            physical_origin='unclassified',width_order=i+1))
    core=rows[0]
    reference_flags=[]
    if not np.isfinite(core['flux_snr']) or core['flux_snr']<5:reference_flags.append('weak_core')
    total_flux=sum(r['flux_5008'] for r in rows)
    if total_flux<=0 or core['flux_5008']/total_flux<.05:reference_flags.append('small_core_flux_fraction')
    if not np.isfinite(core['velocity_error_kms']) or core['velocity_error_kms']>100:reference_flags.append('unstable_core_velocity')
    if len(rows)>1:
        separation=rows[1]['fwhm_kms']-core['fwhm_kms']
        names=list(fit.param_values);cov=fit.covariance
        if cov is not None:
            a,b=names.index('narrow.fwhm_kms'),names.index('wing.fwhm_kms')
            variance=cov[a,a]+cov[b,b]-2*cov[a,b]
            if variance>0 and separation<np.sqrt(variance):reference_flags.append('unstable_width_order')
    reliable=not reference_flags
    fit.metadata.update(oiii_components=rows,oiii_core_reference=dict(
        definition='narrowest fitted gas component; not systemic redshift',reliable=reliable,
        minimum_flux_snr=5.,minimum_flux_fraction=.05,maximum_velocity_error_kms=100.,
        flags=reference_flags,velocity_kms=core['velocity_kms']),
        oiii_measurement_definitions=dict(primary='observed photon profile before detector pixel integration',
            velocity='c log(lambda / vacuum reference wavelength)',quantiles=list(PROBABILITIES),
            relative_reference='narrowest fitted component',fraction_threshold_kms=500.,
            uncertainty='full line-fit covariance conditional on fixed host/continuum and selected model',
            intrinsic_available=bool(descriptor),doublet='aligned velocity coordinates, fixed 5008/4960 ratio',
            uncertainty_cross_continuum='unavailable; fixed-continuum approximation'))
    names=[k for k in fit.param_values if k.startswith(('OIII5007','narrow.','wing.','wing2.'))]
    def all_metrics(parameters):
        result={}
        for frame,lsf in [('observed',descriptor)]+([('intrinsic',None)] if descriptor else []):
            for relative in (False,True):
                reference=parameters['narrow.velocity_kms'] if relative else 0.
                values=profile_distribution(parameters,count,reference=reference,lsf=lsf)
                for key,value in values.items():
                    name=f'oiii_5008_full_{frame}_'+('core_relative_' if relative else 'input_frame_')+key
                    result[name]=value if not relative or reliable else np.nan
                if relative:
                    f5008=values['fraction_beyond_500_kms']
                    identical = not lsf or equal_velocity_lsf
                    f4960=profile_distribution(parameters,count,reference=reference,lsf=lsf,line_id='oiii_4960')['fraction_beyond_500_kms'] if not identical else f5008
                    ratio=fit.metadata['oiii_ratio_5007_4959']
                    fraction = f5008 if identical else (ratio*f5008+f4960)/(ratio+1)
                    result[f'oiii_doublet_full_{frame}_core_relative_fraction_beyond_500_kms']=fraction if reliable else np.nan
        result['oiii_5008_full_flux_input']=sum(parameters[f'OIII5007_{LABELS[i]}.flux'] for i in range(count))
        result['oiii_doublet_full_flux_input']=result['oiii_5008_full_flux_input']*(1+1/fit.metadata['oiii_ratio_5007_4959'])
        return result
    values=all_metrics(fit.param_values);fit.metrics.update(values)
    errors=propagated_errors(fit,all_metrics,names) if compute_covariance else {k:np.nan for k in values}
    fit.metric_errors.update(errors)
    for key in ('w80_kms','w90_kms'):
        full='oiii_5008_full_observed_input_frame_'+key
        fit.metrics['oiii_5008_full_'+key]=fit.metrics[full]
        fit.metric_errors['oiii_5008_full_'+key]=fit.metric_errors[full]
    fit.metadata['oiii_doublet_fraction_equivalence']=('intrinsic_exact; observed_only_if_velocity_LSF_equal' if descriptor else 'exact_fixed_ratio')
    flux_names=[f'OIII5007_{LABELS[i]}.flux' for i in range(count)]
    flux_errors=propagated_errors(fit,lambda p:{k:v for k,v in all_metrics(p).items() if k.endswith('_flux_input')},flux_names) if compute_covariance else {}
    fit.metric_errors.update(flux_errors)
    for feature in ('oiii_5008', 'oiii_doublet'):
        source=feature+'_full_flux_input';target=feature+'_full_flux_cgs'
        scale=spectrum.flux_density_scale_to_cgs
        fit.metrics[target]=fit.metrics[source]*scale if scale is not None else np.nan
        fit.metric_errors[target]=fit.metric_errors[source]*scale if scale is not None else np.nan
    # Primary peaks use the saved observed-profile descriptor. Intrinsic quantiles
    # and widths are separate; adopting a redshift is never part of this function.
    record_fit_peaks(fit,spectrum.z,measure=not defer_peaks)
    fit.metadata['oiii_measurement_version']=1
