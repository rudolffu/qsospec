"""Adaptive Hbeta/[O III] candidates compiled by the shared recipe fitter."""
from dataclasses import asdict, replace
from copy import deepcopy
import time
import numpy as np
from .. import lines
from ..complex_recipes import ComponentRecipe, ComplexRecipe
from ..warnings import FitWarning
from .complexes import (GenericComplexContext, RecipeCoverage, fit_generic_complex,
                        generic_complex_metrics)

C = 299792.458
GROUPS = ('narrow','wing','wing2')
LABELS = ('core','wing','wing2')
VERSION = 'adaptive-oiii-1'


def adaptive_recipe(config, count):
    components=[]
    for i,band in enumerate(config.broad_fwhm_bands_kms,1):
        components.append(ComponentRecipe(f'Hb_broad{i}',('hbeta',),'broad',
            velocity_bounds_kms=config.broad_velocity_bounds_kms, fwhm_bands_kms=(band,)))
    components.append(ComponentRecipe('Hb_narrow',('hbeta',),'narrow',
        velocity_bounds_kms=config.narrow_velocity_bounds_kms,
        fwhm_bands_kms=(config.narrow_fwhm_bounds_kms,)))
    for i in range(count):
        group=GROUPS[i];label=LABELS[i]
        bounds=config.narrow_velocity_bounds_kms if i==0 else (-4000.,4000.)
        width=config.narrow_fwhm_bounds_kms if i==0 else (70.,6000.)
        for prefix,feature in [('OIII5007','oiii_5008'),('OIII4959','oiii_4960')]:
            components.append(ComponentRecipe(f'{prefix}_{label}',(feature,),
                'narrow' if i==0 else 'wing',velocity_group=group,width_group=group,
                velocity_bounds_kms=bounds,fwhm_bands_kms=(width,),
                fixed_ratio_to=f'OIII5007_{label}' if prefix=='OIII4959' else None,
                fixed_ratio=config.oiii_ratio_5007_4959 if prefix=='OIII4959' else None))
    if config.heii_enabled:
        components.append(ComponentRecipe('HeII_broad',('heii_4687',),'broad',
            velocity_bounds_kms=config.broad_velocity_bounds_kms,fwhm_bands_kms=((900.,15000.),)))
    return ComplexRecipe('hbeta_oiii',(),'Hbeta / [O III]',config.window,(config.window,),
        () if config.heii_enabled else (config.heii_mask,),tuple(components),(),
        continuum_mode='residual_linear',coverage_mode='component_adaptive',
        qa_labels=('hbeta','oiii_4960','oiii_5008'))


def residual_diagnostics(wave, residual, mask, core_velocity):
    """Doublet-matched coherent residuals; gaps and masked pixels break runs."""
    wave=np.asarray(wave);mask=np.asarray(mask,bool)&np.isfinite(residual)
    centers=[lines.get(x).vacuum_wavelength for x in ('oiii_4960','oiii_5008')]
    velocities=[C*np.log(wave/x)-core_velocity for x in centers]
    records={};local=[]
    for label,v in zip(('4960','5008'),velocities):
        within=np.abs(v)<=1200.;good=within&mask;local.append(good)
        coverage=(np.count_nonzero(good)/max(np.count_nonzero(within),1))
        complete=bool(np.any(good) and v[mask].min()<=-1200 and v[mask].max()>=1200
                      and coverage>=.8 and np.count_nonzero(good)>=3)
        significant=np.flatnonzero(good & (abs(residual)>3))
        runs=[];run=[]
        step=np.median(np.diff(wave))
        for i in significant:
            if run and (i!=run[-1]+1 or residual[i]*residual[run[-1]]<=0 or wave[i]-wave[run[-1]]>1.5*step):
                if len(run)>=3:runs.append(run)
                run=[]
            run.append(int(i))
        if len(run)>=3:runs.append(run)
        records[label]=dict(n_pixels=int(good.sum()),coverage_fraction=coverage,
            significant_runs=runs,
            complete=complete,mean_chi2=float(np.mean(residual[good]**2)) if np.any(good) else np.nan)
    # Match each 5008 detector pixel to its nearest 4960 velocity sample.
    ids=np.flatnonzero(local[1]);target=wave[ids]*centers[0]/centers[1]
    right=np.clip(np.searchsorted(wave,target),1,len(wave)-1)
    nearest=np.where(abs(wave[right]-target)<abs(wave[right-1]-target),right,right-1)
    step=np.median(np.diff(wave))
    matched=mask[nearest] & (abs(wave[nearest]-target)<=.75*step)
    flags=np.zeros(len(wave),bool);sign=np.zeros(len(wave),int)
    strong=(abs(residual[ids])>3)&(abs(residual[nearest])>2)&(residual[ids]*residual[nearest]>0)&matched
    flags[ids]=strong;sign[ids]=np.sign(residual[ids]).astype(int)
    groups=[];run=[]
    for i in np.flatnonzero(flags):
        if run and (i!=run[-1]+1 or sign[i]!=sign[run[-1]] or wave[i]-wave[run[-1]]>1.5*step):
            if len(run)>=3:groups.append(run)
            run=[]
        run.append(int(i))
    if len(run)>=3:groups.append(run)
    records.update(coherent=bool(groups),runs_5008=groups,
        single_line_structure=any(records[k]['significant_runs'] for k in ('4960','5008')),
        complete=all(records[k]['complete'] for k in ('4960','5008')),
        reference_velocity_kms=float(core_velocity),thresholds=dict(run_pixels=3,
            sigma_5008=3.,sigma_4960=2.,radius_kms=1200.))
    return records


def relevant_bounds(context, theta):
    hits=[]
    for name in context.nonlinear_names:
        if name.startswith(('narrow.','wing.','wing2.')):
            i=context.index[name];tol=1.e-4*(context.upper[i]-context.lower[i])
            if min(theta[i]-context.lower[i],context.upper[i]-theta[i])<=tol:hits.append(name)
    return hits


def _starts(config, count, previous):
    warm={} if previous is None else dict(previous.param_values)
    # The old solution with a nearly absent added component is a nested start.
    group=GROUPS[count-1];label=LABELS[count-1]
    warm.setdefault('narrow.velocity_kms',0.);warm.setdefault('narrow.fwhm_kms',300.)
    starts=[]
    total=max(sum(v for k,v in warm.items() if k.startswith('OIII5007') and k.endswith('.flux')),1.)
    for j,(velocity,width) in enumerate(((0.,900.),(0.,1600.),(-600.,1200.),(600.,1200.),(-1500.,2600.),(1500.,2600.))):
        init=dict(warm)
        if count==1:
            init['narrow.velocity_kms']=(0.,0.,-250.,250.,-600.,600.)[j]
            init['narrow.fwhm_kms']=(200.,650.,400.,400.,1000.,1000.)[j]
        else:
            init[f'{group}.velocity_kms']=velocity
            init[f'{group}.fwhm_kms']=width
            init[f'OIII5007_{label}.flux']=total*(.001 if j==0 else .25)
        starts.append(init)
    return starts


def canonicalize(fit, recipe):
    """Relabel by increasing width, permuting covariance and model definitions."""
    count=sum(c.line_ids==('oiii_5008',) for c in recipe.components)
    order=sorted(range(count),key=lambda i:(fit.param_values[GROUPS[i]+'.fwhm_kms'],
                                           fit.param_values[GROUPS[i]+'.velocity_kms'],i))
    mapping={};component_map={}
    for new,old in enumerate(order):
        for suffix in ('velocity_kms','fwhm_kms'):
            mapping[GROUPS[old]+'.'+suffix]=GROUPS[new]+'.'+suffix
        for prefix in ('OIII5007','OIII4959'):
            component_map[prefix+'_'+LABELS[old]]=prefix+'_'+LABELS[new]
        mapping['OIII5007_'+LABELS[old]+'.flux']='OIII5007_'+LABELS[new]+'.flux'
    original=list(fit.param_values)
    reverse={mapping.get(k,k):k for k in original}
    names=original  # stable public ordering, independent of the optimizer permutation
    permutation=[original.index(reverse[k]) for k in names]
    fit.param_values={k:fit.param_values[reverse[k]] for k in names}
    fit.param_errors={k:fit.param_errors.get(reverse[k],np.nan) for k in names}
    if fit.covariance is not None:fit.covariance=fit.covariance[np.ix_(permutation,permutation)]
    fit.component_models={component_map.get(k,k):v for k,v in fit.component_models.items()}
    components=[]
    for c in recipe.components:
        if c.id in component_map:
            ident=component_map[c.id];group=GROUPS[LABELS.index(ident.split('_',1)[1])]
            c=replace(c,id=ident,velocity_group=group,width_group=group,
                      role='narrow' if ident.endswith('_core') else 'wing',
                      fixed_ratio_to=component_map.get(c.fixed_ratio_to,c.fixed_ratio_to))
        components.append(c)
    # Follow canonical public order and carry the original component's bounds.
    components.sort(key=lambda c:next(i for i,x in enumerate(recipe.components) if x.id==c.id))
    canonical=replace(recipe,components=tuple(components))
    for row in fit.metadata.get('peak_model',{}).get('components',[]):
        row['component_id']=component_map.get(row['component_id'],row['component_id'])
        for key in ('flux_parameter','velocity_parameter','width_parameter'):
            row[key]=mapping.get(row[key],row[key])
        if row['feature'].startswith('oiii_'):row['role']='narrow' if row['component_id'].endswith('_core') else 'wing'
    fit.metadata.update(covariance_parameter_names=names,canonicalization=dict(
        original_to_canonical=mapping,parameter_permutation=permutation,
        definition='ascending intrinsic width when LSF modeled; otherwise observed width'))
    fit.metadata['active_components']=tuple(c.id for c in canonical.components)
    for key in ('component_detection_status','component_coverage_status'):
        fit.metadata[key]=tuple((component_map.get(k,k),v) for k,v in fit.metadata.get(key,()))
    for warning in fit.warnings:
        if warning.context and 'parameter' in warning.context:
            warning.context['parameter']=mapping.get(warning.context['parameter'],warning.context['parameter'])
    # Optimizer coordinates are archived in the candidate record; the public
    # object must not silently expose the old parameter ordering.
    fit.optimizer_result=None
    return canonical


def fit_adaptive_hbeta(spectrum, continuum, config, *, compute_covariance=True, defer_peaks=False):
    begin=time.perf_counter();candidates=[];selected=None;selected_recipe=None
    max_count=3 if config.fit_oiii_wings else 1
    for count in range(1,max_count+1):
        if count==3 and not candidates[-1]['residuals']['coherent']:break
        recipe=adaptive_recipe(config,count)
        # All candidates keep identical masks and all configured Hbeta components.
        coverage=RecipeCoverage('covered',recipe,tuple(c.id for c in recipe.components),(),(),
            1.,int(spectrum.valid_mask.sum()),(config.window,),('hbeta','oiii_4960','oiii_5008'),True,())
        def expand(context, solution, starts, mask):
            scores=sorted(s['chi2'] for s in starts if s['success'])
            reasons=[]
            if len(scores)<2 or scores[1]-scores[0]>1.:reasons.append('starts_disagree')
            if relevant_bounds(context,solution.x):reasons.append('oiii_parameter_at_bound')
            resid=(spectrum.flux-continuum.model-context.model(solution.x,spectrum.wave_rest))/spectrum.err
            core=min((GROUPS[i] for i in range(count)),key=lambda g:solution.x[context.index[g+'.fwhm_kms']])
            diag=residual_diagnostics(spectrum.wave_rest,resid,mask,solution.x[context.index[core+'.velocity_kms']])
            if diag['coherent']:reasons.append('coherent_doublet_residual')
            return reasons
        starts=_starts(config,count,selected)
        fit=fit_generic_complex(spectrum,continuum,recipe,coverage_override=coverage,
            compute_covariance=True,defer_peaks=True,n_starts=6,max_starts=24,
            random_seed=config.oiii_random_seed,initial_values=starts[0],start_values=starts,
            expand_search=expand,optimizer_config=config,forward_resolution=True,measure_metrics=False)
        if fit is None:raise RuntimeError('Adaptive candidate unexpectedly lacked coverage')
        raw_names=list(fit.param_values)
        if not raw_names:
            fit.metadata.update(oiii_profile_mode='adaptive',profile_adequate=False,
                oiii_method_version=VERSION, input_redshift=float(spectrum.z),
                adopted_redshift_unchanged=True, oiii_components=[],
                model_definition=asdict(recipe),
                profile_quality_flags=['insufficient_valid_pixels'],
                candidate_selection=[dict(component_count=count,model_definition=asdict(recipe),
                    parameters={},covariance=None,covariance_status='unavailable',
                    success=False,accepted=False,bic=np.nan,added_component_flux_snr=np.nan,
                    multistart={'seed':config.oiii_random_seed,'starts':[]},
                    rejection_reasons=['insufficient_valid_pixels'])])
            return fit
        # Acceptance uses the newly introduced optimizer component before sorting.
        added='OIII5007_'+LABELS[count-1]+'.flux';error=fit.param_errors.get(added,np.nan)
        snr=fit.param_values[added]/error if np.isfinite(error) and error>0 else np.nan
        context=GenericComplexContext(recipe,coverage.active_component_ids,1.)
        bounds=relevant_bounds(context,np.array([fit.param_values[k] for k in context.names]))
        archive=dict(component_count=count,model_definition=asdict(recipe),parameters=dict(fit.param_values),
            parameter_errors=dict(fit.param_errors),covariance_parameter_names=raw_names,
            covariance=None if fit.covariance is None else fit.covariance.tolist(),
            covariance_status='rank_deficient' if 'covariance_rank_deficient' in fit.warning_codes() else
                'available' if fit.covariance is not None else 'unavailable',
            multistart=deepcopy(fit.metadata['multistart']),chi2=fit.chi2,bic=fit.bic,
            success=fit.success,added_component_flux_snr=float(snr),bounds_hit=bounds,
            line_lsf=fit.metadata['line_lsf'])
        canonical_recipe=canonicalize(fit,recipe)
        residual=(spectrum.flux-continuum.model-fit.model)/spectrum.err
        diag=residual_diagnostics(spectrum.wave_rest,residual,fit.fit_mask,fit.param_values['narrow.velocity_kms'])
        checks=dict(convergence=bool(fit.success))
        if selected is not None:
            checks.update(bic_improvement=bool(selected.bic-fit.bic>=config.wing_bic_delta),
                          added_component_snr=bool(np.isfinite(snr) and snr>=config.wing_min_snr))
        delta=float(selected.bic-fit.bic) if selected is not None else np.nan
        if count==3 and selected is not None:
            core=selected.param_values['narrow.velocity_kms']
            before=residual_diagnostics(spectrum.wave_rest,(spectrum.flux-continuum.model-selected.model)/spectrum.err,fit.fit_mask,core)
            after=residual_diagnostics(spectrum.wave_rest,residual,fit.fit_mask,core)
            checks['improved_both_lines']=bool(all(np.isfinite(after[k]['mean_chi2']) and
                after[k]['mean_chi2']<before[k]['mean_chi2'] for k in ('4960','5008')))
            archive['local_residual_comparison']={'before':before,'after':after}
        accepted=all(checks.values())
        archive.update(accepted=accepted,checks=checks,rejection_reasons=[k for k,v in checks.items() if not v],
            delta_bic=delta,residuals=diag,canonicalization=fit.metadata['canonicalization'],
            width_ratio=fit.param_values.get('wing.fwhm_kms',np.nan)/fit.param_values['narrow.fwhm_kms'],
            centroid_separation_kms=abs(fit.param_values.get('wing.velocity_kms',np.nan)-fit.param_values['narrow.velocity_kms']))
        candidates.append(archive)
        if accepted or selected is None:
            selected=fit;selected_recipe=canonical_recipe
        # A failed two-component candidate can trigger a three-component search;
        # compare the latter directly to the best retained successful baseline.
    selected.selected_model=('core','wing','three_component')[sum(c.line_ids==('oiii_5008',) for c in selected_recipe.components)-1]
    selected.metadata['oiii_ratio_5007_4959']=config.oiii_ratio_5007_4959
    finalize_quality(selected,spectrum,continuum,selected_recipe,candidates)
    from ..oiii_measurements import record_oiii_measurements
    record_oiii_measurements(selected,spectrum,continuum,selected_recipe,compute_covariance=compute_covariance,
                             defer_peaks=defer_peaks)
    if not compute_covariance:
        selected.covariance=None
        selected.param_errors={k:np.nan for k in selected.param_values}
        selected.metric_errors={k:np.nan for k in selected.metrics}
    selected.metadata['oiii_runtime_seconds']=time.perf_counter()-begin
    return selected


def finalize_quality(selected, spectrum, continuum, selected_recipe, candidates):
    """Update diagnostics explicitly; never called implicitly while loading archives."""
    diag=residual_diagnostics(spectrum.wave_rest,(spectrum.flux-continuum.model-selected.model)/spectrum.err,
                              selected.fit_mask,selected.param_values['narrow.velocity_kms'])
    flags=[]
    if not selected.success:flags.append('unsuccessful_search')
    if not diag['complete']:flags.append('incomplete_doublet_coverage')
    if diag['coherent']:flags.append('remaining_coherent_residuals')
    if diag['single_line_structure'] and not diag['coherent']:flags.append('unmatched_line_residual_structure')
    if any(diag[k]['mean_chi2']>4 for k in ('4960','5008')):flags.append('excess_local_residual_chi2')
    chosen=next(c for c in candidates if c['component_count']==sum(x.line_ids==('oiii_5008',) for x in selected_recipe.components))
    if chosen['bounds_hit']:flags.append('boundary_solution')
    if any(c['multistart']['search_failed'] for c in candidates):flags.append('candidate_search_failed')
    if selected.metadata['line_lsf']['status']!='forward_modeled':flags.append('observed_profile_resolution_unavailable')
    selected.metadata.update(oiii_profile_mode='adaptive',oiii_method_version=VERSION,
        model_definition=asdict(selected_recipe),candidate_selection=candidates,
        profile_quality_flags=flags,profile_adequate=bool(selected.success and not any(x in flags for x in
            ('incomplete_doublet_coverage','remaining_coherent_residuals','unmatched_line_residual_structure',
             'excess_local_residual_chi2','boundary_solution','candidate_search_failed'))),
        residual_diagnostics=diag,oiii_ratio_5007_4959=selected.metadata.get('oiii_ratio_5007_4959',2.98),
        component_label_meaning='width ordering; no physical-origin classification',
        input_redshift=float(spectrum.z),adopted_redshift_unchanged=True)
    selected.metadata['profile_adequacy_definition']={'local_mean_chi2_max':4.,'single_line_run_sigma':3.,'single_line_run_pixels':3,'selection_trigger':'matched_doublet_only'}
    selected.warnings=[w for w in selected.warnings if w.code not in {'oiii_'+f for f in flags}]
    for flag in flags:
        selected.warnings.append(FitWarning(code='oiii_'+flag,message=flag.replace('_',' '),severity='info'))
