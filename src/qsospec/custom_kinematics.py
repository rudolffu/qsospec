"""Opt-in serializable recipes and diagnostics for complex gas kinematics.

No registry/default recipe is changed. Velocities use the input spectrum frame.
"""
from dataclasses import asdict, replace
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from .complex_recipes import ComponentRecipe, ComplexRecipe
from .fitting.complexes import fit_generic_complex

C = 299792.458
PROBABILITIES = np.array([.02,.05,.10,.50,.90,.95,.98])


def custom_kinematics(spec):
    """Build a generic recipe from a JSON-compatible dictionary.

    Keys: id, window, components, optional continuum_mode and initial_values.
    Component keys follow ComponentRecipe, including independent velocity/width
    groups and fixed_ratio_to. All components are explicit, never auto-rejected.
    """
    components=[]
    for raw in spec['components']:
        d=dict(raw)
        for key in ('line_ids','velocity_bounds_kms','flux_bounds'):
            if key in d:d[key]=tuple(d[key])
        if 'fwhm_bands_kms' in d:d['fwhm_bands_kms']=tuple(tuple(x) for x in d['fwhm_bands_kms'])
        if d.get('selection_rule'):raise ValueError('Use explicit candidate selection')
        vb=d.get('velocity_bounds_kms',(-1000,1000)); bands=d.get('fwhm_bands_kms',((70,1200),))
        if not np.all(np.isfinite(vb)) or vb[0]>=vb[1]:raise ValueError('Invalid velocity bounds')
        if any(not 0<a<b or not np.isfinite(b) for a,b in bands):raise ValueError('Invalid FWHM bounds')
        components.append(ComponentRecipe(**d))
    ids=[x.id for x in components]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate component IDs')
    for c in components:
        if c.fixed_ratio_to and c.fixed_ratio_to not in ids:raise ValueError('Missing ratio parent')
    for kind in ('velocity','width'):
        groups={}
        for c in components:
            group=getattr(c,f'{kind}_group') or c.kinematic_group or c.id
            bounds=c.velocity_bounds_kms if kind=='velocity' else c.fwhm_bands_kms
            if group in groups and groups[group]!=bounds:raise ValueError('Incompatible tied bounds')
            groups[group]=bounds
    window=tuple(spec['window'])
    return ComplexRecipe(id=spec['id'],aliases=(),label=spec.get('label',spec['id']),
        fit_window=window,fit_windows=(window,),mask_windows=(),components=tuple(components),
        required_line_ids=(),coverage_mode='component_adaptive',min_valid_pixels=30,
        continuum_mode=spec.get('continuum_mode','residual_linear'),backend='generic')


def fit_custom_kinematics(spectrum, continuum, spec, *, n_starts=24, seed=1729,
                          initial_values=None, compute_covariance=True):
    return fit_generic_complex(spectrum,continuum,custom_kinematics(spec),
        n_starts=n_starts,random_seed=seed,initial_values=initial_values or spec.get('initial_values'),
        compute_covariance=compute_covariance,defer_peaks=True)


def component_measurements(result, spec):
    """Resolved components with covariance errors and explicit input-flux units."""
    rows=[]
    for c in custom_kinematics(spec).components:
        vg=c.velocity_group or c.kinematic_group or c.id
        wg=c.width_group or c.kinematic_group or c.id
        fkey=f'{c.fixed_ratio_to or c.id}.flux';vkey=f'{vg}.velocity_kms';wkey=f'{wg}.fwhm_kms'
        if fkey not in result.param_values:continue
        ratio=1./c.fixed_ratio if c.fixed_ratio_to else 1.
        rows.append(dict(component=c.id,line_id=c.line_ids[0],role=c.role,
            flux=result.param_values[fkey]*ratio,flux_error=result.param_errors.get(fkey,np.nan)*ratio,
            velocity_kms=result.param_values[vkey],velocity_error_kms=result.param_errors.get(vkey,np.nan),
            fwhm_kms=result.param_values[wkey],fwhm_error_kms=result.param_errors.get(wkey,np.nan),
            velocity_group=vg,width_group=wg))
    # Stable velocity ordering, preserving the original component/group keys.
    rows.sort(key=lambda x:(x['line_id'],x['velocity_kms'],x['component']))
    for line in set(x['line_id'] for x in rows):
        for i,row in enumerate(x for x in rows if x['line_id']==line):row['velocity_order']=i+1
    return rows


def mixture_kinematics(components, reference_velocity=0., line_id='oiii_5008'):
    """Quantiles of any number of positive wavelength-Gaussian components.

    Line filtering excludes 4960. Metrics are missing for invalid models.
    """
    out={f'v{int(p*100):02d}_kms':np.nan for p in PROBABILITIES}
    out.update(w80_kms=np.nan,w90_kms=np.nan,status='invalid')
    rows=[x for x in components if x['line_id']==line_id]
    if not rows:out['reason']='missing_line';return out
    a=np.array([[x['flux'],x['velocity_kms'],x['fwhm_kms']] for x in rows])
    if np.any(~np.isfinite(a[:,0])) or np.any(a[:,0]<0):out['reason']='invalid_flux';return out
    a=a[a[:,0]>0]
    if not len(a) or np.any(~np.isfinite(a)) or np.any(a[:,2]<=0):out['reason']='invalid_components';return out
    weights=a[:,0]/a[:,0].max();weights=weights/weights.sum();centers=np.exp(a[:,1]/C);sig=centers*a[:,2]/C/2.354820045
    lo=max(np.finfo(float).tiny,np.min(centers-12*sig));hi=np.max(centers+12*sig)
    try:q=np.array([C*np.log(brentq(lambda x:np.dot(weights,ndtr((x-centers)/sig))-p,lo,hi,xtol=1e-14))-reference_velocity for p in PROBABILITIES])
    except ValueError:out['reason']='unbracketed';return out
    out.update({f'v{int(p*100):02d}_kms':float(v) for p,v in zip(PROBABILITIES,q)})
    out.update(w80_kms=float(q[4]-q[2]),w90_kms=float(q[5]-q[1]),status='valid',reason='')
    return out


def kinematic_bounds_hit(result, spec):
    hits={}
    recipe=custom_kinematics(spec)
    for c in recipe.components:
        for name,bounds in [(f'{c.velocity_group or c.kinematic_group or c.id}.velocity_kms',c.velocity_bounds_kms),
                            (f'{c.width_group or c.kinematic_group or c.id}.fwhm_kms',c.fwhm_bands_kms[0])]:
            if name not in result.param_values:continue
            value=result.param_values[name];tol=1e-4*(bounds[1]-bounds[0])
            if abs(value-bounds[1])<=tol or (name.endswith('velocity_kms') and abs(value-bounds[0])<=tol):hits[name]=list(bounds)
    return hits


def select_nested_candidates(candidates, family, *, peer_widths=False, delta_bic=20., min_snr=5.):
    """Select nested explicit candidates, retaining a complete decision trace.

    Each move adds one independent kinematic group or releases a width tie.
    For peer-width alternatives the tied model must be visited before its
    independent-width version. Flux significance is computed for the sum of
    free doublet fluxes belonging to each added kinematic group.
    """
    keys=[k for k in candidates if k.startswith(family)]
    if not keys:return None,[]
    selected=keys[0];visited={selected};audit=[]
    while True:
        oldspec,old=candidates[selected];oldids={c['id'] for c in oldspec['components']};choices=[]
        for key in keys:
            if key in visited:continue
            sp,r=candidates[key];newids={c['id'] for c in sp['components']};added=newids-oldids
            if not oldids.issubset(newids):continue
            newgroups={c['velocity_group'] for c in sp['components'] if c['id'] in added}
            release=not added and len(r.param_values)>len(old.param_values)
            if len(newgroups)!=1 and not release:continue
            if peer_widths and added:
                oiii=[c for c in sp['components'] if c['line_ids'][0]=='oiii_5008']
                if len(oiii)>1 and len({c['width_group'] for c in oiii})>1:continue
            snrs=[]
            for group in newgroups:
                fluxkeys=[c['id']+'.flux' for c in sp['components'] if c['id'] in added and c['velocity_group']==group and not c.get('fixed_ratio_to')]
                names=list(r.param_values);variance=np.nan
                if r.covariance is not None and all(k in names for k in fluxkeys):
                    ix=[names.index(k) for k in fluxkeys];variance=r.covariance[np.ix_(ix,ix)].sum()
                snrs.append(sum(r.param_values.get(k,0.) for k in fluxkeys)/np.sqrt(variance) if variance>0 else 0.)
            snr=min(snrs,default=np.nan)
            accept=bool(old.success and r.success and old.bic-r.bic>=delta_bic and (release or snr>=min_snr))
            audit.append(dict(family=family,baseline=selected,candidate=key,delta_bic=old.bic-r.bic,added_snr=snr,accepted=accept,reason='width_release' if release else 'add_component'))
            if accept:choices.append(key)
        if not choices:break
        selected=min(choices,key=lambda k:candidates[k][1].bic);visited.add(selected)
    return selected,audit


def refit_custom_continuum(spectrum, baseline, config):
    """Warm-refit a selected continuum and its native soft Hgamma constraint.

    Intended for matched noise realizations: retain the selected continuum
    family and accepted continuum pixels while reoptimizing its parameters.
    This never runs host decomposition or Hbeta synchronization.
    """
    from copy import deepcopy
    from .complex_recipes import get
    from .fitting.global_fit import _joint_hgamma_refinement, _fit_global_continuum_fixed
    config=replace(config,power_law=replace(config.power_law,
        mode=baseline.metadata.get('power_law_mode_selected',config.power_law.mode)),
        blue_absorption_clip_enabled=False,clip_passes=0,
        balmer_pseudocontinuum=replace(config.balmer_pseudocontinuum,sync_with_hbeta='never'))
    if baseline.metadata.get('hgamma_joint_status')=='fit':
        previous=deepcopy(baseline)
        hgamma=fit_generic_complex(spectrum,previous,get('oii_nev_neiii_hgamma'),
            compute_covariance=False,defer_peaks=True)
        if hgamma is None or not hgamma.success:raise RuntimeError('hgamma_refit_failed')
        result,_=_joint_hgamma_refinement(spectrum,config,previous,hgamma,False)
        if result.metadata.get('hgamma_joint_status')!='fit':raise RuntimeError('joint_continuum_refit_failed')
        return result
    return _fit_global_continuum_fixed(spectrum,config,compute_covariance=False,
        fit_mask_override=baseline.clip_mask,initial_parameters=baseline.param_values)
