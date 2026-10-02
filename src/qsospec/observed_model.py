"""Deterministic total-model reconstruction for observed-frame comparisons."""
from __future__ import annotations
import numpy as np


def reconstruct_observed_model(result):
    """Return total f_lambda in physical cgs on the original observed grid.

    Each line-complex owns its registered windows. In overlaps the narrowest
    window wins, then recipe ID; local continua are never summed twice. Outside
    those windows the global continuum plus host is the model. Failed registered
    windows are unsupported, even if an overlapping successful fit exists.
    Fit clipping masks do not participate in support or native validity.
    """
    from .complex_recipes import get
    spectrum = result.total_spectrum if result.total_spectrum is not None else result.spectrum
    wave = np.asarray(spectrum.wave_obs, float)
    if np.any(np.diff(wave)<=0): raise ValueError('Observed grid must be strictly increasing')
    scale = spectrum.flux_scale
    if scale is None: raise ValueError('Physical flux scale is required')
    factor = float(scale)/(1+spectrum.z) if spectrum.metadata.flux_frame=='rest' else float(scale)
    continuum = np.asarray(result.continuum.model,float)
    if continuum.shape!=wave.shape: raise ValueError('Continuum grid mismatch')
    host = result.host_model_on_quasar_grid
    host = np.zeros_like(continuum) if host is None else np.asarray(host,float)
    model = continuum+host
    supported = np.isfinite(model)
    failed = np.zeros_like(supported)
    owner = np.full(wave.shape,'',dtype=object)
    windows, claims, evidence = [], [], []
    for recipe_id, fit in sorted(result.line_complexes.items()):
        recipe = get(recipe_id)
        for lo,hi in (recipe.fit_windows or (recipe.fit_window,)):
            lo,hi=lo*(1+spectrum.z),hi*(1+spectrum.z)
            windows.append((lo,hi)); region=(wave>=lo)&(wave<=hi)
            if not fit.success: failed |= region; continue
            claims.append((hi-lo,recipe_id,lo,hi,fit))
        bad = any(any(s in w.code.lower() for s in ('bound','failed','rank_deficient')) for w in fit.warnings)
        for name,value in fit.metrics.items():
            if not name.endswith('_flux_cgs'): continue
            error=fit.metric_errors.get(name,np.nan)
            prefix=name[:-len('_flux_cgs')]
            center=fit.metrics.get(prefix+'_centroid',np.nan)
            if not np.isfinite(center):
                center=np.mean(recipe.fit_window)
            evidence.append(dict(recipe_id=recipe_id,quantity=name,wavelength=float(center)*(1+spectrum.z),
                                 snr=float(value/error) if error is not None and np.isfinite(error) and error>0 else 0.,
                                 usable=bool(fit.success and not bad),narrow=('narrow' in name or 'core' in name)))
    for _,recipe_id,lo,hi,fit in sorted(claims,key=lambda x:(x[0],x[1],x[2])):
        mask=(wave>=lo)&(wave<=hi)&(owner=='')
        values=np.interp(wave,np.asarray(fit.wave_rest)*(1+spectrum.z),fit.model,left=np.nan,right=np.nan)
        model[mask]+=values[mask]; owner[mask]=recipe_id
    supported &= ~failed & np.isfinite(model)
    return dict(wavelength=wave,flux=np.asarray(spectrum.flux)*factor,error=np.asarray(spectrum.err)*factor,
                valid=np.asarray(spectrum.valid_mask,bool),model=model*factor,continuum=(continuum+host)*factor,
                supported=supported,windows=windows,line_evidence=evidence,
                success=bool(result.continuum.success),complex_owner=owner,
                definition='observed_total_v1_narrowest_window_owner',flux_unit='erg s-1 cm-2 Angstrom-1')
