"""Continuously refined peaks of saved native line models (vacuum Angstrom)."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import numpy as np
from scipy.optimize import minimize_scalar

C_KMS = 299792.458
VERSION = 1


def profile_definitions(context):
    """Serialize actual model coordinates, including ties, without fit defaults."""
    from . import lines
    rows = []
    if hasattr(context, 'instances'):
        for ident, component, features, velocity, width in context.instances:
            for feature in features:
                rows.append(dict(component_id=ident, feature=feature, role=component.role,
                    reference_wave=lines.get(feature).vacuum_wavelength, profile=component.profile,
                    flux_parameter=f'{component.fixed_ratio_to or ident}.flux',
                    flux_divisor=component.fixed_ratio if component.fixed_ratio_to else 1.,
                    velocity_parameter=f'{velocity}.velocity_kms', width_parameter=f'{width}.fwhm_kms'))
        return rows
    # Built-in adapters use Gaussian area profiles with explicit shared groups.
    mapping = {'Hb': 'hbeta', 'Ha': 'halpha', 'MgII': 'mgii_blend',
        'OIII5007': 'oiii_5008', 'OIII4959': 'oiii_4960', 'HeII': 'heii_4687',
        'NII6585': 'nii_6585', 'NII6549': 'nii_6550', 'SII6718': 'sii_6718', 'SII6733': 'sii_6733'}
    for ident in context.components(np.asarray(context.initial), np.array([5000.])):
        base = ident.split('_')[0]
        if base not in mapping:
            raise ValueError(f'Unsupported native peak component: {ident}')
        role = 'broad' if '_broad' in ident else 'wing' if '_wing' in ident else 'narrow'
        group = ident if f'{ident}.velocity_kms' in context.names else 'wing' if role == 'wing' else 'narrow'
        flux, divisor = ident, 1.
        if base == 'OIII4959':
            flux = ident.replace('OIII4959', 'OIII5007')
            divisor = context.config.oiii_ratio_5007_4959
        elif base == 'NII6549':
            flux, divisor = 'NII6585', context.config.nii_ratio_6585_6549
        feature = mapping[base]
        rows.append(dict(component_id=ident, feature=feature, role=role,
            reference_wave=lines.get(feature).vacuum_wavelength, profile='gaussian',
            flux_parameter=flux+'.flux', flux_divisor=float(divisor),
            velocity_parameter=group+'.velocity_kms', width_parameter=group+'.fwhm_kms'))
    return rows


def evaluate_profile(wave, parameters, definitions):
    from .fitting.complexes import _profile
    wave = np.asarray(wave, dtype=float)
    output = np.zeros_like(wave)
    for row in definitions:
        width = parameters[row['width_parameter']]
        if width <= 0:
            return np.full_like(wave, np.nan)
        output += parameters[row['flux_parameter']] / row['flux_divisor'] * _profile(
            wave, row['reference_wave'], parameters[row['velocity_parameter']], width, row['profile'])[0]
    return output


def refined_peak(function, bounds, *, seeds=()):
    """Find all gridded maxima, refine each, and report competing peaks."""
    lo, hi = bounds
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return np.nan, 'invalid_window'
    grid = np.unique(np.r_[np.linspace(lo, hi, 1025), [x for x in seeds if lo < x < hi]])
    values = function(grid)
    if not np.isfinite(values).all() or np.max(values) <= 0:
        return np.nan, 'absent_line'
    candidates = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] > 0)) + 1
    peaks = []
    for i in candidates:
        # Optimize a local offset to avoid scipy's wavelength-relative tolerance.
        origin = grid[i]
        result = minimize_scalar(lambda offset: -float(function(np.array([origin+offset]))[0]),
            bounds=(grid[i-1]-origin, grid[i+1]-origin), method='bounded', options={'xatol': 1.e-9})
        peaks.append((float(-result.fun), float(origin+result.x)))
    peaks.sort(reverse=True)
    if not peaks or max(values[0], values[-1]) >= peaks[0][0]:
        return float(grid[np.argmax(values)]), 'boundary_peak'
    if len(peaks) > 1 and peaks[1][0] >= .99*peaks[0][0]:
        return peaks[0][1], 'ambiguous_peak'
    return peaks[0][1], 'available'


def _peak(parameters, rows, bounds):
    centers = [r['reference_wave']*np.exp(parameters[r['velocity_parameter']]/C_KMS) for r in rows]
    seeds = []
    for row, center in zip(rows, centers):
        width = parameters[row['width_parameter']]*center/C_KMS
        seeds.extend(center + width*np.array([-1., -.25, 0., .25, 1.]))
    return refined_peak(lambda w: evaluate_profile(w, parameters, rows), bounds, seeds=seeds)


def _covariance_for(fit, names):
    if fit.covariance is None:
        return None
    order = list(fit.metadata.get('covariance_parameter_names', fit.param_values))
    matrix = np.asarray(fit.covariance)
    if matrix.shape != (len(order), len(order)) or not set(names).issubset(order):
        return None
    ids = [order.index(n) for n in names]
    cov = matrix[np.ix_(ids, ids)]
    if not np.isfinite(cov).all() or any(not np.isfinite(fit.param_errors.get(n, np.nan)) for n in names):
        return None
    if np.any(np.diag(cov) < 0):
        return None
    scale = np.sqrt(np.maximum(np.diag(cov), 1.e-300))
    if np.linalg.eigvalsh(cov/np.outer(scale, scale)).min() < -1.e-8:
        return None
    return cov


def measure_peak(fit, rows, z, bounds):
    """Measure a fixed selection; covariance is conditional on that selection."""
    parameters = fit.param_values
    peak, status = _peak(parameters, rows, bounds)
    names = sorted({r[k] for r in rows for k in ('flux_parameter', 'velocity_parameter', 'width_parameter')})
    covariance = _covariance_for(fit, names)
    gradient = np.zeros(len(names))
    error = np.nan
    uncertainty_status = 'missing_or_nonidentifiable_covariance'
    if status == 'available' and covariance is not None:
        stable = True
        for i, name in enumerate(names):
            # Enough displacement to exceed the peak optimizer's tolerance.
            step = max(abs(parameters[name])*1.e-4, 1.e-3 if name.endswith('kms') else 1.e-6)
            estimates = []
            for factor in (1., 2.):
                plus, minus = dict(parameters), dict(parameters)
                plus[name] += factor*step; minus[name] -= factor*step
                p, ps = _peak(plus, rows, bounds); m, ms = _peak(minus, rows, bounds)
                estimates.append((p-m)/(2*factor*step))
                stable &= ps == ms == 'available'
            gradient[i] = estimates[0]
            stable &= np.isclose(*estimates, rtol=.05, atol=1.e-5)
        variance = float(gradient @ covariance @ gradient)
        if stable and variance > 0 and np.isfinite(variance):
            error = float(np.sqrt(variance)); uncertainty_status = 'available'
        else:
            uncertainty_status = 'unstable_or_zero_peak_derivative'
    elif status != 'available':
        uncertainty_status = status
    flux_gradient = np.array([sum(1/r['flux_divisor'] for r in rows if r['flux_parameter'] == name) for name in names])
    flux = float(sum(parameters[r['flux_parameter']]/r['flux_divisor'] for r in rows))
    flux_var = float(flux_gradient @ covariance @ flux_gradient) if covariance is not None else np.nan
    return dict(peak_rest_angstrom=float(peak), peak_observed_angstrom=float(peak*(1+z)),
        peak_error_rest_angstrom=error, peak_error_observed_angstrom=error*(1+z),
        status=status, uncertainty_status=uncertainty_status, flux=flux,
        flux_error=float(np.sqrt(flux_var)) if flux_var > 0 else np.nan,
        parameter_names=names, peak_gradient=gradient.tolist(), component_ids=[r['component_id'] for r in rows],
        definitions=rows, bounds=list(bounds), uncertainty_method='local_covariance',
        conditioning='fixed_profile_selection_host_continuum_and_input_redshift')


def record_fit_peaks(fit, z, *, definitions=None, bounds=None):
    """Attach additive native measurements; no fitting or change to redshift."""
    if not fit.success:
        return fit
    if definitions is not None:
        fit.metadata['peak_model'] = dict(version=VERSION, components=definitions, z=float(z), bounds=list(bounds))
    model = fit.metadata.get('peak_model')
    if not model:
        fit.metadata['peak_recovery_status'] = 'unsupported_profile_reconstruction'
        return fit
    signature = sha256(json.dumps([fit.param_values, fit.param_errors, model, None if fit.covariance is None else np.asarray(fit.covariance).tolist()], sort_keys=True).encode()).hexdigest()
    if fit.metadata.get('_peak_parameter_state') == signature and 'line_peaks' in fit.metadata:
        return fit
    rows = model['components']
    groups = defaultdict(list)
    for row in rows:
        groups[row['feature']+'_'+row['role']].append(row)
        groups[row['feature']+'_full'].append(row)
    # Distinct full-profile blends, never mixing the C III] neighboring lines.
    for feature, members in [('siiv_oiv', {'siiv_1394','siiv_1403','oiv_1401'}),
                             ('civ_doublet', {'civ_1548','civ_1551'}),
                             ('mgii_doublet', {'mgii_2796','mgii_2804'}),
                             ('oii_doublet', {'oii_3727','oii_3730'})]:
        selected = [r for r in rows if r['feature'] in members]
        if members.issubset({r['feature'] for r in selected}):
            groups[feature+'_full'] = selected
    for key in ('mgii_blend_full', 'civ_blend_full', 'mgii_doublet_full', 'civ_doublet_full'):
        if key in groups:
            original = groups[key]
            areas = [fit.param_values[r['flux_parameter']]/r['flux_divisor'] for r in original]
            groups[key+'_ws22'] = [r for r, area in zip(original, areas) if area >= .05*sum(areas) and area > 0]
    measurements, cache = {}, {}
    for key, selected in groups.items():
        identity = tuple((r['component_id'], r['feature']) for r in selected)
        if not selected:
            continue
        if identity not in cache:
            cache[identity] = measure_peak(fit, selected, z, model['bounds'])
        measurement = dict(cache[identity])
        reference = float({'siiv_oiv': 1399.41, 'civ_doublet': 1549.06, 'mgii_doublet': 2798.75, 'oii_doublet': 3728.48}.get(key.split('_full')[0], selected[0]['reference_wave']))
        measurement['reference_wave'] = reference
        peak = measurement['peak_rest_angstrom']
        measurement['peak_velocity_kms'] = float(C_KMS*np.log(peak/reference)) if peak > 0 else np.nan
        measurement['peak_velocity_error_kms'] = C_KMS*measurement['peak_error_rest_angstrom']/peak if peak > 0 else np.nan
        measurements[key] = measurement
        fit.metrics[f'{key}_peak_selection_flux'] = measurement['flux']
        fit.metric_errors[f'{key}_peak_selection_flux'] = measurement['flux_error']
        for suffix in ('rest_angstrom', 'observed_angstrom', 'velocity_kms'):
            fit.metrics[f'{key}_peak_{suffix}'] = measurement[f'peak_{suffix}']
            error_key = f'peak_error_{suffix}' if suffix != 'velocity_kms' else 'peak_velocity_error_kms'
            fit.metric_errors[f'{key}_peak_{suffix}'] = measurement[error_key]
    fit.metadata['line_peaks'] = dict(version=VERSION, input_redshift=float(z), measurements=measurements)
    fit.metadata['_peak_parameter_state'] = signature
    lo, hi = model['bounds']
    wave = fit.wave_rest
    in_window = (wave >= lo) & (wave <= hi)
    local_wave = wave[in_window]
    step = float(np.median(np.diff(local_wave))) if len(local_wave) > 1 else np.inf
    fit.metadata['peak_pixel_coverage_fraction'] = float(min(1., np.count_nonzero(fit.fit_mask & in_window)/max((hi-lo)/step, 1.)))
    fit.metadata['peak_recovery_status'] = 'available'
    return fit


def recover_line_peaks(result):
    """Explicitly add peaks to a loaded workflow, preserving its adopted frame.

    Legacy reconstruction is accepted only after matching archived component arrays.
    """
    statuses = {}
    for name, fit in result.line_complexes.items():
        if fit.success and 'peak_model' not in fit.metadata:
            recovered = _recover_native_definitions(fit)
            if recovered is not None:
                rows, bounds = recovered
                record_fit_peaks(fit, result.spectrum.z, definitions=rows, bounds=bounds)
                fit.metadata['peak_recovery_provenance'] = 'validated_against_archived_component_arrays'
        record_fit_peaks(fit, result.spectrum.z)
        statuses[name] = fit.metadata.get('peak_recovery_status', 'fit_failed')
    return statuses


def _recover_native_definitions(fit):
    """Validate a candidate description against every archived component array."""
    from dataclasses import replace
    from types import SimpleNamespace
    from . import complex_recipes
    from .fitting.complexes import GenericComplexContext
    recipe_id = fit.metadata.get('recipe_id')
    try:
        recipe = complex_recipes.get(recipe_id) if recipe_id else None
        if recipe is not None and recipe.backend == 'generic':
            context = GenericComplexContext(replace(recipe, continuum_mode='fixed_global'),
                fit.metadata.get('active_components', ()), 1.)
            rows = profile_definitions(context)
            bounds = (min(w[0] for w in recipe.fit_windows), max(w[1] for w in recipe.fit_windows))
        else:
            context = SimpleNamespace(names=list(fit.param_values), initial=list(fit.param_values.values()),
                components=lambda *_: fit.component_models,
                config=SimpleNamespace(oiii_ratio_5007_4959=fit.metadata.get('oiii_ratio_5007_4959', 2.98),
                    nii_ratio_6585_6549=fit.metadata.get('nii_ratio_6585_6549', 2.96)))
            rows = profile_definitions(context)
            selected = fit.wave_rest[fit.fit_mask]
            if not len(selected):
                return None
            bounds = (float(selected.min()), float(selected.max()))
        by_component = defaultdict(list)
        for row in rows:
            by_component[row['component_id']].append(row)
        if not by_component or set(by_component) != {name for name in fit.component_models if not name.startswith('local_continuum_')}:
            return None
        for ident, entries in by_component.items():
            rendered = evaluate_profile(fit.wave_rest, fit.param_values, entries)
            archived = fit.component_models[ident]
            if not np.allclose(rendered, archived, rtol=2.e-6, atol=max(float(np.max(np.abs(archived)))*1.e-8, 1.e-12)):
                return None
        return rows, bounds
    except (KeyError, ValueError, AttributeError, TypeError):
        return None
