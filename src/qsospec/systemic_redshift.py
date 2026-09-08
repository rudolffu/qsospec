"""Optional WS22 systemic-redshift diagnostics; never changes the fitted frame."""
from __future__ import annotations
import numpy as np
from .line_peaks import C_KMS, _covariance_for, record_fit_peaks

# Vacuum wavelengths: Shen+2016 Table 1. Shifts: Tables 2/3 and section 5.
# Negative offsets are blueshifts of the measured line from systemic.
# Isolated C III] uses its own Table 3 row, not the Si III]/Al III] blend row.
WS22_CALIBRATION = {
    'version': 'ws22-shen16-v1',
    'sources': ['https://arxiv.org/html/2209.03987#S4.SS2', 'https://arxiv.org/html/1602.03894#S5'],
    'lines': {
        'hbeta': dict(peak='hbeta_broad', wavelength=4862.68, offset=-109., scatter=400.),
        'oiii': dict(peak='oiii_5008_full', wavelength=5008.24, offset=-48., scatter=56.),
        'caii': dict(peak='caii_3935_full', wavelength=3934.78, offset=0., scatter=0.),
        'oii': dict(peak='oii_blend_full', wavelength=3728.48, offset=8., scatter=46.),
        'mgii': dict(peak='mgii_blend_full_ws22', wavelength=2798.75, offset=-57., scatter=205.),
        'ciii': dict(peak='ciii_1909_full', wavelength=1908.73, offset=-143., scatter=243.),
        'civ': dict(peak='civ_blend_full_ws22', wavelength=1549.06, offset=-27., scatter=415., a=-242., b=-438., logL0=45.),
        'siiv': dict(peak='siiv_oiv_full', wavelength=1399.41, offset=-28., scatter=477., a=-123., b=-345., logL0=45.),
    },
    'excluded_lines': ['heii_1640', 'nev_3427'],
}


def velocity_correction(line, log_l1700=None):
    """Return mean line offset and derivative with respect to log10 L1700."""
    row = WS22_CALIBRATION['lines'][line]
    if 'b' not in row:
        return row['offset'], 0.
    if log_l1700 is None or not np.isfinite(log_l1700):
        return np.nan, np.nan
    if log_l1700 < 44.5:
        return 0., 0.
    return row['offset'] + row['a'] + row['b']*(log_l1700-row['logL0']), row['b']


def luminosity_1700(result):
    """Use the saved power law plus polynomial; no iron, host, or Balmer flux."""
    from astropy.cosmology import FlatLambdaCDM
    spectrum, fit = result.spectrum, result.continuum
    output = dict(status='unavailable', log_l1700=np.nan, error_dex=np.nan,
        source='saved_power_law_plus_polynomial', cosmology='FlatLambdaCDM(H0=70, Om0=0.3)',
        parameter_names=[], gradient=[])
    valid = spectrum.wave_rest[spectrum.valid_mask]
    scale = spectrum.flux_density_scale_to_cgs
    if not len(valid) or not valid.min() <= 1700 <= valid.max() or scale is None or spectrum.z <= 0:
        return output
    components = fit.component_models
    if 'power_law' not in components:
        return output
    flux = float(np.interp(1700., fit.wave_rest, components['power_law']+components.get('polynomial', 0.)))
    if flux <= 0:
        return output
    factor = 4*np.pi*FlatLambdaCDM(H0=70, Om0=.3).luminosity_distance(spectrum.z).to_value('cm')**2*1700*scale
    output.update(status='available', log_l1700=float(np.log10(factor*flux)))
    # Reconstruct only when the saved continuum coordinates are sufficient.
    names = [n for n in fit.param_values if n.startswith(('power_law.', 'polynomial.'))]
    meta = fit.metadata
    if 'power_law_pivot' not in meta or not names:
        return output
    def sample(p):
        pivot = meta['power_law_pivot']
        w = 1700.
        edge = meta.get('power_law_break_wave')
        if meta.get('power_law_mode_selected', 'single') == 'double' and edge is not None and w >= edge:
            value = p['power_law.norm']*(edge/pivot)**p['power_law.slope']*(w/edge)**p['power_law.red_slope']
        else:
            value = p['power_law.norm']*(w/pivot)**p['power_law.slope']
        for name in names:
            if name.startswith('polynomial.c'):
                value += p[name]*((w-meta['polynomial_pivot'])/meta['polynomial_scale'])**int(name.split('.c')[1])
        return value
    try:
        value = sample(fit.param_values)
        if not np.isclose(value, flux, rtol=1.e-3, atol=1.e-10):
            output['covariance_status'] = 'saved_continuum_reconstruction_mismatch'
            return output
        gradient = []
        for name in names:
            step = max(abs(fit.param_values[name])*1.e-5, 1.e-6)
            p, m = dict(fit.param_values), dict(fit.param_values)
            p[name] += step; m[name] -= step
            gradient.append((sample(p)-sample(m))/(2*step*value*np.log(10)))
        cov = _covariance_for(fit, names)
        output.update(parameter_names=names, gradient=gradient)
        variance = np.asarray(gradient) @ cov @ gradient if cov is not None else np.nan
        if np.isfinite(variance) and variance >= 0:
            output['error_dex'] = float(np.sqrt(variance))
    except (KeyError, ValueError):
        output['covariance_status'] = 'unsupported_saved_continuum_coordinates'
    return output


def combine_redshifts(redshifts, covariance):
    """WS22 inverse marginal variance weights; retain covariance in mean error."""
    z, cov = np.asarray(redshifts), np.asarray(covariance)
    weights = 1/np.diag(cov)
    mean = float(weights @ z/weights.sum())
    keep = np.ones(len(z), dtype=bool)
    if len(z) >= 3:
        mad = float(np.median(np.abs(z-np.median(z))))
        if mad > 0:
            keep = np.abs(z-mean) <= 3*mad
    if not np.any(keep):
        return np.nan, np.nan, keep, np.zeros_like(weights)
    weights = np.where(keep, weights, 0.); weights /= weights.sum()
    return float(weights @ z), float(np.sqrt(weights @ cov @ weights)), keep, weights


def estimate_systemic_redshift(result, method='ws22'):
    """Attach and return a diagnostic, leaving the adopted input redshift intact.

    Errors condition on the fitted model; empirical line scatter is added.
    Missing or unstable statistical errors exclude a line from the combination.
    """
    if method != 'ws22':
        raise ValueError("Supported systemic-redshift method: 'ws22'")
    z = float(result.spectrum.z)
    luminosity = luminosity_1700(result)
    mc = getattr(result, 'monte_carlo', {})
    if np.isfinite(mc.get('errors', {}).get('ws22_log_l1700', np.nan)):
        luminosity['error_dex'] = mc['errors']['ws22_log_l1700']
    output = dict(method=method, calibration_version=WS22_CALIBRATION['version'],
        sources=WS22_CALIBRATION['sources'], status='unavailable', input_redshift=z,
        z_sys=None, z_sys_error=None, offset_kms=None, lines={}, luminosity=luminosity,
        uncertainty_assumptions=['conditional_on_fitted_model_and_input_frame',
            'intrinsic_line_scatters_independent', 'unavailable_cross_fit_correlations_assumed_zero'],
        clipping='one_pass_3_unscaled_MAD_about_initial_weighted_mean', adopted=False)
    candidates = []
    for line, calibration in WS22_CALIBRATION['lines'].items():
        found = []
        for recipe, fit in result.line_complexes.items():
            if not fit.success:
                continue
            if 'line_peaks' not in fit.metadata:
                record_fit_peaks(fit, z)
            measurement = fit.metadata.get('line_peaks', {}).get('measurements', {}).get(calibration['peak'])
            if measurement is not None:
                found.append((recipe, fit, measurement))
        if not found:
            output['lines'][line] = dict(status='excluded', reason='compatible_peak_unavailable')
            continue
        # Do not double-count overlapping recipes: prefer smallest valid error.
        recipe, fit, peak = min(found, key=lambda item: item[2]['peak_error_rest_angstrom'] if np.isfinite(item[2]['peak_error_rest_angstrom']) else np.inf)
        row = dict(recipe=recipe, peak_key=calibration['peak'], calibration=dict(calibration),
            peak_rest_angstrom=peak['peak_rest_angstrom'], component_ids=peak['component_ids'],
            status='excluded', reason=None)
        output['lines'][line] = row
        coverage = fit.metadata.get('peak_pixel_coverage_fraction', 0.)
        row['coverage_fraction'] = float(coverage)
        flux_error = peak['flux_error']
        row['flux_snr'] = peak['flux']/flux_error if flux_error > 0 else np.nan
        reason = None
        if peak['status'] != 'available':
            reason = peak['status']
        elif not np.isfinite(peak['peak_error_rest_angstrom']) or peak['peak_error_rest_angstrom'] <= 0:
            reason = 'peak_uncertainty_unavailable'
        elif coverage < .5:
            reason = 'insufficient_coverage'
        elif not row['flux_snr'] > 2.:
            reason = 'line_detection_below_2sigma'
        offset, slope = velocity_correction(line, luminosity['log_l1700'])
        if not np.isfinite(offset):
            reason = 'required_luminosity_unavailable'
        elif slope and not np.isfinite(luminosity['error_dex']):
            reason = 'required_luminosity_uncertainty_unavailable'
        if reason:
            row['reason'] = reason
            continue
        # v = c (z_line-z_sys)/(1+z_sys); preserves the published small-v convention.
        derivative = (1+z)/calibration['wavelength']/(1+offset/C_KMS)
        value = derivative*peak['peak_rest_angstrom']-1
        d_lum = -(1+value)*slope/(C_KMS+offset)
        statistical = (derivative*peak['peak_error_rest_angstrom'])**2
        if slope:
            statistical += (d_lum*luminosity['error_dex'])**2
        intrinsic = (calibration['scatter']*(1+value)/C_KMS)**2
        row.update(status='used', reason=None, z_line=(1+z)*peak['peak_rest_angstrom']/calibration['wavelength']-1,
            z_sys=float(value), correction_kms=float(offset), measurement_error=float(np.sqrt(statistical)),
            intrinsic_error=float(np.sqrt(intrinsic)), total_error=float(np.sqrt(statistical+intrinsic)))
        candidates.append((line, fit, peak, derivative, d_lum, statistical+intrinsic))
    if candidates:
        cov = np.diag([c[5] for c in candidates])
        for i, (_, fit, peak, d, dl, _) in enumerate(candidates):
            for j in range(i):
                _, other_fit, other, od, odl, _ = candidates[j]
                cross = dl*odl*luminosity['error_dex']**2 if dl and odl else 0.
                if fit is other_fit and peak['uncertainty_method'] == other['uncertainty_method'] == 'local_covariance':
                    names = sorted(set(peak['parameter_names']) | set(other['parameter_names']))
                    pcov = _covariance_for(fit, names)
                    if pcov is not None:
                        pg = dict(zip(peak['parameter_names'], peak['peak_gradient']))
                        og = dict(zip(other['parameter_names'], other['peak_gradient']))
                        cross += d*od*np.array([pg.get(n,0.) for n in names]) @ pcov @ np.array([og.get(n,0.) for n in names])
                cov[i,j] = cov[j,i] = cross
        # Matched trials preserve both line-line and line-luminosity correlations.
        draw_vectors = []
        for trial in mc.get('draws', []):
            vector = []
            for line, fit, peak, derivative, dl, _ in candidates:
                record = output['lines'][line]
                wavelength = trial['values'].get(f"{record['recipe']}:{record['peak_key']}_peak_rest_angstrom", np.nan)
                offset, _ = velocity_correction(line, trial['values'].get('ws22_log_l1700', np.nan))
                vector.append((1+z)*wavelength/record['calibration']['wavelength']/(1+offset/C_KMS)-1)
            if np.isfinite(vector).all():
                draw_vectors.append(vector)
        if len(draw_vectors) >= 2:
            cov = np.atleast_2d(np.cov(draw_vectors, rowvar=False))
            cov += np.diag([output['lines'][c[0]]['intrinsic_error']**2 for c in candidates])
            output['uncertainty_method'] = 'matched_existing_bootstrap'
            output['valid_matched_trials'] = len(draw_vectors)
            output['uncertainty_assumptions'].remove('unavailable_cross_fit_correlations_assumed_zero')
        else:
            output['uncertainty_method'] = 'local_covariance'
        for i, candidate in enumerate(candidates):
            record = output['lines'][candidate[0]]
            record['total_error'] = float(np.sqrt(cov[i, i]))
            record['measurement_error'] = float(np.sqrt(max(0., cov[i, i]-record['intrinsic_error']**2)))
        values = [output['lines'][c[0]]['z_sys'] for c in candidates]
        mean, error, keep, weights = combine_redshifts(values, cov)
        for candidate, accepted, weight in zip(candidates, keep, weights):
            row = output['lines'][candidate[0]]
            row['weight'] = float(weight)
            if not accepted:
                row.update(status='excluded', reason='redshift_outlier')
        if np.isfinite(mean):
            output.update(status='available', z_sys=mean, z_sys_error=error,
                offset_kms=C_KMS*(mean-z)/(1+z))
        output['line_order'] = [c[0] for c in candidates]
        output['redshift_covariance'] = cov.tolist()
    result.metadata['systemic_redshift'] = output
    return output
