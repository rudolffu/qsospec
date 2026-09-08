"""Native covariance propagation and matched workflow resampling utilities.

Gaussian errors are local, conditional approximations. Unknown covariance is
never replaced with independent marginal errors.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import numpy as np


def covariance_block(result, scope):
    names = list(result.metadata.get("covariance_parameter_names", result.param_values))
    covariance = result.covariance
    if covariance is not None and len(names) != len(covariance):
        raise ValueError("Covariance needs explicit matching parameter coordinates")
    return {
        "version": 1, "scope": scope, "parameter_names": names,
        "coordinates": "native_linear", "values": None if covariance is None else np.asarray(covariance).tolist(),
        "shape": [len(names), len(names)],
        "status": "unavailable" if covariance is None else "available",
        "method": "local_gaussian", "conditioning": "fixed_continuum" if scope != "continuum" else "fixed_host",
        "noise_scaling": result.metadata.get("covariance_noise_scaling", "residual_scaled"),
        "model_metadata": {key:value for key,value in result.metadata.items()
            if key not in ("covariance_blocks","matched_uncertainty_draws")},
        "fixed_parameter_values": {name:value for name,value in result.param_values.items() if name not in names},
        "warnings": [warning.to_dict() if hasattr(warning,"to_dict") else {"code":warning.code,"context":warning.context} for warning in result.warnings],
        "final_state_id": sha256(np.asarray(result.model, dtype='<f8').tobytes()).hexdigest(),
        "cross_block_covariance": "unavailable",
    }


def restore_covariance(block, parameters):
    if not block or block.get("values") is None:
        return None
    names = block["parameter_names"]
    matrix = np.asarray(block["values"], dtype=float)
    if matrix.shape != (len(names), len(names)) or not set(names).issubset(parameters):
        raise ValueError("Covariance parameter ordering or shape does not match archived parameters")
    output = np.full((len(parameters), len(parameters)), np.nan)
    indices = [list(parameters).index(name) for name in names]
    output[np.ix_(indices, indices)] = matrix
    return output


def propagate(function, values, covariance):
    """Central finite-difference propagation through the actual measurement."""
    theta = np.asarray(values, dtype=float)
    point = np.atleast_1d(function(theta)).astype(float)
    if covariance is None:
        return point, None
    jac = np.empty((len(point), len(theta)))
    for i, value in enumerate(theta):
        step = max(abs(value)*1.e-5, 1.e-6)
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        jac[:, i] = (np.atleast_1d(function(plus))-np.atleast_1d(function(minus)))/(2*step)
    return point, jac @ np.asarray(covariance) @ jac.T


def host_agn_covariance(host, agn, covariance):
    """Transform the *joint* (host, AGN) covariance to (sum, host fraction)."""
    denominator = host+agn
    if denominator <= 0 or covariance is None:
        return None
    jac = np.array([[1., 1.], [agn/denominator**2, -host/denominator**2]])
    return jac @ np.asarray(covariance) @ jac.T


def summarize_matched_draws(draws, failures, requested):
    names = sorted({name for row in draws for name in row["values"]})
    matrix = np.array([[row["values"].get(name, np.nan) for name in names] for row in draws], dtype=float).reshape(len(draws), len(names))
    intervals, errors, counts = {}, {}, {}
    for index, name in enumerate(names):
        good = matrix[:, index][np.isfinite(matrix[:, index])]
        counts[name] = len(good)
        if len(good) >= 1:
            p16, p50, p84 = np.percentile(good, [16., 50., 84.])
            intervals[name] = dict(p16=float(p16), p50=float(p50), p84=float(p84))
            errors[name] = float(np.std(good, ddof=1)) if len(good)>=2 else np.nan
    # A single common subset gives a PSD covariance. Pairwise deletion would
    # combine different trial populations and need not give a PSD matrix.
    covariance_indices = [i for i,name in enumerate(names) if counts[name]>=2]
    covariance_matrix = matrix[:,covariance_indices]
    complete = np.all(np.isfinite(covariance_matrix), axis=1)
    covariance = np.atleast_2d(np.cov(covariance_matrix[complete], rowvar=False)).tolist() if complete.sum() >= 2 and covariance_indices else None
    return {"method": "parametric_bootstrap", "noise_model": "diagonal_pixel_errors",
            "n_requested": int(requested), "n_successful": len(draws), "failures": failures,
            "draws": draws, "measurement_names": names, "measurement_covariance": covariance,
            "covariance_measurement_names": [names[i] for i in covariance_indices],
            "covariance_trial_ids": [row["trial_id"] for row, ok in zip(draws, complete) if ok],
            "valid_trial_counts": counts, "percentiles": intervals, "errors": errors,
            "point_estimate_policy": "original_best_fit_unchanged"}


def trial_seed(seed, object_id, trial_id):
    digest = sha256(str(object_id).encode()).digest()
    return np.random.SeedSequence([int(seed or 0), int.from_bytes(digest[:4], 'little'), int(trial_id)])


def workflow_measurements(result):
    from .systemic_redshift import luminosity_1700
    values = dict(result.metadata.get("continuum_samples", {}))
    values["ws22_log_l1700"] = luminosity_1700(result)["log_l1700"]
    values.update(result.continuum.param_values)
    for recipe, fit in result.line_complexes.items():
        if fit.success:
            values.update({f"{recipe}:{name}": value for name, value in fit.metrics.items()})
            values.update(fit.metrics)
    return {name: float(value) for name, value in values.items() if np.isscalar(value)}


def recover_uncertainties(result, output_path):
    """Write an immutable recovery report; never reinterpret an archived model."""
    path = Path(output_path)
    report = {"version": 1, "object_id": str(result.metadata.get("object_id", "unknown")), "quantities": {}}
    stored = result.metadata.get("continuum_sample_errors", {})
    for name, value in result.metadata.get("continuum_samples", {}).items():
        error = stored.get(name)
        status = "stored_error" if error is not None and np.isfinite(error) else "requires_refit"
        reason = "No saved joint uncertainty for this quantity"
        fit = result.continuum
        pivot = fit.metadata.get("power_law_pivot")
        if status != "stored_error" and name.startswith("f_powerlaw_") and pivot is not None:
            wavelength = float(name.rsplit("_",1)[1])
            mode = fit.metadata.get("power_law_mode_selected", "single")
            if wavelength == pivot and mode == "single":
                error = fit.param_errors.get("power_law.norm")
                status = "recovered_pivot_normalization" if error is not None and np.isfinite(error) else "requires_refit"
            elif fit.covariance is not None:
                names = list(fit.param_values)
                def sample(theta):
                    values = dict(zip(names,theta))
                    norm, slope = values["power_law.norm"], values["power_law.slope"]
                    if mode == "double":
                        edge = fit.metadata["power_law_break_wave"]
                        if wavelength >= edge:
                            return [norm*(edge/pivot)**slope*(wavelength/edge)**values["power_law.red_slope"]]
                    return [norm*(wavelength/pivot)**slope]
                _, cov = propagate(sample,list(fit.param_values.values()),fit.covariance)
                if np.isfinite(cov[0,0]) and cov[0,0]>=0:
                    error=float(np.sqrt(cov[0,0]))
                    status="recovered_saved_covariance"
        report["quantities"][name] = {"value": value, "error": error,
            "status":status, "reason":reason if status == "requires_refit" else None}
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
    return report


def measure_selected_profile(fit, component_ids, *, parameter_draws=None, continuum=None):
    """Measure an explicit subset of a native generic complex.

    FWHM uses the nearest half-maximum crossings enclosing the global peak.
    Selection is fixed. Matched parameter draws must carry trial IDs. EW is
    conditional on the supplied fixed continuum unless joint draws are used.
    """
    from . import complex_recipes
    from .fitting.complexes import GenericComplexContext
    selected = tuple(component_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Supply distinct retained component IDs")
    if not set(selected).issubset(fit.component_models):
        raise ValueError("Selected component IDs are absent from this fit")
    recipe = complex_recipes.get(fit.metadata["recipe_id"])
    if recipe.backend != "generic":
        raise ValueError("Selected-profile reconstruction currently requires a generic complex")
    context = GenericComplexContext(recipe, fit.metadata.get("active_components", [c.id for c in recipe.components]), 1.)
    if not set(context.names).issubset(fit.param_values):
        context = GenericComplexContext(replace(recipe, continuum_mode="fixed_global"),
            fit.metadata.get("active_components", [c.id for c in recipe.components]), 1.)
    names = list(fit.param_values)
    theta = np.array([fit.param_values[name] for name in names])
    wave = np.linspace(float(fit.wave_rest.min()), float(fit.wave_rest.max()), max(4096, len(fit.wave_rest)))
    metric_names = ["flux", "centroid_angstrom", "fwhm_kms", "sigma_kms"]
    if continuum is not None:
        metric_names.append("ew_angstrom")
    def evaluate(values):
        mapping = dict(zip(names, values))
        components = context.components(np.array([mapping[name] for name in context.names]), wave)
        profile = sum((components[name] for name in selected), np.zeros_like(wave))
        flux = float(np.trapezoid(profile, wave))
        if flux <= 0:
            return np.full(len(metric_names), np.nan)
        centroid = float(np.trapezoid(wave*profile,wave)/flux)
        sigma = np.sqrt(max(0.,float(np.trapezoid((wave-centroid)**2*profile,wave)/flux)))*299792.458/centroid
        peak = int(np.argmax(profile)); half = profile[peak]/2.
        left = np.flatnonzero(profile[:peak] < half)
        right = np.flatnonzero(profile[peak+1:] < half)
        fwhm = np.nan
        if len(left) and len(right):
            l = left[-1]; r = peak+1+right[0]
            lo = wave[l]+(half-profile[l])*(wave[l+1]-wave[l])/(profile[l+1]-profile[l])
            hi = wave[r-1]+(half-profile[r-1])*(wave[r]-wave[r-1])/(profile[r]-profile[r-1])
            fwhm = (hi-lo)*299792.458/centroid
        out = [flux,centroid,fwhm,sigma]
        if continuum is not None:
            c = np.interp(wave,fit.wave_rest,continuum)
            out.append(float(np.trapezoid(profile/c,wave)) if np.all(c>0) else np.nan)
        return out
    point, cov = propagate(evaluate, theta, fit.covariance)
    result = {"definition": "selected_profile_v1_nearest_crossings_global_peak", "component_ids": list(selected),
        "values": dict(zip(metric_names,map(float,point))), "errors": {}, "covariance": None if cov is None else cov.tolist(),
        "metric_names": metric_names, "conditioning": "fixed_component_selection_and_continuum",
        "method": "local_gaussian", "width_status": "available" if np.isfinite(point[2]) else "undefined_or_truncated"}
    if cov is not None:
        result["errors"] = {name:float(np.sqrt(cov[i,i])) if cov[i,i]>=0 else np.nan for i,name in enumerate(metric_names)}
    if parameter_draws is not None:
        draws = [{"trial_id":row["trial_id"],"values":dict(zip(metric_names,map(float,evaluate([row["parameters"][name] for name in names]))))} for row in parameter_draws]
        result["bootstrap"] = summarize_matched_draws(draws,[],len(draws))
        result["errors"] = result["bootstrap"]["errors"]
        result["method"] = "matched_parameter_draws"
    fit.metadata.setdefault("selected_profile_measurements", {})["|".join(selected)] = result
    return result


def apply_bootstrap_errors(result):
    """Expose matched bootstrap intervals in the native measurement products."""
    errors=result.monte_carlo.get('errors',{})
    result.metadata['continuum_sample_errors']={name:errors.get(name,np.nan) for name in result.metadata.get('continuum_samples',{})}
    result.metadata['continuum_sample_uncertainty_method']=result.monte_carlo.get('method')
    for recipe,fit in result.line_complexes.items():
        fit.metric_errors={name:errors.get(f'{recipe}:{name}',errors.get(name,np.nan)) for name in fit.metrics}
        fit.metadata['measurement_uncertainty_method']=result.monte_carlo.get('method')
        for key, peak in fit.metadata.get('line_peaks', {}).get('measurements', {}).items():
            error = fit.metric_errors.get(f'{key}_peak_rest_angstrom', np.nan)
            peak['peak_error_rest_angstrom'] = error
            peak['peak_error_observed_angstrom'] = error*(1+result.spectrum.z)
            peak['peak_velocity_error_kms'] = 299792.458*error/peak['peak_rest_angstrom']
            peak['flux_error'] = fit.metric_errors.get(f'{key}_peak_selection_flux', np.nan)
            peak['uncertainty_method'] = result.monte_carlo.get('method')
            peak['uncertainty_status'] = 'available' if np.isfinite(error) and error > 0 else 'insufficient_bootstrap_trials'

        fit.metadata['measurement_intervals']={name:result.monte_carlo.get('percentiles',{}).get(f'{recipe}:{name}',result.monte_carlo.get('percentiles',{}).get(name)) for name in fit.metrics}


def pixel_noise_factor(errors, covariance=None):
    """Prepare the supplied original-grid noise covariance once per bootstrap."""
    errors=np.asarray(errors,dtype=float)
    if covariance is None:
        return np.where(np.isfinite(errors)&(errors>0),errors,0.)
    matrix=np.asarray(covariance,dtype=float)
    if matrix.shape != (len(errors),len(errors)) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix,matrix.T):
        raise ValueError('Pixel covariance must be finite, symmetric and match the original grid')
    eigenvectors=None
    eigenvalues,eigenvectors=np.linalg.eigh(matrix)
    tolerance=np.finfo(float).eps*len(errors)*max(float(np.max(np.abs(eigenvalues))),1.e-300)
    if eigenvalues.min() < -tolerance:
        raise ValueError('Pixel covariance must be positive semidefinite')
    return eigenvectors*np.sqrt(np.maximum(eigenvalues,0.))


def draw_pixel_noise(rng,factor):
    normal=rng.normal(size=factor.shape[0])
    return factor*normal if factor.ndim==1 else factor@normal
