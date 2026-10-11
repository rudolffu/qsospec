"""Local Balmer evidence hypotheses share pixels, nuisance and response.

Known synthetic profiles test recovered physical coefficients and a centered
outflow confound, rather than testing serialization alone.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

import qsospec
from qsospec.fitting.global_fit import C_KMS, _gaussian_area_profile


FAST = dict(broad_width_starts_kms=(1200.0, 3000.0),
            broad_velocity_starts_kms=(0.0,),
            outflow_width_starts_kms=(600.0, 1500.0),
            outflow_velocity_starts_kms=(0.0,))


def _matrix(size, sigma_pixels=2.0):
    offsets = np.arange(-8, 9)
    values = np.exp(-0.5*(offsets/sigma_pixels)**2)
    matrix = sparse.diags(values, offsets, shape=(size, size), format="csr")
    return sparse.diags(1/np.asarray(matrix.sum(axis=1)).ravel()) @ matrix


def _synthetic(line="hbeta", broad_flux=100.0, broad_width=2400.0,
               outflow=False, response=True, noise=True):
    center = qsospec.lines.get(line).vacuum_wavelength
    lo, hi = center*np.exp(np.asarray([-22000., 22000.])/C_KMS)
    wave = np.linspace(lo, hi, 1300)
    z = 0.37
    continuum = 2.0+0.0007*(wave-center)
    line_model = _gaussian_area_profile(wave, 15.0, center, 320.0)
    if broad_flux:
        line_model += _gaussian_area_profile(wave, broad_flux, center*np.exp(100/C_KMS), broad_width)
    if line == "hbeta":
        members = [("oiii_5008", 40.0), ("oiii_4960", 40.0/2.98)]
    else:
        members = [("nii_6585", 12.0), ("nii_6550", 12.0/2.96),
                   ("sii_6718", 6.0), ("sii_6733", 5.0)]
    for name, flux in members:
        line_model += _gaussian_area_profile(wave, flux, qsospec.lines.get(name).vacuum_wavelength, 320.0)
    if outflow:
        line_model += _gaussian_area_profile(wave, 18.0, center, 1400.0)
        for name, flux in members:
            line_model += _gaussian_area_profile(wave, flux*0.7, qsospec.lines.get(name).vacuum_wavelength, 1400.0)
    matrix = _matrix(len(wave))
    true_flux = matrix.dot(continuum+line_model) if response else continuum+line_model
    err = np.full(len(wave), 0.01)
    if noise:
        true_flux = true_flux+np.random.default_rng(17).normal(0.0, err)
    valid = np.ones(len(wave), bool)
    valid[20:23] = False
    spectrum = qsospec.Spectrum.from_arrays(wave, true_flux, err=err, mask=valid,
        wave_frame="rest", flux_unit="relative", z=z)
    operator = qsospec.BandResolutionOperator("synthetic", spectrum.wave_obs,
        np.arange(len(wave)), matrix, provenance=dict(source="known_synthetic_matrix",
        is_object_specific=True, is_approximate=False))
    return spectrum, operator, continuum, line_model


def _config(line, count, family="core", **kwargs):
    return qsospec.BalmerLocalConfig(line=line, broad_count=count,
        nuisance_family=family, require_native_response=True, **FAST, **kwargs)


@pytest.mark.parametrize("line,prefix", [("hbeta", "Hb"), ("halpha", "Ha")])
def test_null_has_no_broad_parameters_and_undefined_width(line, prefix, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Local hypothesis called a global/host/linked-line workflow")
    for name in ("fit_global_continuum", "fit_global_lines", "_build_joint_hgamma_problem"):
        monkeypatch.setattr("qsospec.fitting.global_fit."+name, forbidden)
    spectrum, operator, _, _ = _synthetic(line, broad_flux=0.0)
    config = _config(line, 0)
    continuum, fit = qsospec.fit_balmer_local(spectrum, config, instrumental_response=[operator])
    assert continuum.success and fit.success
    assert not any(name.startswith(prefix+"_broad") for name in fit.param_values)
    assert fit.metrics[prefix+"_broad_flux_input"] == 0
    assert np.isnan(fit.metrics[prefix+"_broad_fwhm_kms"])
    assert np.isnan(fit.metric_errors[prefix+"_broad_flux_input"])
    assert fit.metric_errors[prefix+"_narrow_flux_input"] > 0
    assert fit.metrics[prefix+"_narrow_flux_input"] == pytest.approx(15.0, rel=0.02)
    assert "covariance_rank_deficient" not in fit.warning_codes()
    assert len(fit.metadata["deterministic_multistarts"]) == 1
    assert fit.metadata["broad_detection_assigned"] is False
    assert fit.metadata["host_subtraction"] is False
    assert fit.metadata["balmer_pseudocontinuum"] is False
    assert fit.metadata["hgamma_linked_fit"] is False


@pytest.mark.parametrize("line,prefix", [("hbeta", "Hb"), ("halpha", "Ha")])
def test_zero_and_one_share_likelihood_and_recover_resolution_corrected_width(line, prefix):
    spectrum, operator, _, _ = _synthetic(line)
    c0, c1 = _config(line, 0), _config(line, 1)
    null_continuum, null = qsospec.fit_balmer_local(spectrum, c0, instrumental_response=[operator])
    continuum, fit = qsospec.fit_balmer_local(spectrum, c1, instrumental_response=[operator])
    assert fit.success and null.success
    for name in ("fit_mask_sha256", "data_sha256", "error_sha256", "wavelength_sha256", "instrumental_response", "active_sii_line_ids"):
        assert fit.metadata[name] == null.metadata[name]
    assert fit.metadata["n_free_parameters"] == null.metadata["n_free_parameters"]+3
    assert null.chi2-fit.chi2 > 1000
    assert fit.metrics[prefix+"_broad_fwhm_kms"] == pytest.approx(2400.0, rel=0.02)
    assert fit.metrics[prefix+"_broad_flux_input"] == pytest.approx(100.0, rel=0.02)
    assert fit.param_values["narrow.fwhm_kms"] == pytest.approx(320.0, rel=0.02)
    assert fit.metric_errors[prefix+"_broad_fwhm_kms"] > 0
    assert fit.metric_errors[prefix+"_broad_flux_input"] > 0
    assert fit.metric_errors[prefix+"_broad_ew_rest"] > 0
    assert fit.metadata["resolution_status"] == "verified_native_operator"
    assert fit.metadata["width_definition"] == "intrinsic_single_gaussian_conditional_on_supplied_operator"
    assert len(fit.metadata["deterministic_multistarts"]) == 2
    names = list(fit.param_values)
    assert fit.covariance.shape == (len(names), len(names))
    ci = [names.index("continuum.constant"), names.index("continuum.slope")]
    np.testing.assert_array_equal(continuum.covariance, fit.covariance[np.ix_(ci, ci)])
    np.testing.assert_allclose(fit.model, sum(fit.component_models.values(), np.zeros_like(spectrum.flux)), rtol=0, atol=0)
    residual = ((spectrum.flux-continuum.model-fit.model)/spectrum.err)[fit.fit_mask]
    assert fit.chi2 == pytest.approx(residual@residual, rel=1e-10)
    assert fit.bic == pytest.approx(fit.chi2+len(names)*np.log(fit.fit_mask.sum()), rel=1e-10)


def test_centered_shared_outflow_is_not_forced_into_broad_hbeta():
    spectrum, operator, _, _ = _synthetic(broad_flux=0.0, outflow=True)
    core_config = _config("hbeta", 1)
    _, core = qsospec.fit_balmer_local(spectrum, core_config, instrumental_response=[operator])
    shared_config = _config("hbeta", 0, "shared_outflow")
    _, shared = qsospec.fit_balmer_local(spectrum, shared_config, instrumental_response=[operator])
    _, alternative = qsospec.fit_balmer_local(spectrum, replace(shared_config, broad_count=1), instrumental_response=[operator])
    assert core.success and shared.success and alternative.success
    assert core.metrics["Hb_broad_flux_input"] > 10
    assert shared.param_values["outflow.velocity_kms"] == pytest.approx(0, abs=5)
    assert shared.param_values["outflow.fwhm_kms"] == pytest.approx(1400, rel=0.02)
    assert shared.metrics["Hb_outflow_flux_input"] == pytest.approx(18, rel=0.03)
    assert shared.metrics["Hb_broad_flux_input"] == 0
    assert alternative.bic > shared.bic
    assert alternative.metrics["Hb_broad_flux_input"] < 1.0
    for key in ("fit_mask_sha256", "data_sha256", "error_sha256", "wavelength_sha256"):
        assert shared.metadata[key] == alternative.metadata[key]


def test_native_response_adds_overlapping_bands_and_applies_foreground_factors():
    spectrum, operator, _, _ = _synthetic(noise=False)
    n = len(spectrum.flux)
    wavelength = spectrum.wave_obs
    transmission = np.exp(-0.3*(wavelength/wavelength.mean())**-1.2)
    provenance = dict(source="known_synthetic_foreground_response", is_object_specific=True, is_approximate=False)
    operators = [qsospec.BandResolutionOperator("band1", wavelength, np.arange(n), operator.matrix,
                    output_weights=0.25, input_factor=transmission, output_factor=1/transmission, provenance=provenance),
                 qsospec.BandResolutionOperator("band2", wavelength, np.arange(n), operator.matrix,
                    output_weights=0.75, input_factor=transmission, output_factor=1/transmission, provenance=provenance)]
    config = _config("hbeta", 1)
    # Use a known intrinsic model to construct an independent conjugated
    # response data vector, preserving the already-corrected F_lambda frame.
    _, known = qsospec.fit_balmer_local(spectrum, config, instrumental_response=[operator])
    intrinsic = qsospec.evaluate_balmer_local_model(spectrum.wave_rest, known, config)
    observed = operator.matrix.dot(transmission*intrinsic)/transmission
    corrected = qsospec.Spectrum(spectrum.wave_obs, observed, spectrum.err, spectrum.z,
                                spectrum.metadata, spectrum.mask)
    continuum, fit = qsospec.fit_balmer_local(corrected, config, instrumental_response=operators)
    reconstructed = qsospec.evaluate_balmer_local_model(spectrum.wave_rest, fit, config)
    expected = operator.matrix.dot(transmission*reconstructed)/transmission
    np.testing.assert_allclose(continuum.model+fit.model, expected, atol=1e-12)
    assert fit.metrics["Hb_broad_fwhm_kms"] == pytest.approx(known.metrics["Hb_broad_fwhm_kms"], rel=1e-5)
    assert operators[0].card()["input_factor_sha256"] != qsospec.BandResolutionOperator("unit", wavelength,
        np.arange(n), operator.matrix, provenance=provenance).card()["input_factor_sha256"]


def test_response_operator_cannot_silently_normalize_missing_or_duplicate_weights():
    spectrum, operator, _, _ = _synthetic()
    config = _config("hbeta", 0)
    incomplete = replace(operator, output_weights=0.7)
    with pytest.raises(ValueError, match="sum to one"):
        qsospec.fit_balmer_local(spectrum, config, instrumental_response=[incomplete])
    with pytest.raises(ValueError, match="sum to one"):
        qsospec.fit_balmer_local(spectrum, config, instrumental_response=[operator, replace(operator, name="duplicate")])
    with pytest.raises(ValueError, match="native response"):
        qsospec.fit_balmer_local(spectrum, config)
    with pytest.raises(ValueError, match="native response"):
        qsospec.fit_balmer_local(spectrum, config, instrumental_response=[replace(operator, provenance={})])


def test_signed_native_response_is_preserved_without_clipping_or_normalization():
    spectrum, operator, _, _ = _synthetic(broad_flux=0.0, response=False, noise=False)
    n = len(spectrum.flux)
    signed = sparse.diags([-0.05, 1.1, -0.05], [-1, 0, 1], shape=(n, n), format="csr")
    response = replace(operator, matrix=signed)
    before = response.matrix.copy()
    observed = signed.dot(spectrum.flux)
    data = qsospec.Spectrum(spectrum.wave_obs, observed, spectrum.err, spectrum.z,
                           spectrum.metadata, spectrum.mask)
    config = _config("hbeta", 0)
    continuum, fit = qsospec.fit_balmer_local(data, config, instrumental_response=[response])
    assert fit.success
    assert fit.param_values["narrow.fwhm_kms"] == pytest.approx(320, rel=1e-6)
    assert response.card()["negative_element_count"] == 2*(n-1)
    assert response.card()["row_sum_max"] == pytest.approx(1.05)
    np.testing.assert_array_equal(response.matrix.data, before.data)
    intrinsic = qsospec.evaluate_balmer_local_model(spectrum.wave_rest, fit, config)
    np.testing.assert_allclose(continuum.model+fit.model, signed.dot(intrinsic), atol=1e-12)


def test_float32_coadd_weight_roundoff_is_recorded_not_renormalized():
    spectrum, operator, _, _ = _synthetic(broad_flux=0.0, response=False, noise=False)
    ivars = np.asarray([0.1, 0.2], dtype=np.float32)
    denominator = ivars.sum(dtype=np.float32)
    weights = np.divide(ivars, denominator, out=np.empty(2, dtype=float))
    assert abs(weights.sum()-1) > 1e-8
    operators = [replace(operator, name=f"band{i}", output_weights=w) for i, w in enumerate(weights)]
    data = qsospec.Spectrum(spectrum.wave_obs, operator.matrix.dot(spectrum.flux)*weights.sum(),
                           spectrum.err, spectrum.z, spectrum.metadata, spectrum.mask)
    config = _config("hbeta", 0)
    continuum, fit = qsospec.fit_balmer_local(data, config, instrumental_response=operators)
    assert fit.success
    validation = fit.metadata["resolution_weight_validation"]
    assert validation["absolute_tolerance"] == 4*float(np.finfo(np.float32).eps)
    assert validation["fitted_row_max_abs_deviation"] == abs(weights.sum()-1)
    assert validation["normalization_applied"] is False
    intrinsic = qsospec.evaluate_balmer_local_model(spectrum.wave_rest, fit, config)
    np.testing.assert_allclose(continuum.model+fit.model, operator.matrix.dot(intrinsic)*weights.sum(), atol=1e-12)
    for i, operator in enumerate(operators):
        np.testing.assert_array_equal(operator.output_weights, np.full(len(spectrum.flux), weights[i]))
    with pytest.raises(ValueError, match="sum to one"):
        qsospec.fit_balmer_local(data, config, instrumental_response=[replace(operators[0], output_weights=weights[0]+1e-5), operators[1]])


def test_response_evaluates_native_grid_before_mapping_to_distinct_output_grid():
    spectrum, operator, _, _ = _synthetic(noise=False)
    center = qsospec.lines.get("hbeta").vacuum_wavelength
    native_wave = np.linspace(spectrum.wave_rest.min(), spectrum.wave_rest.max(), 2*len(spectrum.flux)-1)
    intrinsic = 2.0+0.0007*(native_wave-center)
    for flux, line_center, width in ((15.0, center, 320.0),
            (100.0, center*np.exp(100/C_KMS), 2400.0),
            (40.0, qsospec.lines.get("oiii_5008").vacuum_wavelength, 320.0),
            (40.0/2.98, qsospec.lines.get("oiii_4960").vacuum_wavelength, 320.0)):
        intrinsic += _gaussian_area_profile(native_wave, flux, line_center, width)
    native_matrix = _matrix(len(native_wave), sigma_pixels=3.0)[::2, :].tocsr()
    response = replace(operator, input_wave_obs=native_wave*(1+spectrum.z), matrix=native_matrix, input_factor=1.0)
    native_data = qsospec.Spectrum(spectrum.wave_obs, native_matrix.dot(intrinsic), spectrum.err,
                                  spectrum.z, spectrum.metadata, spectrum.mask)
    config = _config("hbeta", 1)
    continuum, fit = qsospec.fit_balmer_local(native_data, config, instrumental_response=[response])
    assert fit.success
    assert response.matrix.shape[0] != response.matrix.shape[1]
    assert fit.param_values["narrow.fwhm_kms"] == pytest.approx(320, rel=1e-5)
    assert fit.metrics["Hb_broad_fwhm_kms"] == pytest.approx(2400, rel=1e-5)
    reconstructed = qsospec.evaluate_balmer_local_model(native_wave, fit, config)
    np.testing.assert_allclose(continuum.model+fit.model, native_matrix.dot(reconstructed), atol=1e-12)


def test_missing_response_remains_explicitly_nonintrinsic_and_no_sigma_fallback():
    spectrum, _, _, _ = _synthetic(response=False)
    config = replace(_config("hbeta", 1), require_native_response=False)
    _, fit = qsospec.fit_balmer_local(spectrum, config)
    assert fit.success
    assert fit.metadata["resolution_status"] == "missing_not_intrinsic"
    assert fit.metadata["width_definition"] == "unconvolved_observed_profile_single_gaussian"
    assert "instrumental_response_missing" in fit.warning_codes()
    with pytest.raises(TypeError, match="BandResolutionOperator"):
        qsospec.fit_balmer_local(spectrum, config, instrumental_response=[np.full(len(spectrum.flux), 100.0)])


def test_intrinsic_helper_and_halpha_wrapper_preserve_exact_parameter_contract():
    spectrum, operator, _, _ = _synthetic("halpha")
    config = _config("halpha", 1)
    continuum, fit = qsospec.fit_halpha_local(spectrum, config, instrumental_response=[operator])
    components = qsospec.evaluate_balmer_local_model(spectrum.wave_rest, fit, config, return_components=True)
    intrinsic = sum(components.values(), np.zeros_like(spectrum.flux))
    np.testing.assert_allclose(operator.matrix.dot(intrinsic), continuum.model+fit.model, atol=1e-12)
    assert not any(name.startswith("local_continuum_") for name in
                   qsospec.evaluate_balmer_local_model(spectrum.wave_rest, fit, config,
                                                      include_continuum=False, return_components=True))
    with pytest.raises(ValueError, match="exact recorded configuration"):
        qsospec.evaluate_balmer_local_model(spectrum.wave_rest, fit, replace(config, broad_count=0))
    with pytest.raises(ValueError, match="halpha configuration"):
        qsospec.fit_halpha_local(spectrum, replace(config, line="hbeta"))


def test_hbeta_legacy_local_default_unchanged_and_zero_count_supported():
    spectrum, _, _, _ = _synthetic(broad_flux=0.0, response=False)
    legacy_config = qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=(), fit_oiii_wings=False)
    continuum, fit = qsospec.fit_hbeta_local(spectrum, legacy_config)
    assert fit.success and continuum.success
    assert not any(name.startswith("Hb_broad") for name in fit.param_values)
    assert fit.metrics["Hb_broad_flux_input"] == 0
    assert np.isnan(fit.metrics["Hb_broad_fwhm_kms"])
    assert tuple(fit.metadata["fit_window"]) == (4640.0, 5100.0)
    native_config = replace(_config("hbeta", 0), require_native_response=False)
    _, native = qsospec.fit_hbeta_local(spectrum, native_config)
    assert tuple(native.metadata["fit_window"]) == pytest.approx(native_config.window)


def test_incomplete_support_is_unobserved_not_a_zero_flux_measurement():
    wave = np.linspace(4840, 4890, 50)
    spectrum = qsospec.Spectrum.from_arrays(wave, np.ones(50), err=np.ones(50)*0.1,
        wave_frame="rest", flux_unit="relative")
    continuum, fit = qsospec.fit_balmer_local(spectrum, qsospec.BalmerLocalConfig(broad_count=0))
    assert not fit.success and not continuum.success
    assert fit.param_values == {} and fit.metrics == {}
    assert "line_complex_not_covered" in fit.warning_codes()
    assert fit.metadata["coverage"]["covered"] is False


@pytest.mark.parametrize("kwargs", [dict(broad_count=2), dict(broad_count=True),
    dict(line="hgamma"), dict(nuisance_family="wing"),
    dict(broad_width_starts_kms=(100.0,)), dict(mask_windows=((4800., 4700.),))])
def test_invalid_local_contracts_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        qsospec.BalmerLocalConfig(**kwargs)
