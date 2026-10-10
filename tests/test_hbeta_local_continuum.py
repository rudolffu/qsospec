"""Native joint local-continuum Hbeta fits preserve physical line models."""

import numpy as np
import pytest

import qsospec
from qsospec.fitting.global_fit import (
    C_KMS, HBETA_WAVE, OIII_4959_WAVE, OIII_5007_WAVE,
    _HbetaContext, _gaussian_area_profile, _hbeta_metrics, _metric_errors,
)


BANDS = {
    1: ((900.0, 20000.0),),
    2: ((900.0, 6000.0), (2500.0, 20000.0)),
    3: ((900.0, 2500.0), (2500.0, 6000.0), (6000.0, 20000.0)),
}


def _spectrum(count=1, *, wing=False):
    wave = np.linspace(4550.0, 5150.0, 1800)
    continuum = 2.0 + 0.0012 * (wave - 4870.0)
    widths = {1: [2400.0], 2: [2200.0, 8000.0], 3: [1600.0, 4000.0, 11000.0]}[count]
    fluxes = {1: [100.0], 2: [70.0, 30.0], 3: [55.0, 30.0, 15.0]}[count]
    line = np.zeros_like(wave)
    for flux, width, velocity in zip(fluxes, widths, [-120.0, 230.0, -400.0]):
        line += _gaussian_area_profile(wave, flux, HBETA_WAVE * np.exp(velocity / C_KMS), width)
    for flux, center in ((12.0, HBETA_WAVE), (30.0, OIII_5007_WAVE), (30.0 / 2.98, OIII_4959_WAVE)):
        line += _gaussian_area_profile(wave, flux, center, 320.0)
    if wing:
        for flux, center in ((35.0, OIII_5007_WAVE), (35.0 / 2.98, OIII_4959_WAVE)):
            line += _gaussian_area_profile(wave, flux, center * np.exp(-500.0 / C_KMS), 1200.0)
    error = np.full_like(wave, 0.003)
    mask = np.ones(wave.size, dtype=bool)
    mask[1200:1205] = False
    spectrum = qsospec.Spectrum.from_arrays(
        wave, continuum + line + np.random.default_rng(17).normal(0.0, error),
        err=error, mask=mask, wave_frame="rest", flux_unit="relative",
    )
    return spectrum, continuum, widths


@pytest.mark.parametrize("count", [1, 2, 3])
def test_local_fit_recovers_continuum_and_lines_with_joint_covariance(count, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Local fitting called a global/Balmer/host workflow")
    for function in ("fit_global_continuum", "fit_global_lines", "_build_joint_hgamma_problem"):
        monkeypatch.setattr("qsospec.fitting.global_fit." + function, forbidden)
    spectrum, known_continuum, widths = _spectrum(count)
    config = qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=BANDS[count], fit_oiii_wings=False)
    continuum, fit = qsospec.fit_hbeta_local(spectrum, config)
    assert config.local_continuum is None
    assert fit.success and continuum.success
    assert fit.metadata["n_broad_components"] == count
    assert fit.metadata["uncertainty_conditioning"] == "joint_local_linear_continuum_and_lines"
    assert "statistical_uncertainty_excludes_continuum_host" not in fit.warning_codes()
    assert "covariance_rank_deficient" not in fit.warning_codes()
    names = list(fit.param_values)
    assert fit.covariance.shape == (3 * count + 6, 3 * count + 6)
    assert fit.metadata["covariance_parameter_names"] == names
    indices = [names.index(name) for name in ("continuum.constant", "continuum.slope")]
    np.testing.assert_array_equal(continuum.covariance, fit.covariance[np.ix_(indices, indices)])
    assert np.any(np.abs(fit.covariance[indices[0], :3 * count]) > 0)
    assert fit.param_values["continuum.constant"] == pytest.approx(2.0, abs=0.008)
    assert fit.param_values["continuum.slope"] == pytest.approx(0.0012, abs=5e-5)
    np.testing.assert_allclose(continuum.model, known_continuum, atol=0.03)
    for index, width in enumerate(widths, 1):
        assert fit.param_values[f"Hb_broad{index}.fwhm_kms"] == pytest.approx(width, rel=0.08)
    assert sum(fit.param_values[f"Hb_broad{i}.flux"] for i in range(1, count + 1)) == pytest.approx(100.0, rel=0.05)
    assert fit.metric_errors["Hb_broad_fwhm_kms"] > 0
    assert fit.metric_errors["Hb_broad_flux_input"] > 0
    assert fit.metric_errors["Hb_broad_ew_rest"] > 0
    expected_mask = spectrum.valid_mask & (spectrum.wave_rest >= 4640.0) & (spectrum.wave_rest <= 5100.0)
    expected_mask &= ~((spectrum.wave_rest >= 4660.0) & (spectrum.wave_rest <= 4715.0))
    np.testing.assert_array_equal(fit.fit_mask, expected_mask)
    np.testing.assert_array_equal(continuum.fit_mask, fit.fit_mask)
    np.testing.assert_array_equal(fit.model, sum(fit.component_models.values(), np.zeros_like(spectrum.flux)))
    assert not any(name.startswith("local_continuum_") for name in fit.component_models)
    residual = (spectrum.flux - continuum.model - fit.model)[fit.fit_mask] / spectrum.err[fit.fit_mask]
    assert fit.chi2 == pytest.approx(float(residual @ residual), rel=1e-10)
    assert fit.bic == pytest.approx(fit.chi2 + len(names) * np.log(fit.fit_mask.sum()))
    assert continuum.metadata["balmer_pseudocontinuum"] is False
    assert continuum.metadata["host_subtraction"] is False
    assert continuum.metadata["hgamma_linked_fit"] is False
    peak_definitions = fit.metadata["peak_model"]["components"]
    assert not any(row["component_id"].startswith("local_continuum_") for row in peak_definitions)


def test_local_continuum_design_and_jacobian_reuse_native_separable_model():
    context = _HbetaContext(qsospec.HbetaComplexConfig(local_continuum="linear", broad_fwhm_bands_kms=BANDS[1]), True, 100.0)
    wave = np.linspace(4640.0, 5100.0, 500)
    theta = context.initial.copy()
    theta[context.index["continuum.constant"]] = 2.0
    theta[context.index["continuum.slope"]] = 0.0012
    linear = np.array([theta[context.index[name]] for name in context.linear_names])
    nonlinear = np.array([theta[context.index[name]] for name in context.nonlinear_names])
    design, derivatives = context.separable_design(nonlinear, wave, True)
    np.testing.assert_allclose(design @ linear, context.model(theta, wave), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(design[:, -2], np.ones_like(wave))
    np.testing.assert_array_equal(design[:, -1], wave - 4870.0)
    for derivative in derivatives:
        np.testing.assert_array_equal(derivative[:, -2:], np.zeros((wave.size, 2)))


def test_equivalent_width_error_uses_joint_continuum_covariance():
    spectrum, _, _ = _spectrum()
    continuum, fit = qsospec.fit_hbeta_local(
        spectrum, qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=BANDS[1], fit_oiii_wings=False)
    )
    context = _HbetaContext(qsospec.HbetaComplexConfig(local_continuum="linear", broad_fwhm_bands_kms=BANDS[1], fit_oiii_wings=False), False, 100.0)
    theta = np.array([fit.param_values[name] for name in context.names])
    level = lambda state: state[context.index["continuum.constant"]] + state[context.index["continuum.slope"]] * (HBETA_WAVE - 4870.0)
    metrics = lambda state: _hbeta_metrics(state, context, level(state), spectrum.z, spectrum.flux_density_scale_to_cgs)
    joint = _metric_errors(theta, fit.covariance, metrics)
    assert fit.metric_errors["Hb_broad_ew_rest"] == joint["Hb_broad_ew_rest"]
    fixed = lambda state: _hbeta_metrics(state, context, level(theta), spectrum.z, spectrum.flux_density_scale_to_cgs)
    conditional = _metric_errors(theta, fit.covariance, fixed)
    assert not np.isclose(joint["Hb_broad_ew_rest"], conditional["Hb_broad_ew_rest"], rtol=1e-5)


def test_local_oiii_wing_uses_existing_native_selection_rules():
    spectrum, _, _ = _spectrum(wing=True)
    continuum, fit = qsospec.fit_hbeta_local(spectrum, qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=BANDS[1]))
    assert fit.success and fit.selected_model == "wing"
    assert fit.metadata["wing_candidate"]["accepted"]
    thresholds = fit.metadata["wing_candidate"]["thresholds"]
    assert thresholds["bic_improvement"] == 20.0
    assert thresholds["snr"] == 5.0
    assert thresholds["fwhm_ratio"] == 2.0
    assert thresholds["velocity_separation_kms"] == 150.0


def test_local_fit_without_covariance_preserves_point_measurements():
    spectrum, _, _ = _spectrum()
    continuum, fit = qsospec.fit_hbeta_local(
        spectrum, qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=BANDS[1], fit_oiii_wings=False),
        compute_covariance=False,
    )
    assert fit.success and continuum.success
    assert fit.covariance is None and continuum.covariance is None
    assert np.isfinite(fit.metrics["Hb_broad_fwhm_kms"])
    assert np.isnan(fit.metric_errors["Hb_broad_fwhm_kms"])


def test_insufficient_local_pixels_remain_explicitly_failed():
    spectrum = qsospec.Spectrum.from_arrays(
        np.linspace(4800.0, 4850.0, 10), np.ones(10), err=np.full(10, 0.1),
        wave_frame="rest", flux_unit="relative",
    )
    continuum, fit = qsospec.fit_hbeta_local(spectrum)
    assert not fit.success and not continuum.success
    assert "window_too_few_pixels" in fit.warning_codes()
    assert fit.param_values == {}
    assert fit.covariance is None and continuum.covariance is None


def test_invalid_local_continuum_mode_fails_explicitly():
    with pytest.raises(ValueError, match="local_continuum"):
        qsospec.HbetaComplexConfig(local_continuum="quadratic")
