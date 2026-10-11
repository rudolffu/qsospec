"""Variable-count H-beta models on the original global-v2 implementation."""

import ast
from pathlib import Path
import subprocess

import numpy as np
import pytest

import qsospec
from qsospec.fitting import global_fit
from qsospec.fitting.global_fit import (
    C_KMS, HBETA_WAVE, OIII_4959_WAVE, OIII_5007_WAVE,
    _HbetaContext, _gaussian_area_profile,
)
from qsospec.global_result import GlobalContinuumResult


BANDS = {
    1: ((900., 20000.),),
    2: ((900., 6000.), (2500., 20000.)),
    3: ((900., 2500.), (2500., 6000.), (6000., 20000.)),
}


@pytest.mark.parametrize("bands", [((900., 2000.),) * 4,
                                  ((0., 2000.),), ((2000., 1000.),),
                                  ((900., np.inf),)])
def test_invalid_component_contract_is_rejected(bands):
    with pytest.raises(ValueError, match="broad FWHM"):
        qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=bands)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_count_changes_all_model_paths_and_derivatives(count):
    config = qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=BANDS[count])
    context = _HbetaContext(config, include_wing=True, flux_scale=100.)
    broad = [name for name in context.names if name.startswith("Hb_broad") and name.endswith(".flux")]
    assert len(broad) == count
    assert sum(context.initial[context.index[name]] for name in broad) == pytest.approx(100.)
    wave = np.linspace(4600., 5150., 700)
    linear, _, nonlinear, _ = context.separable_initial_and_bounds()
    design, derivatives = context.separable_design(nonlinear, wave, True)
    np.testing.assert_allclose(design @ linear, context.model(context.initial, wave), atol=1e-15)
    assert len(context.components(context.initial, wave)) == count + 5
    for index, derivative in enumerate(derivatives):
        plus, minus = nonlinear.copy(), nonlinear.copy()
        plus[index] += .01
        minus[index] -= .01
        finite = (context.separable_design(plus, wave, False)[0] -
                  context.separable_design(minus, wave, False)[0]) / .02
        np.testing.assert_allclose(derivative, finite, rtol=5e-5, atol=1e-11)


def test_default_three_component_model_preserves_original_arithmetic():
    context = _HbetaContext(qsospec.HbetaComplexConfig(), False, 100.)
    expected_initial = [100. * .55, 0., 1700., 100. * .30, 0., 4250., 100. * .15, 0., 13000.,
                        5., 0., 350., 20.]
    np.testing.assert_array_equal(context.initial, expected_initial)
    theta = context.initial.copy()
    theta[[1, 4, 7, 10]] = [-200., 150., -400., 80.]
    wave = np.linspace(4500., 5220., 7201)
    components = context.components(theta, wave)
    original_broad = components["Hb_broad1"] + components["Hb_broad2"] + components["Hb_broad3"]
    np.testing.assert_array_equal(context.broad_profile(theta, wave), original_broad)
    original_model = sum(components.values(), np.zeros_like(wave))
    np.testing.assert_array_equal(context.model(theta, wave), original_model)


@pytest.mark.parametrize("wing,heii", [(False, False), (True, False), (False, True), (True, True)])
def test_three_component_context_is_bit_identical_to_original_pin(wing, heii):
    """Use the immutable original source as the regression reference."""
    repo = Path(__file__).resolve().parents[1]
    try:
        source = subprocess.check_output(
            ["git", "show", "b6a37ab673b12288b474f4864d5e0ddeae7e863d:src/qsospec/fitting/global_fit.py"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("original Git history is unavailable in this source distribution")
    original_class = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.ClassDef) and node.name == "_HbetaContext")
    namespace = dict(vars(global_fit))
    exec(compile(ast.Module(body=[original_class], type_ignores=[]), "original_pinned_hbeta", "exec"), namespace)
    config = qsospec.HbetaComplexConfig(heii_enabled=heii)
    old = namespace["_HbetaContext"](config, wing, 123.456)
    new = _HbetaContext(config, wing, 123.456)
    assert old.names == new.names
    for name in ("initial", "lower", "upper"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    wave = np.linspace(4500., 5220., 913)
    theta = old.initial.copy()
    theta[1], theta[4], theta[7] = -170., 240., -600.
    for method in ("model", "broad_profile"):
        np.testing.assert_array_equal(getattr(old, method)(theta, wave), getattr(new, method)(theta, wave))
    for name, values in old.components(theta, wave).items():
        np.testing.assert_array_equal(values, new.components(theta, wave)[name])
    nonlinear = np.array([theta[old.index[name]] for name in old.nonlinear_names])
    old_design, old_derivatives = old.separable_design(nonlinear, wave, True)
    new_design, new_derivatives = new.separable_design(nonlinear, wave, True)
    np.testing.assert_array_equal(old_design, new_design)
    for old_derivative, new_derivative in zip(old_derivatives, new_derivatives):
        np.testing.assert_array_equal(old_derivative, new_derivative)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_native_fit_recovers_synthetic_profiles_with_formal_covariance(count):
    wave = np.linspace(4550., 5150., 1800)
    continuum = np.full_like(wave, 2.)
    widths = {1: [2400.], 2: [2200., 8000.], 3: [1600., 4000., 11000.]}[count]
    fluxes = {1: [100.], 2: [70., 30.], 3: [55., 30., 15.]}[count]
    line = np.zeros_like(wave)
    for flux, width, velocity in zip(fluxes, widths, [-120., 230., -400.]):
        line += _gaussian_area_profile(wave, flux, HBETA_WAVE * np.exp(velocity / C_KMS), width)
    for flux, center in ((12., HBETA_WAVE), (30., OIII_5007_WAVE), (30./2.98, OIII_4959_WAVE)):
        line += _gaussian_area_profile(wave, flux, center, 320.)
    error = np.full_like(wave, .003)
    spectrum = qsospec.Spectrum.from_arrays(
        wave, continuum + line + np.random.default_rng(17).normal(0., error),
        err=error, wave_frame="rest", survey="desi")
    known = GlobalContinuumResult(
        success=True, status=1, message="known", param_values={}, param_errors={},
        covariance=None, chi2=0., dof=1, reduced_chi2=0., wave_rest=wave.copy(),
        model=continuum, component_models={"power_law": continuum},
        fit_mask=spectrum.valid_mask, clip_mask=spectrum.valid_mask)
    config = qsospec.HbetaComplexConfig(broad_fwhm_bands_kms=BANDS[count], fit_oiii_wings=False)
    fitted = qsospec.fit_hbeta_complex(spectrum, known, config)
    assert fitted.success
    assert fitted.metadata["n_broad_components"] == count
    assert fitted.covariance.shape == (3 * count + 4, 3 * count + 4)
    assert "covariance_rank_deficient" not in fitted.warning_codes()
    assert fitted.metric_errors["Hb_broad_fwhm_kms"] > 0
    assert sum(fitted.param_values[f"Hb_broad{i}.flux"] for i in range(1, count + 1)) == pytest.approx(100., rel=.03)
    for index, width in enumerate(widths, 1):
        assert fitted.param_values[f"Hb_broad{index}.fwhm_kms"] == pytest.approx(width, rel=.06)
    assert fitted.bic == pytest.approx(fitted.chi2 + (3 * count + 4) * np.log(fitted.fit_mask.sum()))

