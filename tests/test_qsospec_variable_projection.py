"""Derivative, fallback, and parity tests for global variable projection."""

from dataclasses import replace

import numpy as np
import pytest

import qsospec
from qsospec.fitting import global_fit
from qsospec.fitting.global_fit import (
    C_KMS,
    _ContinuumContext,
    _HbetaContext,
    _gaussian_area_profile,
    _gaussian_unit_profile_with_derivatives,
)
from qsospec.global_result import GlobalContinuumResult
from qsospec.templates import (
    evaluate_balmer_pseudocontinuum,
    evaluate_balmer_pseudocontinuum_with_derivatives,
    load_balmer_template,
    load_iron_template,
)
from qsospec.templates.iron import (
    evaluate_iron_basis,
    evaluate_iron_basis_with_derivative,
)
from qsospec.solvers.variable_projection import (
    VariableProjectionError,
    _VariableProjectionProblem,
)


def _centered_difference(function, value, step):
    return (function(value + step) - function(value - step)) / (2.0 * step)


def _known_continuum(spectrum, model):
    return GlobalContinuumResult(
        success=True,
        status=1,
        message="known",
        param_values={},
        param_errors={},
        covariance=None,
        chi2=0.0,
        dof=1,
        reduced_chi2=0.0,
        wave_rest=spectrum.wave_rest.copy(),
        model=model.copy(),
        component_models={"power_law": model.copy()},
        fit_mask=spectrum.valid_mask.copy(),
        clip_mask=spectrum.valid_mask.copy(),
    )


@pytest.mark.parametrize(
    ("template_name", "wave"),
    [
        ("vw01", np.linspace(2200.0, 3090.0, 400)),
        ("park22", np.linspace(4435.0, 5535.0, 400)),
    ],
)
def test_iron_width_derivative_matches_centered_difference(template_name, wave):
    template = load_iron_template(template_name)
    _, derivative = evaluate_iron_basis_with_derivative(template, wave, 3000.0)
    finite = _centered_difference(
        lambda width: evaluate_iron_basis(template, wave, width),
        3000.0,
        0.1,
    )
    assert derivative == pytest.approx(finite, rel=2.0e-5, abs=1.0e-12)


@pytest.mark.parametrize(
    "wave",
    [
        np.linspace(1800.0, 3646.0, 500),
        np.linspace(3646.0, 4260.0, 500),
    ],
)
def test_balmer_pseudocontinuum_derivatives_match_centered_differences(wave):
    template = load_balmer_template(provenance="sh95_k13full_ext")
    _, _, _, derivative_fwhm, derivative_velocity = (
        evaluate_balmer_pseudocontinuum_with_derivatives(
            template, wave, 2800.0, 250.0
        )
    )
    finite_fwhm = _centered_difference(
        lambda width: evaluate_balmer_pseudocontinuum(
            template, wave, width, 250.0
        ),
        2800.0,
        0.1,
    )
    finite_velocity = _centered_difference(
        lambda velocity: evaluate_balmer_pseudocontinuum(
            template, wave, 2800.0, velocity
        ),
        250.0,
        0.01,
    )
    assert derivative_fwhm == pytest.approx(
        finite_fwhm, rel=2.0e-5, abs=1.0e-12
    )
    assert derivative_velocity == pytest.approx(
        finite_velocity, rel=2.0e-5, abs=1.0e-12
    )


def test_gaussian_velocity_and_width_derivatives_match_centered_difference():
    wave = np.linspace(4600.0, 5100.0, 1000)
    _, derivative_velocity, derivative_width = _gaussian_unit_profile_with_derivatives(
        wave, 4862.68, -320.0, 2800.0
    )
    finite_velocity = _centered_difference(
        lambda velocity: _gaussian_unit_profile_with_derivatives(
            wave, 4862.68, velocity, 2800.0
        )[0],
        -320.0,
        0.01,
    )
    finite_width = _centered_difference(
        lambda width: _gaussian_unit_profile_with_derivatives(
            wave, 4862.68, -320.0, width
        )[0],
        2800.0,
        0.01,
    )
    assert derivative_velocity == pytest.approx(finite_velocity, rel=2.0e-5, abs=1.0e-12)
    assert derivative_width == pytest.approx(finite_width, rel=2.0e-5, abs=1.0e-12)


def test_continuum_design_derivatives_match_centered_differences():
    wave = np.linspace(1900.0, 5600.0, 1600)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        2.0 * (wave / 3000.0) ** -1.2,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
    )
    context = _ContinuumContext(spectrum, qsospec.GlobalContinuumConfig())
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    design, derivatives = context.separable_design(nonlinear, wave, True)
    assert design.shape[1] == len(context.linear_names)
    for index, derivative in enumerate(derivatives):
        step = 0.01 if "fwhm" in context.nonlinear_names[index] else 1.0e-5

        def evaluate(value):
            trial = nonlinear.copy()
            trial[index] = value
            return context.separable_design(trial, wave, False)[0]

        finite = _centered_difference(evaluate, nonlinear[index], step)
        assert derivative == pytest.approx(finite, rel=5.0e-5, abs=1.0e-11)


def test_hbeta_design_derivatives_match_centered_differences():
    wave = np.linspace(4640.0, 5100.0, 900)
    context = _HbetaContext(qsospec.HbetaComplexConfig(), include_wing=True, flux_scale=100.0)
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    design, derivatives = context.separable_design(nonlinear, wave, True)
    assert design.shape[1] == len(context.linear_names)
    for index, derivative in enumerate(derivatives):
        step = 0.01

        def evaluate(value):
            trial = nonlinear.copy()
            trial[index] = value
            return context.separable_design(trial, wave, False)[0]

        finite = _centered_difference(evaluate, nonlinear[index], step)
        assert derivative == pytest.approx(finite, rel=5.0e-5, abs=1.0e-11)


@pytest.mark.parametrize("active_second_component", [False, True])
def test_reduced_jacobian_matches_finite_difference_with_active_sets(
    active_second_component,
):
    coordinate = np.linspace(-1.0, 1.0, 100)
    nonlinear = np.array([0.3])
    second_amplitude = -0.5 if active_second_component else 0.5
    flux = 2.0 + second_amplitude * np.exp(nonlinear[0] * coordinate)
    err = np.full_like(coordinate, 0.1)

    def evaluator(values, need_derivatives):
        exponential = np.exp(values[0] * coordinate)
        design = np.column_stack([np.ones_like(coordinate), exponential])
        derivative = np.column_stack([np.zeros_like(coordinate), coordinate * exponential])
        return design, (derivative,) if need_derivatives else None

    problem = _VariableProjectionProblem(
        flux,
        err,
        (np.zeros(2), np.full(2, np.inf)),
        evaluator,
    )
    analytic = problem.jacobian(nonlinear)[:, 0]
    finite = _centered_difference(
        lambda value: problem.residual(np.array([value])),
        nonlinear[0],
        1.0e-6,
    )
    assert analytic == pytest.approx(finite, rel=2.0e-5, abs=1.0e-8)
    expected_active = -1 if active_second_component else 0
    assert problem.state(nonlinear, True).linear_active_mask[1] == expected_active


def test_continuum_variable_projection_matches_legacy_joint():
    wave = np.linspace(1900.0, 5600.0, 2400)
    model = 2.5 * (wave / 3000.0) ** -1.25
    model += 70.0 * evaluate_iron_basis(load_iron_template("vw01"), wave, 2600.0)
    model += 55.0 * evaluate_iron_basis(load_iron_template("park22"), wave, 3200.0)
    model += 8.0 * evaluate_balmer_pseudocontinuum(
        load_balmer_template(provenance="sh95_k13full_ext"),
        wave,
        2800.0,
        200.0,
    )
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        model,
        err=np.full_like(wave, 0.03),
        wave_frame="rest",
    )
    base = qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(norm=2.4, slope=-1.1),
        uv_iron=qsospec.IronTemplateConfig.vw01(fwhm_kms=2500.0, amp=60.0),
        optical_iron=qsospec.IronTemplateConfig.park22(fwhm_kms=3400.0, amp=45.0),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            amplitude=7.0,
            fwhm_kms=3000.0,
            velocity_kms=0.0,
        ),
        clip_passes=0,
    )
    optimized = qsospec.fit_global_continuum(
        spectrum, replace(base, optimizer_method="variable_projection")
    )
    legacy = qsospec.fit_global_continuum(
        spectrum, replace(base, optimizer_method="legacy_joint")
    )

    assert optimized.metadata["optimizer_used"] == "variable_projection"
    assert not optimized.metadata["optimizer_fallback"]
    assert optimized.chi2 <= legacy.chi2 + 1.0e-5
    assert optimized.covariance.shape == legacy.covariance.shape
    for name, legacy_value in legacy.param_values.items():
        assert optimized.param_values[name] == pytest.approx(legacy_value, rel=5.0e-3, abs=5.0e-3)


def test_continuum_parity_with_partial_coverage_clipping_and_fixed_balmer_width():
    wave = np.linspace(3300.0, 5200.0, 1800)
    model = 2.2 * (wave / 3000.0) ** -1.1
    model += 45.0 * evaluate_iron_basis(load_iron_template("park22"), wave, 3100.0)
    model += 12.0 * evaluate_balmer_pseudocontinuum(
        load_balmer_template(provenance="sh95_k13full_ext"),
        wave,
        2800.0,
        -150.0,
    )
    flux = model.copy()
    flux[300] += 2.0
    flux[900] -= 2.0
    spectrum = qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.04), wave_frame="rest"
    )
    base = qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=qsospec.IronTemplateConfig.park22(fwhm_kms=3000.0, amp=40.0),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            amplitude=10.0,
            fit_fwhm=False,
            fwhm_kms=2800.0,
        ),
    )
    optimized = qsospec.fit_global_continuum(
        spectrum, replace(base, optimizer_method="variable_projection")
    )
    legacy = qsospec.fit_global_continuum(
        spectrum, replace(base, optimizer_method="legacy_joint")
    )

    assert np.array_equal(optimized.clip_mask, legacy.clip_mask)
    assert optimized.reduced_chi2 <= legacy.reduced_chi2 + 1.0e-5
    assert optimized.metadata["balmer_pseudocontinuum_fwhm_fixed"]
    assert optimized.param_values["power_law.norm"] == pytest.approx(
        legacy.param_values["power_law.norm"], rel=5.0e-3
    )


def test_hbeta_variable_projection_matches_legacy_wing_selection():
    wave = np.linspace(4600.0, 5120.0, 1800)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 100.0, 4862.68, 2500.0)
    line += _gaussian_area_profile(wave, 35.0, 5008.24, 320.0)
    line += _gaussian_area_profile(wave, 35.0 / 2.98, 4960.30, 320.0)
    wing_velocity = -350.0
    line += _gaussian_area_profile(
        wave, 30.0, 5008.24 * np.exp(wing_velocity / C_KMS), 1100.0
    )
    line += _gaussian_area_profile(
        wave, 30.0 / 2.98, 4960.30 * np.exp(wing_velocity / C_KMS), 1100.0
    )
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
    )
    continuum_result = _known_continuum(spectrum, continuum)
    optimized = qsospec.fit_hbeta_complex(
        spectrum,
        continuum_result,
        qsospec.HbetaComplexConfig(optimizer_method="variable_projection"),
    )
    legacy = qsospec.fit_hbeta_complex(
        spectrum,
        continuum_result,
        qsospec.HbetaComplexConfig(optimizer_method="legacy_joint"),
    )

    assert optimized.selected_model == legacy.selected_model == "wing"
    assert optimized.reduced_chi2 <= legacy.reduced_chi2 + 1.0e-4
    assert optimized.metrics["Hb_broad_flux_input"] == pytest.approx(
        legacy.metrics["Hb_broad_flux_input"], rel=5.0e-3
    )
    tolerance = max(5.0, 0.002 * legacy.metrics["Hb_broad_fwhm_kms"])
    assert abs(
        optimized.metrics["Hb_broad_fwhm_kms"] - legacy.metrics["Hb_broad_fwhm_kms"]
    ) <= tolerance


def test_hbeta_parity_for_core_model_with_heii_and_rejected_wing():
    wave = np.linspace(4600.0, 5120.0, 1800)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 90.0, 4862.68, 2200.0)
    line += _gaussian_area_profile(wave, 30.0, 5008.24, 350.0)
    line += _gaussian_area_profile(wave, 30.0 / 2.98, 4960.30, 350.0)
    line += _gaussian_area_profile(wave, 12.0, 4687.02, 1800.0)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
    )
    continuum_result = _known_continuum(spectrum, continuum)
    optimized = qsospec.fit_hbeta_complex(
        spectrum,
        continuum_result,
        qsospec.HbetaComplexConfig(
            heii_enabled=True, optimizer_method="variable_projection"
        ),
    )
    legacy = qsospec.fit_hbeta_complex(
        spectrum,
        continuum_result,
        qsospec.HbetaComplexConfig(
            heii_enabled=True, optimizer_method="legacy_joint"
        ),
    )

    assert optimized.selected_model == legacy.selected_model == "core"
    assert optimized.param_values["HeII_broad.flux"] == pytest.approx(
        legacy.param_values["HeII_broad.flux"], rel=5.0e-3
    )
    assert optimized.metrics["Hb_broad_flux_input"] == pytest.approx(
        legacy.metrics["Hb_broad_flux_input"], rel=5.0e-3
    )


def test_auto_optimizer_falls_back_and_required_variable_projection_raises(monkeypatch):
    wave = np.linspace(4700.0, 5500.0, 600)
    flux = 2.0 * (wave / 3000.0) ** -1.2
    spectrum = qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.05), wave_frame="rest"
    )

    def fail(*args, **kwargs):
        raise VariableProjectionError("forced failure")

    monkeypatch.setattr(global_fit, "_solve_separable_once", fail)
    automatic = qsospec.fit_global_continuum(
        spectrum, qsospec.GlobalContinuumConfig(optimizer_method="auto")
    )
    assert automatic.metadata["optimizer_used"] == "legacy_joint"
    assert automatic.metadata["optimizer_fallback"]
    assert "optimizer_fallback_legacy" in automatic.warning_codes()

    with pytest.raises(VariableProjectionError, match="forced failure"):
        qsospec.fit_global_continuum(
            spectrum,
            qsospec.GlobalContinuumConfig(optimizer_method="variable_projection"),
        )


def test_reduced_two_point_jacobian_mode_is_available():
    wave = np.linspace(4700.0, 5500.0, 600)
    flux = 2.0 * (wave / 3000.0) ** -1.2
    spectrum = qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.05), wave_frame="rest"
    )
    result = qsospec.fit_global_continuum(
        spectrum,
        qsospec.GlobalContinuumConfig(
            optimizer_method="variable_projection",
            jacobian_method="2-point",
        ),
    )
    assert result.success
    assert result.metadata["optimizer_used"] == "variable_projection"
    assert result.metadata["jacobian_method"] == "2-point"
