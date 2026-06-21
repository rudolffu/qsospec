"""Single, broken, and automatically selected global power laws."""

import numpy as np
import pytest

import qsospec
from qsospec.fitting.global_fit import (
    _ContinuumContext,
    _broken_power_law_basis,
)


def _config(mode):
    return qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(mode=mode),
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            enabled=False
        ),
        continuum_windows=((2000.0, 7000.0),),
        mask_windows=(),
        blue_absorption_clip_enabled=False,
        clip_passes=0,
    )


def _spectrum(flux, wave):
    return qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        flux_unit="relative",
    )


def test_broken_power_law_is_continuous_and_recovers_slopes():
    wave = np.linspace(2000.0, 7000.0, 1500)
    basis = _broken_power_law_basis(
        wave,
        pivot=3000.0,
        break_wave=4661.0,
        blue_slope=-1.0,
        red_slope=-3.0,
    )
    result = qsospec.fit_global_continuum(
        _spectrum(3.0 * basis, wave), _config("double")
    )

    assert result.param_values["power_law.slope"] == pytest.approx(
        -1.0, abs=1e-4
    )
    assert result.param_values["power_law.red_slope"] == pytest.approx(
        -3.0, abs=1e-4
    )
    edge = 4661.0
    left = _broken_power_law_basis(
        np.array([edge * (1 - 1e-10)]),
        pivot=3000.0,
        break_wave=edge,
        blue_slope=-1.0,
        red_slope=-3.0,
    )[0]
    right = _broken_power_law_basis(
        np.array([edge]),
        pivot=3000.0,
        break_wave=edge,
        blue_slope=-1.0,
        red_slope=-3.0,
    )[0]
    assert left == pytest.approx(right, rel=1e-9)


def test_auto_selects_double_only_with_strong_bic_evidence():
    wave = np.linspace(2000.0, 7000.0, 1500)
    broken = 3.0 * _broken_power_law_basis(
        wave,
        pivot=3000.0,
        break_wave=4661.0,
        blue_slope=-1.0,
        red_slope=-3.0,
    )
    selected_double = qsospec.fit_global_continuum(
        _spectrum(broken, wave), _config("auto")
    )
    selected_single = qsospec.fit_global_continuum(
        _spectrum(3.0 * (wave / 3000.0) ** -1.5, wave),
        _config("auto"),
    )

    assert selected_double.metadata["power_law_mode_selected"] == "double"
    assert selected_double.metadata["power_law_delta_bic"] >= 10.0
    assert selected_single.metadata["power_law_mode_selected"] == "single"
    assert (
        selected_single.metadata["power_law_selection_reason"]
        == "double_bic_improvement_insufficient"
    )


def test_double_requires_coverage_on_both_sides():
    wave = np.linspace(2000.0, 4300.0, 500)
    with pytest.raises(ValueError, match="each side"):
        qsospec.fit_global_continuum(
            _spectrum(2.0 * (wave / 3000.0) ** -1.2, wave),
            _config("double"),
        )
    fallback = qsospec.fit_global_continuum(
        _spectrum(2.0 * (wave / 3000.0) ** -1.2, wave),
        _config("auto"),
    )
    assert fallback.metadata["power_law_mode_selected"] == "single"
    assert (
        fallback.metadata["power_law_selection_reason"]
        == "double_insufficient_coverage"
    )


def test_double_power_law_slope_derivatives_match_finite_differences():
    wave = np.linspace(2000.0, 7000.0, 600)
    spectrum = _spectrum(2.0 * (wave / 3000.0) ** -1.5, wave)
    context = _ContinuumContext(spectrum, _config("double"))
    nonlinear = np.array(
        [context.initial[context.index[name]] for name in context.nonlinear_names]
    )
    design, derivatives = context.separable_design(
        nonlinear, wave, need_derivatives=True
    )
    assert design.shape[1] == 1
    for index, name in enumerate(context.nonlinear_names):
        if name not in ("power_law.slope", "power_law.red_slope"):
            continue
        step = 1e-6
        upper = nonlinear.copy()
        lower = nonlinear.copy()
        upper[index] += step
        lower[index] -= step
        upper_design, _ = context.separable_design(
            upper, wave, need_derivatives=False
        )
        lower_design, _ = context.separable_design(
            lower, wave, need_derivatives=False
        )
        numerical = (upper_design - lower_design) / (2.0 * step)
        np.testing.assert_allclose(
            derivatives[index], numerical, rtol=2e-6, atol=2e-8
        )
