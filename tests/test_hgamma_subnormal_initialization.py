"""Regression tests for native joint Hgamma initialization at zero flux."""

from types import SimpleNamespace

import numpy as np
import pytest

import qsospec
from qsospec.fitting.complexes import resolve_recipe_coverage
from qsospec.fitting.global_fit import _ContinuumContext, _build_joint_hgamma_problem


def _problem(amplitude, linked_flux):
    wave = np.linspace(3300.0, 4550.0, 180)
    spectrum = qsospec.Spectrum.from_arrays(
        wave, np.full_like(wave, 2.0), err=np.full_like(wave, 0.02),
        wave_frame="rest", flux_unit="relative",
    )
    config = qsospec.GlobalContinuumConfig(
        uv_iron=None, optical_iron=None,
        regional_iron=qsospec.RegionalIronConfig(enabled=False),
        power_law=qsospec.PowerLawConfig(norm=2.0, slope=0.0),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            amplitude=1.0, sync_with_hbeta="never", sync_with_hgamma="soft",
        ),
        continuum_windows=((3300.0, 4260.0),), mask_windows=(),
        blue_absorption_clip_enabled=False, clip_passes=0,
    )
    context = _ContinuumContext(spectrum, config)
    context.initial[context.index["balmer_pseudocontinuum.amp"]] = amplitude
    continuum = SimpleNamespace(clip_mask=spectrum.valid_mask.copy())
    hgamma = SimpleNamespace(
        param_values={"Hgamma_broad.flux": linked_flux},
        fit_mask=spectrum.valid_mask.copy(),
    )
    coverage = resolve_recipe_coverage(
        spectrum, qsospec.recipes.get("oii_nev_neiii_hgamma")
    )
    return _build_joint_hgamma_problem(
        spectrum, config, continuum, hgamma, coverage, context
    )


@pytest.mark.parametrize("linked_flux", [np.nextafter(0.0, 1.0), 1.0e-300, 1.0])
def test_positive_subnormal_amplitude_initializes_finitely(linked_flux):
    amplitude = np.nextafter(0.0, 1.0)
    problem = _problem(amplitude, linked_flux)
    assert problem.ratio * amplitude == 0.0
    expected = float(np.clip(
        np.log10(linked_flux) - np.log10(problem.ratio) - np.log10(amplitude),
        -3.0, 3.0,
    ))
    assert problem.delta_start == expected
    assert np.all(np.isfinite(problem.start))
    assert np.all(np.isfinite(problem.data(problem.start)))
    assert np.all(np.isfinite(problem.prior(problem.start)))
    assert np.all(np.isfinite(problem.jacobian(problem.start)))
    assert problem.lower[-1] == -3.0
    assert problem.upper[-1] == 3.0


@pytest.mark.parametrize("amplitude,linked_flux", [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
def test_zero_flux_initialization_keeps_original_zero_offset(amplitude, linked_flux):
    problem = _problem(amplitude, linked_flux)
    assert problem.delta_start == 0.0
    assert np.all(np.isfinite(problem.data(problem.start)))
    assert np.all(np.isfinite(problem.jacobian(problem.start)))


@pytest.mark.parametrize("amplitude,linked_flux", [(20.0, 27.0), (1.0e-250, 1.0e-251), (1.0, 1.0e8)])
def test_ordinary_positive_initialization_preserves_exact_original_arithmetic(amplitude, linked_flux):
    problem = _problem(amplitude, linked_flux)
    expected = float(np.clip(
        np.log10(linked_flux / (problem.ratio * amplitude)), -3.0, 3.0
    ))
    assert problem.delta_start == expected


@pytest.mark.parametrize("bad_ratio", [0.0, -0.5, np.nan, np.inf, -np.inf])
def test_invalid_physical_anchor_ratio_fails_explicitly(monkeypatch, bad_ratio):
    monkeypatch.setattr(
        "qsospec.templates.balmer.load_balmer_anchor_ratios",
        lambda **kwargs: SimpleNamespace(hgamma_rel_hbeta=bad_ratio),
    )
    with pytest.raises(ValueError, match="anchor ratio must be finite and positive"):
        _problem(1.0, 1.0)


def test_subnormal_initialization_keeps_native_link_and_jacobian():
    problem = _problem(np.nextafter(0.0, 1.0), 1.0)
    state = problem.start.copy()
    amplitude_index = problem.ctx.index["balmer_pseudocontinuum.amp"]
    state[amplitude_index] = 20.0
    state[-1] = 0.2
    _, line_state = problem.unpack(state)
    assert line_state[problem.linked] == problem.ratio * 20.0 * 10.0 ** 0.2
    objective = lambda values: np.r_[problem.data(values), problem.prior(values)]
    jacobian = problem.jacobian(state)
    for index in (amplitude_index, state.size - 1):
        step = max(abs(state[index]) * 1.0e-6, 1.0e-6)
        plus, minus = state.copy(), state.copy()
        plus[index] += step
        minus[index] -= step
        finite = (objective(plus) - objective(minus)) / (2.0 * step)
        np.testing.assert_allclose(jacobian[:, index], finite, rtol=1.0e-5, atol=1.0e-7)
