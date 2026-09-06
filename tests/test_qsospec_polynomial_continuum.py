"""Survey-aware additive polynomial continuum tests."""

from dataclasses import replace

import numpy as np
import pytest

import qsospec


def _config(enabled=None, *, degree=2):
    return qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(norm=2.0, slope=-1.0),
        uv_iron=None,
        optical_iron=None,
        polynomial=qsospec.PolynomialContinuumConfig(
            enabled=enabled,
            degree=degree,
        ),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            enabled=False
        ),
        continuum_windows=((2000.0, 7000.0),),
        mask_windows=(),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
    )


def _spectrum(*, survey=None, wave=None, corrected=False):
    wave = np.linspace(2000.0, 7000.0, 900) if wave is None else wave
    x = (wave - 3000.0) / 3000.0
    flux = (
        2.0 * (wave / 3000.0) ** -1.0
        + 0.3 * x
        - 0.2 * x**2
        + 0.08 * x**3
    )
    unit = {"survey": survey} if survey is not None else {"flux_unit": "relative"}
    return qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.01),
        wave_frame="rest",
        galactic_extinction_corrected=corrected,
        **unit,
    )


def test_polynomial_auto_activates_only_for_explicit_sdss_survey():
    sdss = qsospec.fit_global_continuum(_spectrum(survey="sdss"), _config())
    desi = qsospec.fit_global_continuum(_spectrum(survey="desi"), _config())
    unspecified = qsospec.fit_global_continuum(_spectrum(), _config())

    assert sdss.metadata["polynomial_effective"] is True
    assert sdss.metadata["polynomial_activation_reason"] == "automatic_sdss"
    assert "polynomial" in sdss.component_models
    assert desi.metadata["polynomial_effective"] is False
    assert desi.metadata["polynomial_activation_reason"] == "automatic_non_sdss"
    assert unspecified.metadata["polynomial_effective"] is False


def test_polynomial_preserves_baseline_slope_and_limits_correction():
    result = qsospec.fit_global_continuum(_spectrum(survey="sdss"), _config())
    baseline = qsospec.fit_global_continuum(_spectrum(survey="sdss"), _config(False))

    assert result.success
    assert result.param_values["power_law.slope"] == baseline.param_values["power_law.slope"]
    assert result.param_errors["power_law.slope"] == baseline.param_errors["power_law.slope"]
    assert np.max(np.abs(result.component_models["polynomial"] / baseline.component_models["power_law"])) <= 0.1
    assert result.metadata["polynomial_status"] == "accepted"
    assert result.metadata["polynomial_degree"] == 2
    assert result.metadata["polynomial_pivot"] == 3000.0
    assert result.metadata["polynomial_scale"] == 3000.0
    assert result.metadata["polynomial_rank"] == 2
    assert result.metadata["polynomial_covariance_policy"] == "conditional_on_baseline_slopes"
    assert np.all(np.isnan(result.covariance[-1, :-1]))
    assert result.optimizer_result.x.size == len(result.param_values) - 1
    np.testing.assert_array_equal(result.clip_mask, baseline.clip_mask)


def test_explicit_polynomial_setting_overrides_survey_default():
    forced_off = qsospec.fit_global_continuum(
        _spectrum(survey="sdss"),
        _config(False),
    )
    forced_on = qsospec.fit_global_continuum(
        _spectrum(survey="desi"),
        _config(True),
    )

    assert forced_off.metadata["polynomial_activation_reason"] == "explicit_disabled"
    assert "polynomial" not in forced_off.component_models
    assert forced_on.metadata["polynomial_activation_reason"] == "explicit_enabled"
    assert "polynomial" in forced_on.component_models


def test_polynomial_is_disabled_when_wavelength_leverage_is_too_small():
    wave = np.linspace(3000.0, 3010.0, 100)
    config = replace(
        _config(),
        continuum_windows=((3000.0, 3010.0),),
    )
    result = qsospec.fit_global_continuum(
        _spectrum(survey="sdss", wave=wave),
        config,
    )

    assert result.success
    assert result.metadata["polynomial_effective"] is False
    assert result.metadata["polynomial_status"] == "disabled_insufficient_coverage"
    assert "global_polynomial_disabled_insufficient_coverage" in result.warning_codes()


def test_polynomial_follows_power_law_auto_comparison():
    config = replace(
        _config(True),
        power_law=qsospec.PowerLawConfig(
            mode="auto",
            norm=2.0,
            slope=-1.0,
        ),
    )
    result = qsospec.fit_global_continuum(
        _spectrum(survey="sdss"),
        config,
    )

    assert result.success
    assert "polynomial" in result.component_models
    assert result.metadata["polynomial_effective"] is True
    assert result.metadata["power_law_mode_selected"] in {"single", "double"}
    baseline = qsospec.fit_global_continuum(
        _spectrum(survey="sdss"), replace(config, polynomial=replace(config.polynomial, enabled=False))
    )
    for name in ("power_law_mode_selected", "power_law_single_bic", "power_law_double_bic"):
        if isinstance(baseline.metadata[name], float):
            assert result.metadata[name] == pytest.approx(baseline.metadata[name])
        else:
            assert result.metadata[name] == baseline.metadata[name]


def test_polynomial_component_and_selection_metadata_round_trip(tmp_path):
    run = tmp_path / "sdss_polynomial"
    result = qsospec.fit_object_to_store(
        _spectrum(survey="sdss", corrected=True),
        str(run),
        object_id="sdss-polynomial",
        global_config=_config(),
        complexes=[],
        write_qa=False,
    )
    loaded = qsospec.load_model(str(run), "sdss-polynomial")

    np.testing.assert_allclose(
        loaded.continuum.component_models["polynomial"],
        result.continuum.component_models["polynomial"],
    )
    assert loaded.metadata["polynomial_effective"] is True
    assert loaded.metadata["polynomial_activation_reason"] == "automatic_sdss"
    assert loaded.metadata["polynomial_coefficients"] == pytest.approx(
        result.continuum.metadata["polynomial_coefficients"]
    )
    assert loaded.continuum.metadata["polynomial_baseline_slopes"] == result.continuum.metadata["polynomial_baseline_slopes"]
    figure = loaded.plot_qa()
    axes = [axis for axis in figure.axes if axis.get_ylabel() == "Polynomial\n/ baseline PL"]
    assert len(axes) == 1
    expected = loaded.continuum.component_models["polynomial"] / (
        loaded.metadata["polynomial_baseline_norm"]
        * (loaded.spectrum.wave_rest / 3000) ** loaded.metadata["polynomial_baseline_slopes"]["power_law.slope"]
    )
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), expected)
    import matplotlib.pyplot as plt
    plt.close(figure)


@pytest.mark.parametrize("kwargs", [
    {"max_fraction": 0}, {"max_fraction": np.nan}, {"max_fraction": 1},
    {"max_norm_fraction": -1}, {"auto_delta_bic": -1}, {"auto_delta_bic": np.inf},
])
def test_quadratic_configuration_validation(kwargs):
    assert qsospec.PolynomialContinuumConfig().degree == 2
    with pytest.raises(ValueError):
        qsospec.PolynomialContinuumConfig(**kwargs)


def test_null_auto_case_returns_baseline_arrays_unchanged():
    spectrum = _spectrum(survey="sdss")
    spectrum = replace(spectrum, flux=2 * (spectrum.wave_rest / 3000)**-1)
    baseline = qsospec.fit_global_continuum(spectrum, _config(False))
    result = qsospec.fit_global_continuum(spectrum, _config())
    assert result.metadata["polynomial_status"] == "bic_improvement_insufficient"
    assert "polynomial" not in result.component_models
    np.testing.assert_array_equal(result.model, baseline.model)
    assert result.param_values == baseline.param_values


@pytest.mark.parametrize("method", ["variable_projection", "legacy_joint"])
@pytest.mark.parametrize("degree", [2, 3])
def test_unit_scaling_solver_parity_and_fraction_envelopes(method, degree):
    spectrum = _spectrum()
    config = replace(_config(True, degree=degree), optimizer_method=method)
    baseline = qsospec.fit_global_continuum(spectrum, replace(config, polynomial=replace(config.polynomial, enabled=False)))
    result = qsospec.fit_global_continuum(spectrum, config)
    assert result.metadata["polynomial_effective"]
    assert result.param_values["power_law.slope"] == baseline.param_values["power_law.slope"]
    assert abs(result.param_values["power_law.norm"] / baseline.param_values["power_law.norm"] - 1) <= 0.10000001
    factor = 1e-17
    scaled = replace(spectrum, flux=spectrum.flux * factor, err=spectrum.err * factor)
    scaled_config = replace(config, power_law=replace(config.power_law, norm=2 * factor))
    scaled_result = qsospec.fit_global_continuum(scaled, scaled_config)
    np.testing.assert_allclose(scaled_result.model / factor, result.model, rtol=1e-4, atol=1e-5)
    assert scaled_result.metadata["polynomial_fraction_max"] == pytest.approx(result.metadata["polynomial_fraction_max"], rel=1e-4)


def test_automatic_bic_boundary_and_explicit_bypass():
    spectrum = _spectrum(survey="sdss")
    trial = qsospec.fit_global_continuum(spectrum, _config(True))
    delta = trial.metadata["polynomial_delta_bic"]
    assert delta > 0
    config = _config()
    accepted = qsospec.fit_global_continuum(spectrum, replace(config, polynomial=replace(config.polynomial, auto_delta_bic=delta)))
    rejected = qsospec.fit_global_continuum(spectrum, replace(config, polynomial=replace(config.polynomial, auto_delta_bic=delta + 0.01)))
    assert accepted.metadata["polynomial_effective"]
    assert not rejected.metadata["polynomial_effective"]
    assert trial.metadata["polynomial_selection_reason"] == "explicit_enabled"


def test_candidate_exception_falls_back(monkeypatch):
    from qsospec.fitting import global_fit
    original = global_fit._fit_global_continuum_fixed
    def fail_candidate(*args, **kwargs):
        if kwargs.get("fixed_parameters"):
            raise ValueError("candidate failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(global_fit, "_fit_global_continuum_fixed", fail_candidate)
    result = qsospec.fit_global_continuum(_spectrum(), _config(True))
    assert result.success
    assert result.metadata["polynomial_status"] == "candidate_failed"
    assert "candidate failure" in result.metadata["polynomial_candidate_failure"]


def test_ill_conditioned_candidate_rejected(monkeypatch):
    from qsospec.fitting import global_fit
    original = global_fit._fit_global_continuum_fixed
    def singular_candidate(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("fixed_parameters"):
            result.optimizer_result.jac[:, -1] = result.optimizer_result.jac[:, 0]
        return result
    monkeypatch.setattr(global_fit, "_fit_global_continuum_fixed", singular_candidate)
    result = qsospec.fit_global_continuum(_spectrum(), _config(True))
    assert result.metadata["polynomial_status"] == "ill_conditioned"


def test_signed_quadratic_recovery_given_independent_baseline(monkeypatch):
    from qsospec.fitting import global_fit
    config = _config(True)
    spectrum = _spectrum()
    wave = spectrum.wave_rest
    pure_flux = 2 * (wave / 3000)**-1
    baseline = qsospec.fit_global_continuum(replace(spectrum, flux=pure_flux), _config(False))
    x = (wave - 3000) / 3000
    spectrum = replace(spectrum, flux=pure_flux + 0.015 * x - 0.01 * x**2)
    baseline.chi2 = float(np.sum(((spectrum.flux - baseline.model) / spectrum.err)**2))
    monkeypatch.setattr(global_fit, "_fit_global_continuum_power_law_selection", lambda *a, **k: baseline)
    result = qsospec.fit_global_continuum(spectrum, config)
    assert result.metadata["polynomial_effective"]
    assert result.param_values["polynomial.c1"] == pytest.approx(0.015, abs=1e-8)
    assert result.param_values["polynomial.c2"] == pytest.approx(-0.01, abs=1e-8)


def test_broken_slopes_and_clipped_pixels_are_fixed():
    from qsospec.fitting.global_fit import _broken_power_law_basis
    spectrum = _spectrum()
    wave = spectrum.wave_rest
    flux = 2 * _broken_power_law_basis(wave, pivot=3000, break_wave=4661,
                                     blue_slope=-1, red_slope=-2)
    x = (wave - 3000) / 3000
    flux += 0.01 * x**2
    flux[30] -= 2
    spectrum = replace(spectrum, flux=flux)
    config = replace(_config(True), power_law=qsospec.PowerLawConfig(mode="double"),
                     blue_absorption_clip_enabled=True)
    baseline = qsospec.fit_global_continuum(spectrum, replace(config, polynomial=replace(config.polynomial, enabled=False)))
    result = qsospec.fit_global_continuum(spectrum, config, compute_covariance=False)
    assert result.metadata["polynomial_effective"]
    for name in ("power_law.slope", "power_law.red_slope"):
        assert result.param_values[name] == baseline.param_values[name]
        assert np.isnan(result.param_errors[name])
    assert not result.clip_mask[30]
    np.testing.assert_array_equal(result.clip_mask, baseline.clip_mask)
    assert result.covariance is None
    assert result.metadata["blue_absorption_clip_rejected_pixels"] > 0


def test_missing_power_law_and_conflicting_bounds_fall_back():
    config = replace(_config(True), power_law=qsospec.PowerLawConfig(enabled=False))
    # An otherwise usable baseline without a power law can be supplied by iron.
    config = replace(config, optical_iron=qsospec.IronTemplateConfig.park22())
    result = qsospec.fit_global_continuum(_spectrum(), config)
    assert result.metadata["polynomial_status"] == "baseline_power_law_unavailable"
    config = _config(True)
    config = replace(config, polynomial=replace(config.polynomial, coefficient_bounds=(10, 20)))
    result = qsospec.fit_global_continuum(_spectrum(), config)
    assert result.metadata["polynomial_status"] == "incompatible_bounds"


def test_monte_carlo_refits_baseline_in_each_trial(monkeypatch):
    from qsospec.fitting import global_fit
    original = global_fit._fit_global_continuum_power_law_selection
    baseline_slopes = []
    def record_baseline(*args, **kwargs):
        result = original(*args, **kwargs)
        baseline_slopes.append(result.param_values["power_law.slope"])
        return result
    monkeypatch.setattr(global_fit, "_fit_global_continuum_power_law_selection", record_baseline)
    qsospec.fit_global_lines(
        _spectrum(), _config(True), complexes=[],
        uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=3, random_seed=12),
    )
    assert len(baseline_slopes) == 4
    assert np.ptp(baseline_slopes) > 0
