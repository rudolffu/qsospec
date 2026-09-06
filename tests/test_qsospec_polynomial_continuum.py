"""Survey-aware additive polynomial continuum tests."""

from dataclasses import replace

import numpy as np
import pytest

import qsospec


def _config(enabled=None, *, degree=3):
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


def test_polynomial_recovers_signed_coefficients_and_metadata():
    result = qsospec.fit_global_continuum(_spectrum(survey="sdss"), _config())

    assert result.success
    assert result.param_values["polynomial.c1"] == pytest.approx(0.3, abs=2e-3)
    assert result.param_values["polynomial.c2"] == pytest.approx(-0.2, abs=2e-3)
    assert result.param_values["polynomial.c3"] == pytest.approx(0.08, abs=2e-3)
    assert result.metadata["polynomial_status"] == "enabled"
    assert result.metadata["polynomial_degree"] == 3
    assert result.metadata["polynomial_pivot"] == 3000.0
    assert result.metadata["polynomial_scale"] == 3000.0
    assert result.metadata["polynomial_rank"] == 3


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


def test_polynomial_is_included_in_power_law_auto_comparison():
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
