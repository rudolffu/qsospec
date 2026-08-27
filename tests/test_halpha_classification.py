"""Tests for the opt-in, RGS-aware H-alpha model comparison."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import qsospec
from qsospec.fitting.global_fit import C_KMS, _HalphaContext, _gaussian_area_profile
from qsospec.global_result import GlobalContinuumResult


def _continuum_result(spectrum, model):
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


def _synthetic_halpha(*, broad_flux=0.0, narrow_fwhm=700.0):
    wave = np.linspace(6350.0, 6850.0, 1000)
    continuum = np.ones_like(wave)
    line = np.zeros_like(wave)
    if broad_flux:
        line += _gaussian_area_profile(wave, broad_flux, 6564.61, 3600.0)
    for flux, center in (
        (25.0, 6564.61),
        (18.0, 6585.28),
        (18.0 / 2.96, 6549.85),
        (5.0, 6718.29),
        (4.0, 6732.67),
    ):
        line += _gaussian_area_profile(wave, flux, center, narrow_fwhm)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.03),
        wave_frame="rest",
        flux_unit="relative",
    )
    return spectrum, _continuum_result(spectrum, continuum)


def test_r480_width_conversion_and_unresolved_behavior():
    lsf = qsospec.LineSpreadFunctionConfig(resolving_power=480.0)
    instrumental = C_KMS / 480.0
    assert lsf.instrumental_fwhm_kms == pytest.approx(instrumental)
    observed = qsospec.observed_fwhm_kms(1200.0, instrumental)
    assert observed == pytest.approx(1352.80623630785)
    assert qsospec.intrinsic_fwhm_kms(observed, instrumental) == pytest.approx(1200.0)
    assert qsospec.intrinsic_fwhm_kms(instrumental - 1.0, instrumental) == 0.0


def test_model_grid_bounds_put_intrinsic_1200_at_exact_observed_boundary():
    config = qsospec.HalphaModelSelectionConfig()
    narrow, broad = qsospec.observed_halpha_width_bounds(config, 3)
    boundary = qsospec.observed_fwhm_kms(
        1200.0, config.lsf.instrumental_fwhm_kms
    )
    assert narrow[1] == pytest.approx(boundary)
    assert broad[0][0] == pytest.approx(boundary)
    assert broad[0][1] == pytest.approx(broad[1][0])
    assert broad[1][1] == pytest.approx(broad[2][0])


@pytest.mark.parametrize("count", (0, 1, 2, 3))
def test_halpha_context_supports_zero_through_three_broad_components(count):
    selection = qsospec.HalphaModelSelectionConfig()
    narrow, broad = qsospec.observed_halpha_width_bounds(selection, count)
    config = replace(
        qsospec.HalphaComplexConfig(),
        narrow_fwhm_bounds_kms=narrow,
        broad_fwhm_bands_kms=broad,
    )
    context = _HalphaContext(config, 0.0)
    assert context.n_broad_components == count
    assert sum(name.startswith("Ha_broad") for name in context.linear_names) == count


def test_halpha_config_rejects_too_many_or_non_increasing_bands():
    with pytest.raises(ValueError, match="zero and three"):
        qsospec.HalphaComplexConfig(
            broad_fwhm_bands_kms=((1.0, 2.0),) * 4
        )
    with pytest.raises(ValueError, match="positive and increasing"):
        qsospec.HalphaComplexConfig(broad_fwhm_bands_kms=((2000.0, 1200.0),))


def test_legacy_halpha_default_still_uses_three_broad_components():
    spectrum, continuum = _synthetic_halpha(broad_flux=90.0)
    result = qsospec.fit_halpha_complex(
        spectrum, continuum, compute_covariance=False
    )
    assert result.success
    assert result.metadata["n_broad_components"] == 3
    assert result.selected_model == "three_broad_plus_tied_narrow"


def test_narrow_spectrum_prefers_n0_and_never_assigns_physical_class():
    spectrum, continuum = _synthetic_halpha()
    result = qsospec.fit_halpha_model_grid(
        spectrum, continuum, compute_covariance=False
    )
    assert set(result.candidates) == {"N0", "B1"}
    assert all(candidate.success for candidate in result.candidates.values())
    assert result.minimum_bic_model == "N0"
    record = result.to_record("-42")
    assert record["object_id"] == "-42"
    assert record["selection_status"] == "not_calibrated"
    assert record["physical_class"] is None
    assert np.isnan(record["broad_halpha_fraction_upper_95"])
    assert record["broad_fraction_status"] == "not_calibrated_profile_limit_not_run"


def test_strong_broad_spectrum_favors_a_broad_model():
    spectrum, continuum = _synthetic_halpha(broad_flux=150.0)
    result = qsospec.fit_halpha_model_grid(
        spectrum, continuum, compute_covariance=False
    )
    assert result.minimum_bic_model == "B1"
    assert result.delta_bic_broad > 10.0
    record = result.to_record("7")
    assert 0.0 < record["broad_halpha_fraction_point"] < 1.0
    assert record["broad_fwhm_observed_kms"] == pytest.approx(3600.0, rel=1e-4)
    assert np.isfinite(record["broad_fwhm_intrinsic_kms"])


def test_extended_broad_grid_is_opt_in_for_targeted_diagnostics():
    spectrum, continuum = _synthetic_halpha(broad_flux=150.0)
    config = qsospec.HalphaModelSelectionConfig(
        intrinsic_broad_component_counts=(1, 2, 3)
    )
    result = qsospec.fit_halpha_model_grid(
        spectrum,
        continuum,
        selection_config=config,
        compute_covariance=False,
    )
    assert set(result.candidates) == {"N0", "B1", "B2", "B3"}


def test_diagnostic_sweep_is_strict_and_explicitly_uncalibrated():
    frame = pd.DataFrame(
        {
            "delta_bic_broad": [10.0, 10.1, -5.0, np.nan],
            "narrow_width_secure_below_boundary": [True, True, False, True],
        }
    )
    sweep = qsospec.diagnostic_bic_sweep(frame, anchors=(10.0,))
    row = sweep.iloc[0]
    assert row["n_finite"] == 3
    assert row["n_broad_favored_above_threshold"] == 1
    assert row["n_width_secure_and_broad_not_favored"] == 1
    assert row["selection_status"] == "diagnostic_not_calibrated"
