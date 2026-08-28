from __future__ import annotations

import sys

import numpy as np
import pytest

import qsospec
from qsospec.signed_lines import C_KMS, FWHM_TO_SIGMA


def _profile(wave: np.ndarray, center: float, fwhm_kms: float) -> np.ndarray:
    sigma = center * fwhm_kms / (C_KMS * FWHM_TO_SIGMA)
    return np.exp(-0.5 * ((wave - center) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)


def _spectrum(
    flux: np.ndarray,
    *,
    wave: np.ndarray | None = None,
    err: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> qsospec.Spectrum:
    if wave is None:
        wave = np.linspace(4870.0, 5040.0, len(flux))
    if err is None:
        err = np.full(len(flux), 0.2)
    return qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=err,
        z=0.0,
        wave_frame="rest",
        mask=mask,
        flux_unit="relative",
    )


@pytest.mark.parametrize("injected_flux", [12.0, -12.0])
def test_signed_amplitude_recovers_positive_and_negative_flux(injected_flux: float) -> None:
    wave = np.linspace(4930.0, 4990.0, 241)
    profile = _profile(wave, 4960.30, 900.0)
    result = qsospec.measure_signed_line_amplitude(
        _spectrum(injected_flux * profile, wave=wave, err=np.full(wave.size, 0.05)),
        np.zeros(wave.size),
        4960.30,
        900.0,
        fit_window=(4935.0, 4985.0),
    )
    assert result.success
    assert result.flux == pytest.approx(injected_flux, abs=1e-8)
    assert np.sign(result.snr) == np.sign(injected_flux)


def test_noise_amplitudes_are_not_positive_clipped() -> None:
    wave = np.linspace(4930.0, 4990.0, 121)
    rng = np.random.default_rng(1234)
    fluxes = []
    for _ in range(120):
        noise = rng.normal(0.0, 0.2, wave.size)
        result = qsospec.measure_signed_line_amplitude(
            _spectrum(noise, wave=wave),
            np.zeros(wave.size),
            4960.30,
            900.0,
            fit_window=(4935.0, 4985.0),
        )
        fluxes.append(result.flux)
    values = np.asarray(fluxes)
    assert np.any(values < 0)
    assert np.any(values > 0)
    assert abs(float(np.mean(values))) < 0.35 * float(np.std(values))


def test_uncertainty_matches_weighted_design_matrix() -> None:
    wave = np.linspace(4935.0, 4985.0, 151)
    err = np.linspace(0.1, 0.3, wave.size)
    profile = _profile(wave, 4960.30, 900.0)
    design = np.column_stack([profile, np.ones(wave.size), (wave - wave.mean()) / np.ptp(wave)])
    expected = np.sqrt(np.linalg.inv((design / err[:, None]).T @ (design / err[:, None]))[0, 0])
    result = qsospec.measure_signed_line_amplitude(
        _spectrum(4.0 * profile, wave=wave, err=err),
        np.zeros(wave.size),
        4960.30,
        900.0,
        fit_window=(4935.0, 4985.0),
    )
    assert result.flux_error == pytest.approx(expected, rel=1e-10)


def test_linear_baseline_is_fitted_without_line_bias() -> None:
    wave = np.linspace(4930.0, 4990.0, 181)
    profile = _profile(wave, 4960.30, 900.0)
    baseline = 2.5 + 0.012 * (wave - 4960.0)
    result = qsospec.measure_signed_line_amplitude(
        _spectrum(baseline + 8.0 * profile, wave=wave, err=np.full(wave.size, 0.1)),
        np.zeros(wave.size),
        4960.30,
        900.0,
        fit_window=(4935.0, 4985.0),
    )
    assert result.flux == pytest.approx(8.0, abs=1e-8)
    assert len(result.baseline_coefficients) == 2


def test_masked_invalid_and_excluded_pixels_are_not_used() -> None:
    wave = np.linspace(4930.0, 4990.0, 121)
    profile = _profile(wave, 4960.30, 900.0)
    flux = 5.0 * profile
    err = np.full(wave.size, 0.1)
    mask = np.ones(wave.size, dtype=bool)
    mask[40] = False
    flux[41] = np.nan
    err[42] = -1.0
    excluded = np.zeros(wave.size, dtype=bool)
    excluded[43] = True
    result = qsospec.measure_signed_line_amplitude(
        _spectrum(flux, wave=wave, err=err, mask=mask),
        np.zeros(wave.size),
        4960.30,
        900.0,
        fit_window=(4935.0, 4985.0),
        excluded_mask=excluded,
    )
    for index in (40, 41, 42, 43):
        assert not result.fit_mask[index]
    assert result.flux == pytest.approx(5.0, abs=1e-8)


def test_absent_and_partial_coverage_have_structured_statuses() -> None:
    wave = np.linspace(4950.0, 4970.0, 81)
    spectrum = _spectrum(np.zeros(wave.size), wave=wave)
    absent = qsospec.measure_signed_line_amplitude(
        spectrum, np.zeros(wave.size), 5008.24, 900.0, fit_window=(4990.0, 5020.0)
    )
    assert not absent.success
    assert absent.status == "not_covered"

    partial = qsospec.measure_signed_line_amplitude(
        spectrum, np.zeros(wave.size), 4960.30, 900.0, fit_window=(4940.0, 4965.0)
    )
    assert partial.success
    assert partial.status == "partial_coverage"
    assert "partial_coverage" in partial.warnings


def test_independent_oiii_doublet_recovers_ratio() -> None:
    wave = np.linspace(4925.0, 5040.0, 461)
    main_flux = 29.8
    companion_flux = 10.0
    flux = main_flux * _profile(wave, 5008.24, 800.0)
    flux += companion_flux * _profile(wave, 4960.30, 800.0)
    components = (
        qsospec.SignedLineComponent("main", line_id="oiii_5008"),
        qsospec.SignedLineComponent("companion", line_id="oiii_4960"),
    )
    result = qsospec.fit_local_line_pattern(
        _spectrum(flux, wave=wave, err=np.full(wave.size, 0.05)),
        np.zeros(wave.size),
        components,
        fwhm_kms=800.0,
        fit_windows=((4930.0, 5035.0),),
    )
    assert result.success
    assert result.component_fluxes["main"] / result.component_fluxes["companion"] == pytest.approx(2.98, rel=1e-8)


def test_qsospec_import_does_not_import_mlspecz() -> None:
    assert "mlspecz" not in sys.modules
    assert callable(qsospec.measure_signed_line_amplitude)
