"""Synthetic tests for the measurement-first broad+narrow catalogue model."""

import numpy as np

import qsospec
from qsospec.broad_narrow_measurements import _profile_widths
from qsospec.global_result import GlobalContinuumResult

C_KMS = 299792.458


def _gaussian(wave, area, center, velocity, fwhm):
    shifted = center * (1.0 + velocity / C_KMS)
    sigma = center * fwhm / C_KMS / 2.354820045
    return area * np.exp(-0.5 * ((wave - shifted) / sigma) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma
    )


def _inputs(lo, hi, features, *, residual_constant=0.0, residual_slope=0.0):
    wave = np.linspace(lo, hi, int(2 * (hi - lo)) + 1)
    continuum = np.ones_like(wave)
    pivot = 0.5 * (lo + hi)
    flux = continuum + residual_constant + residual_slope * (wave - pivot)
    for area, center, velocity, fwhm in features:
        flux += _gaussian(wave, area, center, velocity, fwhm)
    err = np.full_like(wave, 0.01)
    spectrum = qsospec.Spectrum.from_arrays(
        wave, flux, err=err, z=0.0, wave_frame="rest", flux_unit="relative"
    )
    global_continuum = GlobalContinuumResult(
        True, 1, "synthetic", {}, {}, None, 0.0, 1, 0.0,
        wave, continuum, {}, np.ones_like(wave, dtype=bool),
        np.ones_like(wave, dtype=bool),
    )
    return spectrum, global_continuum


def test_halpha_fraction_uses_halpha_only_and_covariance():
    spectrum, continuum = _inputs(
        6400.0,
        6800.0,
        (
            (8.0, 6564.61, 25.0, 800.0),
            (12.0, 6564.61, 90.0, 4000.0),
            (80.0, 6585.28, 25.0, 800.0),
            (80.0 / 2.96, 6549.85, 25.0, 800.0),
            (30.0, 6718.29, 25.0, 800.0),
            (20.0, 6732.67, 25.0, 800.0),
        ),
        residual_constant=0.05,
        residual_slope=2.0e-4,
    )
    record, result = qsospec.measure_broad_narrow_complex(
        spectrum, continuum, "halpha"
    )

    assert result is not None and result.success
    assert record["fit_status"] == "complete"
    assert abs(record["narrow_flux"] - 8.0) < 0.1
    assert abs(record["broad_flux"] - 12.0) < 0.1
    assert abs(record["broad_fraction"] - 0.6) < 0.01
    assert np.isfinite(record["broad_fraction_error"])
    assert abs(record["local_continuum_constant"] - 0.05) < 0.01
    assert abs(record["local_continuum_slope"] - 2.0e-4) < 5.0e-5


def test_hbeta_kinematics_are_tied_and_fraction_excludes_oiii():
    spectrum, continuum = _inputs(
        4640.0,
        5100.0,
        (
            (5.0, 4862.68, -30.0, 850.0),
            (5.0, 4862.68, 120.0, 3500.0),
            (40.0, 5008.24, -30.0, 850.0),
            (40.0 / 2.98, 4960.30, -30.0, 850.0),
        ),
    )
    record, result = qsospec.measure_broad_narrow_complex(
        spectrum, continuum, "hbeta"
    )
    assert result is not None and result.success
    assert abs(record["broad_fraction"] - 0.5) < 0.02
    assert "hbeta_narrow.velocity_kms" in result.param_values
    assert "OIII5008_core.velocity_kms" not in result.param_values


def test_mgii_uniform_one_narrow_one_broad_model():
    spectrum, continuum = _inputs(
        2700.0,
        2900.0,
        (
            (3.0, 2798.75, 0.0, 800.0),
            (9.0, 2798.75, 150.0, 5000.0),
        ),
    )
    record, result = qsospec.measure_broad_narrow_complex(
        spectrum, continuum, "mgii"
    )
    assert result is not None and result.success
    assert abs(record["broad_fraction"] - 0.75) < 0.03
    assert {name for name in result.param_values if name.endswith(".flux")} == {
        "MgII_narrow.flux", "MgII_broad.flux"
    }


def test_hei_succeeds_with_zero_pgamma_and_records_joint_fraction():
    spectrum, continuum = _inputs(
        10550.0,
        11150.0,
        (
            (8.0, 10833.3, 20.0, 800.0),
            (2.0, 10833.3, -80.0, 3200.0),
        ),
    )
    record, result = qsospec.measure_broad_narrow_complex(
        spectrum, continuum, "hei_pgamma"
    )
    assert result is not None and result.success
    assert abs(record["hei_broad_fraction"] - 0.2) < 0.04
    assert np.isfinite(record["joint_broad_fraction"])
    assert record["pagamma_narrow_flux"] >= 0.0
    assert record["pagamma_broad_flux"] >= 0.0
    assert "hei_pgamma_narrow.velocity_kms" in result.param_values
    assert "Pagamma_narrow.velocity_kms" not in result.param_values
    assert "hei_pgamma_broad.fwhm_kms" in result.param_values


def test_not_covered_and_continuum_unavailable_are_explicit_nan_not_zero():
    spectrum, continuum = _inputs(3000.0, 3200.0, ())
    record, result = qsospec.measure_broad_narrow_complex(
        spectrum, continuum, "halpha"
    )
    assert result is None
    assert record["fit_status"] == "not_covered"
    assert np.isnan(record["narrow_flux"])

    continuum.success = False
    record, result = qsospec.measure_broad_narrow_complex(
        spectrum, continuum, "mgii"
    )
    assert result is None
    assert record["fit_status"] == "continuum_unavailable"


def test_fixed_r480_width_bounds_and_signed_id_round_trip():
    config = qsospec.BroadNarrowMeasurementConfig()
    instrument = C_KMS / 480.0
    assert np.isclose(config.instrumental_fwhm_kms, instrument)
    assert np.isclose(config.observed_bounds(broad=False)[1], np.hypot(1200.0, instrument))
    assert config.observed_bounds(broad=False)[1] == config.observed_bounds(broad=True)[0]
    assert qsospec.signed_to_uint64_string(-1) == str(2**64 - 1)


def test_all_finite_broad_fractions_are_bounded_and_total_width_is_primary_line_only():
    spectrum, continuum = _inputs(
        6400.0,
        6800.0,
        (
            (10.0, 6564.61, 0.0, 750.0),
            (0.5, 6564.61, 0.0, 6000.0),
            (100.0, 6585.28, 0.0, 750.0),
            (100.0 / 2.96, 6549.85, 0.0, 750.0),
        ),
    )
    record, _ = qsospec.measure_broad_narrow_complex(spectrum, continuum, "halpha")
    assert 0.0 <= record["broad_fraction"] <= 1.0
    assert record["total_profile_fwhm_observed_kms"] < 2000.0


def test_total_profile_flags_disjoint_half_maximum_intervals():
    fwhm, sigma, ambiguous = _profile_widths(
        1.0, 1.0, -5000.0, 5000.0, 700.0, 1400.0
    )
    assert fwhm > 9000.0
    assert sigma > 4000.0
    assert ambiguous
