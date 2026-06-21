"""Unit and survey metadata tests for qsospec."""

import numpy as np
import pytest

import qsospec


def _arrays():
    wave = np.linspace(4800.0, 4920.0, 80)
    line = 5.0 * np.exp(-0.5 * ((wave - 4861.33) / 20.0) ** 2)
    flux = 1.0 + line
    err = np.full_like(wave, 0.05)
    return wave, flux, err


def _config():
    return qsospec.LineComplexConfig(
        center=4861.33,
        window=(4800.0, 4920.0),
        components=[
            qsospec.GaussianComponent(
                name="Hb_broad",
                center=4861.33,
                amp=4.0,
                sigma=22.0,
                bounds={"amp": (0.0, None), "sigma": (5.0, 80.0)},
            )
        ],
        local_continuum="constant",
    )


def test_survey_presets_confirm_cgs_scale_and_normalize_aliases():
    wave, flux, err = _arrays()
    for survey in ["desi", "DESI", "desi-dr1", "desi_edr", "sdss", "SDSS"]:
        spec = qsospec.Spectrum.from_arrays(
            wave, flux, err=err, z=0.0, survey=survey
        )
        expected = "sdss" if survey.lower() == "sdss" else "desi"
        assert spec.metadata.survey == expected
        assert spec.wave_unit == "Angstrom"
        assert spec.flux_unit == "cgs"
        assert spec.flux_scale == 1e-17


def test_explicit_cgs_and_relative_units():
    wave, flux, err = _arrays()
    physical = qsospec.Spectrum.from_arrays(
        wave, flux, err=err, flux_unit="cgs", flux_scale=1e-16
    )
    relative = qsospec.Spectrum.from_arrays(
        wave, flux, err=err, flux_unit="relative"
    )

    assert physical.flux_unit == "cgs"
    assert physical.flux_scale == 1e-16
    assert relative.flux_unit == "relative"
    assert relative.flux_scale is None


def test_units_are_required_and_strictly_validated():
    wave, flux, err = _arrays()
    with pytest.raises(ValueError, match="flux_unit is required"):
        qsospec.Spectrum.from_arrays(wave, flux, err=err)
    with pytest.raises(ValueError, match="must be 'cgs' or 'relative'"):
        qsospec.Spectrum.from_arrays(
            wave, flux, err=err, flux_unit="mystery"
        )
    with pytest.raises(ValueError, match="only valid"):
        qsospec.Spectrum.from_arrays(
            wave, flux, err=err, flux_unit="relative", flux_scale=2.0
        )
    with pytest.raises(ValueError, match="finite and positive"):
        qsospec.Spectrum.from_arrays(
            wave, flux, err=err, flux_unit="cgs", flux_scale=0.0
        )


def test_relative_units_fit_but_do_not_report_cgs_flux():
    wave, flux, err = _arrays()
    spec = qsospec.Spectrum.from_arrays(
        wave, flux, err=err, z=0.0, flux_unit="relative"
    )
    result = qsospec.fit_line_complex(spec, _config())
    row = result.to_table().iloc[0]

    assert result.success
    assert "flux_scale_unknown_cgs_not_reported" in result.warning_codes()
    assert np.isfinite(row["line_flux_input"])
    assert np.isnan(row["line_flux_cgs"])


def test_cgs_line_flux_is_scaled_when_known():
    wave, flux, err = _arrays()
    spec = qsospec.Spectrum.from_arrays(
        wave, flux, err=err, z=0.0, survey="sdss"
    )
    result = qsospec.fit_line_complex(spec, _config())
    row = result.to_table().iloc[0]
    expected_input = row["amp"] * row["sigma"] * np.sqrt(2.0 * np.pi)

    assert np.isclose(row["line_flux_input"], expected_input)
    assert np.isclose(row["line_flux_cgs"], expected_input * 1e-17)


def test_unknown_survey_raises_clear_error():
    wave, flux, err = _arrays()
    with pytest.raises(ValueError, match="Unknown qsospec survey preset"):
        qsospec.Spectrum.from_arrays(
            wave, flux, err=err, z=0.0, survey="mystery"
        )
