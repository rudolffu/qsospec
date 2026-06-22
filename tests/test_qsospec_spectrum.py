"""Tests for qsospec spectrum handling."""

import numpy as np
import pytest

from qsospec import Spectrum, SpectrumMetadata


def test_spectrum_from_arrays_validates_shapes():
    wave = np.arange(5.0)
    flux = np.ones(4)
    err = np.ones(5)

    with pytest.raises(ValueError, match="same shape"):
        Spectrum.from_arrays(wave, flux, err=err, z=0.1, flux_unit="relative")


def test_spectrum_from_arrays_valid_mask_and_rest_frame():
    wave_rest = np.array([4000.0, 4100.0, 4200.0, 4300.0])
    flux = np.array([1.0, np.nan, 3.0, 4.0])
    err = np.array([0.1, 0.1, -1.0, 0.2])
    spec = Spectrum.from_arrays(wave_rest, flux, err=err, z=0.5, wave_frame="rest", flux_unit="relative")

    np.testing.assert_allclose(spec.wave_rest, wave_rest)
    np.testing.assert_allclose(spec.wave_obs, wave_rest * 1.5)
    np.testing.assert_array_equal(spec.valid_mask, [True, False, False, True])
    assert spec.flux_frame == "rest"


def test_spectrum_accepts_ivar():
    wave = np.array([4000.0, 4100.0])
    flux = np.ones(2)
    ivar = np.array([4.0, 0.0])
    spec = Spectrum.from_arrays(wave, flux, ivar=ivar, z=0.0, flux_unit="relative")

    np.testing.assert_allclose(spec.err[0], 0.5)
    assert not spec.valid_mask[1]
    assert spec.flux_frame == "observed"


def test_spectrum_from_arrays_extinction_metadata_and_coordinates():
    wave = np.linspace(3500.0, 4500.0, 10)
    spec = Spectrum.from_arrays(
        wave,
        np.ones_like(wave),
        err=np.full_like(wave, 0.1),
        z=0.1,
        ra=3.97576206,
        dec=56.04931383,
        flux_unit="relative",
    )

    assert spec.metadata.ra == pytest.approx(3.97576206)
    assert spec.metadata.dec == pytest.approx(56.04931383)
    assert not spec.metadata.galactic_extinction_corrected

    copied = Spectrum.from_arrays(
        spec.wave_obs,
        spec.flux,
        err=spec.err,
        z=spec.z,
        metadata=SpectrumMetadata(
            ra=spec.metadata.ra,
            dec=spec.metadata.dec,
            galactic_extinction_corrected=True,
            galactic_extinction={"status": "applied"},
        ),
    )
    assert copied.metadata.galactic_extinction_corrected
    assert copied.metadata.galactic_extinction["status"] == "applied"

    with pytest.raises(ValueError, match="0 <= ra < 360"):
        Spectrum.from_arrays(
            wave,
            np.ones_like(wave),
            err=np.ones_like(wave),
            ra=360.0,
            flux_unit="relative",
        )
