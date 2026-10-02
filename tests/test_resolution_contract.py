import numpy as np
import pandas as pd

from qsospec.io.readers import scan_parquet_spectra
from qsospec.resolution import (
    SpectralResolution,
    additional_template_sigma,
    smooth_constant_resolving_power,
)


def test_resolution_unit_conversions():
    wave = np.array([4000.0, 5000.0])
    model = SpectralResolution(mode="constant_r", values=np.array([2000.0]), is_approximate=True)
    np.testing.assert_allclose(model.sigma_lambda(wave), wave / 2000.0 / 2.354820045)
    assert model.status == "approximate"


def test_optional_per_pixel_resolution_ingestion(tmp_path):
    path = tmp_path / "spectra.parquet"
    pd.DataFrame({
        "object_key": ["one"], "targetid": [1], "redshift": [0.5],
        "wavelength": [np.array([4000.0, 4001.0, 4002.0])],
        "flux": [np.ones(3)], "ivar": [np.ones(3)], "mask": [np.zeros(3, dtype=np.int16)],
        "resolving_power": [np.array([2000.0, 2100.0, 2200.0])],
    }).to_parquet(path, index=False)
    _, spectrum = next(scan_parquet_spectra(str(path)))
    assert spectrum.resolution.mode == "resolving_power"
    assert spectrum.resolution.status == "valid"
    assert spectrum.resolution.is_object_specific


def test_canonical_lsf_sigma_angstrom_alias(tmp_path):
    path = tmp_path / "spectra.parquet"
    sigma = np.array([1.1, 1.2, 1.3])
    pd.DataFrame({
        "object_key": ["one"], "targetid": [1], "redshift": [0.5],
        "wavelength": [np.array([4000.0, 4001.0, 4002.0])],
        "flux": [np.ones(3)], "ivar": [np.ones(3)],
        "mask": [np.zeros(3, dtype=np.int16)],
        "lsf_sigma_angstrom": [sigma],
    }).to_parquet(path, index=False)
    _, spectrum = next(scan_parquet_spectra(str(path)))
    assert spectrum.resolution.mode == "sigma_lambda"
    np.testing.assert_allclose(spectrum.resolution.values, sigma)


def test_canonical_euclid_object_id_precedes_desi_targetid(tmp_path):
    path = tmp_path / "spectra.parquet"
    pd.DataFrame({
        "object_key": ["euclid:minus17"],
        "object_id": [-17],
        "targetid": [9001],
        "redshift": [0.5],
        "wavelength": [np.array([4000.0, 4001.0, 4002.0])],
        "flux": [np.ones(3)],
        "ivar": [np.ones(3)],
        "mask": [np.zeros(3, dtype=np.int16)],
    }).to_parquet(path, index=False)
    _, spectrum = next(scan_parquet_spectra(str(path)))
    assert spectrum.object_id == "-17"
    assert spectrum.targetid == "9001"


def test_missing_resolution_is_explicit(tmp_path):
    path = tmp_path / "spectra.parquet"
    pd.DataFrame({
        "object_key": ["one"], "redshift": [0.5],
        "wavelength": [np.array([4000.0, 4001.0, 4002.0])],
        "flux": [np.ones(3)], "ivar": [np.ones(3)], "mask": [np.zeros(3, dtype=np.int16)],
    }).to_parquet(path, index=False)
    _, spectrum = next(scan_parquet_spectra(str(path)))
    assert spectrum.resolution.status == "missing"


def test_quadrature_broadening_flags_lower_resolution_template():
    sigma, invalid = additional_template_sigma(np.array([2.0, 1.0]), np.array([1.0, 2.0]))
    assert np.isclose(sigma[0], np.sqrt(3.0))
    assert np.isnan(sigma[1])
    assert invalid.tolist() == [False, True]


def test_constant_r_smoothing_backwards_compatible_shape():
    wave = np.geomspace(4000.0, 7000.0, 500)
    flux = np.zeros(500)
    flux[250] = 1.0
    smoothed = smooth_constant_resolving_power(wave, flux, 1000.0)
    assert smoothed.shape == flux.shape
    assert np.nanmax(smoothed) < 1.0
    assert np.isfinite(smoothed).all()


def test_euclid_scale_fit_archives_scale_error_arrays():
    import qsospec

    wave = np.linspace(7600.0, 13490.0, 600)
    host = 0.5 + 0.1 * (wave / 9800.0)
    agn = 2.0 * (wave / 9800.0) ** -1.0
    error = np.full_like(wave, 0.05)
    fit = qsospec.fit_euclid_host_aperture_scale(wave, agn + 0.7 * host, error, host)
    assert fit.success
    assert fit.host_scale_uncertainty_flux.shape == wave.shape
    assert fit.host_subtracted_error_with_scale.shape == wave.shape
    assert np.all(fit.host_subtracted_error_with_scale >= error)
