"""Smoke tests for canonical examples in the public documentation."""

from pathlib import Path
import os
import runpy

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest

import qsospec


@pytest.fixture
def sdss_example():
    return runpy.run_path("examples/fit_sdss_quasar.py")


def _continuum_only_config():
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
    )


def _spectrum():
    wave = np.linspace(3500.0, 5500.0, 800)
    flux = 2.0 * (wave / 4000.0) ** -1.2
    return qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.05),
        z=0.0,
        wave_frame="rest",
        flux_unit="relative",
    )


@pytest.fixture
def synthetic_sdss_fits(tmp_path):
    """Synthetic SDSS-shaped test data, not the real documented spectrum."""
    z = 0.5764288306236267
    wave = np.linspace(3500.0, 5500.0, 800) * (1 + z)
    flux = 2.0 * (wave / 8000.0) ** -1.2
    ivar = np.full_like(wave, 400.0)
    ivar[1] = 0.0
    ivar[2] = -1.0
    ivar[3] = np.nan
    and_mask = np.zeros(wave.size, dtype=np.int32)
    or_mask = and_mask.copy()
    and_mask[4] = 1
    or_mask[5] = 2
    primary = fits.PrimaryHDU()
    primary.header.update(PLUG_RA=135.92077, PLUG_DEC=39.286062, RA=134.05672, DEC=39.419325)
    pixels = fits.BinTableHDU.from_columns([
        fits.Column(name="loglam", format="D", array=np.log10(wave)),
        fits.Column(name="flux", format="D", array=flux),
        fits.Column(name="ivar", format="D", array=ivar),
        fits.Column(name="and_mask", format="J", array=and_mask),
        fits.Column(name="or_mask", format="J", array=or_mask),
    ])
    catalog = fits.BinTableHDU.from_columns([
        fits.Column(name="Z", format="D", array=[z]),
        fits.Column(name="CLASS", format="8A", array=["QSO"]),
        fits.Column(name="ZWARNING", format="J", array=[0]),
    ])
    path = tmp_path / "synthetic_sdss.fits"
    fits.HDUList([primary, pixels, catalog]).writeto(path)
    return path


def test_documented_sdss_quickstart_offline(synthetic_sdss_fits, monkeypatch):
    import qsospec.extinction as extinction

    queries = []

    def fake_dust_query(map_name, data_dir):
        assert map_name == "planck"

        def query(coordinate):
            queries.append((coordinate.ra.deg, coordinate.dec.deg))
            return 0.025

        return query

    monkeypatch.setattr(extinction, "_dust_query", fake_dust_query)
    data = qsospec.read_spectrum(str(synthetic_sdss_fits))
    assert data.redshift == 0.5764288306236267
    assert (data.ra, data.dec) == (135.92077, 39.286062)
    assert data.metadata["survey"] == "sdss"
    assert data.metadata["flux_unit"] == "cgs"
    assert data.metadata["flux_scale"] == 1e-17
    assert data.mask.dtype == bool
    assert not data.mask[4:6].any()
    result = qsospec.fit_global_lines_workflow(
        str(synthetic_sdss_fits),
        global_config=_continuum_only_config(),
        complexes=[],
    )
    assert result.continuum_success
    assert result.line_complexes == {}
    assert result.spectrum.z == data.redshift
    assert result.spectrum.flux_scale == 1e-17
    assert result.spectrum.metadata.survey == "sdss"
    assert not result.host_decomp_enabled
    assert queries == [(data.ra, data.dec)]
    assert result.spectrum.mask[0]
    assert not result.spectrum.mask[1:6].any()
    assert result.spectrum.mask[6:].all()
    dust = result.metadata["galactic_extinction"]
    assert dust["status"] == "applied"
    assert dust["source"] == "planck"
    assert dust["applied_ebv"] == 0.025
    factor = extinction.f99_dereddening_factor(data.wave_obs, 0.025)
    np.testing.assert_allclose(result.spectrum.wave_rest, data.wave_obs / (1 + data.redshift))
    np.testing.assert_allclose(result.spectrum.flux, data.flux * factor * (1 + data.redshift))
    np.testing.assert_allclose(result.spectrum.err[6:], data.uncertainty()[6:] * factor[6:] * (1 + data.redshift))


def test_line_peaks_example_prepares_observed_arrays(synthetic_sdss_fits, monkeypatch, sdss_example):
    import qsospec.extinction as extinction

    monkeypatch.setattr(extinction, "_dust_query", lambda *_: lambda _: 0.025)
    spectrum = sdss_example["prepare_peak_spectrum"](synthetic_sdss_fits)
    result = qsospec.fit_global_lines(spectrum, _continuum_only_config(), complexes=[])
    assert result.continuum_success
    assert spectrum.flux_frame == "rest"
    assert spectrum.metadata.galactic_extinction["status"] == "applied"
    assert spectrum.z == 0.5764288306236267
    np.testing.assert_allclose(spectrum.wave_rest, spectrum.wave_obs / (1 + spectrum.z))


def test_read_first_measurement_handles_unavailable_error(sdss_example):
    from types import SimpleNamespace

    fit = SimpleNamespace(success=True, metrics={"Hb_broad_fwhm_kms": 3000.0},
                          metric_errors={"Hb_broad_fwhm_kms": np.nan})
    result = SimpleNamespace(line_complexes={"hbeta_oiii": fit}, complex_statuses={"hbeta_oiii": "fit"})
    assert sdss_example["read_broad_hbeta"](result) == {
        "status": "available", "fwhm_kms": 3000.0, "fwhm_error_kms": None,
    }
    result.line_complexes = {}
    result.complex_statuses = {"hbeta_oiii": "not_covered"}
    assert sdss_example["read_broad_hbeta"](result) == {"status": "not_covered"}


@pytest.mark.parametrize("mask", [np.array([True, False]), np.array([0, 1], dtype=int)])
def test_workflow_accepts_boolean_valid_and_integer_bad_masks(mask):
    from qsospec.workflows.host.io import SpectrumData
    from qsospec.workflows.host_workflow import _good_mask_from_spectrum_data

    data = SpectrumData(wave_obs=np.array([4000., 4001.]), flux=np.ones(2), error=np.ones(2), mask=mask)
    np.testing.assert_array_equal(_good_mask_from_spectrum_data(data), [True, False])


@pytest.mark.parametrize("suffix", ["csv", "ecsv", "npz"])
def test_file_workflow_preserves_legacy_table_formats(tmp_path, monkeypatch, suffix):
    from astropy.table import Table
    from qsospec.workflows import host_workflow

    columns = {"wavelength": np.array([4000., 4001.]), "flux": np.array([2., 3.]),
               "error": np.array([0.1, 0.1]), "redshift": np.array([0.5, 0.5]),
               "ra": np.array([135., 135.]), "dec": np.array([39., 39.]), "mask": np.array([0, 1])}
    path = tmp_path / f"synthetic_table.{suffix}"
    if suffix == "npz":
        np.savez(path, **columns)
    else:
        Table(columns).write(path, format="csv" if suffix == "csv" else "ascii.ecsv")
    monkeypatch.setattr(host_workflow, "_run_global_fit_with_optional_host", lambda data, **kwargs: data)
    data = qsospec.fit_global_lines_workflow(str(path))
    np.testing.assert_array_equal(data.wave_obs, columns["wavelength"])
    np.testing.assert_array_equal(data.mask, columns["mask"])
    assert data.redshift == 0.5
    assert (data.ra, data.dec) == (135., 39.)


@pytest.mark.docs
@pytest.mark.external_data
@pytest.mark.slow
def test_documented_real_sdss_quickstart(tmp_path, monkeypatch, sdss_example):
    """Opt in with QSOSPEC_SDSS_EXAMPLE_FITS and an installed Planck map."""
    input_path = os.environ.get("QSOSPEC_SDSS_EXAMPLE_FITS")
    if input_path is None:
        pytest.skip("set QSOSPEC_SDSS_EXAMPLE_FITS to the downloaded quickstart FITS")
    import hashlib
    import json

    provenance = json.loads(Path("docs/_static/sdss_quasar_provenance.json").read_text())
    assert hashlib.sha256(Path(input_path).read_bytes()).hexdigest() == provenance["sha256"]
    data = qsospec.read_spectrum(input_path)
    run_directory = tmp_path / "sdss-run"
    result = sdss_example["fit_example"](input_path, run_directory=run_directory)
    assert result.spectrum.z == data.redshift == provenance["redshift"]
    assert (data.ra, data.dec) == (provenance["ra_deg"], provenance["dec_deg"])
    assert result.continuum_success
    assert not result.host_decomp_enabled
    assert result.metadata["galactic_extinction"]["source"] == "planck"
    assert result.metadata["galactic_extinction"]["status"] == "applied"
    assert result.complex_statuses == provenance["complex_statuses"]
    assert set(result.warning_codes()) == set(provenance["warning_codes"])
    measurements = sdss_example["read_broad_hbeta"](result)
    reference = provenance["measurement_verification"]["broad_hbeta"]
    np.testing.assert_allclose(measurements["fwhm_kms"], reference["fwhm_kms"], rtol=2e-3)
    if reference["fwhm_error_kms"] is None:
        assert measurements["fwhm_error_kms"] is None
    else:
        np.testing.assert_allclose(measurements["fwhm_error_kms"], reference["fwhm_error_kms"], rtol=2e-2)
    hbeta = result.line_complexes["hbeta_oiii"]
    assert hbeta.success
    assert hbeta.metadata["line_peaks"]["measurements"]["hbeta_broad"]["component_ids"] == reference["component_ids"]
    assert hbeta.metadata["line_lsf"]["status"] == reference["line_lsf"]["status"]

    def no_refit(*args, **kwargs):
        raise AssertionError("reload must not fit")

    monkeypatch.setattr(qsospec, "fit_object_to_store", no_refit)
    monkeypatch.setattr(qsospec, "fit_global_lines_workflow", no_refit)
    loaded = qsospec.load_model(str(run_directory), sdss_example["OBJECT_ID"])
    assert sdss_example["read_broad_hbeta"](loaded) == measurements
    assert loaded.complex_statuses == result.complex_statuses
    assert loaded.line_complexes["hbeta_oiii"].success == hbeta.success
    assert loaded.spectrum.z == result.spectrum.z


def test_documented_single_object_run_uses_preprocessed_spectrum(tmp_path, sdss_example):
    base = _spectrum()
    spectrum = qsospec.Spectrum.from_arrays(
        base.wave_obs,
        base.flux + 4.0 * np.exp(-0.5 * ((base.wave_rest - 4862.68) / 24.0) ** 2)
        + 2.0 * np.exp(-0.5 * ((base.wave_rest - 5008.24) / 2.0) ** 2)
        + (2.0 / 2.98) * np.exp(-0.5 * ((base.wave_rest - 4960.30) / 2.0) ** 2),
        err=base.err,
        z=base.z,
        wave_frame="rest",
        galactic_extinction_corrected=True,
        flux_unit="relative",
    )
    result = qsospec.fit_object_to_store(
        spectrum,
        str(tmp_path / "run"),
        object_id="docs-object",
        global_config=_continuum_only_config(),
        complexes=["hbeta_oiii"],
        hbeta_config=qsospec.HbetaComplexConfig(
            fit_oiii_wings=False, broad_fwhm_bands_kms=((900.0, 20000.0),),
        ),
        write_qa=False,
    )
    loaded = qsospec.load_model(str(tmp_path / "run"), "docs-object")

    assert Path(result.output_files["manifest"]).exists()
    np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux)
    assert result.metadata["galactic_extinction"]["status"] == "declared_corrected"
    measurement = sdss_example["read_broad_hbeta"](result)
    assert measurement["status"] == "available"
    assert measurement["fwhm_error_kms"] > 0
    assert sdss_example["read_broad_hbeta"](loaded) == measurement
    assert loaded.complex_statuses == result.complex_statuses
    assert loaded.line_complexes["hbeta_oiii"].success


@pytest.mark.docs
@pytest.mark.plotting
@pytest.mark.slow
def test_documented_j001554_example_data_and_preparation(tmp_path):
    data_path = Path("examples/data/spec_J001554.18+560257.5_LJT.csv")
    table = pd.read_csv(data_path)
    spectrum = qsospec.Spectrum.from_arrays(
        table["lam"],
        table["flux"],
        err=table["err"],
        z=0.1684,
        ra=3.97576206,
        dec=56.04931383,
        flux_unit="cgs",
        source=str(data_path),
    )
    result = qsospec.fit_object_to_store(
        spectrum,
        str(tmp_path / "j001554"),
        object_id="J001554.18+560257.5",
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(ebv_override=0.0),
        global_config=_continuum_only_config(),
        complexes=[],
        write_qa=True,
    )

    assert result.continuum_success
    assert result.metadata["galactic_extinction"]["status"] == "applied"
    assert result.spectrum.flux_scale == 1.0
    assert Path(result.output_files["main_qa"]).is_file()
    figure = result.plot_qa()
    assert figure.axes
    run = qsospec.open_run(str(tmp_path / "j001554"))
    archived_figure = run.plot_qa("J001554.18+560257.5")
    assert archived_figure.axes
