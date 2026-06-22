from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qsospec
from qsospec.workflows.host.io import SpectrumData


def _data(**changes):
    wave = np.linspace(3500.0, 7500.0, 64)
    values = {
        "wave_obs": wave,
        "flux": np.ones_like(wave),
        "error": np.full_like(wave, 0.1),
        "ivar": np.full_like(wave, 100.0),
        "redshift": 0.2,
        "object_id": "dust-test",
        "ra": 12.0,
        "dec": -3.0,
    }
    values.update(changes)
    return SpectrumData(**values)


def test_f99_override_propagates_flux_error_and_ivar():
    config = qsospec.GalacticExtinctionConfig(ebv_override=0.1)
    corrected = qsospec.correct_spectrum_data(_data(), config)
    factor = qsospec.f99_dereddening_factor(corrected.wave_obs, 0.1, rv=3.1)
    rest_factor = 1.2

    np.testing.assert_allclose(corrected.flux, factor * rest_factor)
    np.testing.assert_allclose(
        corrected.error, 0.1 * factor * rest_factor
    )
    np.testing.assert_allclose(
        corrected.ivar, 100.0 / (factor * rest_factor) ** 2
    )
    provenance = corrected.metadata["galactic_extinction"]
    assert provenance["source"] == "override"
    assert provenance["raw_ebv"] == pytest.approx(0.1)
    assert provenance["applied_ebv"] == pytest.approx(0.1)
    assert provenance["correction_factor_max"] > 1.0
    assert corrected.metadata["flux_frame"] == "rest"
    assert corrected.metadata["rest_frame_conversion"] == {
        "status": "applied",
        "input_flux_frame": "observed",
        "output_flux_frame": "rest",
        "redshift": 0.2,
        "flux_error_factor": 1.2,
        "inverse_variance_factor": pytest.approx(1.0 / 1.2**2),
    }


def test_planck_alias_and_sfd_recalibration(monkeypatch):
    calls = []

    class Query:
        def __call__(self, coordinate):
            return 0.2

    def fake_query(map_name, data_dir):
        calls.append((map_name, data_dir))
        return Query()

    monkeypatch.setattr("qsospec.extinction._dust_query", fake_query)
    planck_ebv, planck = qsospec.query_galactic_ebv(
        1.0,
        2.0,
        qsospec.GalacticExtinctionConfig(map_name="planck16"),
    )
    sfd_ebv, sfd = qsospec.query_galactic_ebv(
        1.0,
        2.0,
        qsospec.GalacticExtinctionConfig(
            map_name="sfd",
            dustmaps_data_dir="~/dust",
        ),
    )

    assert calls[0][0] == "planck"
    assert calls[0][1] is not None
    assert calls[1][0] == "sfd"
    assert calls[1][1] == str(Path("~/dust").expanduser().resolve())
    assert planck_ebv == pytest.approx(0.2)
    assert planck["map_name"] == "planck"
    assert planck["map_path"].endswith("COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits")
    assert sfd_ebv == pytest.approx(0.172)
    assert sfd["raw_ebv"] == pytest.approx(0.2)
    assert sfd["applied_ebv"] == pytest.approx(0.172)
    assert sfd["map_path"].endswith("/sfd")


def test_negative_ebv_clipping_and_strict_coordinate_validation(monkeypatch):
    class Query:
        def __call__(self, coordinate):
            return -0.01

    monkeypatch.setattr(
        "qsospec.extinction._dust_query",
        lambda map_name, data_dir: Query(),
    )
    ebv, provenance = qsospec.query_galactic_ebv(1.0, 2.0)
    assert ebv == 0.0
    assert provenance["raw_ebv"] == pytest.approx(-0.01)
    assert provenance["warning"] == "negative_ebv_clipped_to_zero"

    with pytest.raises(ValueError, match="requires finite RA and Dec"):
        qsospec.correct_spectrum_data(_data(ra=None, dec=None))
    with pytest.raises(ValueError, match="negative E\\(B-V\\)"):
        qsospec.query_galactic_ebv(
            1.0,
            2.0,
            qsospec.GalacticExtinctionConfig(clip_negative_ebv=False),
        )


def test_correction_is_idempotent_and_rejects_changed_provenance():
    config = qsospec.GalacticExtinctionConfig(ebv_override=0.03)
    corrected = qsospec.correct_spectrum_data(_data(), config)
    repeated = qsospec.correct_spectrum_data(corrected, config)
    assert repeated is corrected

    with pytest.raises(ValueError, match="different Galactic-extinction"):
        qsospec.correct_spectrum_data(
            corrected,
            qsospec.GalacticExtinctionConfig(ebv_override=0.04),
        )


def test_disabled_and_direct_spectrum_helpers():
    disabled = qsospec.GalacticExtinctionConfig(enabled=False)
    corrected = qsospec.correct_spectrum_data(_data(ra=None, dec=None), disabled)
    np.testing.assert_allclose(corrected.flux, 1.2)
    assert corrected.metadata["galactic_extinction"]["status"] == "disabled"
    assert corrected.metadata["flux_frame"] == "rest"

    spectrum = qsospec.Spectrum.from_arrays(
        np.linspace(3500.0, 7500.0, 64),
        np.ones(64),
        err=np.full(64, 0.1),
        z=0.2,
        flux_unit="relative",
    )
    output, provenance = qsospec.correct_spectrum(
        spectrum,
        config=qsospec.GalacticExtinctionConfig(ebv_override=0.05),
    )
    assert np.all(output.flux > 1.2 * spectrum.flux)
    assert provenance["applied_ebv"] == pytest.approx(0.05)
    assert output.flux_frame == "rest"


def test_f99_wavelength_domain_is_enforced():
    with pytest.raises(ValueError, match="outside the F99 supported range"):
        qsospec.f99_dereddening_factor(np.array([900.0, 1200.0]), 0.1)


def test_run_bundle_archives_corrected_arrays_and_provenance(tmp_path):
    data = _data(
        wave_obs=np.linspace(3500.0, 4500.0, 240),
        flux=np.ones(240),
        error=np.full(240, 0.05),
        ivar=None,
        redshift=0.0,
    )
    config = qsospec.GalacticExtinctionConfig(ebv_override=0.05)
    result = qsospec.fit_object_to_store(
        data,
        str(tmp_path / "corrected-run"),
        galactic_extinction_config=config,
        global_config=qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=(qsospec.BalmerPseudoContinuumConfig(enabled=False)),
            clip_passes=0,
        ),
        complexes=[],
        write_qa=False,
    )
    loaded = qsospec.load_model(str(tmp_path / "corrected-run"), "dust-test")

    assert np.all(result.spectrum.flux > 1.0)
    np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux)
    provenance = loaded.metadata["galactic_extinction"]
    assert provenance["status"] == "applied"
    assert provenance["applied_ebv"] == pytest.approx(0.05)
    manifest = qsospec.open_run(str(tmp_path / "corrected-run")).manifest
    assert manifest["configuration"]["galactic_extinction_config"]["ebv_override"] == pytest.approx(0.05)


def test_fit_object_spectrum_is_automatically_prepared(tmp_path):
    spectrum = qsospec.Spectrum.from_arrays(
        np.linspace(3500.0, 4500.0, 240),
        np.ones(240),
        err=np.full(240, 0.05),
        z=0.0,
        ra=12.0,
        dec=-3.0,
        flux_unit="relative",
    )
    result = qsospec.fit_object_to_store(
        spectrum,
        str(tmp_path / "prepared-run"),
        object_id="prepared",
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(ebv_override=0.05),
        global_config=qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=(qsospec.BalmerPseudoContinuumConfig(enabled=False)),
            clip_passes=0,
        ),
        complexes=[],
        write_qa=False,
    )

    assert np.all(result.spectrum.flux > 1.0)
    assert result.metadata["galactic_extinction"]["status"] == "applied"
    assert result.metadata["flux_frame"] == "rest"
    assert result.metadata["rest_frame_conversion"]["redshift"] == 0.0
    assert result.spectrum.metadata.galactic_extinction_corrected
    assert result.spectrum.metadata.ra == pytest.approx(12.0)


def test_prepare_spectrum_declared_corrected_does_not_query_map(monkeypatch):
    spectrum = qsospec.Spectrum.from_arrays(
        np.linspace(3500.0, 4500.0, 64),
        np.ones(64),
        err=np.full(64, 0.1),
        z=0.5,
        galactic_extinction_corrected=True,
        flux_unit="relative",
    )

    def unexpected_query(*args, **kwargs):
        raise AssertionError("dust map should not be queried")

    monkeypatch.setattr("qsospec.extinction.query_galactic_ebv", unexpected_query)
    prepared = qsospec.prepare_spectrum(spectrum)

    np.testing.assert_allclose(prepared.wave_rest, spectrum.wave_obs / 1.5)
    np.testing.assert_allclose(prepared.flux, 1.5 * spectrum.flux)
    np.testing.assert_allclose(prepared.err, 1.5 * spectrum.err)
    assert prepared.flux_frame == "rest"
    assert prepared.metadata.rest_frame_conversion["status"] == "applied"
    assert prepared.metadata.galactic_extinction_corrected
    assert prepared.metadata.galactic_extinction["status"] == "declared_corrected"


def test_prepare_spectrum_reuses_matching_and_rejects_conflicting_correction():
    spectrum = qsospec.Spectrum.from_arrays(
        np.linspace(3500.0, 4500.0, 64),
        np.ones(64),
        err=np.full(64, 0.1),
        z=0.0,
        flux_unit="relative",
    )
    config = qsospec.GalacticExtinctionConfig(ebv_override=0.03)
    prepared = qsospec.prepare_spectrum(spectrum, galactic_extinction_config=config)
    repeated = qsospec.prepare_spectrum(prepared, galactic_extinction_config=config)
    assert repeated is prepared

    with pytest.raises(ValueError, match="different Galactic-extinction"):
        qsospec.prepare_spectrum(
            prepared,
            galactic_extinction_config=qsospec.GalacticExtinctionConfig(ebv_override=0.04),
        )


def test_prepare_spectrum_disabled_and_missing_coordinate_guidance():
    spectrum = qsospec.Spectrum.from_arrays(
        np.linspace(3500.0, 4500.0, 64),
        np.ones(64),
        err=np.full(64, 0.1),
        z=0.25,
        flux_unit="relative",
    )
    disabled = qsospec.prepare_spectrum(
        spectrum,
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
    )
    assert not disabled.metadata.galactic_extinction_corrected
    assert disabled.metadata.galactic_extinction["status"] == "disabled"
    np.testing.assert_allclose(disabled.flux, 1.25 * spectrum.flux)
    np.testing.assert_allclose(disabled.err, 1.25 * spectrum.err)
    assert disabled.flux_frame == "rest"

    with pytest.raises(ValueError, match="galactic_extinction_corrected=True"):
        qsospec.prepare_spectrum(spectrum)


def test_prepare_spectrum_leaves_declared_rest_frame_arrays_unchanged():
    wave = np.linspace(1200.0, 2200.0, 64)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        np.linspace(1.0, 2.0, wave.size),
        err=np.full(wave.size, 0.1),
        z=2.0,
        wave_frame="rest",
        galactic_extinction_corrected=True,
        flux_unit="relative",
    )

    prepared = qsospec.prepare_spectrum(spectrum)

    np.testing.assert_allclose(prepared.wave_rest, wave)
    np.testing.assert_allclose(prepared.flux, spectrum.flux)
    np.testing.assert_allclose(prepared.err, spectrum.err)
    assert prepared.flux_frame == "rest"
    assert prepared.metadata.rest_frame_conversion == {}


def test_prepare_spectrum_rejects_conflicting_frame_provenance():
    wave = np.linspace(1200.0, 2200.0, 64)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        np.ones(wave.size),
        err=np.full(wave.size, 0.1),
        z=1.0,
        wave_frame="rest",
        galactic_extinction_corrected=True,
        flux_unit="relative",
        metadata=qsospec.SpectrumMetadata(
            flux_unit="relative",
            flux_frame="rest",
            galactic_extinction_corrected=True,
            rest_frame_conversion={
                "status": "applied",
                "input_flux_frame": "observed",
                "output_flux_frame": "rest",
                "redshift": 2.0,
                "flux_error_factor": 3.0,
                "inverse_variance_factor": 1.0 / 9.0,
            },
        ),
    )

    with pytest.raises(ValueError, match="conflicting rest-frame"):
        qsospec.prepare_spectrum(spectrum)


def test_batch_records_missing_coordinates_as_failure(tmp_path, monkeypatch):
    source = tmp_path / "missing-coordinates.parquet"
    wave = np.linspace(3500.0, 4500.0, 64)
    pd.DataFrame(
        [
            {
                "TARGETID": "missing-coordinates",
                "WAVELENGTH": wave.tolist(),
                "FLUX": np.ones_like(wave).tolist(),
                "ERROR": np.full_like(wave, 0.1).tolist(),
                "Z": 0.0,
            }
        ]
    ).to_parquet(source, index=False)
    monkeypatch.setattr(
        "qsospec.workflows.batch.preflight_galactic_extinction",
        lambda config: None,
    )

    output = qsospec.fit_batch(
        str(source),
        str(tmp_path / "failed-run"),
        n_workers=1,
        global_config=qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=(qsospec.BalmerPseudoContinuumConfig(enabled=False)),
        ),
        complexes=[],
    )
    failures = qsospec.open_run(str(tmp_path / "failed-run")).read_table("failures").to_pylist()

    assert output.n_failed == 1
    assert "requires finite RA and Dec" in failures[0]["message"]


def test_preflight_propagates_missing_map_error(monkeypatch):
    def missing_map(map_name, data_dir):
        raise FileNotFoundError("missing Planck map")

    monkeypatch.setattr("qsospec.extinction._dust_query", missing_map)
    with pytest.raises(FileNotFoundError, match="missing Planck map"):
        qsospec.preflight_galactic_extinction()

    qsospec.preflight_galactic_extinction(qsospec.GalacticExtinctionConfig(ebv_override=0.0))
