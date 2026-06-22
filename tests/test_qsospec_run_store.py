import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pytest
from astropy.io import fits

import qsospec
from qsospec.io.run_store import RunStore, workflow_payload
from qsospec.workflows.host.io import SpectrumData


def _continuum_config():
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
    )


def _extinction_config():
    return qsospec.GalacticExtinctionConfig(ebv_override=0.0)


def _spectrum_data(object_id="object-1", scale=1.0):
    wave = np.linspace(3500.0, 4500.0, 240)
    flux = scale * 2.0 * (wave / 4000.0) ** -1.1
    return SpectrumData(
        wave_obs=wave,
        flux=flux,
        error=np.full_like(wave, 0.05),
        redshift=0.0,
        object_id=object_id,
        ra=123.4,
        dec=-4.5,
        metadata={"input_file": f"memory-{object_id}"},
    )


def _parquet_input(path, count=2):
    rows = []
    for index in range(count):
        spectrum = _spectrum_data(f"object-{index}", 1.0 + 0.1 * index)
        rows.append(
            {
                "TARGETID": spectrum.object_id,
                "WAVELENGTH": spectrum.wave_obs.tolist(),
                "FLUX": spectrum.flux.tolist(),
                "ERROR": spectrum.error.tolist(),
                "Z": spectrum.redshift,
                "RA": spectrum.ra,
                "DEC": spectrum.dec,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_single_object_bundle_round_trip_catalog_derived_and_qa(tmp_path):
    run = tmp_path / "single"
    result = qsospec.fit_object_to_store(
        _spectrum_data(),
        str(run),
        galactic_extinction_config=_extinction_config(),
        global_config=_continuum_config(),
        complexes=[],
        write_qa=False,
    )

    assert Path(result.output_files["manifest"]).exists()
    assert Path(result.output_files["run_directory"]).is_dir()
    assert "compact_models" not in result.output_files
    assert "host" in result.summary()
    store = qsospec.open_run(str(run))
    loaded = qsospec.load_model(store, "object-1")
    model_table = store.read_table("models")
    np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux)
    np.testing.assert_allclose(loaded.continuum.model, result.continuum.model)
    assert "wave_rest" in model_table.column_names
    assert "wave_obs" not in model_table.column_names
    assert loaded.spectrum.flux_frame == "rest"
    assert sorted(loaded.warning_codes()) == sorted(result.warning_codes())
    assert store.read_table("objects").num_rows == 1
    assert store.read_table("models").num_rows == 1

    catalog = qsospec.build_science_catalog(
        store,
        {
            "power_norm": {
                "section": "continuum_parameter",
                "quantity": "power_law.norm",
            }
        },
    )
    assert np.isfinite(catalog.loc[0, "power_norm"])

    before = store.read_table("models").num_rows
    derived = qsospec.compute_derived_quantities(
        store,
        {
            "test-calibration": lambda context: {
                "quantity": "test_luminosity",
                "value": 42.0,
                "statistical_error": 0.1,
                "intrinsic_scatter": 0.2,
                "total_error": np.hypot(0.1, 0.2),
                "unit": "dex",
            }
        },
    )
    assert derived.loc[0, "value"] == pytest.approx(42.0)
    assert store.read_table("models").num_rows == before

    rendered = qsospec.render_qa(
        store,
        object_ids=["object-1"],
        plot_config=qsospec.GlobalQAPlotConfig(output_format="png"),
    )
    assert Path(rendered["object-1"]["global_plot"]).exists()


def test_balmer_pseudocontinuum_archive_round_trip(tmp_path):
    wave = np.linspace(3300.0, 4300.0, 1000)
    template = qsospec.load_balmer_template(provenance="sh95_k13full_ext")
    flux = 2.0 * (wave / 4000.0) ** -1.1
    flux += 25.0 * qsospec.evaluate_balmer_pseudocontinuum(template, wave, 3200.0, -250.0)
    data = SpectrumData(
        wave_obs=wave,
        flux=flux,
        error=np.full_like(wave, 0.02),
        redshift=0.0,
        object_id="balmer-object",
        metadata={"input_file": "memory-balmer-object"},
    )
    run = tmp_path / "balmer-run"
    result = qsospec.fit_object_to_store(
        data,
        str(run),
        galactic_extinction_config=_extinction_config(),
        global_config=qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
                amplitude=20.0,
                fwhm_kms=3000.0,
            ),
            continuum_windows=((3300.0, 4300.0),),
            mask_windows=(),
            clip_passes=0,
            blue_absorption_clip_enabled=False,
        ),
        complexes=[],
        write_qa=False,
    )
    loaded = qsospec.load_model(str(run), "balmer-object")

    assert set(loaded.continuum.component_models) >= {
        "balmer_bound_free",
        "balmer_high_order_series",
    }
    np.testing.assert_allclose(
        loaded.continuum.model,
        result.continuum.model,
    )
    assert loaded.continuum.metadata["balmer_pseudocontinuum_template_provenance"] == "sh95_k13full_ext"
    measurements = qsospec.open_run(str(run)).read_table("measurements").to_pandas()
    assert "balmer_pseudocontinuum_velocity_kms" in set(measurements["quantity"])


def test_host_masks_round_trip_and_old_schema_rejection(tmp_path):
    data = _spectrum_data("host-mask-object")
    spectrum = qsospec.Spectrum.from_arrays(
        data.wave_obs,
        data.flux,
        err=data.error,
        z=data.redshift,
        wave_frame="rest",
        survey="desi",
    )
    result = qsospec.fit_global_lines(
        spectrum,
        _continuum_config(),
        complexes=[],
    )
    wave = result.spectrum.wave_rest
    result.host_decomp_enabled = True
    result.total_spectrum = result.spectrum
    result.host_model_on_quasar_grid = np.zeros_like(wave)
    result.host_fit_mask = (wave >= 3600.0) & (wave <= 4200.0)
    result.host_emission_mask = (wave >= 3710.0) & (wave <= 3745.0)
    result.metadata.update(
        {
            "object_id": "host-mask-object",
            "host_decomp_enabled": True,
            "host_mask_provenance": "exact",
        }
    )
    run_path = tmp_path / "host-mask-run"
    store = RunStore.create(
        str(run_path),
        configuration={
            "run_host_decomp": True,
            "host_fit_range": [3600.0, 4200.0],
            "host_config": None,
        },
    )
    store.write_payload(
        workflow_payload(
            result,
            run_id=store.run_id,
            object_key="host-mask-object",
            object_id="host-mask-object",
            input_record={
                "source": "memory",
                "row_index": 0,
                "reader": "memory",
                "metadata": {},
            },
        )
    )
    loaded = qsospec.load_model(store, "host-mask-object")
    np.testing.assert_array_equal(loaded.host_fit_mask, result.host_fit_mask)
    np.testing.assert_array_equal(loaded.host_emission_mask, result.host_emission_mask)
    assert loaded.metadata["host_mask_provenance"] == "exact"
    assert store.manifest["schema_version"] == "5"

    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "4"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="requires schema 5"):
        qsospec.open_run(str(run_path))


def test_serial_batch_resume_and_configuration_guard(tmp_path):
    source = tmp_path / "spectra.parquet"
    _parquet_input(source)
    run = tmp_path / "run"
    first = qsospec.fit_batch(
        str(source),
        str(run),
        n_workers=1,
        galactic_extinction_config=_extinction_config(),
        global_config=_continuum_config(),
        complexes=[],
    )
    assert first.n_completed == 2
    assert first.n_failed == 0
    assert Path(first.datasets["objects"]).exists()

    resumed = qsospec.fit_batch(
        str(source),
        str(run),
        n_workers=1,
        galactic_extinction_config=_extinction_config(),
        global_config=_continuum_config(),
        complexes=[],
    )
    assert resumed.n_submitted == 0
    assert resumed.n_skipped == 2
    with pytest.raises(ValueError, match="immutable manifest"):
        qsospec.fit_batch(
            str(source),
            str(run),
            n_workers=1,
            galactic_extinction_config=_extinction_config(),
            global_config=_continuum_config(),
            complexes=["mgii"],
        )


def test_duplicate_object_ids_get_row_safe_qa_names(tmp_path):
    source = tmp_path / "duplicates.parquet"
    rows = []
    for index in range(2):
        spectrum = _spectrum_data("duplicate", 1.0 + 0.1 * index)
        rows.append(
            {
                "TARGETID": "duplicate",
                "WAVELENGTH": spectrum.wave_obs.tolist(),
                "FLUX": spectrum.flux.tolist(),
                "ERROR": spectrum.error.tolist(),
                "Z": spectrum.redshift,
            }
        )
    pd.DataFrame(rows).to_parquet(source, index=False)
    run = tmp_path / "duplicate-run"
    qsospec.fit_batch(
        str(source),
        str(run),
        n_workers=1,
        galactic_extinction_config=_extinction_config(),
        global_config=_continuum_config(),
        complexes=[],
    )
    rendered = qsospec.render_qa(str(run))
    paths = sorted(Path(payload["main_qa"]).name for payload in rendered.values())
    assert len(rendered) == 2
    assert paths == [
        "main_qa_duplicate_row_0.png",
        "main_qa_duplicate_row_1.png",
    ]


def test_parallel_batch_and_deterministic_multi_job_partition(tmp_path):
    source = tmp_path / "spectra.parquet"
    _parquet_input(source, count=4)
    parallel_run = tmp_path / "parallel"
    output = qsospec.fit_batch(
        str(source),
        str(parallel_run),
        n_workers=2,
        galactic_extinction_config=_extinction_config(),
        task_size=1,
        global_config=_continuum_config(),
        complexes=[],
    )
    assert output.n_completed == 4
    assert qsospec.open_run(str(parallel_run)).read_table("models").num_rows == 4

    sharded_run = tmp_path / "sharded"
    counts = []
    for shard_index in (0, 1):
        shard = qsospec.fit_batch(
            str(source),
            str(sharded_run),
            n_workers=1,
            galactic_extinction_config=_extinction_config(),
            num_shards=2,
            shard_index=shard_index,
            finalize=False,
            global_config=_continuum_config(),
            complexes=[],
        )
        counts.append(shard.n_completed)
    assert sum(counts) == 4
    compact = qsospec.finalize_run(str(sharded_run))
    assert len(pd.read_parquet(compact["objects"])) == 4


def test_failure_archive_keeps_input_locator(tmp_path):
    missing = qsospec.SpectrumInput(
        source=str(tmp_path / "missing.fits"),
        object_id="missing",
        redshift=1.0,
    )
    run = tmp_path / "failed"
    output = qsospec.fit_batch(
        [missing],
        str(run),
        n_workers=1,
        galactic_extinction_config=_extinction_config(),
        global_config=_continuum_config(),
        complexes=[],
    )
    assert output.n_failed == 1
    store = qsospec.open_run(str(run))
    assert store.read_table("failures").num_rows == 1
    assert store.read_table("inputs").to_pylist()[0]["source"] == missing.source


def test_parquet_scanner_projects_case_insensitive_vector_columns(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _parquet_input(first, count=2)
    _parquet_input(second, count=1)
    records = list(
        qsospec.scan_parquet_spectra(
            [str(first), str(second)],
            row_indices={str(first): [1], str(second): [0]},
            batch_size=1,
        )
    )
    assert [record[0].row_index for record in records] == [1, 0]
    assert [record[1].object_id for record in records] == [
        "object-1",
        "object-0",
    ]


def test_fits_reader_registry_handles_sdss_lamost_and_iraf(tmp_path):
    wave = np.linspace(4000.0, 4100.0, 20)
    flux = np.linspace(1.0, 2.0, 20)

    sdss = tmp_path / "sdss.fits"
    columns = [
        fits.Column(name="loglam", array=np.log10(wave), format="D"),
        fits.Column(name="flux", array=flux, format="D"),
        fits.Column(name="ivar", array=np.ones_like(flux), format="D"),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(sdss)
    assert qsospec.detect_fits_reader(str(sdss)) == "sdss"
    sdss_data = qsospec.read_spectrum(str(sdss), redshift=0.2)
    np.testing.assert_allclose(sdss_data.wave_obs, wave)
    assert sdss_data.metadata["flux_unit"] == "cgs"
    assert sdss_data.metadata["flux_scale"] == pytest.approx(1e-17)

    lamost = tmp_path / "lamost.fits"
    columns = [
        fits.Column(name="WAVELENGTH", array=wave, format="D"),
        fits.Column(name="FLUX", array=flux, format="D"),
        fits.Column(name="ERROR", array=np.ones_like(flux), format="D"),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(lamost)
    assert qsospec.detect_fits_reader(str(lamost)) == "lamost"
    np.testing.assert_allclose(qsospec.read_spectrum(str(lamost), redshift=0.2).flux, flux)

    iraf = tmp_path / "iraf.fits"
    header = fits.Header()
    header["CRVAL1"] = wave[0]
    header["CDELT1"] = wave[1] - wave[0]
    fits.PrimaryHDU(flux, header=header).writeto(iraf)
    assert qsospec.detect_fits_reader(str(iraf)) == "iraf"
    iraf_data = qsospec.read_spectrum(str(iraf), redshift=0.2)
    np.testing.assert_allclose(iraf_data.wave_obs, wave)
    assert iraf_data.metadata["flux_unit"] == "relative"
    assert iraf_data.metadata["flux_scale"] is None

    calibrated_iraf = qsospec.read_spectrum(
        str(iraf),
        redshift=0.2,
        flux_unit="cgs",
        flux_scale=2e-16,
    )
    assert calibrated_iraf.metadata["flux_unit"] == "cgs"
    assert calibrated_iraf.metadata["flux_scale"] == pytest.approx(2e-16)
    with pytest.raises(ValueError, match="forbidden"):
        qsospec.read_spectrum(
            str(iraf),
            redshift=0.2,
            flux_unit="relative",
            flux_scale=2.0,
        )


def test_manifest_records_schema_and_shard_state(tmp_path):
    run = tmp_path / "manifest"
    qsospec.fit_object_to_store(
        _spectrum_data(),
        str(run),
        galactic_extinction_config=_extinction_config(),
        global_config=_continuum_config(),
        complexes=[],
        write_qa=False,
    )
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"]
    assert manifest["configuration_hash"]
    assert manifest["package_version"]
    assert "hbeta_config" not in manifest["configuration"]
    assert manifest["shard_state"]["models"] == 1
    assert pads.dataset(run / "data" / "models", format="parquet").count_rows() == 1
