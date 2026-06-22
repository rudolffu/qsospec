"""Smoke tests for canonical examples in the public documentation."""

from pathlib import Path

import numpy as np
import pandas as pd

import qsospec


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


def test_documented_array_quickstart():
    result = qsospec.fit_global_lines(
        _spectrum(),
        global_config=_continuum_only_config(),
        complexes=[],
    )

    assert result.continuum_success
    assert result.line_complexes == {}


def test_documented_single_object_run_uses_preprocessed_spectrum(tmp_path):
    base = _spectrum()
    spectrum = qsospec.Spectrum.from_arrays(
        base.wave_obs,
        base.flux,
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
        complexes=[],
        write_qa=False,
    )
    loaded = qsospec.load_model(str(tmp_path / "run"), "docs-object")

    assert Path(result.output_files["manifest"]).exists()
    np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux)
    assert result.metadata["galactic_extinction"]["status"] == "declared_corrected"


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
