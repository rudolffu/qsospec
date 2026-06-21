"""Smoke tests for canonical examples in the public documentation."""

from pathlib import Path

import numpy as np

import qsospec


def _continuum_only_config():
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            enabled=False
        ),
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
        wave_frame="observed",
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
    result = qsospec.fit_object_to_store(
        _spectrum(),
        str(tmp_path / "run"),
        object_id="docs-object",
        global_config=_continuum_only_config(),
        complexes=[],
        write_qa=False,
    )
    loaded = qsospec.load_model(str(tmp_path / "run"), "docs-object")

    assert Path(result.output_files["manifest"]).exists()
    np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux)
    assert result.metadata["galactic_extinction"]["status"] == (
        "caller_preprocessed"
    )


def test_documented_recipe_defaults_match_runtime():
    lya = qsospec.recipes.get("lya_nv")
    civ = qsospec.recipes.get("civ")
    ciii = qsospec.recipes.get("ciii")

    assert lya.fit_window == (1150.0, 1290.0)
    assert [component.multiplicity for component in lya.components] == [2, 1]
    assert civ.fit_window == (1450.0, 1700.0)
    assert civ.components[0].multiplicity == 3
    assert ciii.fit_window == (1700.0, 1970.0)
    assert ciii.components[0].multiplicity == 2
