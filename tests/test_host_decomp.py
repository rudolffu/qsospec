#!/usr/bin/env python
"""Tests for optional pPXF host-decomposition helpers."""

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qsospec.workflows.host.euclid import predict_host_for_euclid_spectrum
from qsospec.workflows.host.io import SpectrumData, inspect_spectrum, read_sparcli_spectrum
from qsospec.workflows.host.plots import _finite_percentile_limits, _host_sed_prediction_on_desi_grid
from qsospec.workflows.host.ppxf_host import (
    HostSED,
    PPXFHostFitResult,
    prepare_desi_for_host_decomp,
    predict_host_sed,
    predict_host_sed_on_grid,
    run_ppxf_host_fit,
    write_host_decomp_outputs,
    _require_ppxf,
)
from qsospec.workflows.host.templates import PPXFTemplateLibrary, load_ppxf_npz_templates


def test_sparcli_parquet_vector_row_selection(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "spectra.parquet"
    df = pd.DataFrame(
        {
            "targetid": ["a", "b"],
            "redshift": [0.1, 0.2],
            "ra": [1.0, 2.0],
            "dec": [3.0, 4.0],
            "wavelength": [np.array([4000.0, 5000.0]), np.array([4100.0, 5100.0])],
            "flux": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "ivar": [np.array([1.0, 4.0]), np.array([9.0, 16.0])],
        }
    )
    df.to_parquet(path)

    spec = read_sparcli_spectrum(str(path), row_index=1)
    report = inspect_spectrum(str(path), row_index=1)

    assert spec.targetid == "b"
    assert spec.redshift == 0.2
    np.testing.assert_allclose(spec.wave_obs, [4100.0, 5100.0])
    assert report["n_valid_pixels"] == 2


def test_preprocessing_preserves_native_mask_gaps_and_observed_artifacts():
    wave = np.arange(3500.0, 9001.0, 1.0)
    flux = np.ones_like(wave)
    ivar = np.ones_like(wave) * 100.0
    mask = np.zeros_like(wave, dtype=int)
    mask[(wave >= 5200.0) & (wave <= 5210.0)] = 1
    spec = SpectrumData(
        wave_obs=wave,
        flux=flux,
        ivar=ivar,
        mask=mask,
        redshift=0.0,
        object_id="masked",
    )

    prep = prepare_desi_for_host_decomp(
        spec,
        fit_range=(3600.0, 7000.0),
    )

    native = prep.mask_provenance
    assert np.all(
        native["original_desi_mask_rejected"][
            (wave >= 5200.0) & (wave <= 5210.0)
        ]
    )
    assert np.all(
        native["observed_artifact_rejected"][
            (wave >= 5570.0) & (wave <= 5585.0)
        ]
    )
    assert not prep.validity_mask_log[
        np.argmin(np.abs(prep.wave_log - 5205.0))
    ]
    assert not prep.validity_mask_log[
        np.argmin(np.abs(prep.wave_log - 5577.0))
    ]
    assert prep.validity_mask_log[
        np.argmin(np.abs(prep.wave_log - 5400.0))
    ]


def _fake_template_library():
    wave = np.linspace(3000.0, 12000.0, 20)
    flux = np.column_stack([np.ones_like(wave), wave / 5100.0])
    return PPXFTemplateLibrary(
        flux=flux,
        wave=wave,
        log_wave=np.log(wave),
        family="test",
        source_path="fake.npz",
        wavelength_coverage=(float(wave.min()), float(wave.max())),
    )


def test_staged_ppxf_expands_broad_masks_clips_spikes_and_keeps_absorption(
    monkeypatch,
):
    wave = np.linspace(3600.0, 7000.0, 1701)
    flux = np.ones_like(wave)
    flux += 0.35 * np.exp(-0.5 * ((wave - 4861.0) / 55.0) ** 2)
    flux[np.argmin(np.abs(wave - 6100.0))] += 2.0
    flux[np.argmin(np.abs(wave - 5175.0))] -= 0.05
    spec = SpectrumData(
        wave_obs=wave,
        flux=flux,
        ivar=np.full_like(wave, 2500.0),
        mask=np.zeros_like(wave, dtype=int),
        redshift=0.0,
        object_id="robust",
    )
    prep = prepare_desi_for_host_decomp(spec, fit_range=(3600.0, 7000.0))
    template_wave = np.linspace(3500.0, 7100.0, 1801)
    templates = PPXFTemplateLibrary(
        flux=np.ones((template_wave.size, 1)),
        wave=template_wave,
        log_wave=np.log(template_wave),
        family="test",
        source_path="fake.npz",
        wavelength_coverage=(3500.0, 7100.0),
    )

    class FakeResult:
        def __init__(self, galaxy, templates_matrix):
            self.bestfit = np.ones_like(galaxy)
            self.weights = np.r_[1.0, np.zeros(templates_matrix.shape[1] - 1)]
            self.sol = np.array([0.0, 150.0])
            self.chi2 = 1.0

    def fake_ppxf(templates_matrix, galaxy, noise, velscale, **kwargs):
        return FakeResult(galaxy, templates_matrix)

    monkeypatch.setattr(
        "qsospec.workflows.host.ppxf_host._require_ppxf",
        lambda: fake_ppxf,
    )
    fit = run_ppxf_host_fit(
        prep,
        templates,
        agn_powerlaw_slopes=(),
        adaptive_line_residual_sigma=2.5,
    )

    broad_wing = (
        (prep.wave_log > 4910.0)
        & (prep.wave_log < 4930.0)
    )
    assert np.any(
        fit.expanded_emission_mask_log[broad_wing]
        & ~fit.initial_emission_mask_log[broad_wing]
    )
    spike = np.argmin(np.abs(prep.wave_log - 6100.0))
    assert fit.residual_clip_mask_log[spike]
    absorption = np.argmin(np.abs(prep.wave_log - 5175.0))
    assert fit.final_goodpixels_mask_log[absorption]
    assert set(fit.noise_rescale_factors) == {"b", "r", "z"}
    assert fit.quality_metrics["clean_pixel_count"] > 0


def test_host_sed_sampling_does_not_extrapolate():
    templates = _fake_template_library()
    prep = SimpleNamespace(normalization=2.0)
    fit = SimpleNamespace(
        templates=templates,
        stellar_template_scales=np.ones(2),
        stellar_weights=np.array([1.0, 0.5]),
        preprocessed=prep,
        warnings=[],
    )

    sed = predict_host_sed(fit)

    assert np.isfinite(sed.samples["fHost_5100"])
    assert np.isnan(sed.samples["fHost_1p6um"])
    assert sed.flags["template_covers_1um"]
    assert not sed.flags["template_covers_1p6um"]


def test_host_sed_prediction_on_quasar_grid_no_extrapolation():
    sed = HostSED(
        wave_rest=np.array([4000.0, 5000.0, 6000.0]),
        host_flux=np.array([1.0, 2.0, 3.0]),
        samples={},
        flags={},
        warnings=[],
    )
    grid = np.array([3500.0, 4500.0, 5500.0, 6500.0])

    host, warnings = predict_host_sed_on_grid(sed, grid)

    assert np.isnan(host[0])
    assert np.isnan(host[-1])
    np.testing.assert_allclose(host[1:3], [1.5, 2.5])
    assert "host_sed_grid_outside_template_coverage" in warnings


def test_desi_diagnostic_host_sed_prediction_is_outside_ppxf_fit():
    wave = np.linspace(3000.0, 8000.0, 51)
    host_model = np.full_like(wave, np.nan)
    fit_region = (wave >= 3600.0) & (wave <= 7000.0)
    host_model[fit_region] = 1.0
    fit = SimpleNamespace(
        preprocessed=SimpleNamespace(wave_rest=wave),
        host_model=host_model,
    )
    sed = HostSED(
        wave_rest=np.linspace(2500.0, 8500.0, 61),
        host_flux=np.ones(61) * 2.0,
        samples={},
        flags={},
        warnings=[],
    )

    predicted = _host_sed_prediction_on_desi_grid(fit, sed)

    assert np.all(np.isfinite(predicted[wave < 3600.0]))
    assert np.all(np.isfinite(predicted[wave > 7000.0]))
    assert np.all(np.isnan(predicted[fit_region]))


def test_plot_percentile_limits_ignore_extreme_spikes():
    limits = _finite_percentile_limits(
        [np.array([0.0, 1.0, 2.0, 3.0, 1000.0])], percentiles=(0.0, 75.0), pad_fraction=0.0
    )

    assert limits == (0.0, 3.0)


def test_euclid_free_scale_is_nonnegative():
    wave = np.linspace(9000.0, 18000.0, 50)
    sed = HostSED(wave_rest=wave, host_flux=np.ones_like(wave), samples={}, flags={}, warnings=[])
    euclid_wave_obs = wave * 1.5
    euclid_flux = np.ones_like(wave) * 3.0

    pred = predict_host_for_euclid_spectrum(
        sed,
        euclid_wave_obs,
        z=0.5,
        euclid_flux=euclid_flux,
        scale_mode="free_scale",
        continuum_windows=[(9500.0, 17000.0)],
    )

    assert pred.scale_factor >= 0
    np.testing.assert_allclose(pred.predicted_host_flux[np.isfinite(pred.predicted_host_flux)], 3.0)


def test_output_schema_writes_summary_files(tmp_path):
    wave = np.linspace(4000.0, 5000.0, 10)
    spec = SpectrumData(wave_obs=wave, flux=np.ones_like(wave), redshift=0.0, object_id="obj")
    prep = SimpleNamespace(
        wave_obs=wave,
        wave_rest=wave,
        flux=np.ones_like(wave),
        error=np.ones_like(wave) * 0.1,
        wave_log=wave,
        redshift=0.0,
    )
    templates = _fake_template_library()
    fit = PPXFHostFitResult(
        preprocessed=prep,
        templates=templates,
        host_model_log=np.ones_like(wave),
        agn_model_log=np.zeros_like(wave),
        total_model_log=np.ones_like(wave),
        residual_log=np.zeros_like(wave),
        host_model=np.ones_like(wave),
        agn_model=np.zeros_like(wave),
        total_model=np.ones_like(wave),
        residual=np.zeros_like(wave),
        stellar_weights=np.ones(2),
        agn_weights=np.zeros(1),
        stellar_template_scales=np.ones(2),
        agn_slopes=np.array([-1.0]),
        stellar_velocity=0.0,
        stellar_sigma=100.0,
        chi2=1.0,
        reduced_chi2=1.0,
        status="success",
    )
    sed = HostSED(
        wave_rest=templates.wave,
        host_flux=np.ones_like(templates.wave),
        samples={},
        flags={"nir_extrapolation_reliable": False},
        warnings=[],
    )
    sed.samples = {
        "fHost_4000": 1.0,
        "fHost_5100": 1.0,
        "fHost_8000": 1.0,
        "fHost_1um": 1.0,
        "fHost_1p6um": np.nan,
        "fHost_2p2um": np.nan,
    }
    sed.flags.update(
        {
            "template_covers_1um": True,
            "template_covers_1p6um": False,
            "template_covers_2p2um": False,
            "nir_extrapolation_not_available": True,
        }
    )

    files, summary = write_host_decomp_outputs(tmp_path, spec, fit, sed, np.zeros_like(wave))

    assert "host_decomp_summary_json" in files
    assert "fHost_5100" in summary
    assert summary["flux_unit"] == "cgs"
    assert summary["flux_scale"] == pytest.approx(1e-17)
    assert np.isfinite(summary["fracHost_4000"])
    assert np.isnan(summary["fracHost_5100"])
    assert np.isnan(summary["fAGN_5100"])
    assert (tmp_path / "desi_host_subtracted.csv").exists()


def test_missing_template_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pPXF template file not found"):
        load_ppxf_npz_templates(template_root=str(tmp_path), template_file="missing.npz", write_report=False)


def test_ppxf_missing_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ppxf"):
            raise ImportError("no ppxf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pPXF is required"):
        _require_ppxf()
