"""Tests for optional host subtraction orchestration before qsospec."""

import numpy as np
import pandas as pd
import pytest

import qsospec
from qsospec.global_result import GlobalContinuumResult, WorkflowResult
from qsospec.io.readers import SpectrumInput
from qsospec.workflows.batch import _fit_spectrum_data
from qsospec.workflows.host.io import SpectrumData
from qsospec.workflows import host_workflow


def _local_config():
    return qsospec.LocalFitConfig(windows=[qsospec.recipes.local_hbeta()])


def test_fit_with_optional_host_decomp_without_ppxf_path(tmp_path):
    pytest.importorskip("pyarrow")
    wave = np.linspace(6200.0, 7600.0, 300)
    rest = wave / 1.45
    flux = 1.0 + 4.0 * np.exp(-0.5 * ((rest - 4861.33) / 22.0) ** 2)
    ivar = np.ones_like(flux) * 100.0
    path = tmp_path / "spectra.parquet"
    pd.DataFrame(
        {
            "targetid": ["obj"],
            "redshift": [0.45],
            "wavelength": [wave],
            "flux": [flux],
            "ivar": [ivar],
            "mask": [np.zeros_like(flux, dtype=int)],
        }
    ).to_parquet(path)

    result = qsospec.fit_with_optional_host_decomp(
        str(path),
        _local_config(),
        row_index=0,
        run_host_decomp=False,
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
    )

    assert result.local_result.success
    assert not result.host_decomp_enabled
    assert result.host_model_on_quasar_grid is None
    assert result.fit_spectrum.flux.shape == wave.shape


def test_global_fit_kind_runs_without_host(tmp_path):
    pytest.importorskip("pyarrow")
    wave = np.linspace(3600.0, 7600.0, 1200)
    rest = wave / 1.2
    continuum = 2.0 * (rest / 3000.0) ** -1.2
    sigma = (2200.0 / 299792.458) * 4862.68 / 2.354820045
    line = 80.0 * np.exp(-0.5 * ((rest - 4862.68) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    path = tmp_path / "global.parquet"
    pd.DataFrame(
        {
            "targetid": ["obj"],
            "redshift": [0.2],
            "wavelength": [wave],
            "flux": [continuum + line],
            "ivar": [np.full_like(wave, 400.0)],
            "mask": [np.zeros_like(wave, dtype=int)],
        }
    ).to_parquet(path)

    result = qsospec.fit_with_optional_host_decomp(
        str(path),
        fit_kind="global",
        row_index=0,
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
        global_config=qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        ),
        hbeta_config=qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )

    assert result.continuum_success
    assert result.legacy_hbeta_success
    assert result.metadata["fit_kind"] == "global"
    assert result.metadata["targetid"] == "obj"
    assert result.metadata["ra"] is None
    assert result.metadata["dec"] is None


def test_batch_uses_shared_bounded_width_update_and_final_components(monkeypatch):
    wave = np.linspace(3600.0, 7000.0, 240)
    data = SpectrumData(
        wave_obs=wave,
        flux=np.full_like(wave, 10.0),
        error=np.ones_like(wave),
        mask=np.zeros_like(wave, dtype=int),
        redshift=0.0,
        object_id="shared-object",
        metadata={"input_file": "memory"},
    )
    calls = []

    def fake_host(spectrum_data, *, pseudocontinuum_width_override_kms=None, source, **kwargs):
        width = float(pseudocontinuum_width_override_kms or 1000.0)
        calls.append(width)
        total = host_workflow._spectrum_from_spectrum_data(spectrum_data, source=source)
        fit = host_workflow._spectrum_from_arrays(
            wave, np.full_like(wave, width / 1000.0), np.ones_like(wave), 0.0,
            np.ones_like(wave, dtype=bool), source, spectrum_data,
        )
        host_fit = type("HostFit", (), {})()
        host_fit.strategy_used = "agn_pseudocontinuum_masked"
        host_fit.strategy_requested = "agn_pseudocontinuum_masked"
        host_fit.strategy_fallback = False
        host_fit.strategy_fallback_reason = None
        host_fit.quality_metrics = {
            "pseudocontinuum_width_initial_kms": width,
            "pseudocontinuum_width_final_kms": width,
            "pseudocontinuum_width_iterations": 1,
            "broad_prefit_status": "success",
        }
        host_fit.preprocessed = type("Prep", (), {
            "wave_rest": wave,
            "normalization": 1.0,
            "mask_provenance": {},
        })()
        host_fit.templates = type("Templates", (), {
            "source_path": "tiny.npz",
            "wavelength_coverage": (3200.0, 23000.0),
            "metadata": {"source_sha256": "a", "template_wave_sha256": "b", "template_matrix_sha256": "c"},
        })()
        host_fit.component_models = {
            "stellar": np.full_like(wave, width),
            "powerlaw": np.ones_like(wave),
            "feii_optical": np.ones_like(wave) * 2,
            "balmer_continuum": np.ones_like(wave) * 3,
            "balmer_high_order": np.ones_like(wave) * 4,
            "agn_total": np.ones_like(wave) * 10,
            "physical_component_total": np.ones_like(wave) * 11,
            "ppxf_bestfit": np.ones_like(wave) * 11,
            "closure_residual": np.zeros_like(wave),
        }
        host_fit.host_model = host_fit.component_models["stellar"]
        host_fit.agn_model = host_fit.component_models["agn_total"]
        host_fit.total_model = host_fit.component_models["ppxf_bestfit"]
        host_fit.host_fit_reliable = True
        host_fit.host_fit_reliability_reasons = []
        host_fit.noise_rescale_factors = {}
        host_fit.status = "success"
        host_fit.reduced_chi2 = 1.0
        host_fit.coverage = type("Coverage", (), {"coverage_class": "full_optical", "feature_coverage": {}})()
        host_fit.ppxf_agn_fraction_flux_global = 0.4
        host_fit.ppxf_high_agn_fraction_warning = False
        host_fit.component_weights = {}
        host_fit.component_metadata = {}
        host_fit.closure_metrics = {"closure_status": "pass"}
        host_fit.warnings = []
        host_fit.host_reconstruction_state = {"host_reconstruction_state_version": "1", "stellar_weights": [width]}
        return total, fit, host_fit, object(), np.full_like(wave, 9.0), fit.flux, np.ones_like(wave, bool), np.zeros_like(wave, bool), []

    def fake_global(spectrum, *args, **kwargs):
        marker = float(np.median(spectrum.flux))
        continuum = GlobalContinuumResult(
            success=True, status=1, message="ok", param_values={"marker": marker},
            param_errors={}, covariance=None, chi2=1.0, dof=1, reduced_chi2=1.0,
            wave_rest=spectrum.wave_rest.copy(), model=np.zeros_like(wave),
            component_models={}, fit_mask=np.ones_like(wave, bool),
            clip_mask=np.ones_like(wave, bool),
        )
        return WorkflowResult(spectrum=spectrum, continuum_initial=continuum, continuum=continuum)

    selections = iter([
        {"status": "reliable", "selected_width_grid_kms": 1200.0},
        {"status": "reliable", "selected_width_grid_kms": 1200.0},
    ])
    monkeypatch.setattr(host_workflow, "_host_subtracted_spectrum", fake_host)
    monkeypatch.setattr(host_workflow, "fit_global_lines", fake_global)
    monkeypatch.setattr(host_workflow, "_final_broad_width_selection", lambda *args: next(selections))
    config = qsospec.HostDecompConfig(
        strategy="agn_pseudocontinuum_masked",
        agn_pseudocontinuum=qsospec.HostAgnPseudoContinuumConfig(maximum_width_iterations=2),
    )
    result, _ = _fit_spectrum_data(
        data,
        descriptor=SpectrumInput(source="memory", object_id="shared-object"),
        run_host_decomp=True,
        template_root="unused",
        template_file="unused.npz",
        host_fit_range=(3600.0, 7000.0),
        host_config=config,
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
        global_config=None,
        hbeta_config=None,
        mgii_config=None,
        halpha_config=None,
        lya_nv_config=None,
        uncertainty_config=qsospec.UncertaintyConfig(),
        complexes=(),
    )
    assert calls == [1000.0, 1200.0]
    assert result.host_fit.quality_metrics["pseudocontinuum_width_iterations"] == 2
    assert result.continuum.param_values["marker"] == pytest.approx(1.2)
    assert np.all(result.host_component_models["stellar"] == 1200.0)
    assert set(result.host_component_models) >= {
        "stellar", "powerlaw", "feii_optical", "balmer_continuum",
        "balmer_high_order", "agn_total", "physical_component_total",
        "ppxf_bestfit", "closure_residual", "host_subtracted_flux",
    }
