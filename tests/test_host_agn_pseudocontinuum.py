"""Tests for the Aydar-inspired masked pseudo-continuum host mode."""

from types import SimpleNamespace

import numpy as np
import pytest

from qsospec import Spectrum
from qsospec.global_result import EmissionComplexResult
from qsospec.workflows.host.agn_templates import (
    build_host_agn_template_bundle,
)
from qsospec.workflows.host.broad_line_prefit import (
    nearest_width_grid_value,
    run_host_broad_line_prefit,
)
from qsospec.workflows.host.config import (
    AYDAR_2026_BROADENING_GRID_KMS,
    AYDAR_2026_POWERLAW_SLOPES_FLAMBDA,
    HostAgnPseudoContinuumConfig,
    HostDecompConfig,
)
from qsospec.workflows.host.coverage import classify_host_coverage
from qsospec.workflows.host.io import SpectrumData
from qsospec.workflows.host.ppxf_host import (
    prepare_spectrum_for_host_decomp,
    run_ppxf_host_fit,
)
from qsospec.workflows.host.templates import PPXFTemplateLibrary


def _fake_complex(prefix, *, fwhm=3420.0, flux=10.0, success=True):
    return EmissionComplexResult(
        success=success,
        status=1 if success else -1,
        message="synthetic",
        selected_model="synthetic",
        param_values={},
        param_errors={},
        covariance=None,
        metrics={
            f"{prefix}_broad_flux_input": flux,
            f"{prefix}_broad_fwhm_kms": fwhm,
            f"{prefix}_broad_velocity_kms": 0.0,
        },
        metric_errors={
            f"{prefix}_broad_flux_input": 1.0,
            f"{prefix}_broad_fwhm_kms": 100.0,
        },
        chi2=1.0,
        dof=10,
        reduced_chi2=0.1,
        bic=2.0,
        wave_rest=np.linspace(6400.0, 6800.0, 10),
        flux_continuum_subtracted=np.zeros(10),
        err=np.ones(10),
        model=np.zeros(10),
        component_models={},
        fit_mask=np.ones(10, dtype=bool),
        warnings=[],
        metadata={"coverage_status": "covered"},
    )


def test_host_strategy_configuration_and_published_grids():
    assert HostDecompConfig().strategy == "masked_simple"
    assert AYDAR_2026_BROADENING_GRID_KMS == (
        1000.0,
        1200.0,
        1400.0,
        1600.0,
        1800.0,
        2000.0,
        2400.0,
        2800.0,
        3400.0,
        4000.0,
        4800.0,
        5800.0,
        7000.0,
        8400.0,
        10000.0,
        11800.0,
    )
    assert AYDAR_2026_POWERLAW_SLOPES_FLAMBDA[0] == -3.0
    assert AYDAR_2026_POWERLAW_SLOPES_FLAMBDA[-1] == 0.0
    assert len(AYDAR_2026_POWERLAW_SLOPES_FLAMBDA) == 31
    with pytest.raises(ValueError, match="strategy"):
        HostDecompConfig(strategy="unknown")
    with pytest.raises(ValueError, match="not implemented"):
        HostDecompConfig(use_regularization=True)
    with pytest.raises(ValueError, match="deprecated"):
        HostDecompConfig(n_iterations=2)
    with pytest.raises(ValueError, match="run_qsospec"):
        HostDecompConfig(run_qsospec=True)
    with pytest.raises(ValueError, match="EuclidHostScaleConfig"):
        HostDecompConfig(euclid_scaling_mode="fixed")


def test_nearest_width_grid_tie_chooses_lower_value():
    assert nearest_width_grid_value(1100.0, (1000.0, 1200.0)) == 1000.0
    assert nearest_width_grid_value(3350.0, AYDAR_2026_BROADENING_GRID_KMS) == 3400.0


def test_broad_prefit_prefers_halpha_then_falls_back_to_hbeta(monkeypatch):
    spectrum = Spectrum.from_arrays(
        np.linspace(3500.0, 6900.0, 500),
        np.ones(500),
        err=np.ones(500) * 0.1,
        z=0.0,
        wave_frame="rest",
        flux_unit="relative",
        galactic_extinction_corrected=True,
    )
    workflow = SimpleNamespace(
        line_complexes={
            "halpha_nii_sii": _fake_complex("Ha", fwhm=3350.0),
            "hbeta_oiii": _fake_complex("Hb", fwhm=4700.0),
        }
    )
    monkeypatch.setattr(
        "qsospec.workflows.host.broad_line_prefit.fit_global_lines",
        lambda *args, **kwargs: workflow,
    )
    selected = run_host_broad_line_prefit(spectrum, width_grid_kms=AYDAR_2026_BROADENING_GRID_KMS)
    assert selected.selected_line == "halpha"
    assert selected.selected_width_grid_kms == 3400.0

    workflow.line_complexes["halpha_nii_sii"] = _fake_complex("Ha", flux=1.0)
    workflow.line_complexes["halpha_nii_sii"].metric_errors["Ha_broad_flux_input"] = 1.0
    selected = run_host_broad_line_prefit(spectrum, width_grid_kms=AYDAR_2026_BROADENING_GRID_KMS)
    assert selected.selected_line == "hbeta"
    assert selected.selected_width_grid_kms == 4800.0


def test_agn_bundle_reuses_physics_and_closes_balmer_branches():
    wave = np.linspace(3300.0, 7100.0, 1200)
    bundle = build_host_agn_template_bundle(
        wave,
        selected_fwhm_kms=4000.0,
        config=HostAgnPseudoContinuumConfig(),
    )
    by_name = {item.name: item for item in bundle.components}
    assert bundle.metadata["host_pseudocontinuum_exact_replication"] is False
    assert bundle.metadata["powerlaw_slope_convention"] == "F_lambda"
    powerlaw = by_name["powerlaw_00_slope_-3.0"].values
    blue = np.argmin(np.abs(wave - 4000.0))
    red = np.argmin(np.abs(wave - 6000.0))
    assert powerlaw[blue] / powerlaw[red] == pytest.approx((wave[blue] / wave[red]) ** -3.0)
    balmer_group = bundle.matrix[:, bundle.group_column_indices["balmer_pseudocontinuum"]]
    np.testing.assert_allclose(
        balmer_group,
        by_name["balmer_continuum"].values + by_name["balmer_high_order"].values,
        rtol=1e-12,
        atol=1e-12,
    )
    optical = by_name["feii_optical"]
    assert np.all(optical.values[wave < optical.wavelength_coverage[0]] == 0)


@pytest.mark.parametrize(
    ("wave_min", "wave_max", "expected"),
    [
        (3550.0, 7050.0, "full_optical"),
        (3550.0, 5550.0, "optical_core"),
        (3550.0, 4550.0, "blue_optical"),
        (3900.0, 4550.0, "insufficient"),
    ],
)
def test_host_coverage_classes(wave_min, wave_max, expected):
    wave = np.linspace(wave_min, wave_max, 1000)
    result = classify_host_coverage(wave, np.ones_like(wave, dtype=bool))
    assert result.coverage_class == expected


def test_new_ppxf_strategy_separates_agn_kinematics_and_closes(monkeypatch):
    wave = np.linspace(3600.0, 7000.0, 900)
    spectrum = SpectrumData(
        wave_obs=wave,
        flux=1.5 * (wave / 5100.0) ** -1.2,
        ivar=np.full_like(wave, 400.0),
        redshift=0.0,
        object_id="synthetic",
    )
    prep = prepare_spectrum_for_host_decomp(spectrum, fit_range=(3600.0, 7000.0))
    template_wave = np.linspace(3500.0, 7100.0, 1000)
    templates = PPXFTemplateLibrary(
        flux=np.ones((template_wave.size, 1)),
        wave=template_wave,
        log_wave=np.log(template_wave),
        family="test",
        source_path="fake.npz",
        wavelength_coverage=(3500.0, 7100.0),
    )
    calls = []

    def fake_ppxf(matrix, galaxy, noise, velscale, **kwargs):
        calls.append(kwargs)
        weights = np.zeros(matrix.shape[1])
        weights[0] = 0.3
        weights[1] = 0.7
        bestfit = matrix @ weights
        return SimpleNamespace(
            bestfit=bestfit,
            weights=weights,
            matrix=matrix,
            sol=[np.array([25.0, 180.0]), np.array([0.0, 0.01])],
            chi2=1.0,
        )

    monkeypatch.setattr("qsospec.workflows.host.ppxf_host._require_ppxf", lambda: fake_ppxf)
    fit = run_ppxf_host_fit(
        prep,
        templates,
        strategy="agn_pseudocontinuum_masked",
        selected_pseudocontinuum_fwhm_kms=4000.0,
        residual_clip_iterations=0,
        minimum_clean_pixels=20,
    )
    assert calls
    assert calls[0]["moments"] == [2, -2]
    assert calls[0]["linear_method"] == "lsq_box"
    assert np.all(calls[0]["component"][:1] == 0)
    assert np.all(calls[0]["component"][1:] == 1)
    assert fit.closure_metrics["closure_status"] == "numerical"
    np.testing.assert_allclose(
        fit.component_models_log["stellar"] + fit.component_models_log["agn_total"],
        fit.component_models_log["ppxf_bestfit"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(fit.agn_weights >= 0)
    assert np.isfinite(fit.ppxf_agn_fraction_flux_global)
