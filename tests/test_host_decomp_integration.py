"""Optional local pPXF/template integration checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qsospec.workflows.host.io import SpectrumData
from qsospec.workflows.host.ppxf_host import (
    predict_host_sed,
    prepare_spectrum_for_host_decomp,
    run_ppxf_host_fit,
)
from qsospec.workflows.host.templates import load_ppxf_npz_templates


def test_local_ppxf_template_fit_smoke():
    if importlib.util.find_spec("ppxf") is None:
        pytest.skip("pPXF is not installed")
    template_path = Path.home() / "tools/ppxf_data/spectra_emiles_9.0.npz"
    if not template_path.exists():
        pytest.skip("local pPXF E-MILES template file not found")

    wave = np.linspace(3600.0, 7000.0, 240)
    flux = 2.0 * (wave / 5100.0) ** -1.0 + 0.15 * np.sin(wave / 500.0)
    ivar = np.full_like(wave, 100.0)
    spec = SpectrumData(wave_obs=wave, flux=flux, ivar=ivar, redshift=0.0, object_id="synthetic")
    templates = load_ppxf_npz_templates(write_report=False)
    prep = prepare_spectrum_for_host_decomp(spec, fit_range=(3700.0, 6800.0))
    fit = run_ppxf_host_fit(prep, templates, quiet=True)
    sed = predict_host_sed(fit)

    assert fit.status == "success"
    assert fit.host_model.shape == prep.wave_rest.shape
    assert sed.flags["template_covers_1um"]


def test_local_ppxf_agn_pseudocontinuum_smoke():
    if importlib.util.find_spec("ppxf") is None:
        pytest.skip("pPXF is not installed")
    template_path = Path.home() / "tools/ppxf_data/spectra_emiles_9.0.npz"
    if not template_path.exists():
        pytest.skip("local pPXF E-MILES template file not found")

    wave = np.linspace(3600.0, 7000.0, 500)
    flux = 1.6 * (wave / 5100.0) ** -1.2
    flux += 0.08 * np.exp(-0.5 * ((wave - 4550.0) / 180.0) ** 2)
    spectrum = SpectrumData(
        wave_obs=wave,
        flux=flux,
        ivar=np.full_like(wave, 400.0),
        redshift=0.0,
        object_id="synthetic-agn-aware",
    )
    templates = load_ppxf_npz_templates(write_report=False)
    prep = prepare_spectrum_for_host_decomp(spectrum, fit_range=(3600.0, 7000.0))
    fit = run_ppxf_host_fit(
        prep,
        templates,
        strategy="agn_pseudocontinuum_masked",
        selected_pseudocontinuum_fwhm_kms=4000.0,
        residual_clip_iterations=0,
        quiet=True,
    )

    assert fit.status == "success"
    assert fit.strategy_used == "agn_pseudocontinuum_masked"
    assert fit.closure_metrics["closure_status"] == "numerical"
    assert "feii_optical" in fit.component_models
    assert "balmer_continuum" in fit.component_models
    assert "balmer_high_order" in fit.component_models
    np.testing.assert_allclose(
        fit.component_models_log["physical_component_total"],
        fit.component_models_log["ppxf_bestfit"],
        rtol=1e-8,
        atol=1e-8,
    )
