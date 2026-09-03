"""Optional local pPXF/template integration checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qsospec.resolution import SpectralResolution
from qsospec.workflows.host.io import SpectrumData
from qsospec.workflows.host.ppxf_host import (
    predict_host_sed,
    prepare_spectrum_for_host_decomp,
    run_ppxf_host_fit,
)
from qsospec.workflows.host.templates import load_ppxf_npz_templates


pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_data,
    pytest.mark.slow,
]


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


def test_emiles_coarser_than_data_is_diagnostic_not_pixel_veto():
    if importlib.util.find_spec("ppxf") is None:
        pytest.skip("pPXF is not installed")
    template_path = Path.home() / "tools/ppxf_data/spectra_emiles_9.0.npz"
    if not template_path.exists():
        pytest.skip("local pPXF E-MILES template file not found")

    wave = np.linspace(3600.0, 7000.0, 500)
    spectrum = SpectrumData(
        wave_obs=wave,
        flux=2.0 * (wave / 5100.0) ** -1.0,
        ivar=np.full_like(wave, 400.0),
        redshift=0.0,
        object_id="synthetic-emiles-resolution",
        resolution=SpectralResolution(
            mode="sigma_lambda",
            values=np.full_like(wave, 0.35),
            wavelength=wave,
            source="synthetic_high_resolution_data",
            is_object_specific=True,
        ),
    )
    templates = load_ppxf_npz_templates(write_report=False)
    prep = prepare_spectrum_for_host_decomp(
        spectrum, fit_range=(3600.0, 7000.0)
    )
    prep.metadata["spectral_resolution"] = spectrum.resolution
    native_wave = prep.wave_rest.copy()
    native_flux = prep.flux.copy()
    native_error = prep.error.copy()
    fit = run_ppxf_host_fit(
        prep,
        templates,
        residual_clip_iterations=0,
        minimum_clean_pixels=20,
        minimum_clean_fraction=0.1,
        minimum_continuum_snr=0.0,
        quiet=True,
    )

    assert fit.status == "success"
    assert fit.quality_metrics["template_coarser_than_data_fraction"] > 0
    assert fit.quality_metrics[
        "template_coarser_than_data_fraction_goodpixels"
    ] > 0
    assert "resolution_approximate_or_missing" not in (
        fit.host_fit_reliability_reasons
    )
    assert fit.stellar_kinematics_resolution_status == (
        "template_resolution_mismatch_not_corrected"
    )
    np.testing.assert_array_equal(prep.wave_rest, native_wave)
    np.testing.assert_array_equal(prep.flux, native_flux)
    np.testing.assert_array_equal(prep.error, native_error)
