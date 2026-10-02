from types import SimpleNamespace

import numpy as np
import pytest

from qsospec.resolution import (
    SpectralResolution,
    additional_template_sigma,
    match_template_resolution_toward_data,
)
from qsospec.workflows.host.config import HostDecompConfig
from qsospec.workflows.host.ppxf_host import (
    _resample_stellar_templates,
    predict_host_sed,
)
from qsospec.workflows.host.preconvolved_templates import (
    build_preconvolved_xsl_product,
    validate_preconvolved_xsl_product,
)
from qsospec.workflows.host.templates import (
    PPXFTemplateLibrary,
    load_ppxf_npz_templates,
    resolve_host_template_profile,
)


def _write_template(path, *, wave, templates, fwhm, ages=None, metals=None):
    if ages is None:
        ages = np.arange(templates.shape[1], dtype=float) + 1
    if metals is None:
        metals = np.array([0.0])
    shaped = np.asarray(templates).reshape(len(wave), len(ages), len(metals))
    np.savez(
        path,
        templates=shaped,
        lam=wave,
        fwhm=fwhm,
        ages=ages,
        metals=metals,
        masses=np.ones((len(ages), len(metals))),
    )


def _prepared(wave, sigma=1.0, object_id="object-1"):
    resolution = SpectralResolution(
        mode="sigma_lambda",
        values=np.full_like(wave, sigma),
        wavelength=wave,
        source="synthetic_object_lsf",
        is_object_specific=True,
    )
    return SimpleNamespace(
        wave_log=wave.copy(),
        redshift=0.0,
        metadata={
            "spectral_resolution": resolution,
            "host_fit_range": [float(wave.min()), float(wave.max())],
            "object_id": object_id,
        },
        warnings=[],
        normalization=2.0,
    )


def test_template_profile_defaults_and_legacy_resolution(tmp_path):
    wave = np.linspace(3500.0, 8000.0, 20)
    flux = np.column_stack([np.ones_like(wave), wave / wave.mean()])
    _write_template(
        tmp_path / "spectra_emiles_9.0.npz",
        wave=wave,
        templates=flux,
        fwhm=np.full_like(wave, 2.5),
    )
    _write_template(
        tmp_path / "spectra_xsl_9.0.npz",
        wave=wave,
        templates=flux,
        fwhm=np.full_like(wave, 0.5),
    )

    default = resolve_host_template_profile(template_root=str(tmp_path))
    assert default.profile_id == "emiles_native"
    assert default.family == "emiles"
    xsl = resolve_host_template_profile(
        template_root=str(tmp_path), template_file="spectra_xsl_9.0.npz"
    )
    assert xsl.profile_id == "xsl_native"
    assert xsl.family == "xsl"
    assert HostDecompConfig(
        template_file="spectra_xsl_9.0.npz",
        template_family="xsl",
        template_profile="xsl_native",
    ).resolution_matching_mode == "object_specific_runtime"

    custom_path = tmp_path / "my_ssps.npz"
    _write_template(
        custom_path,
        wave=wave,
        templates=flux,
        fwhm=np.full_like(wave, 1.0),
    )
    custom = resolve_host_template_profile(
        template_root=str(tmp_path), template_file=custom_path.name
    )
    assert custom.profile_id == "custom_native"

    with pytest.raises(ValueError, match="requires family"):
        resolve_host_template_profile(
            template_root=str(tmp_path),
            template_file="spectra_xsl_9.0.npz",
            template_profile="xsl_native",
            template_family="emiles",
        )
    with pytest.raises(ValueError, match="preserve_native_data"):
        HostDecompConfig(preserve_native_data=False)
    with pytest.raises(ValueError, match="requires resolution_matching_mode"):
        HostDecompConfig(
            template_profile="emiles_native",
            resolution_matching_mode="object_specific_runtime",
        )


def test_one_sided_resolution_match_retains_coarser_pixels():
    wave = np.full(4, 5000.0)
    data = np.array([2.0, 1.0, 1.0, np.nan])
    template = np.array([1.0, 1.0, 2.0, 1.0])
    match = match_template_resolution_toward_data(data, template, wave)
    assert match.additional_sigma_lambda[0] == pytest.approx(np.sqrt(3.0))
    assert match.additional_sigma_lambda[1] == 0.0
    assert match.additional_sigma_lambda[2] == 0.0
    assert match.comparable[2]
    assert match.template_coarser_than_data[2]
    assert match.missing_or_invalid[3]
    assert match.metadata["template_resolution_status"] == (
        "mixed_match_template_coarser_allowed"
    )

    legacy_sigma, legacy_invalid = additional_template_sigma(data, template)
    assert legacy_invalid[2]
    assert np.isnan(legacy_sigma[2])


def test_host_resampling_does_not_reject_coarser_template():
    wave = np.linspace(4000.0, 5000.0, 100)
    flux = np.column_stack([np.ones_like(wave), wave / wave.mean()])
    library = PPXFTemplateLibrary(
        flux=flux,
        wave=wave,
        log_wave=np.log(wave),
        family="emiles",
        source_path="synthetic.npz",
        wavelength_coverage=(wave.min(), wave.max()),
        metadata={"source_sha256": "source"},
        source_resolution_metadata={"fwhm": np.full_like(wave, 2.354820045)},
        profile_id="emiles_native",
    )
    prep = _prepared(wave, sigma=0.3)
    _, _, in_range = _resample_stellar_templates(prep, library)
    assert np.all(in_range)
    assert prep.metadata["template_coarser_than_data_fraction"] == 1.0
    assert prep.metadata["additional_template_sigma_nonzero_fraction"] == 0.0
    assert "template_coarser_than_data" in prep.warnings


def test_preconvolved_xsl_exact_validation_and_native_host_sed(tmp_path):
    wave = np.geomspace(3600.0, 7000.0, 120)
    templates = np.column_stack(
        [
            1.0 + 0.1 * np.sin(wave / 120.0),
            (wave / 5100.0) ** -1.0,
        ]
    )
    source_path = tmp_path / "spectra_xsl_9.0.npz"
    _write_template(
        source_path,
        wave=wave,
        templates=templates,
        fwhm=np.full_like(wave, 0.5),
    )
    source = load_ppxf_npz_templates(
        template_root=str(tmp_path),
        template_file=source_path.name,
        template_family="xsl",
        template_profile="xsl_native",
        write_report=False,
    )
    prep = _prepared(wave, sigma=1.0)
    output = tmp_path / "derived_xsl.npz"
    product = build_preconvolved_xsl_product(
        source,
        prep,
        output_path=output,
        object_key="object-1",
        fit_range=(float(wave.min()), float(wave.max())),
    )
    assert output.exists()
    derived = load_ppxf_npz_templates(
        template_root=str(tmp_path),
        template_file=output.name,
        template_family="xsl",
        template_profile="xsl_preconvolved",
        template_product_kind="preconvolved",
        source_template_file=source_path.name,
        write_report=False,
    )
    validation = validate_preconvolved_xsl_product(
        derived,
        prep,
        fit_range=(float(wave.min()), float(wave.max())),
        object_key="object-1",
    )
    assert validation["preconvolution_validation_status"] == "preconvolved_exact"
    assert derived.fit_source_sha256 != derived.source_library_sha256
    assert np.shares_memory(derived.source_flux, source.source_flux)

    native_prep = _prepared(wave, sigma=1.0)
    cached_prep = _prepared(wave, sigma=1.0)
    native_matrix, native_scales, native_range = _resample_stellar_templates(
        native_prep, source
    )
    cached_matrix, cached_scales, cached_range = _resample_stellar_templates(
        cached_prep, derived
    )
    np.testing.assert_array_equal(native_range, cached_range)
    np.testing.assert_allclose(native_scales, cached_scales, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(native_matrix, cached_matrix, rtol=1e-12, atol=1e-12)

    weights = np.array([0.7, 0.3])
    scales = np.array([1.1, 0.9])
    common = {
        "stellar_weights": weights,
        "stellar_template_scales": scales,
        "preprocessed": prep,
        "warnings": [],
        "strategy_used": "agn_pseudocontinuum_masked",
        "host_reconstruction_state": {},
    }
    native_fit = SimpleNamespace(templates=source, **common)
    cached_fit = SimpleNamespace(templates=derived, **common)
    native_sed = predict_host_sed(native_fit)
    cached_sed = predict_host_sed(cached_fit)
    np.testing.assert_allclose(native_sed.wave_rest, cached_sed.wave_rest)
    np.testing.assert_allclose(native_sed.host_flux, cached_sed.host_flux)
    assert cached_sed.provenance["host_sed_uses_native_source_library"]
    assert product.cache_key == validation["preconvolution_cache_key"]

    wrong = _prepared(wave, sigma=1.2)
    with pytest.raises(ValueError, match="target LSF"):
        validate_preconvolved_xsl_product(
            derived,
            wrong,
            fit_range=(float(wave.min()), float(wave.max())),
            object_key="object-1",
        )


def test_preconvolution_preserves_partial_resolution_validity_mask(tmp_path):
    wave = np.geomspace(3600.0, 7000.0, 80)
    templates = np.column_stack([np.ones_like(wave), wave / wave.mean()])
    source_path = tmp_path / "spectra_xsl_9.0.npz"
    _write_template(
        source_path,
        wave=wave,
        templates=templates,
        fwhm=np.full_like(wave, 0.5),
    )
    source = load_ppxf_npz_templates(
        template_root=str(tmp_path),
        template_file=source_path.name,
        template_family="xsl",
        template_profile="xsl_native",
        write_report=False,
    )
    values = np.ones_like(wave)
    values[12] = np.nan
    prep = _prepared(wave)
    prep.metadata["spectral_resolution"] = SpectralResolution(
        mode="sigma_lambda",
        values=values,
        wavelength=wave,
        source="synthetic_partially_valid_lsf",
        is_object_specific=True,
    )
    output = tmp_path / "partial_lsf_xsl.npz"
    build_preconvolved_xsl_product(
        source,
        prep,
        output_path=output,
        object_key="object-1",
        fit_range=(float(wave.min()), float(wave.max())),
    )
    derived = load_ppxf_npz_templates(
        template_root=str(tmp_path),
        template_file=output.name,
        template_family="xsl",
        template_profile="xsl_preconvolved",
        template_product_kind="preconvolved",
        source_template_file=source_path.name,
        write_report=False,
    )
    _, _, fitted = _resample_stellar_templates(prep, derived)
    assert not fitted[12]
    assert np.sum(fitted) == wave.size - 1
