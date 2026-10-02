"""extended-quasar preset, local coverage, and newly activated lines."""

import numpy as np
import pytest

import qsospec
from qsospec.fitting.complexes import (
    fit_generic_complex,
    resolve_recipe_coverage,
)
from qsospec.global_result import GlobalContinuumResult

C_KMS = 299792.458

# Canonical line ID -> recipe that must provide an enabled fitting path.
INVENTORY_FEATURES = {
    "mgii_blend": "mgii",
    "oii_3727": "oii_nev_neiii_hgamma",
    "oii_3730": "oii_nev_neiii_hgamma",
    "neiii_3870": "oii_nev_neiii_hgamma",
    "hgamma": "oii_nev_neiii_hgamma",
    "hbeta": "hbeta_oiii",
    "oiii_4960": "hbeta_oiii",
    "oiii_5008": "hbeta_oiii",
    "hei_5877": "hei5877",
    "halpha": "halpha_nii_sii",
    "oi_8449": "oi8449",
    "siii_9071": "siii_nir",
    "siii_9533": "siii_nir",
    "padelta": "padelta",
    "hei_10833": "hei10833_pgamma",
    "pagamma": "hei10833_pgamma",
    "oi_11290": "oi11290",
    "pabeta": "pabeta",
}


def _gaussian_area_profile(wave, flux, center, fwhm_kms):
    sigma = fwhm_kms * center / C_KMS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return flux * np.exp(-0.5 * ((wave - center) / sigma) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma
    )


def _spectrum(wave, flux, error, mask=None):
    return qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, error),
        wave_frame="rest",
        flux_unit="relative",
        survey="desi",
        mask=mask,
    )


def _continuum_result(spectrum, model):
    return GlobalContinuumResult(
        success=True,
        status=1,
        message="known",
        param_values={},
        param_errors={},
        covariance=None,
        chi2=0.0,
        dof=1,
        reduced_chi2=0.0,
        wave_rest=spectrum.wave_rest.copy(),
        model=model.copy(),
        component_models={"power_law": model.copy()},
        fit_mask=spectrum.valid_mask.copy(),
        clip_mask=spectrum.valid_mask.copy(),
    )


def _flat_config():
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
        power_law=qsospec.PowerLawConfig(norm=10.0, slope=0.0, mode="single"),
    )


def test_extended_quasar_preset_has_an_enabled_path_for_each_table_feature():
    preset = {recipe.id: recipe for recipe in qsospec.recipes.extended_quasar()}
    assert qsospec.recipes.EXTENDED_QUASAR_PRESET_ID == "extended_quasar_v1"
    assert "paschen_nir" not in preset
    assert {
        "padelta",
        "hei10833_pgamma",
        "oi11290",
        "pabeta",
    } <= set(preset)
    for line_id, recipe_id in INVENTORY_FEATURES.items():
        assert recipe_id in preset
        enabled_lines = {
            member
            for component in preset[recipe_id].components
            if component.enabled
            for member in component.line_ids
        }
        assert line_id in enabled_lines
    assert len({recipe for recipe in INVENTORY_FEATURES.values()}) == 11
    assert {recipe.id for recipe in qsospec.recipes.extended_quasar()} == set(preset)


@pytest.mark.parametrize(
    ("line_id", "recipe_id", "component", "fwhm_kms", "window"),
    [
        ("hei_5877", "hei5877", "HeI5877_narrow", 420.0, (5700.0, 6050.0)),
        ("oi_8449", "oi8449", "OI8449_broad", 2600.0, (8200.0, 8700.0)),
        ("siii_9071", "siii_nir", "SIII9071_narrow", 360.0, (8950.0, 9180.0)),
        ("siii_9533", "siii_nir", "SIII9533_narrow", 360.0, (9390.0, 9680.0)),
    ],
)
def test_newly_activated_nir_lines_are_really_measured(
    line_id, recipe_id, component, fwhm_kms, window
):
    center = qsospec.lines.get(line_id).vacuum_wavelength
    wave = np.linspace(window[0], window[1], 900)
    continuum = np.full_like(wave, 6.0)
    line = _gaussian_area_profile(wave, 42.0, center, fwhm_kms)
    spectrum = _spectrum(wave, continuum + line, 0.04)
    result = fit_generic_complex(
        spectrum, _continuum_result(spectrum, continuum), qsospec.recipes.get(recipe_id)
    )

    assert result.success
    assert component in result.component_models
    role = "narrow" if "narrow" in component else "broad"
    flux = result.metrics[f"{line_id}_{role}_flux_input"]
    centroid = result.metrics[f"{line_id}_{role}_centroid"]
    measured_fwhm = result.metrics[f"{line_id}_{role}_fwhm_kms"]
    assert flux == pytest.approx(42.0, rel=2.0e-3)
    assert abs(centroid - center) < 3.0
    assert measured_fwhm == pytest.approx(fwhm_kms, rel=2.0e-2)
    assert np.isfinite(result.metrics[f"{line_id}_{role}_ew_rest"])
    assert np.isfinite(result.metric_errors[f"{line_id}_{role}_flux_input"])
    assert result.metric_errors[f"{line_id}_{role}_flux_input"] > 0
    peaks = result.metadata["line_peaks"]["measurements"]
    assert f"{line_id}_{role}" in peaks
    assert peaks[f"{line_id}_{role}"]["peak_rest_angstrom"] == pytest.approx(
        center, abs=3.0
    )


def test_siii_each_line_alone_is_fit_eligible_without_the_other():
    wave = np.linspace(8950.0, 9180.0, 600)
    continuum = np.full_like(wave, 5.0)
    line = _gaussian_area_profile(wave, 30.0, 9071.1, 320.0)
    spectrum = _spectrum(wave, continuum + line, 0.03)

    coverage = resolve_recipe_coverage(spectrum, qsospec.recipes.get("siii_nir"))
    assert coverage.covered
    assert "SIII9071_narrow" in coverage.active_component_ids
    assert "SIII9533_narrow" not in coverage.active_component_ids
    assert coverage.component_status_for("SIII9533_narrow") == "not_observed"

    result = fit_generic_complex(
        spectrum, _continuum_result(spectrum, continuum), qsospec.recipes.get("siii_nir")
    )
    assert result.success
    assert result.metrics["siii_9071_narrow_flux_input"] == pytest.approx(30.0, rel=2.0e-3)
    # The off-grid line has no measured value disguised as a detection.
    assert "siii_9533_narrow_flux_input" not in result.metrics
    assert "siii_9533_narrow" not in result.component_models


def test_siii_shared_kinematics_and_independent_fluxes():
    wave = np.linspace(8950.0, 9680.0, 1600)
    continuum = np.full_like(wave, 5.0)
    line = _gaussian_area_profile(wave, 40.0, 9071.1, 330.0)
    line += _gaussian_area_profile(wave, 15.0, 9533.2, 330.0)
    spectrum = _spectrum(wave, continuum + line, 0.03)
    result = fit_generic_complex(
        spectrum, _continuum_result(spectrum, continuum), qsospec.recipes.get("siii_nir")
    )

    assert result.success
    names = list(result.param_values)
    assert names.count("siii_nir_narrow.velocity_kms") == 1
    assert names.count("siii_nir_narrow.fwhm_kms") == 1
    first = result.metrics["siii_9071_narrow_flux_input"]
    second = result.metrics["siii_9533_narrow_flux_input"]
    assert first == pytest.approx(40.0, rel=3.0e-3)
    assert second == pytest.approx(15.0, rel=5.0e-3)
    # The measured doublet ratio is not imposed.
    assert second / first == pytest.approx(15.0 / 40.0, rel=1.0e-2)


def test_siii_9533_paepsilon_blend_quality_and_covariance():
    wave = np.linspace(9390.0, 9680.0, 800)
    continuum = np.full_like(wave, 5.0)
    line = _gaussian_area_profile(wave, 30.0, 9533.2, 320.0)
    line += _gaussian_area_profile(wave, 24.0, 9548.588, 320.0)
    line += _gaussian_area_profile(wave, 18.0, 9548.588, 2400.0)
    spectrum = _spectrum(wave, continuum + line, 0.02)
    resolved = fit_generic_complex(
        spectrum, _continuum_result(spectrum, continuum), qsospec.recipes.get("siii_nir")
    )
    assert resolved.success
    assert resolved.covariance is not None
    quality = resolved.metadata["blend_quality"]["siii_9533_paepsilon"]
    assert quality["status"] == "resolved"
    assert np.isfinite(resolved.metric_errors["siii_9533_narrow_flux_input"])
    assert resolved.metric_errors["siii_9533_narrow_flux_input"] > 0

    coarse_wave = np.linspace(9390.0, 9680.0, 80)
    coarse_line = _gaussian_area_profile(coarse_wave, 30.0, 9533.2, 900.0)
    coarse_line += _gaussian_area_profile(coarse_wave, 25.0, 9548.588, 900.0)
    coarse = _spectrum(coarse_wave, np.full_like(coarse_wave, 5.0) + coarse_line, 0.02)
    unresolved = fit_generic_complex(
        coarse,
        _continuum_result(coarse, np.full_like(coarse_wave, 5.0)),
        qsospec.recipes.get("siii_nir"),
    )
    assert unresolved.covariance is not None
    quality = unresolved.metadata["blend_quality"]["siii_9533_paepsilon"]
    assert quality["status"] == "unresolved_or_unreliable"
    assert abs(quality["flux_correlation"]) >= 0.95 or not np.isfinite(
        quality["flux_correlation"]
    )
    assert not np.isfinite(
        unresolved.metric_errors["siii_9533_narrow_flux_input"]
    )
    assert "siii_9533_paepsilon_blend_unresolved" in unresolved.warning_codes()


def test_partial_nir_coverage_fits_pabeta_and_oi11290_without_padelta():
    wave = np.linspace(10800.0, 13200.0, 500)
    continuum = np.full_like(wave, 4.0)
    line = _gaussian_area_profile(wave, 60.0, 12821.6, 2600.0)
    line += _gaussian_area_profile(wave, 35.0, 11290.0, 2600.0)
    spectrum = _spectrum(wave, continuum + line, 0.03)

    assert resolve_recipe_coverage(
        spectrum, qsospec.recipes.get("pabeta")
    ).status == "covered"
    assert resolve_recipe_coverage(
        spectrum, qsospec.recipes.get("oi11290")
    ).status == "covered"
    padelta = resolve_recipe_coverage(spectrum, qsospec.recipes.get("padelta"))
    assert padelta.status == "not_covered"
    assert not padelta.active_component_ids

    result = qsospec.fit_global_lines(
        spectrum,
        _flat_config(),
        complexes=("pabeta", "oi11290", "padelta"),
    )
    assert result.line_complexes["pabeta"].success
    assert result.line_complexes["oi11290"].success
    assert "padelta" not in result.line_complexes
    assert result.metadata["complex_statuses"]["padelta"] == "not_covered"


def test_masked_line_core_fails_the_local_support_check():
    wave = np.linspace(10800.0, 13200.0, 500)
    continuum = np.full_like(wave, 4.0)
    core = (wave >= 12780.0) & (wave <= 12865.0)
    mask = ~core
    spectrum = _spectrum(wave, continuum, 0.03, mask=mask)

    coverage = resolve_recipe_coverage(spectrum, qsospec.recipes.get("pabeta"))
    assert not coverage.covered
    assert coverage.component_status_for("Pabeta_narrow") == "masked_core"


def test_compact_and_umbrella_recipes_cannot_double_count():
    wave = np.linspace(9900.0, 13060.0, 800)
    spectrum = _spectrum(wave, np.full_like(wave, 4.0), 0.05)
    with pytest.raises(ValueError, match="overlapping_complex_recipes"):
        qsospec.fit_global_lines(
            spectrum, _flat_config(), complexes=("paschen_nir", "pabeta")
        )
    legacy = resolve_recipe_coverage(spectrum, qsospec.recipes.get("paschen_nir"))
    assert legacy.status == "covered"  # umbrella keeps its envelope policy
    compact = resolve_recipe_coverage(
        spectrum, qsospec.recipes.get("hei10833_pgamma")
    )
    assert compact.covered
    assert "HeI10833_broad" in compact.active_component_ids


def test_hei_pgamma_narrow_broad_components_and_noise_rejection():
    wave = np.linspace(10700.0, 11060.0, 1000)
    continuum = np.full_like(wave, 8.0)
    line = _gaussian_area_profile(wave, 60.0, 10833.3, 2500.0)
    line += _gaussian_area_profile(wave, 20.0, 10833.3, 500.0)
    line += _gaussian_area_profile(wave, 45.0, 10941.1, 2500.0)
    line += _gaussian_area_profile(wave, 12.0, 10941.1, 500.0)
    spectrum = _spectrum(wave, continuum + line, 0.05)
    result = fit_generic_complex(
        spectrum,
        _continuum_result(spectrum, continuum),
        qsospec.recipes.get("hei10833_pgamma"),
    )
    assert result.success
    assert result.param_values["HeI10833_broad.flux"] == pytest.approx(60.0, rel=2.0e-3)
    assert result.param_values["HeI10833_narrow.flux"] == pytest.approx(20.0, rel=5.0e-3)
    assert result.param_values["Pagamma_broad.flux"] == pytest.approx(45.0, rel=2.0e-3)
    assert result.param_values["Pagamma_narrow.flux"] == pytest.approx(12.0, rel=5.0e-3)
    detection = dict(result.metadata["component_detection_status"])
    assert all(status == "detected" for status in detection.values())

    rng = np.random.default_rng(11)
    noise_spectrum = _spectrum(
        wave, continuum + rng.normal(0.0, 0.05, wave.size), 0.05
    )
    noise = fit_generic_complex(
        noise_spectrum,
        _continuum_result(noise_spectrum, continuum),
        qsospec.recipes.get("hei10833_pgamma"),
    )
    detection = dict(noise.metadata["component_detection_status"])
    assert all(status != "detected" for status in detection.values())


def test_nir_compact_measurements_survive_run_store_reload(tmp_path):
    wave = np.linspace(8900.0, 9750.0, 800)
    continuum = np.full_like(wave, 5.0)
    line = _gaussian_area_profile(wave, 38.0, 9071.1, 330.0)
    line += _gaussian_area_profile(wave, 22.0, 9533.2, 330.0)
    line += _gaussian_area_profile(wave, 14.0, 9017.384, 330.0)
    line += _gaussian_area_profile(wave, 19.0, 9548.588, 330.0)
    from qsospec.workflows.host.io import SpectrumData

    data = SpectrumData(
        wave_obs=wave,
        flux=continuum + line,
        error=np.full_like(wave, 0.03),
        redshift=0.0,
        object_id="nir-siii",
        metadata={"input_file": "memory-nir-siii"},
    )
    result = qsospec.fit_object_to_store(
        data,
        str(tmp_path / "run"),
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(ebv_override=0.0),
        global_config=_flat_config(),
        complexes=("siii_nir",),
        write_qa=False,
    )
    fit = result.line_complexes["siii_nir"]
    assert fit.success
    assert fit.metrics["siii_9071_narrow_flux_input"] == pytest.approx(38.0, rel=5.0e-3)
    assert fit.metrics["siii_9533_narrow_flux_input"] == pytest.approx(22.0, rel=8.0e-3)

    loaded = qsospec.load_model(str(tmp_path / "run"), "nir-siii")
    loaded_fit = loaded.line_complexes["siii_nir"]
    assert loaded_fit.metrics["siii_9071_narrow_flux_input"] == pytest.approx(
        fit.metrics["siii_9071_narrow_flux_input"], rel=1.0e-9
    )
    assert np.isfinite(loaded_fit.metric_errors["siii_9071_narrow_flux_input"])
    assert loaded_fit.metadata["component_coverage_status"]
    assert loaded_fit.covariance is not None
