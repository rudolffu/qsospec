from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import qsospec
from qsospec.fitting.complexes import (
    GenericComplexContext,
    fit_generic_complex,
    resolve_recipe_coverage,
)
from qsospec.global_result import GlobalContinuumResult
from qsospec.workflows.host_workflow import _host_decomp_decision
from qsospec.templates import load_balmer_template


def _continuum_config(balmer_pseudocontinuum):
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=balmer_pseudocontinuum,
        clip_passes=0,
    )


def _spectrum(lo=3300.0, hi=4500.0):
    wave = np.linspace(lo, hi, 1400)
    flux = 2.0 * (wave / 4000.0) ** -1.1
    if lo < 4260.0 and hi > 3500.0:
        flux += 12.0 * qsospec.evaluate_balmer_pseudocontinuum(
            load_balmer_template(provenance="sh95_k13full_ext"),
            wave,
            3200.0,
        )
    return qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        survey="desi",
    )


def test_line_registry_resolves_vacuum_and_historical_aliases():
    assert qsospec.lines.resolve("OIII5007") == "oiii_5008"
    assert qsospec.lines.resolve("Hβ") == "hbeta"
    assert qsospec.lines.get("oiii_5008").vacuum_wavelength == pytest.approx(5008.24)
    assert qsospec.lines.get("pabeta").reference is not None
    with pytest.raises(ValueError, match="unknown_line_id"):
        qsospec.lines.get("definitely-not-a-line")


def test_recipe_registry_is_immutable_and_overrides_by_copy():
    recipe = qsospec.recipes.get("civ")
    changed = recipe.with_component("CIV_broad", velocity_bounds_kms=(-4500.0, 2500.0))
    assert recipe.components[0].velocity_bounds_kms == (-5000.0, 3000.0)
    assert recipe.components[0].multiplicity == 3
    assert changed.components[0].velocity_bounds_kms == (-4500.0, 2500.0)
    with pytest.raises(FrozenInstanceError):
        recipe.label = "changed"
    with pytest.raises(ValueError, match="unknown_complex_recipe"):
        qsospec.recipes.get("unknown")


def test_continuum_only_has_no_hbeta_or_aggregate_summary_verdict():
    result = qsospec.fit_global_lines(
        _spectrum(),
        _continuum_config(qsospec.BalmerPseudoContinuumConfig(amplitude=12.0)),
        complexes=[],
    )
    assert result.continuum_success
    assert result.hbeta is None
    assert result.hbeta_initial is None
    assert result.line_complexes == {}
    assert result.metadata["balmer_pseudocontinuum_fwhm_source"] == "free_global_fit"
    assert result.metadata["hbeta_sync_requested"] is False
    assert "success" not in result.summary()
    assert "complete_success" not in result.summary()
    assert "legacy_hbeta_success" not in result.summary()


def test_hbeta_absent_auto_keeps_free_width_and_warns():
    result = qsospec.fit_global_lines(
        _spectrum(),
        _continuum_config(qsospec.BalmerPseudoContinuumConfig(amplitude=12.0)),
        complexes=None,
    )
    assert result.hbeta is None
    assert result.metadata["balmer_pseudocontinuum_fwhm_source"] == "free_global_fit"
    assert result.metadata["hbeta_sync_attempted"] is False
    assert "hbeta_sync_skipped_not_covered" in result.warning_codes()
    assert "balmer_pseudocontinuum_fwhm_free_no_hbeta_anchor" in result.warning_codes()


def test_never_and_fixed_balmer_width_policies_do_not_synchronize():
    never = qsospec.fit_global_lines(
        _spectrum(),
        _continuum_config(qsospec.BalmerPseudoContinuumConfig(amplitude=12.0, sync_with_hbeta="never")),
        complexes=[],
    )
    fixed = qsospec.fit_global_lines(
        _spectrum(),
        _continuum_config(qsospec.BalmerPseudoContinuumConfig(amplitude=12.0, fit_fwhm=False, fwhm_kms=4100.0)),
        complexes=[],
    )
    assert never.metadata["hbeta_sync_requested"] is False
    assert never.metadata["balmer_pseudocontinuum_fwhm_source"] == "free_global_fit"
    assert fixed.metadata["hbeta_sync_requested"] is False
    assert fixed.metadata["balmer_pseudocontinuum_fwhm_source"] == "fixed_config"
    assert fixed.metadata["balmer_pseudocontinuum_fwhm_kms"] == pytest.approx(4100.0)


def test_require_without_hbeta_warns_and_continues():
    result = qsospec.fit_global_lines(
        _spectrum(),
        _continuum_config(qsospec.BalmerPseudoContinuumConfig(amplitude=12.0, sync_with_hbeta="require")),
    )
    assert result.continuum_success
    assert "hbeta_sync_required_unmet" in result.warning_codes()


def test_component_adaptive_coverage_and_nir_blend_metadata():
    spectrum = _spectrum(10700.0, 11060.0)
    recipe = qsospec.recipes.get("paschen_nir")
    coverage = resolve_recipe_coverage(spectrum, recipe)
    assert coverage.status == "not_covered"
    assert coverage.coverage_fraction < 0.8
    assert not coverage.active_component_ids


@pytest.mark.parametrize(
    ("recipe_name", "upper", "expected_status"),
    [
        ("ciii", 1916.0, "covered"),
        ("ciii", 1915.9, "not_covered"),
        ("oii_nev_neiii_hgamma", 4216.0, "partially_covered"),
        ("oii_nev_neiii_hgamma", 4215.9, "not_covered"),
    ],
)
def test_exact_eighty_percent_window_coverage(recipe_name, upper, expected_status):
    lower = qsospec.recipes.get(recipe_name).fit_window[0]
    coverage = resolve_recipe_coverage(_spectrum(lower, upper), qsospec.recipes.get(recipe_name))
    assert coverage.status == expected_status


@pytest.mark.parametrize(
    ("recipe_name", "component_id", "multiplicity", "bands"),
    [
        (
            "ciii",
            "CIII_broad",
            2,
            ((900.0, 3500.0), (3500.0, 15000.0)),
        ),
        (
            "civ",
            "CIV_broad",
            3,
            ((900.0, 2500.0), (2500.0, 6000.0), (6000.0, 20000.0)),
        ),
    ],
)
def test_uv_recipe_shapes(recipe_name, component_id, multiplicity, bands):
    recipe = qsospec.recipes.get(recipe_name)
    component = recipe.components[0]
    assert recipe.auto_enabled
    assert component.id == component_id
    assert component.multiplicity == multiplicity
    assert component.fwhm_bands_kms == bands
    assert all(item.role == "broad" for item in recipe.components)


@pytest.mark.parametrize("recipe_name", ["ciii", "civ"])
def test_generic_uv_recipe_recovers_profile_metrics(recipe_name):
    recipe = qsospec.recipes.get(recipe_name)
    wave = np.linspace(*recipe.fit_window, 1200)
    continuum_model = np.full_like(wave, 2.0)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum_model,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        flux_unit="relative",
    )
    context = GenericComplexContext(recipe, (recipe.components[0].id,), 50.0)
    theta = context.initial.copy()
    for name, value in zip(context.linear_names, (60.0, 30.0, 15.0)):
        theta[context.index[name]] = value
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum_model + context.model(theta, wave),
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        flux_unit="relative",
    )
    continuum = GlobalContinuumResult(
        True,
        1,
        "known",
        {},
        {},
        None,
        0.0,
        1,
        0.0,
        wave.copy(),
        continuum_model,
        {"power_law": continuum_model},
        spectrum.valid_mask.copy(),
        spectrum.valid_mask.copy(),
    )
    result = fit_generic_complex(spectrum, continuum, recipe)
    feature = recipe.components[0].line_ids[0]
    assert result is not None and result.success
    assert result.metrics[f"{feature}_broad_flux_input"] == pytest.approx(
        sum((60.0, 30.0, 15.0)[: recipe.components[0].multiplicity]),
        rel=2.0e-3,
    )
    assert np.isfinite(result.metrics[f"{feature}_broad_fwhm_kms"])
    assert f"{feature}_broad_flux_input" in result.metric_errors


@pytest.mark.parametrize(
    ("redshift", "enabled", "reason"),
    [
        (1.199, True, None),
        (1.2, False, "redshift_at_or_above_1.2"),
        (None, False, "missing_redshift"),
    ],
)
def test_host_redshift_gate(redshift, enabled, reason):
    assert _host_decomp_decision(True, redshift) == (enabled, reason)


@pytest.mark.parametrize(("maximum", "disabled"), [(3599.9, True), (3600.0, True), (3600.1, False)])
def test_balmer_components_follow_maximum_rest_wavelength(maximum, disabled):
    wave = np.linspace(3000.0, maximum, 500)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        2.0 * (wave / 3500.0) ** -1.0,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        flux_unit="relative",
    )
    result = qsospec.fit_global_lines(
        spectrum,
        qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            continuum_windows=((3000.0, maximum),),
            mask_windows=(),
            clip_passes=0,
        ),
        complexes=[],
    )
    assert result.continuum.metadata["balmer_components_disabled_short_coverage"] is disabled
    assert ("balmer_components_disabled_short_coverage" in result.continuum.warning_codes()) is disabled


def test_generic_fixed_ratio_compilation():
    recipe = qsospec.recipes.get("halpha_nii_sii")
    context = GenericComplexContext(
        recipe,
        tuple(component.id for component in recipe.components if component.enabled),
        100.0,
    )
    theta = context.initial.copy()
    wave = np.linspace(6500.0, 6630.0, 2000)
    components = context.components(theta, wave)
    ratio = np.trapezoid(components["NII6585"], wave) / np.trapezoid(components["NII6550"], wave)
    assert ratio == pytest.approx(2.96, rel=2.0e-3)


@pytest.mark.parametrize(
    ("recipe_name", "profile"),
    [("ciii", "gaussian"), ("civ", "lorentzian")],
)
def test_generic_profile_derivatives_match_centered_differences(recipe_name, profile):
    recipe = qsospec.recipes.get(recipe_name)
    recipe = replace(
        recipe,
        components=(replace(recipe.components[0], profile=profile),),
    )
    context = GenericComplexContext(recipe, (recipe.components[0].id,), 50.0)
    nonlinear = np.asarray([context.initial[context.index[name]] for name in context.nonlinear_names])
    wave = np.linspace(1480.0, 1620.0, 500)
    design, derivatives = context.separable_design(nonlinear, wave, True)
    for index in range(nonlinear.size):
        step = max(abs(nonlinear[index]) * 1.0e-5, 1.0e-4)
        plus = nonlinear.copy()
        minus = nonlinear.copy()
        plus[index] += step
        minus[index] -= step
        plus_design, _ = context.separable_design(plus, wave, False)
        minus_design, _ = context.separable_design(minus, wave, False)
        finite_difference = (plus_design - minus_design) / (2.0 * step)
        assert derivatives[index] == pytest.approx(finite_difference, rel=3.0e-4, abs=1.0e-9)
