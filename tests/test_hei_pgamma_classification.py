"""Tests for the R=480 He I 10833 + Pa-gamma N0/B1 comparison."""

import numpy as np
import pytest

import qsospec
from qsospec.fitting.global_fit import _gaussian_area_profile
from qsospec.global_result import GlobalContinuumResult
from qsospec.narrow_line_evidence import evaluate_provisional_narrow_evidence


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


def _synthetic_hei_pgamma(
    *,
    hei_narrow_flux=30.0,
    pagamma_narrow_flux=0.0,
    hei_broad_flux=0.0,
    pagamma_broad_flux=0.0,
    narrow_intrinsic_fwhm=500.0,
    broad_intrinsic_fwhm=3500.0,
    err=0.03,
    add_noise=False,
):
    wave = np.linspace(10520.0, 11180.0, 1321)
    continuum = np.ones_like(wave)
    line = np.zeros_like(wave)
    instrument = qsospec.LineSpreadFunctionConfig().instrumental_fwhm_kms
    narrow_observed = qsospec.observed_fwhm_kms(
        narrow_intrinsic_fwhm, instrument
    )
    broad_observed = qsospec.observed_fwhm_kms(broad_intrinsic_fwhm, instrument)
    for flux, center in (
        (hei_narrow_flux, 10833.31),
        (pagamma_narrow_flux, 10941.09),
    ):
        if flux:
            line += _gaussian_area_profile(wave, flux, center, narrow_observed)
    for flux, center in (
        (hei_broad_flux, 10833.31),
        (pagamma_broad_flux, 10941.09),
    ):
        if flux:
            line += _gaussian_area_profile(wave, flux, center, broad_observed)
    flux = continuum + line
    if add_noise:
        flux = flux + np.random.default_rng(1).normal(0.0, err, len(wave))
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, err),
        wave_frame="rest",
        flux_unit="relative",
    )
    return spectrum, _continuum_result(spectrum, continuum)


def test_recipe_has_shared_kinematics_and_independent_nonnegative_fluxes():
    n0 = qsospec.hei_pgamma_classification_recipe(include_broad=False)
    b1 = qsospec.hei_pgamma_classification_recipe(include_broad=True)
    assert n0.fit_window == (10550.0, 11150.0)
    assert [component.role for component in n0.components] == ["narrow", "narrow"]
    assert {component.kinematic_group for component in n0.components} == {
        "hei_pgamma_narrow"
    }
    assert len(b1.components) == 4
    assert {component.kinematic_group for component in b1.components[2:]} == {
        "hei_pgamma_broad"
    }
    assert all(component.flux_bounds == (0.0, None) for component in b1.components)
    assert all(component.fixed_ratio_to is None for component in b1.components)


def test_r480_bounds_use_the_intrinsic_1200_boundary():
    config = qsospec.HeIPagammaModelSelectionConfig()
    assert config.lsf.instrumental_fwhm_kms == pytest.approx(624.5676208333333)
    assert config.observed_narrow_bounds_kms[1] == pytest.approx(1352.80623630785)
    assert config.observed_broad_bounds_kms[0] == pytest.approx(
        config.observed_narrow_bounds_kms[1]
    )


@pytest.mark.parametrize("pagamma_flux", (0.0, 0.2))
def test_secure_narrow_hei_does_not_require_pagamma_detection(pagamma_flux):
    spectrum, continuum = _synthetic_hei_pgamma(
        pagamma_narrow_flux=pagamma_flux,
        add_noise=True,
    )
    pair = qsospec.fit_hei_pgamma_model_pair(spectrum, continuum)
    assert pair is not None
    assert pair.n0.success and pair.b1.success
    record = pair.to_record(-7)
    assert record["hei_narrow_snr"] >= 5.0
    assert not np.isfinite(record["pagamma_narrow_snr"]) or record[
        "pagamma_narrow_snr"
    ] < 5.0
    assert record["narrow_width_secure_below_boundary"]
    assert record["minimum_bic_model"] == "N0"
    assert record["provisional_narrow_evidence"]
    assert record["broad_fwhm_observed_kms"] >= record["observed_narrow_boundary_kms"]
    assert np.isfinite(record["broad_fwhm_intrinsic_kms"])
    assert record["physical_class"] is None


def test_broad_hei_cannot_pass_merely_because_pagamma_is_weak():
    spectrum, continuum = _synthetic_hei_pgamma(
        hei_narrow_flux=8.0,
        hei_broad_flux=100.0,
        pagamma_narrow_flux=0.0,
        pagamma_broad_flux=0.0,
    )
    pair = qsospec.fit_hei_pgamma_model_pair(spectrum, continuum)
    assert pair is not None
    record = pair.to_record(8)
    assert record["delta_bic_broad"] >= 10.0
    assert record["hei_broad_snr"] >= 3.0
    assert record["broad_evidence"]
    assert not record["provisional_narrow_evidence"]


def test_model_pair_returns_none_when_local_window_is_not_covered():
    wave = np.linspace(8000.0, 9000.0, 400)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        np.ones_like(wave),
        err=np.full_like(wave, 0.1),
        wave_frame="rest",
        flux_unit="relative",
    )
    continuum = _continuum_result(spectrum, np.ones_like(wave))
    assert qsospec.fit_hei_pgamma_model_pair(spectrum, continuum) is None


def test_exact_evidence_thresholds_and_active_bound_veto():
    base = {
        "n0_success": True,
        "b1_success": True,
        "narrow_line_snr": 5.0,
        "narrow_width_secure_below_boundary": True,
        "narrow_kinematic_bound_hit": False,
        "delta_bic_broad": 0.0,
        "hei_broad_snr": 4.99,
        "pagamma_broad_snr": np.nan,
        "n0_residual_abs_p95_sigma": 2.0,
        "n0_residual_abs_max_sigma": 4.0,
        "b1_residual_abs_p95_sigma": 1.9,
    }
    result = evaluate_provisional_narrow_evidence(
        base, broad_snr_fields=("hei_broad_snr", "pagamma_broad_snr")
    )
    assert result["provisional_narrow_evidence"]
    broad_boundary = {
        **base,
        "delta_bic_broad": 10.0,
        "hei_broad_snr": 5.0,
        "b1_residual_abs_p95_sigma": 1.5,
    }
    result = evaluate_provisional_narrow_evidence(
        broad_boundary,
        broad_snr_fields=("hei_broad_snr", "pagamma_broad_snr"),
    )
    assert result["broad_evidence"]
    assert not result["provisional_narrow_evidence"]
    bound = evaluate_provisional_narrow_evidence(
        {**base, "narrow_kinematic_bound_hit": True},
        broad_snr_fields=("hei_broad_snr",),
    )
    assert not bound["provisional_narrow_evidence"]

    b1_preferred_without_secure_broad = evaluate_provisional_narrow_evidence(
        {
            **base,
            "delta_bic_broad": 20.0,
            "hei_broad_snr": 4.0,
            "b1_residual_abs_p95_sigma": 1.0,
        },
        broad_snr_fields=("hei_broad_snr",),
    )
    assert not b1_preferred_without_secure_broad["n0_bic_preferred"]
    assert b1_preferred_without_secure_broad["provisional_narrow_evidence"]

    poor_residuals = evaluate_provisional_narrow_evidence(
        {**base, "n0_residual_abs_max_sigma": 8.01},
        broad_snr_fields=("hei_broad_snr",),
    )
    assert poor_residuals["evidence_status"] == "poor_fit"
    assert not poor_residuals["provisional_narrow_evidence"]


def test_existing_paschen_recipe_remains_broad_only():
    recipe = qsospec.recipes.get("paschen_nir")
    roles = {
        component.id: component.role
        for component in recipe.components
        if component.id in {"HeI10833_broad", "Pagamma_broad"}
    }
    assert roles == {"HeI10833_broad": "broad", "Pagamma_broad": "broad"}
