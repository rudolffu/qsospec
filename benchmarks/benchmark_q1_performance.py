"""Staged timing benchmark for the Q1/performance changes.

Deterministic synthetic spectra only. The script reports medians over repeats,
cold/warm iron-cache timings, objective and convolution counts, and separates
optimizer cost from post-fit diagnostics and serialization. It does not assert
speedups; compare runs before/after numerical optimizations with identical
masks, errors, seeds, bounds, and convergence tolerances.

BLAS threads are pinned for the single-process comparison through the
environment before NumPy is imported. Nothing here changes machine-wide
settings inside the library.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

import qsospec  # noqa: E402
from qsospec.fitting.complexes import fit_generic_complex, resolve_recipe_coverage  # noqa: E402
from qsospec.fitting.global_fit import (  # noqa: E402
    _ContinuumContext,
    _build_joint_hgamma_problem,
)
from qsospec.global_result import GlobalContinuumResult  # noqa: E402
from qsospec.templates.iron import (  # noqa: E402
    clear_iron_caches,
    evaluate_iron_basis,
    evaluate_iron_basis_with_derivative,
)
from qsospec.templates import load_balmer_template, load_iron_template  # noqa: E402

C_KMS = 299792.458


def _gaussian_area_profile(wave, flux, center, fwhm_kms):
    sigma = fwhm_kms * center / C_KMS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return flux * np.exp(-0.5 * ((wave - center) / sigma) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma
    )


def _median_timing(function, repeats):
    samples = []
    value = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": float(np.median(samples)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
        "repeats": repeats,
    }, value


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


def _synthetic_optical_spectrum(n=1600):
    wave = np.linspace(3600.0, 7000.0, n)
    power = 2.0 * (wave / 4000.0) ** -1.0
    lines = np.zeros_like(wave)
    for flux, center, fwhm in (
        (80.0, 4862.68, 2600.0),
        (30.0, 4862.68, 420.0),
        (120.0, 5008.24, 420.0),
        (40.0, 4960.30, 420.0),
        (150.0, 6564.61, 2600.0),
        (45.0, 6564.61, 420.0),
        (60.0, 6585.28, 420.0),
    ):
        lines += _gaussian_area_profile(wave, flux, center, fwhm)
    rng = np.random.default_rng(7)
    flux = power + lines + rng.normal(0.0, 0.01, wave.size)
    return qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.02), wave_frame="rest",
        flux_unit="relative", survey="desi",
    )


def _synthetic_rgs_spectrum(n=500):
    wave = np.linspace(10800.0, 13200.0, n)
    power = 8.0 * (wave / 12000.0) ** -0.5
    lines = _gaussian_area_profile(wave, 60.0, 12821.6, 2600.0)
    lines += _gaussian_area_profile(wave, 35.0, 11290.0, 2600.0)
    rng = np.random.default_rng(11)
    flux = power + lines + rng.normal(0.0, 0.02, wave.size)
    return qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.04), wave_frame="rest",
        flux_unit="relative", survey="desi",
    )


def _flat_config(soft=False, continuum_windows=None):
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
        power_law=qsospec.PowerLawConfig(norm=10.0, slope=0.0, mode="single"),
        iron_width_coupling="soft" if soft else "independent",
        continuum_windows=continuum_windows
        or qsospec.GlobalContinuumConfig().continuum_windows,
    )


def _count_convolutions(function):
    import qsospec.templates.iron as iron_module

    original = iron_module._linear_convolve
    counter = Counter()

    def counting(padded, kernel):
        counter["convolutions"] += 1
        return original(padded, kernel)

    iron_module._linear_convolve = counting
    try:
        value = function()
    finally:
        iron_module._linear_convolve = original
    return value, dict(counter)


def benchmark(repeats=3):
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "qsospec": getattr(qsospec, "__version__", "unknown"),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "stages": {},
    }
    stages = report["stages"]

    # 1. Iron evaluation (value only / value + derivative), cold and warm cache.
    template = load_iron_template("park22")
    wave = np.linspace(4000.0, 5500.0, 1200)

    def cold_value():
        clear_iron_caches()
        return evaluate_iron_basis(template, wave, 3000.0)

    timing, _ = _median_timing(cold_value, repeats)
    timing["cache"] = "cold"
    stages["iron_value_only"] = timing

    evaluate_iron_basis(template, wave, 3000.0)
    timing, _ = _median_timing(
        lambda: evaluate_iron_basis(template, wave, 3000.0), repeats
    )
    timing["cache"] = "warm"
    stages["iron_value_only_warm"] = timing

    def cold_derivative():
        clear_iron_caches()
        return evaluate_iron_basis_with_derivative(template, wave, 3000.0)

    timing, _ = _median_timing(cold_derivative, repeats)
    timing["cache"] = "cold"
    stages["iron_value_and_derivative"] = timing

    # 2. Ordinary continuum fit and 3. continuum with the optional width prior.
    continuum_wave = np.linspace(2000.0, 5500.0, 900)
    continuum_context = _ContinuumContext(
        qsospec.Spectrum.from_arrays(
            continuum_wave,
            2.0 * (continuum_wave / 4000.0) ** -1.0,
            err=np.full_like(continuum_wave, 0.05),
            wave_frame="rest",
            flux_unit="relative",
        ),
        qsospec.GlobalContinuumConfig(
            clip_passes=0,
            blue_absorption_clip_enabled=False,
            balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        ),
    )
    truth = continuum_context.initial.copy()
    for name, value in (
        ("uv_iron.fwhm_kms", 1800.0),
        ("optical_iron.fwhm_kms", 6000.0),
    ):
        truth[continuum_context.index[name]] = value
    continuum_spectrum = replace(
        continuum_context.spectrum,
        flux=continuum_context.model(truth, continuum_wave),
    )
    ordinary_config = qsospec.GlobalContinuumConfig(
        clip_passes=0,
        blue_absorption_clip_enabled=False,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
    )

    def ordinary_fit():
        return qsospec.fit_global_continuum(continuum_spectrum, ordinary_config)

    timing, ordinary = _median_timing(ordinary_fit, repeats)
    timing["optimizer_used"] = ordinary.metadata["optimizer_used"]
    timing["nonlinear_nfev"] = ordinary.metadata["nonlinear_nfev"]
    timing["linear_solve_count"] = ordinary.metadata["linear_solve_count"]
    stages["ordinary_continuum_fit"] = timing

    soft_config = replace(
        ordinary_config,
        iron_width_coupling="soft",
        optimizer_method="variable_projection",
    )

    def soft_fit():
        return qsospec.fit_global_continuum(continuum_spectrum, soft_config)

    timing, soft = _median_timing(soft_fit, repeats)
    timing["optimizer_used"] = soft.metadata["optimizer_used"]
    timing["nonlinear_nfev"] = soft.metadata["nonlinear_nfev"]
    timing["linear_solve_count"] = soft.metadata["linear_solve_count"]
    timing["prior_penalty"] = soft.metadata["prior_penalty"]
    stages["continuum_fit_with_width_prior"] = timing

    # 4. Joint Hgamma refinement on an intermediate optical state.
    gamma_wave = np.linspace(3300.0, 4550.0, 650)
    balmer = 20.0 * qsospec.evaluate_balmer_pseudocontinuum(
        load_balmer_template(provenance="sh95_k13full_ext"), gamma_wave, 3500.0, 0.0
    )
    gamma_flux = (
        2.0 + balmer + _gaussian_area_profile(gamma_wave, 27.0, 4341.68, 3600.0)
    )
    gamma_spectrum = qsospec.Spectrum.from_arrays(
        gamma_wave, gamma_flux, err=np.full_like(gamma_wave, 0.015),
        wave_frame="rest", flux_unit="relative",
    )
    gamma_config = qsospec.GlobalContinuumConfig(
        uv_iron=qsospec.IronTemplateConfig.vw01(fwhm_kms=3000.0),
        optical_iron=qsospec.IronTemplateConfig.park22(fwhm_kms=3000.0),
        power_law=qsospec.PowerLawConfig(norm=2.0, slope=0.0, mode="single"),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            amplitude=20.0, fwhm_kms=3500.0, sync_with_hbeta="never",
            sync_with_hgamma="soft",
        ),
        continuum_windows=((3300.0, 4260.0),),
        mask_windows=(),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
    )
    gamma_result = qsospec.fit_global_lines(
        gamma_spectrum, gamma_config, complexes=("oii_nev_neiii_hgamma",)
    )
    recipe = qsospec.recipes.get("oii_nev_neiii_hgamma")
    coverage = resolve_recipe_coverage(gamma_spectrum, recipe)
    joint_context = _ContinuumContext(
        gamma_spectrum,
        gamma_config,
        initial_parameters=gamma_result.continuum.param_values,
    )
    joint_problem = _build_joint_hgamma_problem(
        gamma_spectrum,
        gamma_config,
        gamma_result.continuum,
        gamma_result.line_complexes["oii_nev_neiii_hgamma"],
        coverage,
        joint_context,
    )

    def joint_metrics():
        theta = joint_problem.start.copy()
        return joint_problem.jacobian(theta)

    timing, _ = _median_timing(joint_metrics, repeats)
    timing["n_data_pixels"] = joint_problem.n_data
    timing["n_parameters"] = joint_problem.start.size
    stages["joint_hgamma_model_and_jacobian"] = timing

    # 5. Line optimization, 6. diagnostics, and 7. native serialization.
    line_wave = np.linspace(12600.0, 13100.0, 900)
    line_flux = 10.0 + _gaussian_area_profile(line_wave, 120.0, 12821.6, 2500.0)
    line_flux += _gaussian_area_profile(line_wave, 20.0, 12821.6, 400.0)
    line_spectrum = qsospec.Spectrum.from_arrays(
        line_wave, line_flux, err=np.full_like(line_wave, 0.05),
        wave_frame="rest", flux_unit="relative",
    )
    line_continuum = _continuum_result(line_spectrum, np.full_like(line_wave, 10.0))
    pabeta_recipe = qsospec.recipes.get("pabeta")

    def line_fit():
        return fit_generic_complex(line_spectrum, line_continuum, pabeta_recipe)

    timing, line_result = _median_timing(line_fit, repeats)
    timing["optimizer_used"] = line_result.metadata["optimizer_used"]
    timing["nonlinear_nfev"] = line_result.metadata["nonlinear_nfev"]
    stages["line_optimization"] = timing

    from qsospec.line_peaks import record_fit_peaks
    from qsospec.fitting.complexes import GenericComplexContext, generic_complex_metrics

    diagnostic_fit = fit_generic_complex(
        line_spectrum, line_continuum, pabeta_recipe
    )
    diagnostic_context = GenericComplexContext(
        pabeta_recipe, diagnostic_fit.metadata["active_components"], 1.0
    )
    diagnostic_theta = np.array(
        [diagnostic_fit.param_values[name] for name in diagnostic_context.names]
    )

    def diagnostics():
        return generic_complex_metrics(
            diagnostic_context, diagnostic_theta, line_continuum, line_spectrum
        )

    def peak_diagnostics():
        diagnostic_fit.metadata.pop("_peak_parameter_state", None)
        diagnostic_fit.metadata.pop("line_peaks", None)
        return record_fit_peaks(
            diagnostic_fit,
            0.0,
            definitions=diagnostic_fit.metadata["peak_model"]["components"],
            bounds=diagnostic_fit.metadata["peak_model"]["bounds"],
        )

    timing, _ = _median_timing(diagnostics, repeats)
    stages["metric_calculation"] = timing
    timing, _ = _median_timing(peak_diagnostics, repeats)
    stages["peak_uncertainty_calculation"] = timing

    from qsospec.io.run_store import workflow_payload

    serialization = qsospec.fit_global_lines(
        line_spectrum,
        _flat_config(continuum_windows=((12650.0, 13050.0),)),
        complexes=("pabeta",),
    )

    def serialize():
        return workflow_payload(
            serialization,
            run_id="benchmark",
            object_key="benchmark-0",
            object_id="benchmark-0",
            input_record={"source": "synthetic"},
        )

    timing, _ = _median_timing(serialize, repeats)
    stages["native_serialization"] = timing

    # 8. End-to-end no-host optical fit.
    optical_spectrum = _synthetic_optical_spectrum()

    def optical_fit():
        return qsospec.fit_global_lines(
            optical_spectrum,
            replace(
                ordinary_config,
                balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
                    enabled=False
                ),
            ),
            complexes=("hbeta_oiii", "halpha_nii_sii"),
        )

    timing, optical_result = _median_timing(optical_fit, repeats)
    timing["statuses"] = dict(optical_result.metadata["complex_statuses"])
    stages["end_to_end_optical_fit"] = timing

    # 9. End-to-end 500-bin RGS-like no-host no-Balmer fit.
    rgs_spectrum = _synthetic_rgs_spectrum()

    def rgs_fit():
        return qsospec.fit_global_lines(
            rgs_spectrum,
            _flat_config(continuum_windows=((10800.0, 13200.0),)),
            complexes=("pabeta", "oi11290", "padelta"),
        )

    timing, rgs_result = _median_timing(rgs_fit, repeats)
    timing["statuses"] = dict(rgs_result.metadata["complex_statuses"])
    stages["end_to_end_rgs_like_no_balmer_fit"] = timing

    # Convolution counts for the ordinary fit (cold cache).
    clear_iron_caches()
    _, counts = _count_convolutions(ordinary_fit)
    stages["ordinary_continuum_fit"]["convolutions_cold"] = counts.get(
        "convolutions", 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = benchmark(repeats=max(1, arguments.repeats))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
