"""Diagnostic plots for optional pPXF host-decomposition workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np

from .euclid import EuclidHostPrediction
from .ppxf_host import HostSED, PPXFHostFitResult


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _finite_percentile_limits(values, percentiles=(1.0, 99.0), pad_fraction=0.08):
    arrays = [np.ravel(np.asarray(value, dtype=float)) for value in values if value is not None]
    if not arrays:
        return None
    data = np.concatenate(arrays)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    lo, hi = np.nanpercentile(data, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = abs(lo) * pad_fraction if lo != 0 else 1.0
    else:
        pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _host_sed_prediction_on_input_grid(
    fit: PPXFHostFitResult,
    host_sed: Optional[HostSED],
) -> Optional[np.ndarray]:
    if host_sed is None:
        return None
    wave = fit.preprocessed.wave_rest
    predicted = np.interp(wave, host_sed.wave_rest, host_sed.host_flux, left=np.nan, right=np.nan)
    finite_fit_host = np.isfinite(fit.host_model)
    outside_fit = np.ones_like(wave, dtype=bool)
    if np.any(finite_fit_host):
        fit_min = np.nanmin(wave[finite_fit_host])
        fit_max = np.nanmax(wave[finite_fit_host])
        outside_fit = (wave < fit_min) | (wave > fit_max)
    return np.where(outside_fit & np.isfinite(predicted), predicted, np.nan)


def _host_sed_prediction_on_desi_grid(
    fit: PPXFHostFitResult,
    host_sed: Optional[HostSED],
) -> Optional[np.ndarray]:
    warnings.warn(
        "_host_sed_prediction_on_desi_grid is deprecated; use "
        "_host_sed_prediction_on_input_grid instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _host_sed_prediction_on_input_grid(fit, host_sed)


def plot_ppxf_host_fit(
    fit: PPXFHostFitResult,
    output_path: str,
    host_sed: Optional[HostSED] = None,
    workflow_result=None,
) -> str:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    wave = fit.preprocessed.wave_rest
    axes[0].plot(wave, fit.preprocessed.flux, color="0.2", lw=0.8, label="input spectrum")
    axes[0].plot(wave, fit.host_model, color="tab:green", lw=1.2, label="stellar host")
    predicted_host = _host_sed_prediction_on_input_grid(fit, host_sed)
    if predicted_host is not None and np.any(np.isfinite(predicted_host)):
        axes[0].plot(wave, predicted_host, color="tab:green", lw=1.0, ls="--", label="host SED prediction")
    axes[0].plot(
        wave,
        fit.agn_model,
        color="tab:orange",
        lw=1.2,
        label="AGN pseudo-continuum",
    )
    component_styles = {
        "powerlaw": ("tab:red", "--", "power law"),
        "feii_optical": ("tab:purple", ":", "optical Fe II"),
        "feii_uv": ("mediumpurple", ":", "UV Fe II"),
        "balmer_continuum": ("goldenrod", "-.", "Balmer continuum"),
        "balmer_high_order": ("darkgoldenrod", ":", "high-order Balmer"),
    }
    for name, (color, linestyle, label) in component_styles.items():
        values = fit.component_models.get(name)
        if values is None or not np.any(np.isfinite(values) & (values != 0)):
            continue
        axes[0].plot(
            wave,
            values,
            color=color,
            ls=linestyle,
            lw=0.9,
            label=label,
        )
    axes[0].plot(
        wave,
        fit.total_model,
        color="tab:blue",
        lw=1.2,
        label="pPXF best fit",
    )
    if workflow_result is not None:
        final_model = np.asarray(workflow_result.continuum.model, dtype=float).copy()
        for complex_result in workflow_result.line_complexes.values():
            if complex_result.success:
                final_model += np.asarray(complex_result.model, dtype=float)
        final_wave = np.asarray(workflow_result.spectrum.wave_rest, dtype=float)
        final_model = np.interp(
            wave,
            final_wave,
            final_model,
            left=np.nan,
            right=np.nan,
        )
        axes[0].plot(
            wave,
            final_model,
            color="0.05",
            lw=1.0,
            ls="--",
            label="final host-subtracted qsospec model",
        )
    masked = fit.preprocessed.emission_mask
    if np.any(masked):
        ymin, ymax = np.nanpercentile(fit.preprocessed.flux, [2, 98])
        axes[0].fill_between(wave, ymin, ymax, where=masked, color="tab:red", alpha=0.12, label="masked lines")
    final_good = fit.final_goodpixels_mask_log
    if final_good is not None:
        final_good_native = np.interp(
            wave,
            fit.preprocessed.wave_log,
            np.asarray(final_good, dtype=float),
            left=0.0,
            right=0.0,
        ) > 0.5
        axes[0].scatter(
            wave[final_good_native],
            fit.preprocessed.flux[final_good_native],
            s=2,
            color="0.15",
            alpha=0.15,
            label="final pPXF good pixels",
        )
    flux_limits = _finite_percentile_limits(
        [fit.preprocessed.flux, fit.host_model, predicted_host, fit.agn_model, fit.total_model],
        percentiles=(1.0, 99.0),
    )
    if flux_limits is not None:
        axes[0].set_ylim(*flux_limits)
    quality = fit.quality_metrics
    width = quality.get("pseudocontinuum_width_final_kms")
    prefit_line = quality.get("broad_prefit_line")
    coverage = fit.coverage.coverage_class if fit.coverage is not None else "unavailable"
    fagn = fit.ppxf_agn_fraction_flux_global
    closure = fit.closure_metrics.get("closure_relative_to_normalization")
    preprocessed_metadata = getattr(fit.preprocessed, "metadata", {})
    text_lines = [
        f"strategy: {fit.strategy_used}",
        (
            f"stellar templates: {quality.get('stellar_template_profile', fit.templates.profile_id)} "
            f"({quality.get('stellar_template_product_kind', fit.templates.product_kind)})"
        ),
        f"native data preserved: {quality.get('native_data_preserved', True)}",
        f"coverage: {coverage}",
        f"broad prefit: {prefit_line or 'unavailable'}; width={width or 'n/a'} km/s",
        f"global AGN fraction: {fagn:.3f}" if np.isfinite(fagn) else "global AGN fraction: unavailable",
        f"closure/norm: {closure:.2e}" if closure is not None and np.isfinite(closure) else "closure/norm: unavailable",
        (
            f"data LSF: {quality.get('resolution_status', 'unavailable')} / "
            f"{preprocessed_metadata.get('resolution_source', 'unspecified')}"
        ),
        (
            f"template resolution: "
            f"{quality.get('template_resolution_status', 'unavailable')}"
        ),
        (
            "template coarser on good pixels: "
            f"{quality.get('template_coarser_than_data_fraction_goodpixels', np.nan):.3f}"
            if np.isfinite(
                quality.get(
                    "template_coarser_than_data_fraction_goodpixels", np.nan
                )
            )
            else "template coarser on good pixels: unavailable"
        ),
        (
            "one-sided convolution fraction: "
            f"{quality.get('additional_template_sigma_nonzero_fraction', np.nan):.3f}"
            if np.isfinite(
                quality.get("additional_template_sigma_nonzero_fraction", np.nan)
            )
            else "one-sided convolution fraction: unavailable"
        ),
        (
            "preconvolution: "
            f"{quality.get('preconvolution_validation_status', 'not applicable')}"
        ),
        f"host continuum reliable: {fit.host_continuum_reliable}",
        f"host fraction reliable: {fit.host_fraction_reliable}",
        f"absorption subtraction: {fit.host_absorption_subtraction_status}",
        f"stellar kinematics: {fit.stellar_kinematics_resolution_status}",
        f"reliable: {fit.host_fit_reliable}",
    ]
    axes[0].text(
        0.01,
        0.98,
        "\n".join(text_lines),
        transform=axes[0].transAxes,
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "0.8"},
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=4, fontsize=7)
    axes[0].set_ylabel("Flux density [input units]")
    standardized = np.divide(
        fit.residual,
        fit.preprocessed.error,
        out=np.full_like(fit.residual, np.nan),
        where=np.isfinite(fit.preprocessed.error) & (fit.preprocessed.error > 0),
    )
    if final_good is not None:
        standardized[~final_good_native] = np.nan
    axes[1].plot(wave, standardized, color="0.25", lw=0.8)
    axes[1].axhline(0.0, color="0.7", lw=0.8)
    axes[1].axhline(3.0, color="0.7", lw=0.7, ls=":")
    axes[1].axhline(-3.0, color="0.7", lw=0.7, ls=":")
    residual_limits = _finite_percentile_limits([standardized], percentiles=(1.0, 99.0))
    if residual_limits is not None:
        axes[1].set_ylim(*residual_limits)
    axes[1].set_xlabel("Rest wavelength [Angstrom]")
    axes[1].set_ylabel(r"$(data-model)/\sigma$")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_desi_ppxf_fit(
    fit: PPXFHostFitResult,
    output_path: str,
    host_sed: Optional[HostSED] = None,
) -> str:
    warnings.warn(
        "plot_desi_ppxf_fit is deprecated; use plot_ppxf_host_fit instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_ppxf_host_fit(fit, output_path, host_sed)


def plot_host_sed_prediction(sed: HostSED, output_path: str) -> str:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(sed.wave_rest, sed.host_flux, color="tab:green", lw=1.0)
    for wave, label in [(5100, "5100A"), (10000, "1.0um"), (16000, "1.6um"), (22000, "2.2um")]:
        ax.axvline(wave, color="0.7", ls="--", lw=0.8)
        ax.text(wave, 0.98, label, rotation=90, transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=8)
    limits = _finite_percentile_limits([sed.host_flux], percentiles=(1.0, 99.0))
    if limits is not None:
        ax.set_ylim(*limits)
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("Host flux density [input units]")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_euclid_prediction(prediction: EuclidHostPrediction, output_path: str) -> str:
    plt = _setup_matplotlib()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    if prediction.euclid_flux is not None:
        ax.plot(prediction.wave_obs, prediction.euclid_flux, color="0.2", lw=0.8, label="Euclid spectrum")
    ax.plot(prediction.wave_obs, prediction.predicted_host_flux, color="tab:green", lw=1.0, label="predicted host")
    if prediction.host_subtracted_flux is not None:
        ax.plot(prediction.wave_obs, prediction.host_subtracted_flux, color="tab:blue", lw=0.8, label="host-subtracted")
    limits = _finite_percentile_limits(
        [prediction.euclid_flux, prediction.predicted_host_flux, prediction.host_subtracted_flux],
        percentiles=(1.0, 99.0),
    )
    if limits is not None:
        ax.set_ylim(*limits)
    ax.set_xlabel("Observed wavelength [Angstrom]")
    ax.set_ylabel("Flux density [input units]")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
