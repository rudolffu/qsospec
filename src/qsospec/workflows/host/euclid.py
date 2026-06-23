"""Euclid host-contamination prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .ppxf_host import HostSED


_C_KMS = 299792.458


@dataclass
class EuclidHostPrediction:
    wave_obs: np.ndarray
    wave_rest: np.ndarray
    euclid_flux: Optional[np.ndarray]
    predicted_host_flux: np.ndarray
    host_subtracted_flux: Optional[np.ndarray]
    scale_factor: float
    scale_mode: str
    warnings: list


@dataclass(frozen=True)
class EuclidHostScaleConfig:
    """Configuration for a fixed-host-shape Euclid aperture-scale fit."""

    fit_range: Tuple[float, float] = (7500.0, 13500.0)
    continuum_windows: Tuple[Tuple[float, float], ...] = (
        (7600.0, 8350.0),
        (8550.0, 8950.0),
        (9160.0, 9400.0),
        (9635.0, 9850.0),
        (10250.0, 10550.0),
        (11480.0, 12480.0),
        (13150.0, 13500.0),
    )
    safe_line_masks: Tuple[Tuple[float, float], ...] = (
        (8350.0, 8550.0),
        (8980.0, 9160.0),
        (9430.0, 9635.0),
        (9850.0, 10250.0),
        (10550.0, 11150.0),
        (11150.0, 11480.0),
        (12480.0, 13150.0),
    )
    permitted_lines: Tuple[float, ...] = (
        8447.0,
        10050.0,
        10833.0,
        10941.0,
        11290.0,
        12820.0,
    )
    forbidden_lines: Tuple[float, ...] = (9069.0, 9532.0)
    permitted_velocity_half_width: float = 10000.0
    forbidden_velocity_half_width: float = 3000.0
    resolving_power: float = 400.0
    host_bound_continuum_resolving_power: float = 100.0
    host_bound_safety_fraction: float = 0.98
    host_bound_percentile: float = 10.0
    break_wavelength: float = 9800.0
    alpha1_bounds: Tuple[float, float] = (-3.0, 0.5)
    alpha2_bounds: Tuple[float, float] = (-1.8, 0.5)
    slope_seeds: Tuple[float, ...] = (-1.5, -1.0, -0.6, 0.0)
    minimum_pixels_per_window: int = 10
    minimum_windows_per_break_side: int = 2
    error_floor_fraction: float = 0.03
    robust_loss: str = "soft_l1"
    minimum_clean_pixels: int = 80
    minimum_clean_coverage: float = 1500.0
    minimum_clean_windows: int = 2
    minimum_continuum_snr: float = 2.0
    maximum_reduced_chi2: float = 5.0
    negligible_host_fraction: float = 0.05
    bound_tolerance_fraction: float = 0.01


@dataclass
class EuclidHostScaleFit:
    """Result of a fixed-host-shape Euclid aperture-scale fit."""

    status: str
    success: bool
    model_type: str
    host_scale: float
    host_scale_error: float
    host_scale_max: float
    agn_amplitude: float
    agn_amplitude_error: float
    alpha1: float
    alpha1_error: float
    alpha2: float
    alpha2_error: float
    scaled_host_flux: np.ndarray
    agn_continuum_flux: np.ndarray
    total_continuum_flux: np.ndarray
    host_subtracted_flux: np.ndarray
    smooth_euclid_continuum: np.ndarray
    smooth_host_continuum: np.ndarray
    clean_mask: np.ndarray
    line_mask: np.ndarray
    represented_window_mask: np.ndarray
    n_clean_pixels: int
    n_clean_windows: int
    n_blue_windows: int
    n_red_windows: int
    clean_wavelength_coverage: float
    median_continuum_snr: float
    reduced_chi2: float
    host_scale_upper_bound_hit: bool
    slope_bound_hit: bool
    host_fraction_median: float
    reliable: bool
    reliability_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    parameter_errors: Dict[str, float] = field(default_factory=dict)


def _continuum_mask(wave_rest: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(wave_rest, dtype=bool)
    for lo, hi in windows:
        mask |= (wave_rest >= lo) & (wave_rest <= hi)
    return mask


def _nonnegative_scale(model: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(model) & np.isfinite(flux) & (model > 0)
    if not np.any(good):
        return 0.0
    denom = float(np.sum(model[good] ** 2))
    if denom <= 0:
        return 0.0
    return max(0.0, float(np.sum(model[good] * flux[good]) / denom))


def _constant_resolution_smooth(
    wave: np.ndarray,
    values: np.ndarray,
    resolving_power: float,
) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(wave) & np.isfinite(values) & (wave > 0)
    if np.count_nonzero(finite) < 10:
        return np.full_like(values, np.nan)
    order = np.argsort(wave[finite])
    source_wave = wave[finite][order]
    source_values = values[finite][order]
    log_wave = np.log(source_wave)
    dlog = float(np.nanmedian(np.diff(log_wave)))
    if not np.isfinite(dlog) or dlog <= 0:
        return np.full_like(values, np.nan)
    grid = np.arange(log_wave[0], log_wave[-1] + 0.5 * dlog, dlog)
    sampled = np.interp(grid, log_wave, source_values)
    sigma_log = 1.0 / (
        float(resolving_power) * 2.0 * np.sqrt(2.0 * np.log(2.0))
    )
    smoothed = gaussian_filter1d(sampled, sigma_log / dlog, mode="nearest")
    output = np.full_like(values, np.nan)
    output[finite] = np.interp(log_wave, grid, smoothed)
    return output


def euclid_nir_line_mask(
    wave_rest: np.ndarray,
    config: Optional[EuclidHostScaleConfig] = None,
) -> np.ndarray:
    """Return the conservative NIR emission-line mask for host scaling."""

    cfg = config or EuclidHostScaleConfig()
    wave = np.asarray(wave_rest, dtype=float)
    mask = np.zeros_like(wave, dtype=bool)
    for lower, upper in cfg.safe_line_masks:
        mask |= (wave >= lower) & (wave <= upper)
    resolution_half_width_factor = 2.0 / float(cfg.resolving_power)
    for centers, velocity in (
        (cfg.permitted_lines, cfg.permitted_velocity_half_width),
        (cfg.forbidden_lines, cfg.forbidden_velocity_half_width),
    ):
        for center in centers:
            half_width = max(
                40.0,
                float(center) * resolution_half_width_factor,
                float(center) * float(velocity) / _C_KMS,
            )
            mask |= (
                (wave >= float(center) - half_width)
                & (wave <= float(center) + half_width)
            )
    return mask


def _power_law(
    wave: np.ndarray,
    amplitude: float,
    alpha1: float,
    alpha2: float,
    model_type: str,
    break_wavelength: float,
) -> np.ndarray:
    x = np.asarray(wave, dtype=float) / float(break_wavelength)
    if model_type == "broken_power_law":
        return float(amplitude) * np.where(
            wave < break_wavelength,
            x**float(alpha1),
            x**float(alpha2),
        )
    return float(amplitude) * x**float(alpha1)


def _failed_scale_fit(
    wave: np.ndarray,
    flux: np.ndarray,
    host: np.ndarray,
    clean_mask: np.ndarray,
    line_mask: np.ndarray,
    represented_window_mask: np.ndarray,
    status: str,
    warnings: list[str],
) -> EuclidHostScaleFit:
    nan_array = np.full_like(wave, np.nan, dtype=float)
    return EuclidHostScaleFit(
        status=status,
        success=False,
        model_type="unavailable",
        host_scale=np.nan,
        host_scale_error=np.nan,
        host_scale_max=np.nan,
        agn_amplitude=np.nan,
        agn_amplitude_error=np.nan,
        alpha1=np.nan,
        alpha1_error=np.nan,
        alpha2=np.nan,
        alpha2_error=np.nan,
        scaled_host_flux=nan_array.copy(),
        agn_continuum_flux=nan_array.copy(),
        total_continuum_flux=nan_array.copy(),
        host_subtracted_flux=nan_array.copy(),
        smooth_euclid_continuum=nan_array.copy(),
        smooth_host_continuum=nan_array.copy(),
        clean_mask=clean_mask,
        line_mask=line_mask,
        represented_window_mask=represented_window_mask,
        n_clean_pixels=int(np.count_nonzero(clean_mask)),
        n_clean_windows=int(np.count_nonzero(represented_window_mask)),
        n_blue_windows=0,
        n_red_windows=0,
        clean_wavelength_coverage=0.0,
        median_continuum_snr=np.nan,
        reduced_chi2=np.nan,
        host_scale_upper_bound_hit=False,
        slope_bound_hit=False,
        host_fraction_median=np.nan,
        reliable=False,
        reliability_reasons=[status],
        warnings=warnings,
    )


def fit_euclid_host_aperture_scale(
    wave_rest: np.ndarray,
    euclid_flux: np.ndarray,
    euclid_error: np.ndarray,
    host_flux_r400: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    desi_host_fit_reliable: bool = True,
    config: Optional[EuclidHostScaleConfig] = None,
) -> EuclidHostScaleFit:
    """Fit a Euclid aperture scale for a fixed DESI-derived host shape."""

    cfg = config or EuclidHostScaleConfig()
    wave = np.asarray(wave_rest, dtype=float)
    flux = np.asarray(euclid_flux, dtype=float)
    error = np.asarray(euclid_error, dtype=float)
    host = np.asarray(host_flux_r400, dtype=float)
    if not (wave.shape == flux.shape == error.shape == host.shape):
        raise ValueError("Euclid host-scale inputs must have identical shapes.")
    valid = (
        np.ones_like(wave, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != wave.shape:
        raise ValueError("valid_mask must match the Euclid wavelength shape.")

    line_mask = euclid_nir_line_mask(wave, cfg)
    window_mask = _continuum_mask(wave, cfg.continuum_windows)
    base_clean = (
        valid
        & np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
        & np.isfinite(host)
        & (wave >= cfg.fit_range[0])
        & (wave <= cfg.fit_range[1])
        & window_mask
        & (~line_mask)
    )
    represented = np.zeros(len(cfg.continuum_windows), dtype=bool)
    for index, (lower, upper) in enumerate(cfg.continuum_windows):
        represented[index] = (
            np.count_nonzero(base_clean & (wave >= lower) & (wave <= upper))
            >= cfg.minimum_pixels_per_window
        )
    n_blue = sum(
        bool(present) and upper <= cfg.break_wavelength
        for present, (_, upper) in zip(represented, cfg.continuum_windows)
    )
    n_red = sum(
        bool(present) and lower >= cfg.break_wavelength
        for present, (lower, _) in zip(represented, cfg.continuum_windows)
    )
    model_type = (
        "broken_power_law"
        if n_blue >= cfg.minimum_windows_per_break_side
        and n_red >= cfg.minimum_windows_per_break_side
        else "single_power_law"
    )
    if not np.any(base_clean):
        return _failed_scale_fit(
            wave,
            flux,
            host,
            base_clean,
            line_mask,
            represented,
            "no_clean_euclid_pixels",
            ["no_clean_euclid_pixels"],
        )

    smooth_flux = _constant_resolution_smooth(
        wave, flux, cfg.host_bound_continuum_resolving_power
    )
    smooth_host = _constant_resolution_smooth(
        wave, host, cfg.host_bound_continuum_resolving_power
    )
    bound_pixels = (
        base_clean
        & np.isfinite(smooth_flux)
        & (smooth_flux > 0)
        & np.isfinite(smooth_host)
        & (smooth_host > 0)
    )
    if not np.any(bound_pixels):
        return _failed_scale_fit(
            wave,
            flux,
            host,
            base_clean,
            line_mask,
            represented,
            "host_scale_bound_unavailable",
            ["host_scale_bound_unavailable"],
        )
    host_scale_max = float(
        cfg.host_bound_safety_fraction
        * np.nanpercentile(
            smooth_flux[bound_pixels] / smooth_host[bound_pixels],
            cfg.host_bound_percentile,
        )
    )
    if not np.isfinite(host_scale_max) or host_scale_max <= 0:
        return _failed_scale_fit(
            wave,
            flux,
            host,
            base_clean,
            line_mask,
            represented,
            "invalid_host_scale_bound",
            ["invalid_host_scale_bound"],
        )

    continuum_for_floor = np.where(
        np.isfinite(smooth_flux), np.abs(smooth_flux), 0.0
    )
    effective_error = np.sqrt(
        error**2 + (cfg.error_floor_fraction * continuum_for_floor) ** 2
    )
    clean = base_clean & np.isfinite(effective_error) & (effective_error > 0)
    x_wave = wave[clean]
    x_flux = flux[clean]
    x_error = effective_error[clean]
    x_host = host[clean]
    amplitude_guess = max(
        float(np.nanmedian(np.clip(x_flux, 0.0, np.inf))),
        np.finfo(float).eps,
    )
    amplitude_upper = max(
        10.0 * float(np.nanmax(np.abs(x_flux))),
        10.0 * amplitude_guess,
        1.0,
    )

    def unpack(theta):
        if model_type == "broken_power_law":
            return theta[0], theta[1], theta[2], theta[3]
        return theta[0], theta[1], theta[2], theta[2]

    def residual(theta):
        scale, amplitude, alpha1, alpha2 = unpack(theta)
        agn = _power_law(
            x_wave,
            amplitude,
            alpha1,
            alpha2,
            model_type,
            cfg.break_wavelength,
        )
        return (x_flux - scale * x_host - agn) / x_error

    if model_type == "broken_power_law":
        lower = np.array(
            [0.0, 0.0, cfg.alpha1_bounds[0], cfg.alpha2_bounds[0]]
        )
        upper = np.array(
            [
                host_scale_max,
                amplitude_upper,
                cfg.alpha1_bounds[1],
                cfg.alpha2_bounds[1],
            ]
        )
        starts = [
            np.array(
                [
                    0.5 * host_scale_max,
                    amplitude_guess,
                    alpha1,
                    alpha2,
                ]
            )
            for alpha1 in cfg.slope_seeds
            for alpha2 in cfg.slope_seeds
            if cfg.alpha1_bounds[0] <= alpha1 <= cfg.alpha1_bounds[1]
            and cfg.alpha2_bounds[0] <= alpha2 <= cfg.alpha2_bounds[1]
        ]
    else:
        lower = np.array([0.0, 0.0, cfg.alpha1_bounds[0]])
        upper = np.array(
            [host_scale_max, amplitude_upper, cfg.alpha1_bounds[1]]
        )
        starts = [
            np.array([0.5 * host_scale_max, amplitude_guess, alpha])
            for alpha in cfg.slope_seeds
            if cfg.alpha1_bounds[0] <= alpha <= cfg.alpha1_bounds[1]
        ]
    fits = []
    for start in starts:
        try:
            fits.append(
                least_squares(
                    residual,
                    np.clip(start, lower + 1e-10, upper - 1e-10),
                    bounds=(lower, upper),
                    loss=cfg.robust_loss,
                    max_nfev=3000,
                )
            )
        except ValueError:
            continue
    successful = [
        result
        for result in fits
        if result.success and np.all(np.isfinite(result.x))
    ]
    if not successful:
        return _failed_scale_fit(
            wave,
            flux,
            host,
            clean,
            line_mask,
            represented,
            "euclid_host_scale_optimization_failed",
            ["euclid_host_scale_optimization_failed"],
        )
    result = min(successful, key=lambda item: float(np.sum(residual(item.x) ** 2)))
    host_scale, amplitude, alpha1, alpha2 = unpack(result.x)
    agn_full = _power_law(
        wave,
        amplitude,
        alpha1,
        alpha2,
        model_type,
        cfg.break_wavelength,
    )
    scaled_host = host_scale * host
    total_model = scaled_host + agn_full
    subtracted = flux - scaled_host
    normalized_residual = residual(result.x)
    dof = max(int(normalized_residual.size - result.x.size), 1)
    reduced_chi2 = float(
        np.sum(normalized_residual**2) / dof
    )

    errors = np.full(result.x.size, np.nan)
    try:
        covariance = np.linalg.inv(result.jac.T @ result.jac)
        covariance *= float(np.sum(normalized_residual**2) / dof)
        errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    except np.linalg.LinAlgError:
        pass
    tolerance = cfg.bound_tolerance_fraction
    lower_distance = (result.x - lower) / np.maximum(upper - lower, 1e-12)
    upper_distance = (upper - result.x) / np.maximum(upper - lower, 1e-12)
    parameter_bound_hit = (lower_distance <= tolerance) | (
        upper_distance <= tolerance
    )
    errors[parameter_bound_hit] = np.nan
    host_bound_hit = bool(
        host_scale >= host_scale_max * (1.0 - tolerance)
    )
    slope_indices = (2, 3) if model_type == "broken_power_law" else (2,)
    slope_bound_hit = bool(np.any(parameter_bound_hit[list(slope_indices)]))

    clean_wave = wave[clean]
    clean_coverage = (
        float(np.nanmax(clean_wave) - np.nanmin(clean_wave))
        if clean_wave.size
        else 0.0
    )
    median_snr = float(
        np.nanmedian(np.abs(flux[clean]) / effective_error[clean])
    )
    positive_fraction_pixels = (
        clean & np.isfinite(smooth_flux) & (smooth_flux > 0)
    )
    host_fraction = (
        float(
            np.nanmedian(
                scaled_host[positive_fraction_pixels]
                / smooth_flux[positive_fraction_pixels]
            )
        )
        if np.any(positive_fraction_pixels)
        else np.nan
    )
    host_scale_error = float(errors[0])
    negligible_zero_host = (
        (
            (
                np.isfinite(host_scale_error)
                and host_scale <= 2.0 * host_scale_error
            )
            or host_scale
            <= cfg.bound_tolerance_fraction * host_scale_max
        )
        and np.isfinite(host_fraction)
        and host_fraction < cfg.negligible_host_fraction
    )
    reasons = []
    if not desi_host_fit_reliable:
        reasons.append("desi_host_fit_unreliable")
    if np.count_nonzero(clean) < cfg.minimum_clean_pixels:
        reasons.append("too_few_clean_pixels")
    if clean_coverage < cfg.minimum_clean_coverage:
        reasons.append("clean_coverage_below_threshold")
    if np.count_nonzero(represented) < cfg.minimum_clean_windows:
        reasons.append("too_few_clean_windows")
    if median_snr < cfg.minimum_continuum_snr:
        reasons.append("continuum_snr_below_threshold")
    if reduced_chi2 > cfg.maximum_reduced_chi2:
        reasons.append("reduced_chi2_above_threshold")
    if host_bound_hit:
        reasons.append("host_scale_upper_bound_hit")
    if slope_bound_hit:
        reasons.append("agn_slope_bound_hit")
    if negligible_zero_host:
        reasons.append("host_consistent_with_zero_and_negligible")

    parameter_errors = {
        "host_scale": host_scale_error,
        "agn_amplitude": float(errors[1]),
        "alpha1": float(errors[2]),
        "alpha2": (
            float(errors[3])
            if model_type == "broken_power_law"
            else np.nan
        ),
    }
    return EuclidHostScaleFit(
        status="success",
        success=True,
        model_type=model_type,
        host_scale=float(host_scale),
        host_scale_error=host_scale_error,
        host_scale_max=host_scale_max,
        agn_amplitude=float(amplitude),
        agn_amplitude_error=float(errors[1]),
        alpha1=float(alpha1),
        alpha1_error=float(errors[2]),
        alpha2=float(alpha2) if model_type == "broken_power_law" else np.nan,
        alpha2_error=(
            float(errors[3])
            if model_type == "broken_power_law"
            else np.nan
        ),
        scaled_host_flux=scaled_host,
        agn_continuum_flux=agn_full,
        total_continuum_flux=total_model,
        host_subtracted_flux=subtracted,
        smooth_euclid_continuum=smooth_flux,
        smooth_host_continuum=smooth_host,
        clean_mask=clean,
        line_mask=line_mask,
        represented_window_mask=represented,
        n_clean_pixels=int(np.count_nonzero(clean)),
        n_clean_windows=int(np.count_nonzero(represented)),
        n_blue_windows=int(n_blue),
        n_red_windows=int(n_red),
        clean_wavelength_coverage=clean_coverage,
        median_continuum_snr=median_snr,
        reduced_chi2=reduced_chi2,
        host_scale_upper_bound_hit=host_bound_hit,
        slope_bound_hit=slope_bound_hit,
        host_fraction_median=host_fraction,
        reliable=not reasons,
        reliability_reasons=reasons,
        warnings=[],
        parameters={
            "host_scale": float(host_scale),
            "agn_amplitude": float(amplitude),
            "alpha1": float(alpha1),
            "alpha2": (
                float(alpha2)
                if model_type == "broken_power_law"
                else np.nan
            ),
        },
        parameter_errors=parameter_errors,
    )


def predict_host_for_euclid_spectrum(
    desi_host_sed: HostSED,
    euclid_wave_obs: np.ndarray,
    z: float,
    euclid_flux: Optional[np.ndarray] = None,
    scale_mode: str = "free_scale",
    aperture_scale: Optional[float] = None,
    continuum_windows: Sequence[Tuple[float, float]] = ((10000.0, 12000.0), (14500.0, 17000.0)),
) -> EuclidHostPrediction:
    """Interpolate a DESI-derived host SED onto a Euclid observed grid."""

    wave_obs = np.asarray(euclid_wave_obs, dtype=float)
    wave_rest = wave_obs / (1.0 + float(z))
    warnings = []
    host = np.interp(wave_rest, desi_host_sed.wave_rest, desi_host_sed.host_flux, left=np.nan, right=np.nan)
    if np.any(~np.isfinite(host)):
        warnings.append("euclid_grid_extends_beyond_host_sed_coverage")

    mode = scale_mode.lower()
    if mode == "desi_fiber_scaled":
        scale = 1.0 if aperture_scale is None else float(aperture_scale)
    elif mode == "image_prior_scale":
        if aperture_scale is None:
            raise ValueError("image_prior_scale requires aperture_scale.")
        scale = float(aperture_scale)
    elif mode == "free_scale":
        if euclid_flux is None:
            warnings.append("free_scale_without_euclid_flux_uses_unity")
            scale = 1.0 if aperture_scale is None else float(aperture_scale)
        else:
            mask = _continuum_mask(wave_rest, continuum_windows)
            scale = _nonnegative_scale(host, np.asarray(euclid_flux, dtype=float), mask)
    else:
        raise ValueError(f"Unknown Euclid scale_mode: {scale_mode}")

    scaled_host = host * scale
    flux = None if euclid_flux is None else np.asarray(euclid_flux, dtype=float)
    subtracted = None if flux is None else flux - scaled_host
    return EuclidHostPrediction(
        wave_obs=wave_obs,
        wave_rest=wave_rest,
        euclid_flux=flux,
        predicted_host_flux=scaled_host,
        host_subtracted_flux=subtracted,
        scale_factor=scale,
        scale_mode=mode,
        warnings=warnings,
    )


def write_euclid_prediction(prediction: EuclidHostPrediction, output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "euclid_host_prediction.csv"
    data = {
        "wave_obs": prediction.wave_obs,
        "wave_rest": prediction.wave_rest,
        "predicted_host_flux": prediction.predicted_host_flux,
    }
    if prediction.euclid_flux is not None:
        data["euclid_flux"] = prediction.euclid_flux
    if prediction.host_subtracted_flux is not None:
        data["host_subtracted_flux"] = prediction.host_subtracted_flux
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)
