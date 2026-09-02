"""Coverage classification for host-galaxy decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HostCoverageConfig

HOST_FEATURE_WINDOWS: dict[str, tuple[float, float]] = {
    "ca_hk": (3920.0, 3995.0),
    "d4000_blue": (3850.0, 3950.0),
    "d4000_red": (4000.0, 4100.0),
    "g_band": (4280.0, 4325.0),
    "hbeta_absorption": (4800.0, 4920.0),
    "mg_b": (5100.0, 5350.0),
    "na_d": (5840.0, 5940.0),
}


@dataclass(frozen=True)
class HostCoverageResult:
    """Materialized wavelength leverage for one host fit."""

    coverage_class: str
    rest_min: float
    rest_max: float
    fit_span_angstrom: float
    valid_pixel_count: int
    valid_fraction: float
    feature_coverage: dict[str, bool]
    reasons: tuple[str, ...]


def _window_support(
    wave: np.ndarray,
    valid: np.ndarray,
    window: tuple[float, float],
    config: HostCoverageConfig,
) -> tuple[bool, float, int]:
    lo, hi = map(float, window)
    inside = (wave >= lo) & (wave <= hi)
    available = int(np.count_nonzero(inside))
    count = int(np.count_nonzero(inside & valid))
    fraction = float(count / available) if available else 0.0
    valid_wave = wave[inside & valid]
    endpoints = bool(
        valid_wave.size
        and valid_wave.min() <= lo + config.endpoint_tolerance_angstrom
        and valid_wave.max() >= hi - config.endpoint_tolerance_angstrom
    )
    supported = bool(
        endpoints
        and fraction >= config.minimum_valid_fraction
        and count >= min(config.minimum_valid_pixels, max(10, available // 3))
    )
    return supported, fraction, count


def classify_host_coverage(
    wave_rest: np.ndarray,
    valid_mask: np.ndarray,
    config: HostCoverageConfig | None = None,
) -> HostCoverageResult:
    """Classify actual valid rest-frame support, independent of redshift."""

    cfg = config or HostCoverageConfig()
    wave = np.asarray(wave_rest, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(wave) & (wave > 0)
    valid_wave = wave[valid]
    if valid_wave.size:
        rest_min = float(np.nanmin(valid_wave))
        rest_max = float(np.nanmax(valid_wave))
    else:
        rest_min = rest_max = float("nan")
    feature_coverage = {
        name: _window_support(wave, valid, window, cfg)[0] for name, window in HOST_FEATURE_WINDOWS.items()
    }
    class_windows = (
        ("full_optical", cfg.full_optical_range),
        ("optical_core", cfg.optical_core_range),
        ("blue_optical", cfg.blue_optical_range),
    )
    selected = "insufficient"
    selected_fraction = 0.0
    selected_count = 0
    reasons = []
    for name, window in class_windows:
        supported, fraction, count = _window_support(wave, valid, window, cfg)
        if supported:
            selected = name
            selected_fraction = fraction
            selected_count = count
            break
    if selected == "insufficient":
        _, selected_fraction, selected_count = _window_support(wave, valid, cfg.blue_optical_range, cfg)
        reasons.append("blue_optical_requirement_not_met")
    elif selected == "blue_optical":
        reasons.append("limited_wavelength_leverage")
    if selected_count < cfg.minimum_valid_pixels:
        reasons.append("too_few_valid_host_pixels")
    return HostCoverageResult(
        coverage_class=selected,
        rest_min=rest_min,
        rest_max=rest_max,
        fit_span_angstrom=(float(rest_max - rest_min) if np.isfinite(rest_min) and np.isfinite(rest_max) else 0.0),
        valid_pixel_count=int(np.count_nonzero(valid)),
        valid_fraction=float(selected_fraction),
        feature_coverage=feature_coverage,
        reasons=tuple(dict.fromkeys(reasons)),
    )
