"""Explicit spectral-resolution models and template-matching utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

_C_KMS = 299792.458


@dataclass(frozen=True)
class SpectralResolution:
    """Generic spectral-resolution description on an optional wavelength grid."""

    mode: str = "missing"
    values: np.ndarray | None = None
    wavelength: np.ndarray | None = None
    source: str = "unspecified"
    version: str | None = None
    is_object_specific: bool = False
    is_approximate: bool = False
    banded_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        allowed = {"missing", "resolving_power", "sigma_lambda", "fwhm_lambda", "sigma_kms", "banded_matrix", "external_curve", "constant_r"}
        if self.mode not in allowed:
            raise ValueError(f"Unknown resolution mode: {self.mode}")
        if self.mode == "banded_matrix" and self.banded_matrix is None:
            raise ValueError("banded_matrix mode requires a matrix")
        if self.mode not in {"missing", "banded_matrix"} and self.values is None:
            raise ValueError(f"{self.mode} requires resolution values")
        if self.wavelength is not None and self.values is not None:
            wave = np.asarray(self.wavelength, float)
            values = np.asarray(self.values, float)
            if values.size not in {1, wave.size}:
                raise ValueError("Resolution values must be scalar or match wavelength")

    @property
    def status(self) -> str:
        if self.mode == "missing":
            return "missing"
        return "approximate" if self.is_approximate or self.mode == "constant_r" else "valid"

    @property
    def coverage(self) -> tuple[float, float] | None:
        if self.wavelength is None:
            return None
        wave = np.asarray(self.wavelength, float)
        finite = wave[np.isfinite(wave)]
        return (float(finite.min()), float(finite.max())) if finite.size else None

    def sigma_lambda(self, wavelength: np.ndarray) -> np.ndarray:
        wave = np.asarray(wavelength, float)
        if self.mode == "missing":
            return np.full_like(wave, np.nan)
        if self.mode == "banded_matrix":
            raise ValueError("A banded matrix cannot be reduced to sigma_lambda without a declared convention")
        source_wave = np.asarray(self.wavelength, float) if self.wavelength is not None else wave
        raw = np.asarray(self.values, float)
        value = np.full_like(wave, raw.item()) if raw.size == 1 else np.interp(wave, source_wave, raw, left=np.nan, right=np.nan)
        mode = "resolving_power" if self.mode in {"constant_r", "external_curve"} else self.mode
        if mode == "resolving_power":
            return wave / value / 2.354820045
        if mode == "sigma_lambda":
            return value
        if mode == "fwhm_lambda":
            return value / 2.354820045
        if mode == "sigma_kms":
            return wave * value / _C_KMS
        raise ValueError(f"Cannot convert resolution mode {self.mode}")

    def metadata(self) -> Mapping[str, Any]:
        return {"resolution_mode": self.mode, "resolution_source": self.source, "resolution_version": self.version, "resolution_is_object_specific": self.is_object_specific, "resolution_is_approximate": self.is_approximate, "resolution_status": self.status, "resolution_coverage": self.coverage}


def additional_template_sigma(data_sigma_lambda: np.ndarray, template_sigma_lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature broadening and invalid mask when templates are lower resolution."""
    data = np.asarray(data_sigma_lambda, float)
    template = np.asarray(template_sigma_lambda, float)
    if data.shape != template.shape:
        raise ValueError("Data and template resolution arrays must match")
    invalid = ~np.isfinite(data) | ~np.isfinite(template) | (data <= 0) | (template < 0) | (template > data)
    sigma = np.sqrt(np.maximum(0.0, data**2 - template**2))
    sigma[invalid] = np.nan
    return sigma, invalid


def smooth_constant_resolving_power(wavelength: np.ndarray, values: np.ndarray, resolving_power: float) -> np.ndarray:
    """Smooth values on a log-wavelength grid with a Gaussian constant R."""
    wave = np.asarray(wavelength, float)
    flux = np.asarray(values, float)
    if wave.shape != flux.shape or resolving_power <= 0:
        raise ValueError("Valid same-shape arrays and positive resolving_power required")
    finite = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    output = np.full_like(flux, np.nan)
    if finite.sum() < 3:
        return output
    order = np.argsort(wave[finite])
    log_wave = np.log(wave[finite][order])
    vals = flux[finite][order]
    step = float(np.median(np.diff(log_wave)))
    grid = np.arange(log_wave[0], log_wave[-1] + step / 2, step)
    sampled = np.interp(grid, log_wave, vals)
    sigma_log = 1.0 / (float(resolving_power) * 2.354820045)
    smoothed = gaussian_filter1d(sampled, sigma_log / step, mode="nearest")
    output_indices = np.flatnonzero(finite)[order]
    output[output_indices] = np.interp(log_wave, grid, smoothed)
    return output
