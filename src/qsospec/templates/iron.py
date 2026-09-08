"""Iron-template objects and grid preparation for qsospec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..warnings import FitWarning


C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class IronTemplateError(ValueError):
    """Exception carrying a stable warning/error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class IronTemplate:
    """Normalized iron-template spectrum."""

    name: str
    wave_rest: np.ndarray
    flux: np.ndarray
    wave_unit: str = "Angstrom"
    flux_unit: str = "arbitrary area-normalized"
    reference: Optional[str] = None
    source_path: Optional[str] = None
    coverage: Optional[Tuple[float, float]] = None
    notes: List[str] = field(default_factory=list)
    normalization: str = "area"
    native_fwhm_kms: float = 0.0
    native_width_status: str = "unknown"
    native_width_source: Optional[str] = None

    def __post_init__(self) -> None:
        self.wave_rest = np.asarray(self.wave_rest, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if not np.isfinite(self.native_fwhm_kms) or self.native_fwhm_kms < 0:
            raise ValueError("Iron-template native FWHM must be finite and non-negative.")
        if self.coverage is None and self.wave_rest.size:
            self.coverage = (float(self.wave_rest.min()), float(self.wave_rest.max()))


@dataclass
class PreparedIronTemplate:
    """Iron template evaluated on one local fitting grid."""

    template: IronTemplate
    basis: np.ndarray
    fwhm_kms: float
    warnings: List[FitWarning] = field(default_factory=list)

    @property
    def has_overlap(self) -> bool:
        return bool(np.any(np.isfinite(self.basis) & (self.basis != 0)))

    @property
    def flux_integral_unit_amp(self) -> float:
        return float("nan")


def _log_grid(wave_min: float, wave_max: float, velocity_step_kms: float) -> np.ndarray:
    dlog = velocity_step_kms / C_KMS
    return np.exp(np.arange(np.log(wave_min), np.log(wave_max) + 0.5 * dlog, dlog))


def _broaden_template_with_derivative(
    template: IronTemplate,
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
    width_mode: str = "legacy",
):
    resolved = resolve_iron_width(template, fwhm_kms, width_mode)
    convolution_fwhm = resolved["kernel_fwhm_kms"]
    native_fwhm = template.native_fwhm_kms if resolved["requested_width_mode"] == "target" else 0.0
    wave = template.wave_rest
    flux = template.flux
    # Keep the original sampling phase; explicit zero padding prevents the
    # convolution output from exceeding the input for very broad kernels.
    grid = _log_grid(float(wave.min()), float(wave.max()), velocity_step_kms)
    sampled = np.interp(grid, wave, flux, left=0.0, right=0.0)
    if convolution_fwhm == 0:
        return grid, sampled, np.zeros_like(sampled)
    sigma_pix = (convolution_fwhm / FWHM_TO_SIGMA) / float(velocity_step_kms)
    half = max(1, int(np.ceil(4.0 * sigma_pix)))
    x = np.arange(-half, half + 1, dtype=float)
    raw_kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    raw_derivative = raw_kernel * x**2 / sigma_pix**3
    kernel_sum = raw_kernel.sum()
    kernel = raw_kernel / kernel_sum
    kernel_derivative_sigma = (
        raw_derivative * kernel_sum - raw_kernel * raw_derivative.sum()
    ) / kernel_sum**2
    convolution_derivative = (
        float(fwhm_kms) / convolution_fwhm if native_fwhm > 0 else 1.0
    )
    sigma_derivative_fwhm = convolution_derivative / (
        FWHM_TO_SIGMA * float(velocity_step_kms)
    )
    kernel_derivative_fwhm = kernel_derivative_sigma * sigma_derivative_fwhm
    return (
        grid,
        np.convolve(np.pad(sampled, half), kernel, mode="same")[half:-half],
        np.convolve(np.pad(sampled, half), kernel_derivative_fwhm, mode="same")[half:-half],
    )


def _broaden_template(template: IronTemplate, fwhm_kms: float, velocity_step_kms: float = 25.0):
    grid, broadened, _ = _broaden_template_with_derivative(
        template, fwhm_kms, velocity_step_kms
    )
    return grid, broadened


def _apply_coverage_taper(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    coverage = template.coverage or (float(template.wave_rest.min()), float(template.wave_rest.max()))
    inside = (wave_rest_fit >= coverage[0]) & (wave_rest_fit <= coverage[1])
    values[~inside] = 0.0
    span = float(coverage[1] - coverage[0])
    taper_width = min(100.0, max(20.0, 0.05 * span))
    left = (wave_rest_fit >= coverage[0]) & (wave_rest_fit < coverage[0] + taper_width)
    right = (wave_rest_fit > coverage[1] - taper_width) & (wave_rest_fit <= coverage[1])
    if np.any(left):
        phase = (wave_rest_fit[left] - coverage[0]) / taper_width
        values[left] *= 0.5 - 0.5 * np.cos(np.pi * phase)
    if np.any(right):
        phase = (coverage[1] - wave_rest_fit[right]) / taper_width
        values[right] *= 0.5 - 0.5 * np.cos(np.pi * phase)
    return values


def evaluate_iron_basis(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
) -> np.ndarray:
    """Return a broadened iron-template basis on a rest-frame fitting grid."""

    wave_rest_fit = np.asarray(wave_rest_fit, dtype=float)
    broadened_wave, broadened_flux = _broaden_template(template, fwhm_kms, velocity_step_kms=velocity_step_kms)
    basis = np.interp(wave_rest_fit, broadened_wave, broadened_flux, left=0.0, right=0.0)
    return _apply_coverage_taper(template, wave_rest_fit, basis)


def evaluate_iron_basis_with_derivative(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the broadened iron basis and its FWHM derivative."""

    wave_rest_fit = np.asarray(wave_rest_fit, dtype=float)
    broadened_wave, broadened_flux, derivative_flux = _broaden_template_with_derivative(
        template,
        fwhm_kms,
        velocity_step_kms=velocity_step_kms,
    )
    basis = np.interp(wave_rest_fit, broadened_wave, broadened_flux, left=0.0, right=0.0)
    derivative = np.interp(
        wave_rest_fit, broadened_wave, derivative_flux, left=0.0, right=0.0
    )
    return (
        _apply_coverage_taper(template, wave_rest_fit, basis),
        _apply_coverage_taper(template, wave_rest_fit, derivative),
    )


def prepare_iron_template(
    template: IronTemplate,
    wave_rest_fit: np.ndarray,
    window: Tuple[float, float],
    fwhm_kms: float,
    velocity_step_kms: float = 25.0,
) -> PreparedIronTemplate:
    """Broaden and interpolate an iron template onto a fit grid."""

    wave_rest_fit = np.asarray(wave_rest_fit, dtype=float)
    warnings: List[FitWarning] = []
    coverage = template.coverage or (float(template.wave_rest.min()), float(template.wave_rest.max()))
    lo, hi = map(float, window)
    context = {"template": template.name, "window": (lo, hi), "coverage": coverage}

    if hi < coverage[0] or lo > coverage[1]:
        warnings.append(
            FitWarning(
                code="iron_template_no_overlap",
                message="Iron template has no overlap with the requested local fitting window.",
                severity="error",
                context=context,
            )
        )
        return PreparedIronTemplate(template, np.zeros_like(wave_rest_fit), fwhm_kms, warnings)

    if lo < coverage[0] or hi > coverage[1]:
        warnings.append(
            FitWarning(
                code="iron_template_partial_coverage",
                message="Iron template only partially covers the requested local fitting window.",
                context=context,
            )
        )

    basis = evaluate_iron_basis(template, wave_rest_fit, fwhm_kms, velocity_step_kms=velocity_step_kms)
    return PreparedIronTemplate(template, basis, float(fwhm_kms), warnings)


def resolve_iron_width(template, value, mode="legacy"):
    """Resolve requested width without treating an unknown native width as zero.

    Legacy Verner widths are targets; empirical legacy widths are kernels.
    The derivative at the zero-kernel boundary is defined in kernel coordinates.
    """
    if mode not in ("legacy", "kernel", "target"):
        raise ValueError("width mode must be legacy, kernel, or target")
    if not np.isfinite(value) or value < 0:
        raise ValueError("Iron width must be finite and non-negative")
    status = template.native_width_status
    if template.native_fwhm_kms > 0 and status == "unknown":
        status = "known_gaussian_equivalent"
    known = status in ("known_gaussian_equivalent", "unbroadened")
    native = float(template.native_fwhm_kms) if known else None
    requested = ("target" if template.native_fwhm_kms > 0 else "kernel") if mode == "legacy" else mode
    if requested == "target":
        if native is None:
            raise ValueError("Target width requires a justified native Gaussian-equivalent width")
        if value < native:
            raise IronTemplateError("iron_template_below_native_resolution", "Target width cannot sharpen the native template")
        kernel = np.sqrt((value-native)*(value+native))
    else:
        kernel = float(value)
    return {"requested_width_mode": requested, "configuration_width_mode": mode,
            "requested_fwhm_kms": float(value), "kernel_fwhm_kms": float(kernel),
            "native_fwhm_kms": native, "native_width_status": status,
            "native_width_source": template.native_width_source,
            "target_fwhm_kms": float(np.hypot(native, kernel)) if known else None,
            "effective_width_status": "gaussian_equivalent" if known else "unavailable"}


def evaluate_iron_kernel(template, wave, kernel_fwhm_kms, *, taper=True):
    """Evaluate full-template convolution and derivative in kernel coordinates."""
    grid, flux, derivative = _broaden_template_with_derivative(
        template, kernel_fwhm_kms, width_mode="kernel")
    values = tuple(np.interp(wave, grid, item, left=0., right=0.) for item in (flux, derivative))
    return tuple(_apply_coverage_taper(template, np.asarray(wave), item) for item in values) if taper else values


def regional_weights(wave, uv_interval, optical_interval):
    """C2 partition of unity for ordered, nonoverlapping handoffs."""
    b0, b1 = uv_interval
    o0, o1 = optical_interval
    if not 0 < b0 < b1 <= o0 < o1:
        raise ValueError("Regional handoffs must be positive, ordered and nonoverlapping")
    def smooth(interval):
        x = np.clip((np.asarray(wave)-interval[0])/(interval[1]-interval[0]), 0., 1.)
        return x**3*(10.+x*(-15.+6.*x))
    u, o = 1.-smooth(uv_interval), smooth(optical_interval)
    return u, np.maximum(0., 1.-u-o), o


def resolve_regional_intervals(uv, optical, uv_max_kernel, optical_max_kernel,
                               uv_interval=(3300.,3450.), optical_interval=(4100.,4250.)):
    """Resolve fixed four-sigma support guards from maximum allowed kernels."""
    if not np.isfinite(uv_max_kernel + optical_max_kernel):
        raise ValueError("A regional bridge requires finite maximum kernels")
    upper = min(uv_interval[1], uv.coverage[1]*np.exp(-4*uv_max_kernel/(FWHM_TO_SIGMA*C_KMS)))
    lower = max(optical_interval[0], optical.coverage[0]*np.exp(4*optical_max_kernel/(FWHM_TO_SIGMA*C_KMS)))
    return (upper-(uv_interval[1]-uv_interval[0]), upper), (lower, lower+(optical_interval[1]-optical_interval[0]))
