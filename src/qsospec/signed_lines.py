"""Signed fixed-profile measurements for local spectral diagnostics.

The functions in this module deliberately perform only linear measurements.
They do not interpret line identities or apply survey-specific thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from . import lines
from .spectrum import Spectrum, require_rest_frame_flux

C_KMS = 299792.458
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class SignedLineComponent:
    """One fixed-profile component in a local linear pattern."""

    label: str
    rest_wavelength: float | None = None
    line_id: str | None = None
    fwhm_kms: float | None = None
    velocity_kms: float | None = None
    fixed_ratio_to: str | None = None
    fixed_ratio: float | None = None
    non_negative: bool = False

    def wavelength(self) -> float:
        if self.line_id is not None:
            return float(lines.get(self.line_id).vacuum_wavelength)
        if self.rest_wavelength is None:
            raise ValueError(f"Component {self.label!r} needs line_id or rest_wavelength")
        value = float(self.rest_wavelength)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid rest wavelength for {self.label!r}: {value}")
        return value


@dataclass(frozen=True)
class SignedLineAmplitudeResult:
    """Result of a fixed-centre, fixed-width signed line measurement."""

    success: bool
    status: str
    message: str
    rest_wavelength: float
    velocity_kms: float
    fwhm_kms: float
    flux: float
    flux_error: float
    snr: float
    baseline_coefficients: tuple[float, ...]
    chi2: float
    dof: int
    reduced_chi2: float
    n_valid_pixels: int
    coverage_fraction: float
    fit_mask: np.ndarray
    model: np.ndarray
    line_model: np.ndarray
    baseline_model: np.ndarray
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalLinePatternResult:
    """Result of a fixed-kinematics local line-pattern measurement."""

    success: bool
    status: str
    message: str
    component_fluxes: Mapping[str, float]
    component_flux_errors: Mapping[str, float]
    component_snrs: Mapping[str, float]
    baseline_coefficients: tuple[float, ...]
    chi2: float
    dof: int
    reduced_chi2: float
    bic: float
    n_parameters: int
    n_valid_pixels: int
    coverage_fraction: float
    fit_mask: np.ndarray
    model: np.ndarray
    line_model: np.ndarray
    baseline_model: np.ndarray
    component_models: Mapping[str, np.ndarray]
    covariance: np.ndarray | None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _continuum_array(continuum: Any, shape: tuple[int, ...]) -> np.ndarray:
    values = continuum.model if hasattr(continuum, "model") else continuum
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"continuum shape {array.shape} does not match spectrum {shape}")
    return array


def _window_mask(wave: np.ndarray, windows: Sequence[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(wave.shape, dtype=bool)
    for lower, upper in windows:
        lo, hi = float(lower), float(upper)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid fit window {(lower, upper)!r}")
        mask |= (wave >= lo) & (wave <= hi)
    return mask


def _profile(
    wave: np.ndarray,
    rest_wavelength: float,
    fwhm_kms: float,
    velocity_kms: float,
) -> np.ndarray:
    fwhm = float(fwhm_kms)
    velocity = float(velocity_kms)
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("fwhm_kms must be positive and finite")
    if not np.isfinite(velocity):
        raise ValueError("velocity_kms must be finite")
    center = float(rest_wavelength) * (1.0 + velocity / C_KMS)
    sigma = center * fwhm / (C_KMS * FWHM_TO_SIGMA)
    u = (np.asarray(wave, dtype=float) - center) / sigma
    # Unit-integral Gaussian: fitted coefficient is integrated line flux.
    return np.exp(-0.5 * u * u) / (np.sqrt(2.0 * np.pi) * sigma)


def _failure_pattern(
    spectrum: Spectrum,
    *,
    status: str,
    message: str,
    fit_mask: np.ndarray,
    coverage_fraction: float,
    metadata: Mapping[str, Any],
    warnings: Sequence[str] = (),
) -> LocalLinePatternResult:
    nan_model = np.full(spectrum.flux.shape, np.nan, dtype=float)
    return LocalLinePatternResult(
        success=False,
        status=status,
        message=message,
        component_fluxes={},
        component_flux_errors={},
        component_snrs={},
        baseline_coefficients=(),
        chi2=np.nan,
        dof=0,
        reduced_chi2=np.nan,
        bic=np.nan,
        n_parameters=0,
        n_valid_pixels=int(np.count_nonzero(fit_mask)),
        coverage_fraction=float(coverage_fraction),
        fit_mask=fit_mask,
        model=nan_model.copy(),
        line_model=nan_model.copy(),
        baseline_model=nan_model.copy(),
        component_models={},
        covariance=None,
        warnings=tuple(warnings),
        metadata=dict(metadata),
    )


def fit_local_line_pattern(
    spectrum: Spectrum,
    continuum: Any,
    components: Sequence[SignedLineComponent],
    *,
    fwhm_kms: float,
    velocity_kms: float = 0.0,
    baseline: str = "linear",
    fit_windows: Sequence[tuple[float, float]] | None = None,
    excluded_mask: np.ndarray | None = None,
    minimum_valid_pixels: int = 5,
) -> LocalLinePatternResult:
    """Fit a fixed line pattern and local baseline by weighted linear LS.

    Independent component amplitudes are signed unless ``non_negative`` is
    explicitly requested. Components with ``fixed_ratio_to`` are folded into
    the referenced component's design column and add no free parameter.
    """

    require_rest_frame_flux(spectrum)
    if not components:
        raise ValueError("components must not be empty")
    labels = [str(component.label) for component in components]
    if len(set(labels)) != len(labels):
        raise ValueError("component labels must be unique")
    by_label = {component.label: component for component in components}
    for component in components:
        if component.fixed_ratio_to is not None:
            if component.fixed_ratio_to not in by_label:
                raise ValueError(
                    f"Unknown fixed_ratio_to={component.fixed_ratio_to!r} for {component.label!r}"
                )
            ratio = component.fixed_ratio
            if ratio is None or not np.isfinite(float(ratio)):
                raise ValueError(f"Component {component.label!r} needs a finite fixed_ratio")
            if by_label[component.fixed_ratio_to].fixed_ratio_to is not None:
                raise ValueError("fixed-ratio chains are not supported")

    default_fwhm = float(fwhm_kms)
    default_velocity = float(velocity_kms)
    component_profiles: dict[str, np.ndarray] = {}
    for component in components:
        component_profiles[component.label] = _profile(
            spectrum.wave_rest,
            component.wavelength(),
            default_fwhm if component.fwhm_kms is None else float(component.fwhm_kms),
            default_velocity if component.velocity_kms is None else float(component.velocity_kms),
        )

    if fit_windows is None:
        centres = np.asarray([component.wavelength() for component in components])
        broadest = max(
            default_fwhm if component.fwhm_kms is None else float(component.fwhm_kms)
            for component in components
        )
        pad = max(20.0, 6.0 * float(np.max(centres)) * broadest / (C_KMS * FWHM_TO_SIGMA))
        fit_windows = ((float(np.min(centres) - pad), float(np.max(centres) + pad)),)
    windows = tuple((float(lo), float(hi)) for lo, hi in fit_windows)
    requested = _window_mask(spectrum.wave_rest, windows)
    continuum_array = _continuum_array(continuum, spectrum.flux.shape)
    valid = requested & spectrum.valid_mask & np.isfinite(continuum_array)
    if excluded_mask is not None:
        excluded = np.asarray(excluded_mask, dtype=bool)
        if excluded.shape != spectrum.flux.shape:
            raise ValueError("excluded_mask must match the spectrum shape")
        valid &= ~excluded
    n_requested = int(np.count_nonzero(requested))
    coverage_fraction = float(np.count_nonzero(valid) / n_requested) if n_requested else 0.0
    metadata = {
        "fit_windows": [list(window) for window in windows],
        "baseline": str(baseline),
        "default_fwhm_kms": default_fwhm,
        "default_velocity_kms": default_velocity,
        "component_definitions": [
            {
                "label": component.label,
                "rest_wavelength": component.wavelength(),
                "fwhm_kms": default_fwhm if component.fwhm_kms is None else float(component.fwhm_kms),
                "velocity_kms": default_velocity if component.velocity_kms is None else float(component.velocity_kms),
                "fixed_ratio_to": component.fixed_ratio_to,
                "fixed_ratio": component.fixed_ratio,
                "non_negative": bool(component.non_negative),
            }
            for component in components
        ],
    }
    if n_requested == 0:
        return _failure_pattern(
            spectrum,
            status="not_covered",
            message="No spectral pixels fall in the requested window.",
            fit_mask=valid,
            coverage_fraction=0.0,
            metadata=metadata,
        )
    if np.count_nonzero(valid) < int(minimum_valid_pixels):
        return _failure_pattern(
            spectrum,
            status="insufficient_valid_pixels",
            message="Too few finite, unmasked, positive-error pixels in the requested window.",
            fit_mask=valid,
            coverage_fraction=coverage_fraction,
            metadata=metadata,
        )

    free_components = [component for component in components if component.fixed_ratio_to is None]
    columns: list[np.ndarray] = []
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for component in free_components:
        column = component_profiles[component.label].copy()
        for tied in components:
            if tied.fixed_ratio_to == component.label:
                column += float(tied.fixed_ratio) * component_profiles[tied.label]
        columns.append(column[valid])
        names.append(component.label)
        lower.append(0.0 if component.non_negative else -np.inf)
        upper.append(np.inf)

    wave_valid = spectrum.wave_rest[valid]
    pivot = float(np.mean(wave_valid))
    scale = max(float(np.ptp(wave_valid)), 1.0)
    baseline_mode = str(baseline).lower()
    if baseline_mode == "constant":
        columns.append(np.ones(wave_valid.size))
        names.append("baseline_constant")
        lower.append(-np.inf)
        upper.append(np.inf)
    elif baseline_mode == "linear":
        columns.extend([np.ones(wave_valid.size), (wave_valid - pivot) / scale])
        names.extend(["baseline_constant", "baseline_slope"])
        lower.extend([-np.inf, -np.inf])
        upper.extend([np.inf, np.inf])
    elif baseline_mode != "none":
        raise ValueError("baseline must be 'none', 'constant', or 'linear'")

    design = np.column_stack(columns)
    target = spectrum.flux[valid] - continuum_array[valid]
    error = spectrum.err[valid]
    weighted_design = design / error[:, None]
    weighted_target = target / error
    n_parameters = int(design.shape[1])
    if design.shape[0] <= n_parameters:
        return _failure_pattern(
            spectrum,
            status="insufficient_valid_pixels",
            message="The requested window has no positive degrees of freedom.",
            fit_mask=valid,
            coverage_fraction=coverage_fraction,
            metadata=metadata,
        )

    constrained = any(np.isfinite(lower)) or any(np.isfinite(upper))
    if constrained:
        solved = lsq_linear(
            weighted_design,
            weighted_target,
            bounds=(np.asarray(lower), np.asarray(upper)),
            method="trf",
        )
        coefficients = np.asarray(solved.x, dtype=float)
        success = bool(solved.success)
        message = str(solved.message)
    else:
        coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        success = int(rank) == n_parameters
        message = "weighted linear least squares"

    normal = weighted_design.T @ weighted_design
    covariance = np.linalg.pinv(normal, hermitian=True)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    model_valid = design @ coefficients
    residual = (target - model_valid) / error
    chi2 = float(np.sum(residual**2))
    dof = int(design.shape[0] - n_parameters)
    reduced_chi2 = float(chi2 / dof) if dof > 0 else np.nan
    bic = float(chi2 + n_parameters * np.log(design.shape[0]))

    coefficient_map = dict(zip(names, coefficients))
    error_map = dict(zip(names, errors))
    full_component_models: dict[str, np.ndarray] = {}
    component_fluxes: dict[str, float] = {}
    component_errors: dict[str, float] = {}
    component_snrs: dict[str, float] = {}
    free_index = {component.label: index for index, component in enumerate(free_components)}
    for component in components:
        parent = component.fixed_ratio_to or component.label
        ratio = float(component.fixed_ratio) if component.fixed_ratio_to is not None else 1.0
        value = float(coefficient_map[parent] * ratio)
        uncertainty = float(error_map[parent] * abs(ratio))
        component_fluxes[component.label] = value
        component_errors[component.label] = uncertainty
        component_snrs[component.label] = value / uncertainty if uncertainty > 0 else np.nan
        full_component_models[component.label] = value * component_profiles[component.label]

    baseline_coefficients = tuple(
        float(coefficient_map[name]) for name in names if name.startswith("baseline_")
    )
    baseline_full = np.zeros(spectrum.flux.shape, dtype=float)
    if "baseline_constant" in coefficient_map:
        baseline_full += coefficient_map["baseline_constant"]
    if "baseline_slope" in coefficient_map:
        baseline_full += coefficient_map["baseline_slope"] * (spectrum.wave_rest - pivot) / scale
    line_full = sum(full_component_models.values(), np.zeros(spectrum.flux.shape, dtype=float))
    model_full = line_full + baseline_full
    outside = ~requested
    for array in (line_full, baseline_full, model_full):
        array[outside] = np.nan

    warnings: list[str] = []
    status = "ok"
    requested_bounds = [(lo, hi) for lo, hi in windows]
    valid_wave = spectrum.wave_rest[spectrum.valid_mask]
    if valid_wave.size and any(lo < valid_wave.min() or hi > valid_wave.max() for lo, hi in windows):
        status = "partial_coverage"
        warnings.append("partial_coverage")
    if not success:
        status = "fit_failed"
        warnings.append("rank_deficient_or_solver_failure")
    metadata = {
        **metadata,
        "parameter_names": names,
        "pivot_wavelength": pivot,
        "baseline_scale": scale,
        "requested_bounds": requested_bounds,
        "covariance_note": "Known per-pixel errors; covariance is inverse weighted normal matrix.",
        "free_component_indices": free_index,
    }
    return LocalLinePatternResult(
        success=success,
        status=status,
        message=message,
        component_fluxes=component_fluxes,
        component_flux_errors=component_errors,
        component_snrs=component_snrs,
        baseline_coefficients=baseline_coefficients,
        chi2=chi2,
        dof=dof,
        reduced_chi2=reduced_chi2,
        bic=bic,
        n_parameters=n_parameters,
        n_valid_pixels=int(design.shape[0]),
        coverage_fraction=coverage_fraction,
        fit_mask=valid,
        model=model_full,
        line_model=line_full,
        baseline_model=baseline_full,
        component_models=full_component_models,
        covariance=covariance,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def measure_signed_line_amplitude(
    spectrum: Spectrum,
    continuum: Any,
    rest_wavelength: float,
    fwhm_kms: float,
    *,
    velocity_kms: float = 0.0,
    baseline: str = "linear",
    fit_window: tuple[float, float] | None = None,
    excluded_mask: np.ndarray | None = None,
    profile: str = "gaussian",
    minimum_valid_pixels: int = 5,
) -> SignedLineAmplitudeResult:
    """Measure a fixed Gaussian line with an unbiased signed amplitude."""

    if str(profile).lower() != "gaussian":
        raise ValueError("Only profile='gaussian' is currently supported")
    component = SignedLineComponent(
        label="line",
        rest_wavelength=float(rest_wavelength),
        fwhm_kms=float(fwhm_kms),
        velocity_kms=float(velocity_kms),
    )
    windows = None if fit_window is None else (fit_window,)
    pattern = fit_local_line_pattern(
        spectrum,
        continuum,
        (component,),
        fwhm_kms=float(fwhm_kms),
        velocity_kms=float(velocity_kms),
        baseline=baseline,
        fit_windows=windows,
        excluded_mask=excluded_mask,
        minimum_valid_pixels=minimum_valid_pixels,
    )
    return SignedLineAmplitudeResult(
        success=pattern.success,
        status=pattern.status,
        message=pattern.message,
        rest_wavelength=float(rest_wavelength),
        velocity_kms=float(velocity_kms),
        fwhm_kms=float(fwhm_kms),
        flux=float(pattern.component_fluxes.get("line", np.nan)),
        flux_error=float(pattern.component_flux_errors.get("line", np.nan)),
        snr=float(pattern.component_snrs.get("line", np.nan)),
        baseline_coefficients=pattern.baseline_coefficients,
        chi2=pattern.chi2,
        dof=pattern.dof,
        reduced_chi2=pattern.reduced_chi2,
        n_valid_pixels=pattern.n_valid_pixels,
        coverage_fraction=pattern.coverage_fraction,
        fit_mask=pattern.fit_mask,
        model=pattern.model,
        line_model=pattern.component_models.get("line", pattern.line_model),
        baseline_model=pattern.baseline_model,
        warnings=pattern.warnings,
        metadata=pattern.metadata,
    )


def measure_signed_line_grid(
    spectrum: Spectrum,
    continuum: Any,
    rest_wavelength: float,
    fwhm_grid_kms: Sequence[float],
    *,
    velocity_grid_kms: Sequence[float] = (0.0,),
    baseline: str = "linear",
    fit_window: tuple[float, float] | None = None,
    excluded_mask: np.ndarray | None = None,
    minimum_valid_pixels: int = 5,
) -> tuple[SignedLineAmplitudeResult, tuple[Mapping[str, float | str | bool], ...]]:
    """Evaluate a deterministic fixed width/velocity grid and select min-BIC."""

    trials: list[tuple[SignedLineAmplitudeResult, float]] = []
    records: list[Mapping[str, float | str | bool]] = []
    for fwhm in fwhm_grid_kms:
        for velocity in velocity_grid_kms:
            result = measure_signed_line_amplitude(
                spectrum,
                continuum,
                rest_wavelength,
                float(fwhm),
                velocity_kms=float(velocity),
                baseline=baseline,
                fit_window=fit_window,
                excluded_mask=excluded_mask,
                minimum_valid_pixels=minimum_valid_pixels,
            )
            n_parameters = 1 + len(result.baseline_coefficients)
            # Width/velocity grid selection contributes one discrete effective
            # parameter when comparing to a fixed single-trial measurement.
            bic = (
                result.chi2 + (n_parameters + 1) * np.log(result.n_valid_pixels)
                if result.success and result.n_valid_pixels > 0
                else np.inf
            )
            trials.append((result, float(bic)))
            records.append(
                {
                    "fwhm_kms": float(fwhm),
                    "velocity_kms": float(velocity),
                    "success": bool(result.success),
                    "status": result.status,
                    "flux": result.flux,
                    "flux_error": result.flux_error,
                    "snr": result.snr,
                    "bic_with_grid_penalty": float(bic),
                }
            )
    if not trials:
        raise ValueError("fwhm_grid_kms and velocity_grid_kms must not be empty")
    selected = min(trials, key=lambda item: item[1])[0]
    return selected, tuple(records)
