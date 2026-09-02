"""Nonrecursive broad-Balmer prefit for host-template width selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...config import (
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    UncertaintyConfig,
)
from ...fitting.global_fit import fit_global_lines
from ...global_result import EmissionComplexResult
from ...spectrum import Spectrum
from .config import HostBroadLinePrefitConfig


@dataclass(frozen=True)
class HostBroadLinePrefitResult:
    """Auditable outcome of the nuisance broad-line prefit."""

    status: str
    selected_line: str | None
    fwhm_kms: float | None
    fwhm_error_kms: float | None
    fwhm_snr: float | None
    flux: float | None
    flux_error: float | None
    flux_snr: float | None
    velocity_kms: float | None
    at_parameter_bound: bool
    selected_width_grid_kms: float | None
    fallback_used: bool
    fallback_reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def nearest_width_grid_value(width_kms: float, grid: Iterable[float]) -> float:
    """Return the nearest grid value, choosing the lower value on a tie."""

    width = float(width_kms)
    values = tuple(float(value) for value in grid)
    if not np.isfinite(width) or width <= 0 or not values:
        raise ValueError("A positive width and non-empty grid are required.")
    return min(values, key=lambda value: (abs(value - width), value))


def _bound_parameters(result: EmissionComplexResult) -> tuple[str, ...]:
    parameters = []
    for warning in result.warnings:
        if warning.code != "parameter_at_bound":
            continue
        parameter = str(warning.context.get("parameter", ""))
        if "broad" in parameter.lower() and ("fwhm" in parameter.lower() or "velocity" in parameter.lower()):
            parameters.append(parameter)
    return tuple(parameters)


def _line_diagnostic(
    line: str,
    result: EmissionComplexResult | None,
    config: HostBroadLinePrefitConfig,
) -> dict[str, Any]:
    prefix = "Ha" if line == "halpha" else "Hb"
    if result is None:
        return {"line": line, "usable": False, "reason": "not_selected_or_not_covered"}
    flux = float(result.metrics.get(f"{prefix}_broad_flux_input", np.nan))
    flux_error = float(result.metric_errors.get(f"{prefix}_broad_flux_input", np.nan))
    fwhm = float(result.metrics.get(f"{prefix}_broad_fwhm_kms", np.nan))
    fwhm_error = float(result.metric_errors.get(f"{prefix}_broad_fwhm_kms", np.nan))
    velocity = float(result.metrics.get(f"{prefix}_broad_velocity_kms", np.nan))
    flux_snr = flux / flux_error if np.isfinite(flux_error) and flux_error > 0 else np.nan
    fwhm_snr = fwhm / fwhm_error if np.isfinite(fwhm_error) and fwhm_error > 0 else np.nan
    bound_parameters = _bound_parameters(result)
    reason = "reliable"
    usable = bool(result.success)
    if not result.success:
        reason = "fit_failed"
    elif not np.isfinite(flux) or flux <= 0:
        usable, reason = False, "nonpositive_flux"
    elif not np.isfinite(fwhm) or fwhm <= 0:
        usable, reason = False, "nonpositive_fwhm"
    elif not np.isfinite(flux_snr) or flux_snr < config.minimum_flux_snr:
        usable, reason = False, "low_flux_snr"
    elif not np.isfinite(fwhm_snr) or fwhm_snr < config.minimum_fwhm_snr:
        usable, reason = False, "low_fwhm_snr"
    elif config.reject_parameter_bounds and bound_parameters:
        usable, reason = False, "broad_kinematics_at_bound"
    return {
        "line": line,
        "usable": bool(usable),
        "reason": reason,
        "fit_success": bool(result.success),
        "flux": flux,
        "flux_error": flux_error,
        "flux_snr": float(flux_snr),
        "fwhm_kms": fwhm,
        "fwhm_error_kms": fwhm_error,
        "fwhm_snr": float(fwhm_snr),
        "velocity_kms": velocity,
        "bound_parameters": bound_parameters,
        "coverage_status": result.metadata.get("coverage_status"),
        "warning_codes": result.warning_codes(),
    }


def _fallback_result(
    config: HostBroadLinePrefitConfig,
    width_grid_kms: Iterable[float],
    *,
    reason: str,
    diagnostics: dict[str, Any],
) -> HostBroadLinePrefitResult:
    if config.fallback_policy == "fail":
        raise RuntimeError(f"No reliable broad-Balmer prefit: {reason}")
    selected = None
    status = "fallback_masked_simple"
    if config.fallback_policy == "fixed_width":
        selected = nearest_width_grid_value(float(config.fixed_fallback_fwhm_kms), width_grid_kms)
        status = "fallback_fixed_width"
    return HostBroadLinePrefitResult(
        status=status,
        selected_line=None,
        fwhm_kms=None,
        fwhm_error_kms=None,
        fwhm_snr=None,
        flux=None,
        flux_error=None,
        flux_snr=None,
        velocity_kms=None,
        at_parameter_bound=False,
        selected_width_grid_kms=selected,
        fallback_used=True,
        fallback_reason=reason,
        diagnostics=diagnostics,
    )


def run_host_broad_line_prefit(
    spectrum: Spectrum,
    *,
    config: HostBroadLinePrefitConfig | None = None,
    width_grid_kms: Iterable[float],
    global_config: GlobalContinuumConfig | None = None,
    hbeta_config: HbetaComplexConfig | None = None,
    halpha_config: HalphaComplexConfig | None = None,
) -> HostBroadLinePrefitResult:
    """Fit Hα/Hβ directly with qsospec and select a nuisance width.

    This helper calls only the low-level global fitter. It never invokes the
    optional-host workflow and therefore cannot recurse into pPXF.
    """

    cfg = config or HostBroadLinePrefitConfig()
    if not cfg.enabled:
        return _fallback_result(
            cfg,
            width_grid_kms,
            reason="prefit_disabled",
            diagnostics={"prefit_enabled": False},
        )
    requested_recipes = tuple("halpha_nii_sii" if line == "halpha" else "hbeta_oiii" for line in cfg.preferred_lines)
    try:
        workflow = fit_global_lines(
            spectrum,
            global_config,
            hbeta_config,
            None,
            halpha_config,
            UncertaintyConfig(covariance=True, monte_carlo_trials=0),
            complexes=requested_recipes,
        )
    except Exception as exc:  # noqa: BLE001 - optimizer failures trigger policy
        return _fallback_result(
            cfg,
            width_grid_kms,
            reason="prefit_exception",
            diagnostics={"exception": f"{type(exc).__name__}: {exc}"},
        )
    line_results = {
        "halpha": workflow.line_complexes.get("halpha_nii_sii"),
        "hbeta": workflow.line_complexes.get("hbeta_oiii"),
    }
    diagnostics = {line: _line_diagnostic(line, line_results.get(line), cfg) for line in cfg.preferred_lines}
    for line in cfg.preferred_lines:
        diagnostic = diagnostics[line]
        if not diagnostic["usable"]:
            continue
        selected_width = nearest_width_grid_value(diagnostic["fwhm_kms"], width_grid_kms)
        return HostBroadLinePrefitResult(
            status="success",
            selected_line=line,
            fwhm_kms=float(diagnostic["fwhm_kms"]),
            fwhm_error_kms=float(diagnostic["fwhm_error_kms"]),
            fwhm_snr=float(diagnostic["fwhm_snr"]),
            flux=float(diagnostic["flux"]),
            flux_error=float(diagnostic["flux_error"]),
            flux_snr=float(diagnostic["flux_snr"]),
            velocity_kms=float(diagnostic["velocity_kms"]),
            at_parameter_bound=bool(diagnostic["bound_parameters"]),
            selected_width_grid_kms=float(selected_width),
            fallback_used=False,
            fallback_reason=None,
            diagnostics=diagnostics,
        )
    return _fallback_result(
        cfg,
        width_grid_kms,
        reason="no_reliable_broad_line",
        diagnostics=diagnostics,
    )
