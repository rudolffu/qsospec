"""Instrument-aware H-alpha model comparison for narrow-line classification.

This module is intentionally opt-in.  The established ``fit_halpha_complex``
API continues to report observed widths and to use its legacy three-broad-
component configuration.  The helpers here compare a narrow-only model with a
single flexible broad component by default, after converting intrinsic
velocity-width bounds to the observed frame for a Gaussian constant-R LSF.
Two- and three-broad-component alternatives remain opt-in diagnostics.

No physical class is assigned here.  In particular, the broad H-alpha flux
fraction is an uncalibrated continuous diagnostic until injection/recovery and
independent high-resolution decompositions justify a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import HalphaComplexConfig
from .fitting.global_fit import fit_halpha_complex
from .global_result import EmissionComplexResult, GlobalContinuumResult
from .narrow_line_evidence import evaluate_provisional_narrow_evidence
from .spectrum import Spectrum


C_KMS = 299792.458


def _residual_statistics(result: EmissionComplexResult) -> Dict[str, float]:
    mask = np.asarray(result.fit_mask, dtype=bool)
    if not np.any(mask):
        return {
            "residual_rms_sigma": np.nan,
            "residual_median_sigma": np.nan,
            "residual_abs_p95_sigma": np.nan,
            "residual_abs_max_sigma": np.nan,
        }
    residual = (
        result.flux_continuum_subtracted[mask] - result.model[mask]
    ) / result.err[mask]
    finite = residual[np.isfinite(residual)]
    if not finite.size:
        return {
            "residual_rms_sigma": np.nan,
            "residual_median_sigma": np.nan,
            "residual_abs_p95_sigma": np.nan,
            "residual_abs_max_sigma": np.nan,
        }
    return {
        "residual_rms_sigma": float(np.sqrt(np.mean(finite**2))),
        "residual_median_sigma": float(np.median(finite)),
        "residual_abs_p95_sigma": float(
            np.percentile(np.abs(finite), 95.0)
        ),
        "residual_abs_max_sigma": float(np.max(np.abs(finite))),
    }


@dataclass(frozen=True)
class LineSpreadFunctionConfig:
    """Gaussian line-spread function with constant resolving power."""

    kind: str = "constant_resolving_power_gaussian"
    resolving_power: float = 480.0

    def __post_init__(self) -> None:
        if self.kind != "constant_resolving_power_gaussian":
            raise ValueError(
                "LineSpreadFunctionConfig.kind must be "
                "'constant_resolving_power_gaussian'."
            )
        if not np.isfinite(self.resolving_power) or self.resolving_power <= 0:
            raise ValueError("resolving_power must be positive and finite.")

    @property
    def instrumental_fwhm_kms(self) -> float:
        return float(C_KMS / self.resolving_power)


@dataclass(frozen=True)
class HalphaModelSelectionConfig:
    """Configuration for the opt-in RGS H-alpha model grid."""

    lsf: LineSpreadFunctionConfig = field(
        default_factory=LineSpreadFunctionConfig
    )
    intrinsic_narrow_fwhm_bounds_kms: Tuple[float, float] = (0.0, 1200.0)
    intrinsic_broad_component_counts: Tuple[int, ...] = (1,)
    width_upper_sigma: float = 2.0

    def __post_init__(self) -> None:
        lower, upper = self.intrinsic_narrow_fwhm_bounds_kms
        if lower < 0 or upper <= lower:
            raise ValueError(
                "intrinsic_narrow_fwhm_bounds_kms must be non-negative and "
                "increasing."
            )
        counts = tuple(self.intrinsic_broad_component_counts)
        if not counts or len(set(counts)) != len(counts):
            raise ValueError("intrinsic_broad_component_counts must be unique.")
        if any(count not in (1, 2, 3) for count in counts):
            raise ValueError("H-alpha model comparison supports 1-3 broad components.")
        if not np.isfinite(self.width_upper_sigma) or self.width_upper_sigma <= 0:
            raise ValueError("width_upper_sigma must be positive and finite.")


def observed_fwhm_kms(
    intrinsic_fwhm_kms: float,
    instrumental_fwhm_kms: float,
) -> float:
    """Return the Gaussian observed FWHM from intrinsic and LSF widths."""

    intrinsic = float(intrinsic_fwhm_kms)
    instrumental = float(instrumental_fwhm_kms)
    if intrinsic < 0 or instrumental <= 0:
        raise ValueError("FWHM values must be intrinsic >= 0 and instrumental > 0.")
    return float(np.hypot(intrinsic, instrumental))


def intrinsic_fwhm_kms(
    observed_width_kms: float,
    instrumental_fwhm_kms: float,
) -> float:
    """Quadrature-deconvolve a Gaussian LSF, returning zero if unresolved."""

    observed = float(observed_width_kms)
    instrumental = float(instrumental_fwhm_kms)
    if observed < 0 or instrumental <= 0:
        raise ValueError("FWHM values must be observed >= 0 and instrumental > 0.")
    return float(np.sqrt(max(observed * observed - instrumental * instrumental, 0.0)))


def _intrinsic_bands(component_count: int) -> Tuple[Tuple[float, float], ...]:
    if component_count == 1:
        return ((1200.0, 20000.0),)
    if component_count == 2:
        return ((1200.0, 6000.0), (6000.0, 20000.0))
    if component_count == 3:
        return (
            (1200.0, 2500.0),
            (2500.0, 6000.0),
            (6000.0, 20000.0),
        )
    raise ValueError("component_count must be 1, 2, or 3.")


def observed_halpha_width_bounds(
    config: HalphaModelSelectionConfig,
    component_count: int,
) -> tuple[Tuple[float, float], Tuple[Tuple[float, float], ...]]:
    """Return observed-frame narrow and broad bounds for one model."""

    instrumental = config.lsf.instrumental_fwhm_kms
    narrow = tuple(
        observed_fwhm_kms(value, instrumental)
        for value in config.intrinsic_narrow_fwhm_bounds_kms
    )
    broad = tuple(
        (
            observed_fwhm_kms(lower, instrumental),
            observed_fwhm_kms(upper, instrumental),
        )
        for lower, upper in _intrinsic_bands(component_count)
    ) if component_count else ()
    return (float(narrow[0]), float(narrow[1])), broad


@dataclass
class HalphaModelGridResult:
    """Narrow-only and selected broad H-alpha fits on an identical window."""

    candidates: Dict[str, EmissionComplexResult]
    selection_config: HalphaModelSelectionConfig
    best_broad_model: Optional[str]
    minimum_bic_model: Optional[str]
    delta_bic_broad: float

    @property
    def narrow(self) -> EmissionComplexResult:
        return self.candidates["N0"]

    @property
    def best_broad(self) -> Optional[EmissionComplexResult]:
        if self.best_broad_model is None:
            return None
        return self.candidates[self.best_broad_model]

    def to_record(self, object_id: object) -> Dict[str, object]:
        """Return one flat, provenance-friendly model-comparison row."""

        instrumental = self.selection_config.lsf.instrumental_fwhm_kms
        narrow = self.narrow
        observed = float(narrow.param_values.get("narrow.fwhm_kms", np.nan))
        observed_error = float(
            narrow.param_errors.get("narrow.fwhm_kms", np.nan)
        )
        intrinsic = (
            intrinsic_fwhm_kms(observed, instrumental)
            if np.isfinite(observed)
            else np.nan
        )
        upper_observed = (
            observed + self.selection_config.width_upper_sigma * observed_error
            if np.isfinite(observed) and np.isfinite(observed_error)
            else np.nan
        )
        intrinsic_upper = (
            intrinsic_fwhm_kms(max(upper_observed, 0.0), instrumental)
            if np.isfinite(upper_observed)
            else np.nan
        )

        broad = self.best_broad
        broad_flux = (
            float(broad.metrics.get("Ha_broad_flux_input", np.nan))
            if broad is not None else np.nan
        )
        narrow_flux = (
            float(broad.metrics.get("Ha_narrow_flux_input", np.nan))
            if broad is not None else np.nan
        )
        denominator = broad_flux + narrow_flux
        broad_fraction = (
            broad_flux / denominator
            if np.isfinite(denominator) and denominator > 0
            else np.nan
        )
        broad_to_narrow = (
            broad_flux / narrow_flux
            if np.isfinite(narrow_flux) and narrow_flux > 0
            else np.nan
        )
        broad_error = (
            float(broad.metric_errors.get("Ha_broad_flux_input", np.nan))
            if broad is not None else np.nan
        )
        broad_observed_width = (
            float(broad.param_values.get("Ha_broad1.fwhm_kms", np.nan))
            if broad is not None else np.nan
        )
        broad_observed_width_error = (
            float(broad.param_errors.get("Ha_broad1.fwhm_kms", np.nan))
            if broad is not None else np.nan
        )
        broad_intrinsic_width = (
            intrinsic_fwhm_kms(broad_observed_width, instrumental)
            if np.isfinite(broad_observed_width) else np.nan
        )
        narrow_error = (
            float(broad.metric_errors.get("Ha_narrow_flux_input", np.nan))
            if broad is not None else np.nan
        )
        narrow_only_flux = float(
            narrow.metrics.get("Ha_narrow_flux_input", np.nan)
        )
        narrow_only_error = float(
            narrow.metric_errors.get("Ha_narrow_flux_input", np.nan)
        )

        narrow_bound_hit = False
        for warning in narrow.warnings:
            if warning.code != "parameter_at_bound":
                continue
            parameter = str(warning.context.get("parameter", ""))
            side = int(warning.context.get("bound_side", 0) or 0)
            if parameter == "narrow.velocity_kms":
                narrow_bound_hit = True
            if parameter == "narrow.fwhm_kms" and side > 0:
                narrow_bound_hit = True

        record: Dict[str, object] = {
            "object_id": str(object_id),
            "complex_name": "halpha",
            "fit_status": (
                "complete"
                if all(result.success for result in self.candidates.values())
                else "partial_failure"
            ),
            "minimum_bic_model": self.minimum_bic_model,
            "best_broad_model": self.best_broad_model,
            "delta_bic_broad": self.delta_bic_broad,
            "resolving_power": self.selection_config.lsf.resolving_power,
            "instrumental_fwhm_kms": instrumental,
            "intrinsic_narrow_boundary_kms": (
                self.selection_config.intrinsic_narrow_fwhm_bounds_kms[1]
            ),
            "observed_narrow_boundary_kms": observed_fwhm_kms(
                self.selection_config.intrinsic_narrow_fwhm_bounds_kms[1],
                instrumental,
            ),
            "narrow_fwhm_observed_kms": observed,
            "narrow_fwhm_observed_error_kms": observed_error,
            "narrow_fwhm_intrinsic_kms": intrinsic,
            "narrow_fwhm_intrinsic_upper_2sigma_kms": intrinsic_upper,
            "narrow_width_secure_below_boundary": bool(
                np.isfinite(intrinsic_upper)
                and intrinsic_upper
                < self.selection_config.intrinsic_narrow_fwhm_bounds_kms[1]
            ),
            "narrow_kinematic_bound_hit": narrow_bound_hit,
            "narrow_line_id": "halpha",
            "narrow_line_flux": narrow_only_flux,
            "narrow_line_flux_error": narrow_only_error,
            "narrow_line_snr": (
                narrow_only_flux / narrow_only_error
                if np.isfinite(narrow_only_error) and narrow_only_error > 0
                else np.nan
            ),
            "narrow_halpha_flux_narrow_model": narrow_only_flux,
            "narrow_halpha_flux_error_narrow_model": narrow_only_error,
            "narrow_halpha_snr_narrow_model": (
                narrow_only_flux / narrow_only_error
                if np.isfinite(narrow_only_error) and narrow_only_error > 0
                else np.nan
            ),
            "broad_halpha_flux_best_broad_model": broad_flux,
            "broad_halpha_flux_error_best_broad_model": broad_error,
            "narrow_halpha_flux_best_broad_model": narrow_flux,
            "narrow_halpha_flux_error_best_broad_model": narrow_error,
            "broad_halpha_snr_best_broad_model": (
                broad_flux / broad_error
                if np.isfinite(broad_error) and broad_error > 0
                else np.nan
            ),
            "broad_line_snr": (
                broad_flux / broad_error
                if np.isfinite(broad_error) and broad_error > 0
                else np.nan
            ),
            "broad_fwhm_observed_kms": broad_observed_width,
            "broad_fwhm_observed_error_kms": broad_observed_width_error,
            "broad_fwhm_intrinsic_kms": broad_intrinsic_width,
            "broad_halpha_fraction_point": broad_fraction,
            "broad_to_narrow_halpha_ratio": broad_to_narrow,
            "broad_halpha_fraction_upper_95": np.nan,
            "broad_fraction_status": "not_calibrated_profile_limit_not_run",
            "selection_status": "not_calibrated",
            "physical_class": None,
            "lsf_assumption": (
                "Gaussian constant-R effective-resolution approximation; fitted widths "
                "remain observed-frame widths"
            ),
        }
        for model_name, result in self.candidates.items():
            prefix = model_name.lower()
            record[f"{prefix}_success"] = bool(result.success)
            record[f"{prefix}_bic"] = float(result.bic)
            record[f"{prefix}_chi2"] = float(result.chi2)
            record[f"{prefix}_dof"] = int(result.dof)
            record[f"{prefix}_selected_model"] = result.selected_model
            record.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in _residual_statistics(result).items()
                }
            )
        record.update(
            evaluate_provisional_narrow_evidence(
                record,
                broad_snr_fields=("broad_line_snr",),
            )
        )
        return record


def fit_halpha_model_grid(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    *,
    base_config: Optional[HalphaComplexConfig] = None,
    selection_config: Optional[HalphaModelSelectionConfig] = None,
    compute_covariance: bool = True,
) -> HalphaModelGridResult:
    """Fit narrow-only and configured broad H-alpha alternatives.

    Width parameters remain observed-frame values, matching qsospec's existing
    output contract.  The bounds are obtained by quadrature-combining the
    requested intrinsic bounds with the Gaussian instrumental FWHM.  The
    production classification default is the economical N0-versus-B1 pair;
    B2/B3 remain available only for targeted diagnostic refits.
    """

    cfg = selection_config or HalphaModelSelectionConfig()
    base = base_config or HalphaComplexConfig()
    counts: Sequence[int] = (0, *cfg.intrinsic_broad_component_counts)
    candidates: Dict[str, EmissionComplexResult] = {}
    for count in counts:
        narrow_bounds, broad_bounds = observed_halpha_width_bounds(cfg, count)
        candidate_config = replace(
            base,
            narrow_fwhm_bounds_kms=narrow_bounds,
            broad_fwhm_bands_kms=broad_bounds,
        )
        name = f"B{count}" if count else "N0"
        result = fit_halpha_complex(
            spectrum,
            continuum_result,
            candidate_config,
            compute_covariance=compute_covariance,
        )
        result.metadata.update(
            {
                "model_grid_name": name,
                "width_parameterization": "observed",
                "intrinsic_width_bounds_transformed_by_gaussian_lsf": True,
                "resolving_power": cfg.lsf.resolving_power,
                "instrumental_fwhm_kms": cfg.lsf.instrumental_fwhm_kms,
                "intrinsic_narrow_fwhm_bounds_kms": (
                    cfg.intrinsic_narrow_fwhm_bounds_kms
                ),
                "observed_narrow_fwhm_bounds_kms": narrow_bounds,
                "intrinsic_broad_fwhm_bands_kms": (
                    _intrinsic_bands(count) if count else ()
                ),
                "observed_broad_fwhm_bands_kms": broad_bounds,
                "broad_fraction_threshold_calibrated": False,
            }
        )
        candidates[name] = result

    finite = {
        name: result.bic
        for name, result in candidates.items()
        if result.success and np.isfinite(result.bic)
    }
    minimum_bic_model = min(finite, key=finite.get) if finite else None
    broad_finite = {name: value for name, value in finite.items() if name != "N0"}
    best_broad_model = min(broad_finite, key=broad_finite.get) if broad_finite else None
    delta_bic = (
        float(candidates["N0"].bic - candidates[best_broad_model].bic)
        if best_broad_model is not None
        and candidates["N0"].success
        and np.isfinite(candidates["N0"].bic)
        else np.nan
    )
    return HalphaModelGridResult(
        candidates=candidates,
        selection_config=cfg,
        best_broad_model=best_broad_model,
        minimum_bic_model=minimum_bic_model,
        delta_bic_broad=delta_bic,
    )


def diagnostic_bic_sweep(
    records,
    anchors: Sequence[float] = (-20.0, -10.0, 0.0, 10.0, 20.0),
):
    """Return diagnostic counts without declaring a physical class."""

    import pandas as pd

    frame = pd.DataFrame(records)
    rows = []
    for threshold in anchors:
        delta = pd.to_numeric(frame.get("delta_bic_broad"), errors="coerce")
        width = frame.get("narrow_width_secure_below_boundary", False)
        width = pd.Series(width, index=frame.index).fillna(False).astype(bool)
        finite = np.isfinite(delta)
        rows.append(
            {
                "delta_bic_broad_threshold": float(threshold),
                "n_finite": int(finite.sum()),
                "n_broad_favored_above_threshold": int(
                    (finite & (delta > threshold)).sum()
                ),
                "n_width_secure_and_broad_not_favored": int(
                    (finite & width & (delta <= threshold)).sum()
                ),
                "selection_status": "diagnostic_not_calibrated",
            }
        )
    return pd.DataFrame(rows)
