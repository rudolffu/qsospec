"""Instrument-aware He I 10833 + Pa-gamma narrow-line model comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .complex_recipes import ComplexRecipe, ComponentRecipe
from .fitting.complexes import fit_generic_complex
from .global_result import EmissionComplexResult, GlobalContinuumResult
from .halpha_classification import (
    LineSpreadFunctionConfig,
    intrinsic_fwhm_kms,
    observed_fwhm_kms,
)
from .narrow_line_evidence import evaluate_provisional_narrow_evidence
from .spectrum import Spectrum
from .warnings import FitWarning

HEI_PGAMMA_WINDOW = (10550.0, 11150.0)


@dataclass(frozen=True)
class HeIPagammaModelSelectionConfig:
    """Configuration for the RGS He I/Pa-gamma N0-versus-B1 pair."""

    lsf: LineSpreadFunctionConfig = field(default_factory=LineSpreadFunctionConfig)
    fit_window: tuple[float, float] = HEI_PGAMMA_WINDOW
    intrinsic_narrow_fwhm_bounds_kms: tuple[float, float] = (0.0, 1200.0)
    intrinsic_broad_fwhm_bounds_kms: tuple[float, float] = (1200.0, 10000.0)
    width_upper_sigma: float = 2.0
    min_coverage_fraction: float = 0.8
    min_valid_pixels: int = 30

    def __post_init__(self) -> None:
        if self.fit_window[1] <= self.fit_window[0]:
            raise ValueError("fit_window must be increasing.")
        for name, bounds in (
            ("intrinsic_narrow_fwhm_bounds_kms", self.intrinsic_narrow_fwhm_bounds_kms),
            ("intrinsic_broad_fwhm_bounds_kms", self.intrinsic_broad_fwhm_bounds_kms),
        ):
            if bounds[0] < 0 or bounds[1] <= bounds[0]:
                raise ValueError(f"{name} must be non-negative and increasing.")
        if self.intrinsic_broad_fwhm_bounds_kms[0] != self.intrinsic_narrow_fwhm_bounds_kms[1]:
            raise ValueError("Narrow and broad intrinsic bounds must share the 1200 km/s boundary.")
        if self.width_upper_sigma <= 0 or not np.isfinite(self.width_upper_sigma):
            raise ValueError("width_upper_sigma must be positive and finite.")
        if not 0 < self.min_coverage_fraction <= 1:
            raise ValueError("min_coverage_fraction must be in (0, 1].")
        if self.min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be positive.")

    @property
    def observed_narrow_bounds_kms(self) -> tuple[float, float]:
        instrument = self.lsf.instrumental_fwhm_kms
        return tuple(
            observed_fwhm_kms(value, instrument)
            for value in self.intrinsic_narrow_fwhm_bounds_kms
        )

    @property
    def observed_broad_bounds_kms(self) -> tuple[float, float]:
        instrument = self.lsf.instrumental_fwhm_kms
        return tuple(
            observed_fwhm_kms(value, instrument)
            for value in self.intrinsic_broad_fwhm_bounds_kms
        )


def _component(
    component_id: str,
    line_id: str,
    role: str,
    group: str,
    bounds: tuple[float, float],
    velocity_bounds: tuple[float, float],
) -> ComponentRecipe:
    return ComponentRecipe(
        id=component_id,
        line_ids=(line_id,),
        role=role,
        flux_bounds=(0.0, None),
        velocity_bounds_kms=velocity_bounds,
        fwhm_bands_kms=(bounds,),
        kinematic_group=group,
    )


def hei_pgamma_classification_recipe(
    config: HeIPagammaModelSelectionConfig | None = None,
    *,
    include_broad: bool,
) -> ComplexRecipe:
    """Build the dedicated local recipe without altering ``paschen_nir``."""

    cfg = config or HeIPagammaModelSelectionConfig()
    components = [
        _component(
            "HeI10833_narrow", "hei_10833", "narrow",
            "hei_pgamma_narrow", cfg.observed_narrow_bounds_kms,
            (-1000.0, 1000.0),
        ),
        _component(
            "Pagamma_narrow", "pagamma", "narrow",
            "hei_pgamma_narrow", cfg.observed_narrow_bounds_kms,
            (-1000.0, 1000.0),
        ),
    ]
    if include_broad:
        components.extend(
            (
                _component(
                    "HeI10833_broad", "hei_10833", "broad",
                    "hei_pgamma_broad", cfg.observed_broad_bounds_kms,
                    (-2000.0, 2000.0),
                ),
                _component(
                    "Pagamma_broad", "pagamma", "broad",
                    "hei_pgamma_broad", cfg.observed_broad_bounds_kms,
                    (-2000.0, 2000.0),
                ),
            )
        )
    return ComplexRecipe(
        id="hei_pgamma_narrow_classification",
        aliases=("hei_pgamma_classification",),
        label="He I 10833 / Pa-gamma narrow-line classification",
        fit_window=cfg.fit_window,
        fit_windows=(cfg.fit_window,),
        mask_windows=(),
        components=tuple(components),
        required_line_ids=("hei_10833", "pagamma"),
        coverage_mode="full",
        min_coverage_fraction=cfg.min_coverage_fraction,
        min_valid_pixels=cfg.min_valid_pixels,
        continuum_mode="fixed_global",
        qa_labels=("hei_10833", "pagamma"),
        auto_enabled=False,
        priority=0,
        backend="generic",
        exclusive_group="hei_pgamma_narrow_classification",
    )


def _metric(result: EmissionComplexResult | None, key: str) -> float:
    return float(result.metrics.get(key, np.nan)) if result is not None else np.nan


def _metric_error(result: EmissionComplexResult | None, key: str) -> float:
    return float(result.metric_errors.get(key, np.nan)) if result is not None else np.nan


def _snr(value: float, error: float) -> float:
    return float(value / error) if np.isfinite(error) and error > 0 else np.nan


def _residual_statistics(result: EmissionComplexResult) -> dict[str, float]:
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
        "residual_abs_p95_sigma": float(np.percentile(np.abs(finite), 95.0)),
        "residual_abs_max_sigma": float(np.max(np.abs(finite))),
    }


def _relevant_narrow_bound_hit(result: EmissionComplexResult) -> bool:
    for warning in result.warnings:
        if warning.code != "parameter_at_bound":
            continue
        parameter = str(warning.context.get("parameter", ""))
        side = int(warning.context.get("bound_side", 0) or 0)
        if parameter == "hei_pgamma_narrow.velocity_kms":
            return True
        if parameter == "hei_pgamma_narrow.fwhm_kms" and side > 0:
            return True
    return False


@dataclass
class HeIPagammaModelPairResult:
    """N0 and B1 fits for the local He I/Pa-gamma window."""

    n0: EmissionComplexResult
    b1: EmissionComplexResult
    selection_config: HeIPagammaModelSelectionConfig

    @property
    def delta_bic_broad(self) -> float:
        if not self.n0.success or not self.b1.success:
            return np.nan
        if not np.isfinite(self.n0.bic) or not np.isfinite(self.b1.bic):
            return np.nan
        return float(self.n0.bic - self.b1.bic)

    def to_record(self, object_id: object) -> dict[str, Any]:
        cfg = self.selection_config
        instrument = cfg.lsf.instrumental_fwhm_kms
        observed = float(
            self.n0.param_values.get("hei_pgamma_narrow.fwhm_kms", np.nan)
        )
        observed_error = float(
            self.n0.param_errors.get("hei_pgamma_narrow.fwhm_kms", np.nan)
        )
        intrinsic = (
            intrinsic_fwhm_kms(observed, instrument)
            if np.isfinite(observed) else np.nan
        )
        observed_upper = (
            observed + cfg.width_upper_sigma * observed_error
            if np.isfinite(observed) and np.isfinite(observed_error) else np.nan
        )
        intrinsic_upper = (
            intrinsic_fwhm_kms(max(observed_upper, 0.0), instrument)
            if np.isfinite(observed_upper) else np.nan
        )
        broad_observed = float(
            self.b1.param_values.get("hei_pgamma_broad.fwhm_kms", np.nan)
        )
        broad_observed_error = float(
            self.b1.param_errors.get("hei_pgamma_broad.fwhm_kms", np.nan)
        )
        broad_intrinsic = (
            intrinsic_fwhm_kms(broad_observed, instrument)
            if np.isfinite(broad_observed) else np.nan
        )

        hei_narrow = _metric(self.n0, "hei_10833_narrow_flux_input")
        hei_narrow_error = _metric_error(self.n0, "hei_10833_narrow_flux_input")
        pgamma_narrow = _metric(self.n0, "pagamma_narrow_flux_input")
        pgamma_narrow_error = _metric_error(self.n0, "pagamma_narrow_flux_input")
        hei_broad = _metric(self.b1, "hei_10833_broad_flux_input")
        hei_broad_error = _metric_error(self.b1, "hei_10833_broad_flux_input")
        pgamma_broad = _metric(self.b1, "pagamma_broad_flux_input")
        pgamma_broad_error = _metric_error(self.b1, "pagamma_broad_flux_input")
        hei_narrow_b1 = _metric(self.b1, "hei_10833_narrow_flux_input")
        pgamma_narrow_b1 = _metric(self.b1, "pagamma_narrow_flux_input")
        hei_total = hei_narrow_b1 + hei_broad
        joint_broad = hei_broad + pgamma_broad
        joint_narrow = hei_narrow_b1 + pgamma_narrow_b1
        joint_total = joint_broad + joint_narrow

        record: dict[str, Any] = {
            "object_id": str(object_id),
            "complex_name": "hei_pgamma",
            "fit_status": (
                "complete" if self.n0.success and self.b1.success else "fit_failed"
            ),
            "n0_success": bool(self.n0.success),
            "b1_success": bool(self.b1.success),
            "n0_bic": float(self.n0.bic),
            "b1_bic": float(self.b1.bic),
            "delta_bic_broad": self.delta_bic_broad,
            "minimum_bic_model": (
                "N0" if self.n0.bic <= self.b1.bic else "B1"
            ) if (
                self.n0.success
                and self.b1.success
                and np.isfinite(self.n0.bic)
                and np.isfinite(self.b1.bic)
            ) else None,
            "resolving_power": cfg.lsf.resolving_power,
            "instrumental_fwhm_kms": instrument,
            "intrinsic_narrow_boundary_kms": cfg.intrinsic_narrow_fwhm_bounds_kms[1],
            "observed_narrow_boundary_kms": cfg.observed_narrow_bounds_kms[1],
            "narrow_fwhm_observed_kms": observed,
            "narrow_fwhm_observed_error_kms": observed_error,
            "narrow_fwhm_intrinsic_kms": intrinsic,
            "narrow_fwhm_intrinsic_upper_2sigma_kms": intrinsic_upper,
            "broad_fwhm_observed_kms": broad_observed,
            "broad_fwhm_observed_error_kms": broad_observed_error,
            "broad_fwhm_intrinsic_kms": broad_intrinsic,
            "narrow_width_secure_below_boundary": bool(
                np.isfinite(intrinsic_upper)
                and intrinsic_upper < cfg.intrinsic_narrow_fwhm_bounds_kms[1]
            ),
            "narrow_kinematic_bound_hit": _relevant_narrow_bound_hit(self.n0),
            "narrow_line_id": "hei_10833",
            "narrow_line_flux": hei_narrow,
            "narrow_line_flux_error": hei_narrow_error,
            "narrow_line_snr": _snr(hei_narrow, hei_narrow_error),
            "hei_narrow_flux": hei_narrow,
            "hei_narrow_flux_error": hei_narrow_error,
            "hei_narrow_snr": _snr(hei_narrow, hei_narrow_error),
            "pagamma_narrow_flux": pgamma_narrow,
            "pagamma_narrow_flux_error": pgamma_narrow_error,
            "pagamma_narrow_snr": _snr(pgamma_narrow, pgamma_narrow_error),
            "hei_broad_flux": hei_broad,
            "hei_broad_flux_error": hei_broad_error,
            "hei_broad_snr": _snr(hei_broad, hei_broad_error),
            "pagamma_broad_flux": pgamma_broad,
            "pagamma_broad_flux_error": pgamma_broad_error,
            "pagamma_broad_snr": _snr(pgamma_broad, pgamma_broad_error),
            "hei_broad_fraction_point": (
                hei_broad / hei_total
                if np.isfinite(hei_total) and hei_total > 0 else np.nan
            ),
            "joint_broad_fraction_point": (
                joint_broad / joint_total
                if np.isfinite(joint_total) and joint_total > 0 else np.nan
            ),
            "broad_fraction_status": "not_calibrated_profile_limit_not_run",
            "n0_warning_codes": tuple(w.code for w in self.n0.warnings),
            "b1_warning_codes": tuple(w.code for w in self.b1.warnings),
            "n0_n_valid_pixels": int(self.n0.metadata.get("n_valid_pixels", 0) or 0),
            "b1_n_valid_pixels": int(self.b1.metadata.get("n_valid_pixels", 0) or 0),
            "fit_window_lower": cfg.fit_window[0],
            "fit_window_upper": cfg.fit_window[1],
            "lsf_assumption": (
                "Gaussian constant-R effective-resolution approximation; fitted widths "
                "remain observed-frame widths"
            ),
        }
        record.update(
            {
                f"n0_{key}": value
                for key, value in _residual_statistics(self.n0).items()
            }
        )
        record.update(
            {
                f"b1_{key}": value
                for key, value in _residual_statistics(self.b1).items()
            }
        )
        record.update(
            evaluate_provisional_narrow_evidence(
                record,
                broad_snr_fields=("hei_broad_snr", "pagamma_broad_snr"),
            )
        )
        return record


def fit_hei_pgamma_model_pair(
    spectrum: Spectrum,
    continuum_result: GlobalContinuumResult,
    *,
    selection_config: HeIPagammaModelSelectionConfig | None = None,
    compute_covariance: bool = True,
) -> HeIPagammaModelPairResult | None:
    """Fit the dedicated local N0/B1 pair, or return ``None`` if uncovered."""

    cfg = selection_config or HeIPagammaModelSelectionConfig()
    n0 = fit_generic_complex(
        spectrum,
        continuum_result,
        hei_pgamma_classification_recipe(cfg, include_broad=False),
        compute_covariance=compute_covariance,
    )
    b1 = fit_generic_complex(
        spectrum,
        continuum_result,
        hei_pgamma_classification_recipe(cfg, include_broad=True),
        compute_covariance=compute_covariance,
    )
    if n0 is None and b1 is None:
        return None
    if n0 is None or b1 is None:
        raise RuntimeError("He I/Pa-gamma N0 and B1 coverage decisions disagree.")
    for name, result in (("N0", n0), ("B1", b1)):
        result.metadata.update(
            {
                "model_grid_name": name,
                "width_parameterization": "observed",
                "intrinsic_width_bounds_transformed_by_gaussian_lsf": True,
                "resolving_power": cfg.lsf.resolving_power,
                "instrumental_fwhm_kms": cfg.lsf.instrumental_fwhm_kms,
                "intrinsic_narrow_fwhm_bounds_kms": cfg.intrinsic_narrow_fwhm_bounds_kms,
                "observed_narrow_fwhm_bounds_kms": cfg.observed_narrow_bounds_kms,
                "intrinsic_broad_fwhm_bounds_kms": cfg.intrinsic_broad_fwhm_bounds_kms,
                "observed_broad_fwhm_bounds_kms": cfg.observed_broad_bounds_kms,
                "decomposition_dependent": True,
                "pagamma_detection_required": False,
                "broad_fraction_threshold_calibrated": False,
            }
        )
        if not any(w.code == "nir_he10833_pgamma_blend" for w in result.warnings):
            result.warnings.append(
                FitWarning(
                    code="nir_he10833_pgamma_blend",
                    message="He I 10833 and Pa-gamma broad wings are decomposition-dependent.",
                    severity="info",
                )
            )
    return HeIPagammaModelPairResult(n0=n0, b1=b1, selection_config=cfg)
