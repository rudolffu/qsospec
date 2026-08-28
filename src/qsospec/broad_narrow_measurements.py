"""Uniform measurement-first broad+narrow decompositions for Euclid RGS.

This module deliberately reports continuous measurements.  It does not choose
a broad/narrow class and it does not compare alternative component families.
Every covered complex is fitted with one narrow and one broad permitted-line
component on top of the archived global continuum plus a residual straight
line.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import lines
from .complex_recipes import ComplexRecipe, ComponentRecipe
from .fitting.complexes import fit_generic_complex, resolve_recipe_coverage
from .global_result import EmissionComplexResult, GlobalContinuumResult
from .halpha_classification import (
    LineSpreadFunctionConfig,
    intrinsic_fwhm_kms,
    observed_fwhm_kms,
)
from .spectrum import Spectrum

MEASUREMENT_SCHEMA_VERSION = "broad_narrow_r480_v1"
COMPLEX_ORDER = ("halpha", "hbeta", "mgii", "hei_pgamma")


@dataclass(frozen=True)
class BroadNarrowMeasurementConfig:
    """Locked scientific configuration for the first measurement catalogue."""

    lsf: LineSpreadFunctionConfig = field(default_factory=LineSpreadFunctionConfig)
    intrinsic_narrow_fwhm_bounds_kms: tuple[float, float] = (0.0, 1200.0)
    intrinsic_broad_fwhm_bounds_kms: tuple[float, float] = (1200.0, 20000.0)
    hei_intrinsic_broad_fwhm_bounds_kms: tuple[float, float] = (1200.0, 15000.0)
    narrow_velocity_bounds_kms: tuple[float, float] = (-1000.0, 1000.0)
    broad_velocity_bounds_kms: tuple[float, float] = (-2000.0, 2000.0)
    local_continuum_mode: str = "residual_linear"
    compute_covariance: bool = True

    def __post_init__(self) -> None:
        if self.local_continuum_mode != "residual_linear":
            raise ValueError("The v1 measurement model requires residual_linear continuum.")
        if self.intrinsic_narrow_fwhm_bounds_kms != (0.0, 1200.0):
            raise ValueError("The v1 intrinsic narrow-width interval is locked to 0-1200 km/s.")
        for bounds in (
            self.intrinsic_broad_fwhm_bounds_kms,
            self.hei_intrinsic_broad_fwhm_bounds_kms,
        ):
            if bounds[0] != 1200.0 or bounds[1] <= bounds[0]:
                raise ValueError("Broad-width bounds must start at 1200 km/s and increase.")

    @property
    def instrumental_fwhm_kms(self) -> float:
        return self.lsf.instrumental_fwhm_kms

    def observed_bounds(self, *, broad: bool, hei: bool = False) -> tuple[float, float]:
        bounds = (
            self.hei_intrinsic_broad_fwhm_bounds_kms
            if broad and hei
            else self.intrinsic_broad_fwhm_bounds_kms
            if broad
            else self.intrinsic_narrow_fwhm_bounds_kms
        )
        return tuple(observed_fwhm_kms(value, self.instrumental_fwhm_kms) for value in bounds)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["instrumental_fwhm_kms"] = self.instrumental_fwhm_kms
        return payload


@dataclass(frozen=True)
class _ComplexDefinition:
    name: str
    label: str
    primary_line_id: str
    fit_window: tuple[float, float]
    required_line_ids: tuple[str, ...]
    coverage_mode: str
    min_coverage_fraction: float
    min_valid_pixels: int
    components: tuple[ComponentRecipe, ...]
    narrow_group: str
    broad_group: str
    narrow_component: str
    broad_component: str


def _component(
    component_id: str,
    line_id: str,
    role: str,
    group: str,
    width_bounds: tuple[float, float],
    velocity_bounds: tuple[float, float],
    **kwargs: Any,
) -> ComponentRecipe:
    return ComponentRecipe(
        id=component_id,
        line_ids=(line_id,),
        role=role,
        flux_bounds=(0.0, None),
        velocity_bounds_kms=velocity_bounds,
        fwhm_bands_kms=(width_bounds,),
        kinematic_group=group,
        **kwargs,
    )


def _definition(name: str, config: BroadNarrowMeasurementConfig) -> _ComplexDefinition:
    nwidth = config.observed_bounds(broad=False)
    bwidth = config.observed_bounds(broad=True, hei=name == "hei_pgamma")
    nv = config.narrow_velocity_bounds_kms
    bv = config.broad_velocity_bounds_kms
    if name == "halpha":
        ng, bg = "halpha_narrow", "halpha_broad"
        components = (
            _component("Ha_narrow", "halpha", "narrow", ng, nwidth, nv, required=True),
            _component("Ha_broad", "halpha", "broad", bg, bwidth, bv, required=True),
            _component("NII6585", "nii_6585", "narrow", ng, nwidth, nv),
            _component(
                "NII6550", "nii_6550", "narrow", ng, nwidth, nv,
                fixed_ratio_to="NII6585", fixed_ratio=2.96,
            ),
            _component("SII6718", "sii_6718", "narrow", ng, nwidth, nv),
            _component("SII6733", "sii_6733", "narrow", ng, nwidth, nv),
        )
        return _ComplexDefinition(
            name, "H-alpha + [N II] + [S II]", "halpha", (6400.0, 6800.0),
            ("halpha", "nii_6550", "nii_6585"), "component_adaptive", 0.6, 30,
            components, ng, bg, "Ha_narrow", "Ha_broad",
        )
    if name == "hbeta":
        ng, bg = "hbeta_narrow", "hbeta_broad"
        components = (
            _component("Hb_narrow", "hbeta", "narrow", ng, nwidth, nv, required=True),
            _component("Hb_broad", "hbeta", "broad", bg, bwidth, bv, required=True),
            _component("OIII5008_core", "oiii_5008", "narrow", ng, nwidth, nv),
            _component(
                "OIII4960_core", "oiii_4960", "narrow", ng, nwidth, nv,
                fixed_ratio_to="OIII5008_core", fixed_ratio=2.98,
            ),
        )
        return _ComplexDefinition(
            name, "H-beta + [O III]", "hbeta", (4640.0, 5100.0),
            ("hbeta", "oiii_4960", "oiii_5008"), "full", 0.8, 30,
            components, ng, bg, "Hb_narrow", "Hb_broad",
        )
    if name == "mgii":
        ng, bg = "mgii_narrow", "mgii_broad"
        components = (
            _component("MgII_narrow", "mgii_blend", "narrow", ng, nwidth, nv, required=True),
            _component("MgII_broad", "mgii_blend", "broad", bg, bwidth, bv, required=True),
        )
        return _ComplexDefinition(
            name, "Mg II effective blend", "mgii_blend", (2700.0, 2900.0),
            ("mgii_blend",), "full", 0.8, 20, components,
            ng, bg, "MgII_narrow", "MgII_broad",
        )
    if name == "hei_pgamma":
        ng, bg = "hei_pgamma_narrow", "hei_pgamma_broad"
        components = (
            _component("HeI10833_narrow", "hei_10833", "narrow", ng, nwidth, nv, required=True),
            _component("HeI10833_broad", "hei_10833", "broad", bg, bwidth, bv, required=True),
            _component("Pagamma_narrow", "pagamma", "narrow", ng, nwidth, nv),
            _component("Pagamma_broad", "pagamma", "broad", bg, bwidth, bv),
        )
        return _ComplexDefinition(
            name, "He I 10833 + Pa-gamma", "hei_10833", (10550.0, 11150.0),
            ("hei_10833",), "component_adaptive", 0.8, 30, components,
            ng, bg, "HeI10833_narrow", "HeI10833_broad",
        )
    raise KeyError(f"Unsupported broad+narrow complex: {name!r}")


def broad_narrow_recipe(
    complex_name: str,
    config: BroadNarrowMeasurementConfig | None = None,
) -> ComplexRecipe:
    """Return the locked one-narrow plus one-broad recipe for a complex."""

    cfg = config or BroadNarrowMeasurementConfig()
    definition = _definition(complex_name, cfg)
    return ComplexRecipe(
        id=f"broad_narrow_{definition.name}",
        aliases=(),
        label=definition.label,
        fit_window=definition.fit_window,
        fit_windows=(definition.fit_window,),
        mask_windows=(),
        components=definition.components,
        required_line_ids=definition.required_line_ids,
        coverage_mode=definition.coverage_mode,
        min_coverage_fraction=definition.min_coverage_fraction,
        min_valid_pixels=definition.min_valid_pixels,
        continuum_mode=cfg.local_continuum_mode,
        qa_labels=definition.required_line_ids,
        auto_enabled=False,
        priority=0,
        backend="generic",
        exclusive_group=f"broad_narrow_{definition.name}",
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _covariance_quantity(
    result: EmissionComplexResult,
    gradients: Mapping[str, float],
) -> float:
    if result.covariance is None:
        return np.nan
    names = tuple(result.param_values)
    covariance = np.asarray(result.covariance, dtype=float)
    if covariance.shape != (len(names), len(names)):
        return np.nan
    gradient = np.zeros(len(names), dtype=float)
    for name, value in gradients.items():
        if name not in result.param_values or not np.isfinite(value):
            return np.nan
        gradient[names.index(name)] = float(value)
    variance = float(gradient @ covariance @ gradient)
    return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan


def _profile_widths(
    narrow_flux: float,
    broad_flux: float,
    narrow_velocity: float,
    broad_velocity: float,
    narrow_fwhm: float,
    broad_fwhm: float,
) -> tuple[float, float, bool]:
    values = np.asarray(
        [narrow_flux, broad_flux, narrow_velocity, broad_velocity, narrow_fwhm, broad_fwhm],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or narrow_flux + broad_flux <= 0:
        return np.nan, np.nan, False
    velocity = np.linspace(-30000.0, 30000.0, 30001)
    profile = np.zeros_like(velocity)
    for flux, center, fwhm in (
        (narrow_flux, narrow_velocity, narrow_fwhm),
        (broad_flux, broad_velocity, broad_fwhm),
    ):
        sigma = max(float(fwhm), 1.0e-6) / 2.354820045
        profile += flux * np.exp(-0.5 * ((velocity - center) / sigma) ** 2) / (
            np.sqrt(2.0 * np.pi) * sigma
        )
    if not np.any(profile > 0):
        return np.nan, np.nan, False
    half = 0.5 * float(np.max(profile))
    above = profile >= half
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    stops = np.flatnonzero(above & ~np.r_[above[1:], False])
    ambiguous = len(starts) != 1
    left_index, right_index = int(starts[0]), int(stops[-1])
    left = float(velocity[left_index])
    right = float(velocity[right_index])
    total = narrow_flux + broad_flux
    mean = (narrow_flux * narrow_velocity + broad_flux * broad_velocity) / total
    variance = (
        narrow_flux * ((narrow_fwhm / 2.354820045) ** 2 + (narrow_velocity - mean) ** 2)
        + broad_flux * ((broad_fwhm / 2.354820045) ** 2 + (broad_velocity - mean) ** 2)
    ) / total
    return float(right - left), float(np.sqrt(max(variance, 0.0))), bool(ambiguous)


def _residual_statistics(result: EmissionComplexResult) -> dict[str, float]:
    mask = np.asarray(result.fit_mask, dtype=bool)
    residual = (result.flux_continuum_subtracted[mask] - result.model[mask]) / result.err[mask]
    residual = residual[np.isfinite(residual)]
    if not residual.size:
        return {key: np.nan for key in (
            "residual_rms_sigma", "residual_median_sigma",
            "residual_abs_p95_sigma", "residual_abs_max_sigma",
        )}
    return {
        "residual_rms_sigma": float(np.sqrt(np.mean(residual**2))),
        "residual_median_sigma": float(np.median(residual)),
        "residual_abs_p95_sigma": float(np.percentile(np.abs(residual), 95.0)),
        "residual_abs_max_sigma": float(np.max(np.abs(residual))),
    }


def signed_to_uint64_string(object_id: object) -> str:
    """Round-trip a signed Euclid identifier through its uint64 decimal form."""

    value = int(object_id)
    return str(int(np.asarray(value, dtype=np.int64).view(np.uint64)))


def _empty_record(
    complex_name: str,
    status: str,
    message: str,
    config: BroadNarrowMeasurementConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BroadNarrowMeasurementConfig()
    definition = _definition(complex_name, cfg)
    missing = {
        name: np.nan
        for name in (
            "narrow_flux", "narrow_flux_error", "narrow_flux_snr",
            "broad_flux", "broad_flux_error", "broad_flux_snr",
            "total_flux", "total_flux_error", "broad_fraction",
            "broad_fraction_error", "broad_to_narrow_ratio",
            "narrow_velocity_kms", "narrow_velocity_kms_error",
            "broad_velocity_kms", "broad_velocity_kms_error",
            "narrow_fwhm_observed_kms", "narrow_fwhm_observed_kms_error",
            "broad_fwhm_observed_kms", "broad_fwhm_observed_kms_error",
            "narrow_fwhm_intrinsic_approx_kms", "broad_fwhm_intrinsic_approx_kms",
            "total_equivalent_width_rest", "total_profile_fwhm_observed_kms",
            "total_profile_sigma_kms", "chi2", "reduced_chi2", "bic",
            "residual_rms_sigma", "residual_median_sigma",
            "residual_abs_p95_sigma", "residual_abs_max_sigma",
        )
    }
    return {
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "complex_name": complex_name,
        "primary_line_id": definition.primary_line_id,
        "fit_status": status,
        "fit_success": False,
        "fit_message": message,
        "resolving_power": cfg.lsf.resolving_power,
        "instrumental_fwhm_kms": cfg.instrumental_fwhm_kms,
        "local_continuum_mode": cfg.local_continuum_mode,
        "local_continuum_pivot": 0.5 * sum(definition.fit_window),
        "narrow_fwhm_observed_lower_bound_kms": cfg.observed_bounds(broad=False)[0],
        "narrow_fwhm_observed_upper_bound_kms": cfg.observed_bounds(broad=False)[1],
        "broad_fwhm_observed_lower_bound_kms": cfg.observed_bounds(
            broad=True, hei=complex_name == "hei_pgamma"
        )[0],
        "broad_fwhm_observed_upper_bound_kms": cfg.observed_bounds(
            broad=True, hei=complex_name == "hei_pgamma"
        )[1],
        **missing,
    }


def measurement_record(
    complex_name: str,
    result: EmissionComplexResult,
    config: BroadNarrowMeasurementConfig | None = None,
    continuum: GlobalContinuumResult | None = None,
) -> dict[str, Any]:
    """Convert a successful or failed generic fit to the common long schema."""

    cfg = config or BroadNarrowMeasurementConfig()
    definition = _definition(complex_name, cfg)
    if not result.success:
        record = _empty_record(complex_name, "fit_failed", result.message, cfg)
        record.update({
            "fit_status_code": int(result.status),
            "coverage_fraction": float(result.metadata.get("coverage_fraction", np.nan)),
            "n_valid_pixels": int(result.metadata.get("n_valid_pixels", 0) or 0),
            "warning_codes": tuple(result.warning_codes()),
        })
        return record

    narrow_flux_name = f"{definition.narrow_component}.flux"
    broad_flux_name = f"{definition.broad_component}.flux"
    narrow_flux = float(result.param_values.get(narrow_flux_name, np.nan))
    broad_flux = float(result.param_values.get(broad_flux_name, np.nan))
    narrow_error = float(result.param_errors.get(narrow_flux_name, np.nan))
    broad_error = float(result.param_errors.get(broad_flux_name, np.nan))
    total_flux = narrow_flux + broad_flux
    broad_fraction = _safe_ratio(broad_flux, total_flux)
    if np.isfinite(broad_fraction) and not (-1.0e-10 <= broad_fraction <= 1.0 + 1.0e-10):
        raise ValueError(f"broad fraction outside [0,1]: {broad_fraction}")
    if np.isfinite(broad_fraction):
        broad_fraction = float(np.clip(broad_fraction, 0.0, 1.0))
    total_error = _covariance_quantity(
        result, {narrow_flux_name: 1.0, broad_flux_name: 1.0}
    )
    fraction_error = (
        _covariance_quantity(
            result,
            {
                narrow_flux_name: -broad_flux / total_flux**2,
                broad_flux_name: narrow_flux / total_flux**2,
            },
        )
        if np.isfinite(total_flux) and total_flux > 0 else np.nan
    )
    nvel_name = f"{definition.narrow_group}.velocity_kms"
    bvel_name = f"{definition.broad_group}.velocity_kms"
    nwidth_name = f"{definition.narrow_group}.fwhm_kms"
    bwidth_name = f"{definition.broad_group}.fwhm_kms"
    nvel = float(result.param_values.get(nvel_name, np.nan))
    bvel = float(result.param_values.get(bvel_name, np.nan))
    nwidth = float(result.param_values.get(nwidth_name, np.nan))
    bwidth = float(result.param_values.get(bwidth_name, np.nan))
    profile_fwhm, profile_sigma, profile_ambiguous = _profile_widths(
        narrow_flux, broad_flux, nvel, bvel, nwidth, bwidth
    )
    primary_wave = lines.get(definition.primary_line_id).vacuum_wavelength
    continuum_global = (
        float(np.interp(primary_wave, continuum.wave_rest, continuum.model))
        if continuum is not None else np.nan
    )
    if not np.isfinite(continuum_global):
        # Public callers may serialize a result without retaining its continuum.
        # The generic narrow-line EW provides an exact fallback unless its flux
        # is zero.
        metric_prefix = f"{definition.primary_line_id}_narrow"
        ew_narrow = float(result.metrics.get(f"{metric_prefix}_ew_rest", np.nan))
        if np.isfinite(ew_narrow) and ew_narrow != 0:
            continuum_global = narrow_flux / ew_narrow
    local_continuum = float(result.param_values.get("continuum.constant", 0.0)) + float(
        result.param_values.get("continuum.slope", 0.0)
    ) * (primary_wave - 0.5 * sum(definition.fit_window))
    continuum_total = continuum_global + local_continuum
    total_ew = total_flux / continuum_total if np.isfinite(continuum_total) and continuum_total > 0 else np.nan

    warnings = tuple(result.warning_codes())
    bound_parameters = tuple(
        str(warning.context.get("parameter", ""))
        for warning in result.warnings
        if warning.code == "parameter_at_bound"
    )
    bound_set = set(bound_parameters)
    active_component_ids = tuple(result.metadata.get("active_components", ()))
    active_component_set = set(active_component_ids)
    active_optional_lines = tuple(sorted({
        line_id
        for component in definition.components
        if component.id in active_component_set
        for line_id in component.line_ids
        if line_id not in definition.required_line_ids
    }))
    record: dict[str, Any] = {
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "complex_name": complex_name,
        "primary_line_id": definition.primary_line_id,
        "fit_status": "complete",
        "fit_success": True,
        "fit_status_code": int(result.status),
        "fit_message": result.message,
        "coverage_fraction": float(result.metadata.get("coverage_fraction", np.nan)),
        "n_valid_pixels": int(result.metadata.get("n_valid_pixels", np.count_nonzero(result.fit_mask))),
        "valid_wavelength_min": float(np.min(result.wave_rest[result.fit_mask])),
        "valid_wavelength_max": float(np.max(result.wave_rest[result.fit_mask])),
        "active_component_ids": active_component_ids,
        "disabled_component_ids": tuple(result.metadata.get("disabled_components", ())),
        "covered_line_ids": tuple(result.metadata.get("covered_line_ids", ())),
        "active_optional_lines": active_optional_lines,
        "warning_codes": warnings,
        "active_bound_parameters": bound_parameters,
        "narrow_velocity_at_bound": nvel_name in bound_set,
        "narrow_width_at_bound": nwidth_name in bound_set,
        "broad_velocity_at_bound": bvel_name in bound_set,
        "broad_width_at_bound": bwidth_name in bound_set,
        "covariance_rank_deficient": "covariance_rank_deficient" in warnings,
        "narrow_flux": narrow_flux,
        "narrow_flux_error": narrow_error,
        "narrow_flux_snr": _safe_ratio(narrow_flux, narrow_error),
        "broad_flux": broad_flux,
        "broad_flux_error": broad_error,
        "broad_flux_snr": _safe_ratio(broad_flux, broad_error),
        "total_flux": total_flux,
        "total_flux_error": total_error,
        "broad_fraction": broad_fraction,
        "broad_fraction_error": fraction_error,
        "broad_to_narrow_ratio": _safe_ratio(broad_flux, narrow_flux),
        "narrow_velocity_kms": nvel,
        "narrow_velocity_kms_error": float(result.param_errors.get(nvel_name, np.nan)),
        "broad_velocity_kms": bvel,
        "broad_velocity_kms_error": float(result.param_errors.get(bvel_name, np.nan)),
        "narrow_fwhm_observed_kms": nwidth,
        "narrow_fwhm_observed_kms_error": float(result.param_errors.get(nwidth_name, np.nan)),
        "broad_fwhm_observed_kms": bwidth,
        "broad_fwhm_observed_kms_error": float(result.param_errors.get(bwidth_name, np.nan)),
        "narrow_fwhm_intrinsic_approx_kms": intrinsic_fwhm_kms(nwidth, cfg.instrumental_fwhm_kms),
        "broad_fwhm_intrinsic_approx_kms": intrinsic_fwhm_kms(bwidth, cfg.instrumental_fwhm_kms),
        "total_equivalent_width_rest": total_ew,
        "total_profile_fwhm_observed_kms": profile_fwhm,
        "total_profile_sigma_kms": profile_sigma,
        "total_profile_fwhm_ambiguous": profile_ambiguous,
        "chi2": float(result.chi2),
        "dof": int(result.dof),
        "reduced_chi2": float(result.reduced_chi2),
        "bic": float(result.bic),
        "optimizer_requested": result.metadata.get("optimizer_requested"),
        "optimizer_used": result.metadata.get("optimizer_used"),
        "nonlinear_nfev": int(result.metadata.get("nonlinear_nfev", 0) or 0),
        "nonlinear_njev": int(result.metadata.get("nonlinear_njev", 0) or 0),
        "optimizer_fallback": bool(result.metadata.get("optimizer_fallback", False)),
        "local_continuum_mode": cfg.local_continuum_mode,
        "local_continuum_pivot": 0.5 * sum(definition.fit_window),
        "local_continuum_constant": float(result.param_values.get("continuum.constant", np.nan)),
        "local_continuum_constant_error": float(result.param_errors.get("continuum.constant", np.nan)),
        "local_continuum_slope": float(result.param_values.get("continuum.slope", np.nan)),
        "local_continuum_slope_error": float(result.param_errors.get("continuum.slope", np.nan)),
        "resolving_power": cfg.lsf.resolving_power,
        "instrumental_fwhm_kms": cfg.instrumental_fwhm_kms,
        "narrow_fwhm_observed_lower_bound_kms": cfg.observed_bounds(broad=False)[0],
        "narrow_fwhm_observed_upper_bound_kms": cfg.observed_bounds(broad=False)[1],
        "broad_fwhm_observed_lower_bound_kms": cfg.observed_bounds(broad=True, hei=complex_name == "hei_pgamma")[0],
        "broad_fwhm_observed_upper_bound_kms": cfg.observed_bounds(broad=True, hei=complex_name == "hei_pgamma")[1],
    }
    record.update(_residual_statistics(result))
    # Explicit component measurements make the long table self-sufficient.
    for key, value in result.metrics.items():
        if key.endswith(("_flux_input", "_fwhm_kms")):
            record[f"component_{key}"] = float(value)
    if complex_name == "hei_pgamma":
        for prefix, component_n, component_b in (
            ("hei", "HeI10833_narrow.flux", "HeI10833_broad.flux"),
            ("pagamma", "Pagamma_narrow.flux", "Pagamma_broad.flux"),
        ):
            fn = float(result.param_values.get(component_n, np.nan))
            fb = float(result.param_values.get(component_b, np.nan))
            record[f"{prefix}_narrow_flux"] = fn
            record[f"{prefix}_narrow_flux_error"] = float(
                result.param_errors.get(component_n, np.nan)
            )
            record[f"{prefix}_broad_flux"] = fb
            record[f"{prefix}_broad_flux_error"] = float(
                result.param_errors.get(component_b, np.nan)
            )
            record[f"{prefix}_broad_fraction"] = _safe_ratio(fb, fn + fb)
            record[f"{prefix}_broad_fraction_error"] = (
                _covariance_quantity(
                    result,
                    {
                        component_n: -fb / (fn + fb) ** 2,
                        component_b: fn / (fn + fb) ** 2,
                    },
                )
                if np.isfinite(fn + fb) and fn + fb > 0 else np.nan
            )
        joint_n = record["hei_narrow_flux"] + record["pagamma_narrow_flux"]
        joint_b = record["hei_broad_flux"] + record["pagamma_broad_flux"]
        record["joint_broad_fraction"] = _safe_ratio(joint_b, joint_n + joint_b)
        joint_total = joint_n + joint_b
        record["joint_broad_fraction_error"] = (
            _covariance_quantity(
                result,
                {
                    "HeI10833_narrow.flux": -joint_b / joint_total**2,
                    "Pagamma_narrow.flux": -joint_b / joint_total**2,
                    "HeI10833_broad.flux": joint_n / joint_total**2,
                    "Pagamma_broad.flux": joint_n / joint_total**2,
                },
            )
            if np.isfinite(joint_total) and joint_total > 0 else np.nan
        )
    return record


def measure_broad_narrow_complex(
    spectrum: Spectrum,
    continuum: GlobalContinuumResult,
    complex_name: str,
    config: BroadNarrowMeasurementConfig | None = None,
) -> tuple[dict[str, Any], EmissionComplexResult | None]:
    """Fit one complex and return its common measurement row and model result."""

    cfg = config or BroadNarrowMeasurementConfig()
    if not continuum.success or continuum.model.shape != spectrum.flux.shape:
        return _empty_record(
            complex_name, "continuum_unavailable", "Archived global continuum is unavailable.", cfg
        ), None
    recipe = broad_narrow_recipe(complex_name, cfg)
    coverage = resolve_recipe_coverage(spectrum, recipe)
    if coverage.status == "not_covered":
        record = _empty_record(
            complex_name, "not_covered", "Complex is outside valid coverage.", cfg
        )
        record.update({
            "coverage_fraction": coverage.coverage_fraction,
            "n_valid_pixels": coverage.n_valid_pixels,
            "warning_codes": tuple(warning.code for warning in coverage.warnings),
        })
        return record, None
    result = fit_generic_complex(
        spectrum,
        continuum,
        recipe,
        compute_covariance=cfg.compute_covariance,
        coverage_override=coverage,
    )
    if result is None:
        return _empty_record(
            complex_name, "not_covered", "Complex is outside valid coverage.", cfg
        ), None
    return measurement_record(complex_name, result, cfg, continuum), result


def measure_broad_narrow_complexes(
    spectrum: Spectrum,
    continuum: GlobalContinuumResult,
    complexes: tuple[str, ...] = COMPLEX_ORDER,
    config: BroadNarrowMeasurementConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, EmissionComplexResult]]:
    """Fit requested complexes in stable order without assigning a class."""

    cfg = config or BroadNarrowMeasurementConfig()
    records: list[dict[str, Any]] = []
    results: dict[str, EmissionComplexResult] = {}
    for complex_name in complexes:
        record, result = measure_broad_narrow_complex(
            spectrum, continuum, complex_name, cfg
        )
        records.append(record)
        if result is not None:
            results[complex_name] = result
    return records, results
