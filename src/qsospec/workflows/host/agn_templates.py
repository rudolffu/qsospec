"""AGN pseudo-continuum templates for pPXF host decomposition.

The evaluators in this module deliberately reuse qsospec's production Fe II
and Balmer physics.  The resulting basis is inspired by Aydar et al. (2026),
but it is not an exact reproduction of template files that are not distributed
with that work.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from typing import Any

import numpy as np

from ...templates import (
    evaluate_balmer_pseudocontinuum_with_derivatives,
    evaluate_iron_basis,
    load_balmer_template,
    load_iron_template,
)
from .config import HostAgnPseudoContinuumConfig


@lru_cache(maxsize=16)
def _cached_builtin_iron_template(template_name: str):
    """Cache immutable bundled iron resources between object fits."""

    return load_iron_template(template_name)


def _wave_cache_payload(wave: np.ndarray) -> tuple[int, bytes]:
    contiguous = np.ascontiguousarray(wave, dtype=np.float64)
    return contiguous.size, contiguous.tobytes()


@lru_cache(maxsize=32)
def _cached_iron_basis(
    template_name: str,
    selected_fwhm_kms: float,
    wave_size: int,
    wave_bytes: bytes,
) -> np.ndarray:
    wave = np.frombuffer(wave_bytes, dtype=np.float64, count=wave_size)
    template = _cached_builtin_iron_template(template_name)
    values = evaluate_iron_basis(template, wave, selected_fwhm_kms)
    values.setflags(write=False)
    return values


@lru_cache(maxsize=32)
def _cached_balmer_basis(
    log10_ne: int,
    selected_fwhm_kms: float,
    temperature_k: float,
    tau_edge: float,
    wave_size: int,
    wave_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = np.frombuffer(wave_bytes, dtype=np.float64, count=wave_size)
    template = load_balmer_template(
        log10_ne=log10_ne,
        n_min=6,
        provenance="sh95_k13full_ext",
    )
    combined, bound_free, high_order, _, _ = evaluate_balmer_pseudocontinuum_with_derivatives(
        template,
        wave,
        selected_fwhm_kms,
        0.0,
        temperature_k=temperature_k,
        tau_edge=tau_edge,
    )
    for values in (combined, bound_free, high_order):
        values.setflags(write=False)
    return combined, bound_free, high_order


@dataclass(frozen=True)
class HostAgnTemplateComponent:
    """One physical component represented in the pPXF AGN basis."""

    name: str
    category: str
    values: np.ndarray
    wavelength: np.ndarray
    normalization: float
    intrinsic_fwhm_kms: float | None
    selected_fwhm_kms: float | None
    source_id: str
    source_reference: str
    wavelength_coverage: tuple[float, float]
    included_in_global_fagn: bool
    linear_group: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HostAgnTemplateBundle:
    """Linear pPXF matrix and physical component bookkeeping."""

    matrix: np.ndarray
    column_names: tuple[str, ...]
    column_categories: tuple[str, ...]
    components: tuple[HostAgnTemplateComponent, ...]
    group_column_indices: dict[str, int]
    support_mask: np.ndarray
    metadata: dict[str, Any]


def _normalization_interval(wave: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    valid_wave = np.asarray(wave, dtype=float)[np.asarray(valid_mask, dtype=bool)]
    return float(np.nanmin(valid_wave)), float(np.nanmax(valid_wave))


def _normalize(values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    selected = np.asarray(valid_mask, dtype=bool) & np.isfinite(values) & (values != 0)
    if not np.any(selected):
        return np.zeros_like(values), 1.0
    factor = float(np.nanmedian(np.abs(values[selected])))
    if not np.isfinite(factor) or factor <= 0:
        factor = float(np.nanmax(np.abs(values[selected])))
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0
    return values / factor, factor


def _template_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return sha256(contiguous.view(np.uint8)).hexdigest()


def _apply_instrumental_lsf(
    values: np.ndarray,
    wave_rest: np.ndarray,
    *,
    redshift: float,
    spectral_resolution: Any,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Apply the data LSF once to a physically prebroadened AGN template."""

    matrix = np.asarray(values, dtype=float)
    metadata: dict[str, Any] = {
        "instrumental_broadening_applied": False,
        "instrumental_broadening_status": "resolution_missing",
    }
    valid = np.ones(len(wave_rest), dtype=bool)
    if spectral_resolution is None or getattr(spectral_resolution, "status", None) != "valid":
        return matrix, metadata, valid
    if getattr(spectral_resolution, "mode", None) == "banded_matrix":
        metadata["instrumental_broadening_status"] = "unsupported_banded_resolution_matrix"
        return matrix, metadata, valid
    observed_wave = np.asarray(wave_rest, dtype=float) * (1.0 + float(redshift))
    sigma_lambda = np.asarray(spectral_resolution.sigma_lambda(observed_wave), dtype=float)
    pixel_width = np.gradient(np.asarray(wave_rest, dtype=float))
    sigma_pixels = np.divide(
        sigma_lambda / (1.0 + float(redshift)),
        pixel_width,
        out=np.zeros_like(pixel_width),
        where=np.isfinite(sigma_lambda) & np.isfinite(pixel_width) & (pixel_width > 0),
    )
    valid &= np.isfinite(sigma_pixels) & (sigma_pixels >= 0)
    try:
        from ppxf.ppxf_util import gaussian_filter1d as variable_gaussian_filter1d
    except ImportError:
        metadata["instrumental_broadening_status"] = "variable_filter_unavailable"
        return matrix, metadata, valid
    broadened = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        broadened[:, column] = variable_gaussian_filter1d(matrix[:, column], np.where(valid, sigma_pixels, 0.0))
    metadata.update(
        {
            "instrumental_broadening_applied": True,
            "instrumental_broadening_status": "applied_once",
            "instrumental_sigma_pixels_min": float(np.nanmin(sigma_pixels[valid])),
            "instrumental_sigma_pixels_max": float(np.nanmax(sigma_pixels[valid])),
        }
    )
    return broadened, metadata, valid


def _powerlaw_components(
    wave: np.ndarray,
    valid_mask: np.ndarray,
    config: HostAgnPseudoContinuumConfig,
) -> Iterable[HostAgnTemplateComponent]:
    interval = _normalization_interval(wave, valid_mask)
    for index, slope in enumerate(config.powerlaw_slopes):
        raw = (wave / float(config.powerlaw_pivot_angstrom)) ** float(slope)
        values, normalization = _normalize(raw, valid_mask)
        yield HostAgnTemplateComponent(
            name=f"powerlaw_{index:02d}_slope_{float(slope):+.1f}",
            category="agn_powerlaw",
            values=values,
            wavelength=wave,
            normalization=normalization,
            intrinsic_fwhm_kms=None,
            selected_fwhm_kms=None,
            source_id="analytic_flambda_powerlaw",
            source_reference="Aydar et al. 2026, A&A, 710, A141",
            wavelength_coverage=interval,
            included_in_global_fagn=True,
            linear_group=f"powerlaw_{index:02d}",
            metadata={
                "slope_flambda": float(slope),
                "pivot_angstrom": float(config.powerlaw_pivot_angstrom),
                "normalization_method": "median_absolute_over_valid_fit_interval",
                "normalization_interval": interval,
            },
        )


def build_host_agn_template_bundle(
    wave_rest: np.ndarray,
    *,
    selected_fwhm_kms: float,
    valid_mask: np.ndarray | None = None,
    config: HostAgnPseudoContinuumConfig | None = None,
    redshift: float = 0.0,
    spectral_resolution: Any = None,
) -> HostAgnTemplateBundle:
    """Build the non-negative linear AGN basis on the pPXF log grid."""

    cfg = config or HostAgnPseudoContinuumConfig()
    wave = np.asarray(wave_rest, dtype=float)
    valid = np.ones_like(wave, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).copy()
    valid &= np.isfinite(wave) & (wave > 0)
    if not np.isfinite(selected_fwhm_kms) or selected_fwhm_kms <= 0:
        raise ValueError("selected_fwhm_kms must be positive and finite.")
    components = list(_powerlaw_components(wave, valid, cfg))
    interval = _normalization_interval(wave, valid)
    wave_size, wave_bytes = _wave_cache_payload(wave)

    optical = _cached_builtin_iron_template(cfg.optical_feii_template)
    optical_raw = _cached_iron_basis(
        cfg.optical_feii_template,
        float(selected_fwhm_kms),
        wave_size,
        wave_bytes,
    )
    optical_support = (wave >= float(optical.coverage[0])) & (wave <= float(optical.coverage[1])) & valid
    if np.any(optical_support):
        optical_values, optical_norm = _normalize(optical_raw, optical_support)
        optical_interval = _normalization_interval(wave, optical_support)
        components.append(
            HostAgnTemplateComponent(
                name="feii_optical",
                category="agn_feii_optical",
                values=optical_values,
                wavelength=wave,
                normalization=optical_norm,
                intrinsic_fwhm_kms=None,
                selected_fwhm_kms=float(selected_fwhm_kms),
                source_id=optical.source_path or optical.name,
                source_reference=optical.reference or "Boroson & Green 1992",
                wavelength_coverage=tuple(map(float, optical.coverage)),
                included_in_global_fagn=True,
                linear_group="feii_optical",
                metadata={
                    "physical_broadening_applied_once": True,
                    "native_resolution_status": "unknown_assumed_negligible",
                    "source_sha256": _template_hash(np.column_stack([optical.wave_rest, optical.flux])),
                    "normalization_method": ("median_absolute_over_supported_fit_interval"),
                    "normalization_interval": optical_interval,
                },
            )
        )

    if cfg.uv_feii_template is not None:
        ultraviolet = _cached_builtin_iron_template(cfg.uv_feii_template)
        uv_support = (wave >= float(ultraviolet.coverage[0])) & (wave <= float(ultraviolet.coverage[1])) & valid
        if np.any(uv_support):
            uv_raw = _cached_iron_basis(
                cfg.uv_feii_template,
                float(selected_fwhm_kms),
                wave_size,
                wave_bytes,
            )
            uv_values, uv_norm = _normalize(uv_raw, uv_support)
            uv_interval = _normalization_interval(wave, uv_support)
            components.append(
                HostAgnTemplateComponent(
                    name="feii_uv",
                    category="agn_feii_uv",
                    values=uv_values,
                    wavelength=wave,
                    normalization=uv_norm,
                    intrinsic_fwhm_kms=None,
                    selected_fwhm_kms=float(selected_fwhm_kms),
                    source_id=ultraviolet.source_path or ultraviolet.name,
                    source_reference=ultraviolet.reference or "Vestergaard & Wilkes 2001",
                    wavelength_coverage=tuple(map(float, ultraviolet.coverage)),
                    included_in_global_fagn=True,
                    linear_group="feii_uv",
                    metadata={
                        "physical_broadening_applied_once": True,
                        "native_resolution_status": "unknown_assumed_negligible",
                        "source_sha256": _template_hash(np.column_stack([ultraviolet.wave_rest, ultraviolet.flux])),
                        "normalization_method": "median_absolute_over_supported_fit_interval",
                        "normalization_interval": uv_interval,
                    },
                )
            )

    if cfg.balmer_enabled:
        balmer = load_balmer_template(
            log10_ne=cfg.balmer_log10_ne,
            n_min=6,
            provenance="sh95_k13full_ext",
        )
        combined, bound_free, high_order = _cached_balmer_basis(
            int(cfg.balmer_log10_ne),
            float(selected_fwhm_kms),
            float(cfg.balmer_temperature_k),
            float(cfg.balmer_tau_edge),
            wave_size,
            wave_bytes,
        )
        combined_values, balmer_norm = _normalize(combined, valid & (combined != 0))
        common = {
            "physical_broadening_applied_once": True,
            "velocity_kms": 0.0,
            "temperature_k": float(cfg.balmer_temperature_k),
            "tau_edge": float(cfg.balmer_tau_edge),
            "log10_ne": int(cfg.balmer_log10_ne),
            "n_min": int(balmer.n_min),
            "n_max": int(balmer.n_max),
            "source_sha256": _template_hash(
                np.column_stack(
                    [
                        balmer.n_upper,
                        balmer.wavelength_vacuum,
                        balmer.rel_flux_hbeta,
                    ]
                )
            ),
            "normalization_method": "shared_pseudocontinuum_median_absolute",
            "normalization_interval": interval,
        }
        components.extend(
            [
                HostAgnTemplateComponent(
                    name="balmer_continuum",
                    category="agn_balmer_continuum",
                    values=bound_free / balmer_norm,
                    wavelength=wave,
                    normalization=balmer_norm,
                    intrinsic_fwhm_kms=None,
                    selected_fwhm_kms=float(selected_fwhm_kms),
                    source_id=balmer.source_path,
                    source_reference="Storey & Hummer 1995; Kovačević et al. 2013",
                    wavelength_coverage=interval,
                    included_in_global_fagn=True,
                    linear_group="balmer_pseudocontinuum",
                    metadata={**common, "branch": "bound_free"},
                ),
                HostAgnTemplateComponent(
                    name="balmer_high_order",
                    category="agn_balmer_high_order",
                    values=high_order / balmer_norm,
                    wavelength=wave,
                    normalization=balmer_norm,
                    intrinsic_fwhm_kms=None,
                    selected_fwhm_kms=float(selected_fwhm_kms),
                    source_id=balmer.source_path,
                    source_reference="Storey & Hummer 1995; Kovačević et al. 2013",
                    wavelength_coverage=interval,
                    included_in_global_fagn=True,
                    linear_group="balmer_pseudocontinuum",
                    metadata={**common, "branch": "high_order"},
                ),
            ]
        )
        if not np.allclose(
            combined_values,
            components[-2].values + components[-1].values,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError("Balmer pseudo-continuum branch closure failed.")

    groups: dict[str, list[HostAgnTemplateComponent]] = {}
    for component in components:
        groups.setdefault(component.linear_group, []).append(component)
    columns = []
    column_names = []
    column_categories = []
    group_column_indices: dict[str, int] = {}
    for group, grouped in groups.items():
        group_column_indices[group] = len(columns)
        columns.append(np.sum([item.values for item in grouped], axis=0))
        column_names.append(group)
        categories = sorted({item.category for item in grouped})
        column_categories.append("+".join(categories))
    matrix = np.column_stack(columns) if columns else np.zeros((len(wave), 0))
    matrix, lsf_metadata, lsf_valid = _apply_instrumental_lsf(
        matrix,
        wave,
        redshift=redshift,
        spectral_resolution=spectral_resolution,
    )
    # Apply the same instrumental operator to diagnostic subcomponents.
    component_matrix = np.column_stack([item.values for item in components])
    component_matrix, _, _ = _apply_instrumental_lsf(
        component_matrix,
        wave,
        redshift=redshift,
        spectral_resolution=spectral_resolution,
    )
    for group, column in group_column_indices.items():
        grouped = [item for item in components if item.linear_group == group]
        supported = np.ones_like(wave, dtype=bool)
        if grouped and all("feii" in item.category for item in grouped):
            supported = np.zeros_like(wave, dtype=bool)
            for item in grouped:
                supported |= (wave >= item.wavelength_coverage[0]) & (wave <= item.wavelength_coverage[1])
        matrix[~supported, column] = 0.0
        for index, item in enumerate(components):
            if item.linear_group == group:
                component_matrix[~supported, index] = 0.0
    components = [
        HostAgnTemplateComponent(
            **{
                **item.__dict__,
                "values": component_matrix[:, index],
                "metadata": {**item.metadata, **lsf_metadata},
            }
        )
        for index, item in enumerate(components)
    ]
    support = valid & lsf_valid & np.any(np.isfinite(matrix), axis=1)
    metadata = {
        "host_pseudocontinuum_method": "aydar2026_inspired",
        "host_pseudocontinuum_exact_replication": False,
        "powerlaw_grid_source": "Aydar et al. 2026 Table A.1 and Section 2.1",
        "powerlaw_grid_replication_status": "exact_slopes",
        "powerlaw_slope_convention": "F_lambda",
        "broadening_grid_source": "Aydar et al. 2026 Table A.1",
        "optical_feii_source": optical.source_path,
        "optical_feii_sha256": _template_hash(np.column_stack([optical.wave_rest, optical.flux])),
        "balmer_source": ("qsospec KD13/Storey-Hummer implementation" if cfg.balmer_enabled else None),
        "selected_fwhm_kms": float(selected_fwhm_kms),
        "intrinsic_template_cache_grid_sha256": sha256(wave_bytes).hexdigest(),
        "intrinsic_template_cache_key": {
            "selected_fwhm_kms": float(selected_fwhm_kms),
            "balmer_log10_ne": int(cfg.balmer_log10_ne),
            "balmer_temperature_k": float(cfg.balmer_temperature_k),
            "balmer_tau_edge": float(cfg.balmer_tau_edge),
            "normalization": "median_absolute_over_valid_support",
        },
        "normalization_interval": interval,
        "matrix_sha256": _template_hash(matrix),
        **lsf_metadata,
    }
    return HostAgnTemplateBundle(
        matrix=matrix,
        column_names=tuple(column_names),
        column_categories=tuple(column_categories),
        components=tuple(components),
        group_column_indices=group_column_indices,
        support_mask=support,
        metadata=metadata,
    )
