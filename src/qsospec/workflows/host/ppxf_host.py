"""Optional pPXF host fitting plus qsospec handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_propagation, gaussian_filter1d

from .agn_templates import HostAgnTemplateBundle, build_host_agn_template_bundle
from .config import (
    DEFAULT_LINE_CENTERS,
    DEFAULT_OBSERVED_ARTIFACT_WINDOWS,
    HostAgnPseudoContinuumConfig,
    HostCoverageConfig,
)
from .coverage import HostCoverageResult, classify_host_coverage
from .io import SpectrumData
from .templates import (
    PPXF_NPZ_LOADER_VERSION,
    SAMPLE_WAVELENGTHS,
    TEMPLATE_FLATTENING_CONVENTION,
    PPXFTemplateLibrary,
)
from .preconvolved_templates import validate_preconvolved_xsl_product
from ...resolution import match_template_resolution_toward_data


_C_KMS = 299792.458
HOST_RECONSTRUCTION_STATE_VERSION = "2"
HOST_SED_METHOD = "stellar_template_weighted_sed"
HOST_SED_METHOD_VERSION = "1"


@dataclass
class PreprocessedSpectrum:
    """Spectrum prepared for pPXF host fitting."""

    wave_obs: np.ndarray
    wave_rest: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    ivar: Optional[np.ndarray]
    fit_mask: np.ndarray
    emission_mask: np.ndarray
    wave_log: np.ndarray
    log_wave: np.ndarray
    flux_log: np.ndarray
    noise_log: np.ndarray
    emission_mask_log: np.ndarray
    validity_mask_log: np.ndarray
    artifact_mask_log: np.ndarray
    normalization: float
    redshift: float
    velscale: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    mask_provenance: Dict[str, np.ndarray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PPXFHostFitResult:
    """Output of the first pPXF host-continuum fit."""

    preprocessed: PreprocessedSpectrum
    templates: PPXFTemplateLibrary
    host_model_log: np.ndarray
    agn_model_log: np.ndarray
    total_model_log: np.ndarray
    residual_log: np.ndarray
    host_model: np.ndarray
    agn_model: np.ndarray
    total_model: np.ndarray
    residual: np.ndarray
    stellar_weights: np.ndarray
    agn_weights: np.ndarray
    stellar_template_scales: np.ndarray
    agn_slopes: np.ndarray
    stellar_velocity: float
    stellar_sigma: float
    chi2: float
    reduced_chi2: float
    status: str
    initial_emission_mask_log: Optional[np.ndarray] = None
    expanded_emission_mask_log: Optional[np.ndarray] = None
    residual_clip_mask_log: Optional[np.ndarray] = None
    final_goodpixels_mask_log: Optional[np.ndarray] = None
    noise_rescale_factors: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    host_fit_reliable: bool = True
    host_fit_reliability_reasons: List[str] = field(default_factory=list)
    host_continuum_reliable: bool = True
    host_fraction_reliable: bool = True
    host_absorption_subtraction_status: str = "available"
    stellar_kinematics_resolution_status: str = "unavailable"
    stellar_population_resolution_status: str = "unavailable"
    host_sed_prediction_reliable: bool = True
    warnings: List[str] = field(default_factory=list)
    strategy_requested: str = "masked_simple"
    strategy_used: str = "masked_simple"
    strategy_fallback: bool = False
    strategy_fallback_reason: Optional[str] = None
    component_models_log: Dict[str, np.ndarray] = field(default_factory=dict)
    component_models: Dict[str, np.ndarray] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)
    component_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage: Optional[HostCoverageResult] = None
    ppxf_agn_fraction_flux_global: float = np.nan
    ppxf_high_agn_fraction_warning: bool = False
    closure_metrics: Dict[str, Any] = field(default_factory=dict)
    host_reconstruction_state: Dict[str, Any] = field(default_factory=dict)
    ppxf_result: Any = None


@dataclass
class HostSED:
    """Template-weighted host SED prediction."""

    wave_rest: np.ndarray
    host_flux: np.ndarray
    samples: Dict[str, float]
    flags: Dict[str, bool]
    warnings: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


class HostSEDReconstructionError(RuntimeError):
    """Structured failure to restore a stellar HostSED from compact state."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class HostReconstructionState:
    """Compact, versioned state required to reproduce a stellar HostSED."""

    host_reconstruction_state_version: str
    stellar_weights: List[float]
    stellar_template_scales: List[float]
    preprocessing_normalization: float
    template_family: str
    template_file_name: str
    template_file_sha256: str
    template_profile: str
    template_product_kind: str
    fit_template_file_name: str
    fit_template_file_sha256: str
    fit_template_wave_sha256: str
    fit_template_matrix_sha256: str
    source_template_file_name: str
    source_template_file_sha256: str
    source_template_wave_sha256: str
    source_template_matrix_sha256: str
    template_axis_metadata: Dict[str, Any]
    template_grid_id: str
    template_wave_sha256: str
    template_matrix_sha256: str
    template_loader_version: str
    template_original_shape: List[int]
    template_flattening_order_convention: str
    selected_template_indices: Optional[List[int]]
    host_sed_method: str
    host_sed_method_version: str
    host_sed_wavelength_min: float
    host_sed_wavelength_max: float
    strategy_used: str
    fit_redshift: float
    fit_normalization_convention: str
    fit_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HostDecompWorkflowResult:
    """Combined pPXF + qsospec decomposition result."""

    spectrum: SpectrumData
    ppxf_result: PPXFHostFitResult
    host_sed: HostSED
    host_subtracted_flux: np.ndarray
    qsospec_status: str
    qsospec_result_path: Optional[str]
    output_files: Dict[str, str]
    summary: Dict[str, Any]


def _require_ppxf():
    try:
        from ppxf.ppxf import ppxf
    except Exception as exc:
        raise RuntimeError(
            "pPXF is required for host decomposition but is not importable. "
            "Install it locally with `pip install ppxf` and keep template data outside qsospec."
        ) from exc
    return ppxf


def make_emission_line_mask(
    wave_rest: np.ndarray,
    line_mask_widths: Optional[Dict[str, float]] = None,
    broad_line_mask_widths: Optional[Dict[str, float]] = None,
    use_broad_masks: bool = True,
) -> np.ndarray:
    """Return a boolean mask for emission-line regions on a rest-frame grid.

    Widths are interpreted as velocity half-widths in km/s.
    """

    wave_rest = np.asarray(wave_rest, dtype=float)
    mask = np.zeros_like(wave_rest, dtype=bool)
    widths = dict(line_mask_widths or {})
    broad_widths = dict(broad_line_mask_widths or {})
    for name, center in DEFAULT_LINE_CENTERS.items():
        width = widths.get(name, widths.get("default", 800.0))
        if use_broad_masks and name in {"MgII", "Hdelta", "Hgamma", "Hbeta", "Halpha"}:
            width = broad_widths.get(name, broad_widths.get("default", width))
        delta = float(center) * float(width) / _C_KMS
        mask |= (wave_rest >= center - delta) & (wave_rest <= center + delta)
    return mask


def _window_mask(wave: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(np.asarray(wave, dtype=float), dtype=bool)
    for lower, upper in windows:
        mask |= (wave >= float(lower)) & (wave <= float(upper))
    return mask


def _log_resample(
    wave: np.ndarray,
    flux: np.ndarray,
    noise: np.ndarray,
    emission_mask: np.ndarray,
    source_indices: np.ndarray,
    native_spacing: float,
    max_native_gap_pixels: float,
):
    wave = np.asarray(wave, dtype=float)
    log_wave = np.linspace(np.log(wave[0]), np.log(wave[-1]), len(wave))
    wave_log = np.exp(log_wave)
    wave_log[0] = wave[0]
    wave_log[-1] = wave[-1]
    log_wave[0] = np.log(wave[0])
    log_wave[-1] = np.log(wave[-1])
    flux_log = np.interp(wave_log, wave, flux)
    noise_log = np.interp(wave_log, wave, noise)
    mask_float = np.interp(wave_log, wave, emission_mask.astype(float))
    emission_mask_log = mask_float > 0.1
    validity_mask_log = np.zeros_like(wave_log, dtype=bool)
    index_break = np.diff(source_indices) > 1
    wavelength_break = np.diff(wave) > float(max_native_gap_pixels) * native_spacing
    boundaries = np.flatnonzero(index_break | wavelength_break) + 1
    for segment in np.split(np.arange(len(wave)), boundaries):
        if segment.size < 2:
            continue
        validity_mask_log |= (
            (wave_log >= wave[segment[0]]) & (wave_log <= wave[segment[-1]])
        )
    velscale = float(np.diff(log_wave).mean() * _C_KMS)
    return (
        wave_log,
        log_wave,
        flux_log,
        noise_log,
        emission_mask_log,
        validity_mask_log,
        velscale,
    )


def prepare_spectrum_for_host_decomp(
    spectrum: SpectrumData,
    redshift: Optional[float] = None,
    fit_range: Tuple[float, float] = (3600.0, 7000.0),
    line_mask_widths: Optional[Dict[str, float]] = None,
    broad_line_mask_widths: Optional[Dict[str, float]] = None,
    use_broad_line_masks: bool = True,
    observed_artifact_windows: Sequence[Tuple[float, float]] = (
        DEFAULT_OBSERVED_ARTIFACT_WINDOWS
    ),
    max_native_gap_pixels: float = 3.0,
    systematic_error_floor_fraction: float = 0.02,
) -> PreprocessedSpectrum:
    """Clean, rest-frame, mask, normalize, and log-resample a spectrum."""

    z = spectrum.redshift if redshift is None else redshift
    if z is None:
        raise ValueError("A redshift is required for host decomposition.")
    z = float(z)

    wave_obs = np.asarray(spectrum.wave_obs, dtype=float)
    flux = np.asarray(spectrum.flux, dtype=float)
    err = np.asarray(spectrum.uncertainty(), dtype=float)
    finite_valid = (
        np.isfinite(wave_obs)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (wave_obs > 0)
        & (err > 0)
    )
    valid = finite_valid.copy()
    if spectrum.ivar is not None:
        ivar = np.asarray(spectrum.ivar, dtype=float)
        valid &= np.isfinite(ivar) & (ivar > 0)
    else:
        ivar = None
    input_mask_rejected = np.zeros_like(valid)
    if spectrum.mask is not None:
        mask = np.asarray(spectrum.mask)
        input_mask_rejected = mask != 0
        valid &= ~input_mask_rejected
    artifact_rejected = _window_mask(wave_obs, observed_artifact_windows)
    valid &= ~artifact_rejected
    source_indices = np.flatnonzero(valid)
    native_differences = np.diff(wave_obs[np.isfinite(wave_obs)])
    native_spacing = float(np.nanmedian(native_differences[native_differences > 0]))
    if not np.isfinite(native_spacing) or native_spacing <= 0:
        native_spacing = 1.0

    warnings_out: List[str] = []
    if np.sum(valid) < 20:
        warnings_out.append("few_valid_pixels_after_cleaning")

    wave_obs = wave_obs[valid]
    flux = flux[valid]
    err = err[valid]
    ivar_clean = ivar[valid] if ivar is not None else None
    order = np.argsort(wave_obs)
    wave_obs = wave_obs[order]
    flux = flux[order]
    err = err[order]
    if ivar_clean is not None:
        ivar_clean = ivar_clean[order]
    source_indices = source_indices[order]

    wave_rest = wave_obs / (1.0 + z)
    fit_mask = (wave_rest >= fit_range[0]) & (wave_rest <= fit_range[1])
    if np.sum(fit_mask) < 20:
        raise ValueError(
            f"Too few pixels in rest-frame fit range {fit_range}: {int(np.sum(fit_mask))}"
        )

    emission_mask = make_emission_line_mask(
        wave_rest,
        line_mask_widths=line_mask_widths,
        broad_line_mask_widths=broad_line_mask_widths,
        use_broad_masks=use_broad_line_masks,
    )
    fit_wave = wave_rest[fit_mask]
    fit_flux = flux[fit_mask]
    fit_err = err[fit_mask]
    fit_emission_mask = emission_mask[fit_mask]

    normalization = float(np.nanmedian(np.abs(fit_flux[np.isfinite(fit_flux)])))
    if not np.isfinite(normalization) or normalization <= 0:
        normalization = 1.0
        warnings_out.append("normalization_fallback_to_one")
    systematic_floor = (
        float(systematic_error_floor_fraction)
        * float(np.nanmedian(np.abs(fit_flux[np.isfinite(fit_flux)])))
    )
    fit_err = np.sqrt(fit_err**2 + systematic_floor**2)
    err[fit_mask] = fit_err
    fit_flux_norm = fit_flux / normalization
    fit_err_norm = np.clip(fit_err / normalization, 1e-12, np.inf)

    (
        wave_log,
        log_wave,
        flux_log,
        noise_log,
        emission_mask_log,
        validity_mask_log,
        velscale,
    ) = _log_resample(
        fit_wave,
        fit_flux_norm,
        fit_err_norm,
        fit_emission_mask,
        source_indices[fit_mask],
        native_spacing,
        max_native_gap_pixels,
    )

    return PreprocessedSpectrum(
        wave_obs=wave_obs,
        wave_rest=wave_rest,
        flux=flux,
        error=err,
        ivar=ivar_clean,
        fit_mask=fit_mask,
        emission_mask=emission_mask,
        wave_log=wave_log,
        log_wave=log_wave,
        flux_log=flux_log,
        noise_log=noise_log,
        emission_mask_log=emission_mask_log,
        validity_mask_log=validity_mask_log,
        artifact_mask_log=_window_mask(
            wave_log * (1.0 + z),
            observed_artifact_windows,
        ),
        normalization=normalization,
        redshift=z,
        velscale=velscale,
        metadata={
            **dict(spectrum.metadata),
            "object_id": spectrum.object_id or spectrum.targetid,
            "observed_artifact_windows": [
                [float(lower), float(upper)]
                for lower, upper in observed_artifact_windows
            ],
            "systematic_error_floor_fraction": float(
                systematic_error_floor_fraction
            ),
            "systematic_error_floor": float(systematic_floor),
            "max_native_gap_pixels": float(max_native_gap_pixels),
            "host_fit_range": [float(fit_range[0]), float(fit_range[1])],
            "native_data_preserved": True,
        },
        mask_provenance={
            "input_mask_rejected": input_mask_rejected,
            "invalid_or_nonpositive_error_rejected": ~finite_valid,
            "observed_artifact_rejected": artifact_rejected,
            "native_valid": valid,
            "log_grid_valid": validity_mask_log,
            "artifact_mask_log": _window_mask(
                wave_log * (1.0 + z),
                observed_artifact_windows,
            ),
        },
        warnings=warnings_out,
    )


def prepare_desi_for_host_decomp(*args, **kwargs) -> PreprocessedSpectrum:
    """Deprecated alias for :func:`prepare_spectrum_for_host_decomp`."""

    warnings.warn(
        "prepare_desi_for_host_decomp is deprecated; use "
        "prepare_spectrum_for_host_decomp instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_spectrum_for_host_decomp(*args, **kwargs)


def _build_agn_basis(wave: np.ndarray, slopes: Sequence[float]) -> np.ndarray:
    pivot = 5100.0
    basis = []
    for slope in slopes:
        vec = (np.asarray(wave, dtype=float) / pivot) ** float(slope)
        scale = np.nanmedian(np.abs(vec[np.isfinite(vec)]))
        basis.append(vec / (scale if scale > 0 else 1.0))
    return np.column_stack(basis) if basis else np.zeros((len(wave), 0))


def _resample_stellar_templates(prep: PreprocessedSpectrum, templates: PPXFTemplateLibrary):
    wave = prep.wave_log
    in_range = (
        (wave >= templates.wavelength_coverage[0])
        & (wave <= templates.wavelength_coverage[1])
    )
    if np.sum(in_range) < len(wave):
        warnings.warn(
            "Template wavelength coverage does not span the full pPXF fit range.",
            RuntimeWarning,
    )
    if templates.product_kind == "preconvolved":
        validation_start = perf_counter()
        validation = validate_preconvolved_xsl_product(
            templates,
            prep,
            fit_range=tuple(prep.metadata.get("host_fit_range", ())),
            object_key=(
                prep.metadata.get("host_preconvolution_object_key")
                or prep.metadata.get("object_id")
            ),
        )
        validation["preconvolved_validation_seconds"] = float(
            perf_counter() - validation_start
        )
        matrix = np.asarray(templates.fit_flux, dtype=float).copy()
        if matrix.shape != (len(wave), templates.n_templates):
            raise ValueError(
                "Preconvolved XSL matrix shape does not match the exact pPXF grid."
            )
        prep.metadata.update(validation)
    else:
        matrix = np.empty((len(wave), templates.n_templates), dtype=float)
        for j in range(templates.n_templates):
            matrix[:, j] = np.interp(
                wave,
                templates.fit_wave,
                templates.fit_flux[:, j],
                left=0.0,
                right=0.0,
            )
    resolution = prep.metadata.get("spectral_resolution")
    if resolution is None:
        prep.metadata["host_fit_resolution_status"] = "missing"
        prep.metadata["template_resolution_status"] = "data_resolution_missing"
    else:
        prep.metadata.update(resolution.metadata())
        prep.metadata["host_fit_resolution_status"] = resolution.status
        if resolution.status == "valid" and resolution.mode == "banded_matrix":
            prep.metadata["host_fit_resolution_status"] = "unsupported_banded_resolution_matrix"
            prep.metadata["template_resolution_status"] = "data_resolution_missing"
        elif resolution.status == "valid":
            lsf_interpolation_start = perf_counter()
            observed = wave * (1.0 + prep.redshift)
            data_sigma = resolution.sigma_lambda(observed) / (1.0 + prep.redshift)
            template_fwhm = templates.source_resolution_metadata.get("fwhm")
            if template_fwhm is None:
                prep.metadata["template_resolution_status"] = (
                    "template_resolution_unknown"
                )
                prep.warnings.append("template_resolution_metadata_missing")
            else:
                fwhm = np.asarray(template_fwhm, dtype=float)
                if fwhm.size == 1:
                    template_sigma = np.full_like(wave, float(fwhm.item()))
                else:
                    template_sigma = np.interp(
                        wave,
                        np.asarray(templates.source_wave, dtype=float),
                        fwhm,
                        left=np.nan,
                        right=np.nan,
                    )
                template_sigma /= 2.354820045
                prep.metadata["lsf_interpolation_seconds"] = float(
                    perf_counter() - lsf_interpolation_start
                )
                match = match_template_resolution_toward_data(
                    data_sigma,
                    template_sigma,
                    wave,
                )
                in_range &= match.comparable
                prep.metadata.update(match.metadata)
                prep.metadata["template_resolution_comparable_mask"] = (
                    match.comparable.tolist()
                )
                prep.metadata["template_coarser_than_data_mask"] = (
                    match.template_coarser_than_data.tolist()
                )
                coarser = match.template_coarser_than_data & in_range
                if np.any(coarser):
                    prep.metadata["template_coarser_wave_min"] = float(
                        np.min(wave[coarser])
                    )
                    prep.metadata["template_coarser_wave_max"] = float(
                        np.max(wave[coarser])
                    )
                    if templates.metadata.get("template_coarser_action", "warn") == "warn":
                        prep.warnings.append("template_coarser_than_data")
                else:
                    prep.metadata["template_coarser_wave_min"] = np.nan
                    prep.metadata["template_coarser_wave_max"] = np.nan
                pixel_width = np.gradient(wave)
                broadening = np.nan_to_num(
                    match.additional_sigma_lambda, nan=0.0
                )
                sigma_pixels = np.divide(
                    broadening,
                    pixel_width,
                    out=np.zeros_like(broadening),
                    where=pixel_width > 0,
                )
                if templates.product_kind == "preconvolved":
                    prep.metadata["template_resolution_status"] = (
                        "preconvolved_exact"
                    )
                    prep.metadata["preconvolution_residual_match_status"] = (
                        "not_required"
                    )
                else:
                    try:
                        from ppxf.ppxf_util import (
                            gaussian_filter1d as variable_gaussian_filter1d,
                        )

                        runtime_convolution_start = perf_counter()
                        for j in range(matrix.shape[1]):
                            matrix[:, j] = variable_gaussian_filter1d(
                                matrix[:, j], sigma_pixels
                            )
                        prep.metadata["runtime_convolution_seconds"] = float(
                            perf_counter() - runtime_convolution_start
                        )
                    except Exception:
                        prep.metadata["host_fit_resolution_status"] = (
                            "invalid_variable_broadening_unavailable"
                        )
                prep.metadata["effective_data_sigma_lambda"] = data_sigma.tolist()
                prep.metadata["effective_template_sigma_lambda"] = template_sigma.tolist()
                prep.metadata["additional_template_sigma_lambda"] = broadening.tolist()
    prep.metadata.update(
        {
            "stellar_template_profile": templates.profile_id,
            "stellar_template_family": templates.family,
            "stellar_template_product_kind": templates.product_kind,
            "stellar_template_fit_file": templates.fit_source_path,
            "stellar_template_fit_sha256": templates.fit_source_sha256,
            "stellar_template_source_file": templates.source_library_path,
            "stellar_template_source_sha256": templates.source_library_sha256,
            "native_data_preserved": True,
            "template_resolution_matching_mode": templates.metadata.get(
                "resolution_matching_mode"
            ),
        }
    )
    scales = np.nanmedian(np.abs(matrix), axis=0)
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    return matrix / scales, scales, in_range


def _interpolate_component(
    preprocessed: PreprocessedSpectrum,
    values_log: np.ndarray,
) -> np.ndarray:
    values = np.interp(
        preprocessed.wave_rest,
        preprocessed.wave_log,
        np.asarray(values_log, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    return values * preprocessed.normalization


def _closure_diagnostics(
    bestfit: np.ndarray,
    physical_total: np.ndarray,
    normalization: float,
    *,
    explained_by_legacy_polynomial: bool,
) -> Dict[str, Any]:
    residual = np.asarray(bestfit, dtype=float) - np.asarray(physical_total, dtype=float)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {
            "closure_rms": np.nan,
            "closure_median_absolute": np.nan,
            "closure_p95_absolute": np.nan,
            "closure_max_absolute": np.nan,
            "closure_relative_to_normalization": np.nan,
            "closure_status": "unavailable",
        }
    absolute = np.abs(finite)
    rms = float(np.sqrt(np.mean(finite**2)))
    relative = rms / max(abs(float(normalization)), np.finfo(float).eps)
    if relative <= 1.0e-6:
        status = "numerical"
    elif explained_by_legacy_polynomial:
        status = "legacy_polynomial_not_decomposed"
    else:
        status = "mismatch"
    return {
        "closure_rms": rms,
        "closure_median_absolute": float(np.median(absolute)),
        "closure_p95_absolute": float(np.percentile(absolute, 95.0)),
        "closure_max_absolute": float(np.max(absolute)),
        "closure_relative_to_normalization": float(relative),
        "closure_status": status,
    }


def _global_agn_fraction(
    wave: np.ndarray,
    stellar: np.ndarray,
    agn: np.ndarray,
    good_mask: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    mask = (
        np.asarray(good_mask, dtype=bool)
        & np.isfinite(wave)
        & np.isfinite(stellar)
        & np.isfinite(agn)
    )
    count = int(np.count_nonzero(mask))
    available = int(np.count_nonzero(np.isfinite(wave)))
    metadata = {
        "ppxf_agn_fraction_wave_min": float(np.nanmin(wave[mask])) if count else np.nan,
        "ppxf_agn_fraction_wave_max": float(np.nanmax(wave[mask])) if count else np.nan,
        "ppxf_agn_fraction_valid_pixel_count": count,
        "ppxf_agn_fraction_valid_fraction": float(count / available) if available else 0.0,
        "ppxf_agn_fraction_definition_version": "flux_integral_v1",
    }
    if count < 2:
        metadata["ppxf_agn_fraction_status"] = "insufficient_pixels"
        return np.nan, metadata
    order = np.argsort(wave[mask])
    selected_wave = wave[mask][order]
    agn_integral = float(np.trapezoid(agn[mask][order], selected_wave))
    denominator = float(
        np.trapezoid((agn[mask] + stellar[mask])[order], selected_wave)
    )
    if not np.isfinite(denominator) or denominator <= 0:
        metadata["ppxf_agn_fraction_status"] = "invalid_denominator"
        return np.nan, metadata
    metadata["ppxf_agn_fraction_status"] = "available"
    return float(agn_integral / denominator), metadata


def run_ppxf_host_fit(
    preprocessed: PreprocessedSpectrum,
    templates: PPXFTemplateLibrary,
    agn_powerlaw_slopes: Sequence[float] = (-2.0, -1.5, -1.0, -0.5, 0.0),
    polynomial_degree: int = 4,
    multiplicative_polynomial_degree: int = 0,
    adaptive_broad_line_max_velocity: float = 10000.0,
    adaptive_line_residual_sigma: float = 3.0,
    residual_clip_sigma: float = 4.5,
    residual_clip_iterations: int = 2,
    residual_clip_dilation_pixels: int = 2,
    max_noise_rescale: float = 5.0,
    minimum_clean_fraction: float = 0.35,
    minimum_clean_pixels: int = 200,
    minimum_continuum_snr: float = 2.0,
    maximum_clipped_fraction: float = 0.25,
    strategy: str = "masked_simple",
    strategy_requested: Optional[str] = None,
    strategy_fallback: bool = False,
    strategy_fallback_reason: Optional[str] = None,
    agn_pseudocontinuum_config: Optional[HostAgnPseudoContinuumConfig] = None,
    selected_pseudocontinuum_fwhm_kms: Optional[float] = None,
    coverage_config: Optional[HostCoverageConfig] = None,
    quiet: bool = True,
) -> PPXFHostFitResult:
    """Run a staged, robust pPXF stellar-host fit."""

    ppxf = _require_ppxf()
    try:
        from importlib.metadata import version as package_version

        ppxf_version = package_version("ppxf")
    except Exception:  # noqa: BLE001 - provenance is best effort
        ppxf_version = "unknown"
    warnings_out = list(preprocessed.warnings) + list(templates.warnings)
    if strategy not in {"masked_simple", "agn_pseudocontinuum_masked"}:
        raise ValueError(f"Unknown host strategy: {strategy!r}")
    stellar_template_start = perf_counter()
    stellar_matrix, stellar_scales, in_template_range = _resample_stellar_templates(preprocessed, templates)
    stellar_template_seconds = perf_counter() - stellar_template_start
    warnings_out = list(
        dict.fromkeys(
            [*warnings_out, *preprocessed.warnings, *templates.warnings]
        )
    )
    agn_bundle: Optional[HostAgnTemplateBundle] = None
    agn_template_seconds = 0.0
    if strategy == "agn_pseudocontinuum_masked":
        if selected_pseudocontinuum_fwhm_kms is None:
            raise ValueError(
                "agn_pseudocontinuum_masked requires a selected broad-line width."
            )
        agn_cfg = agn_pseudocontinuum_config or HostAgnPseudoContinuumConfig()
        agn_template_start = perf_counter()
        agn_bundle = build_host_agn_template_bundle(
            preprocessed.wave_log,
            selected_fwhm_kms=float(selected_pseudocontinuum_fwhm_kms),
            valid_mask=preprocessed.validity_mask_log & in_template_range,
            config=agn_cfg,
            redshift=preprocessed.redshift,
            spectral_resolution=preprocessed.metadata.get("spectral_resolution"),
        )
        agn_template_seconds = perf_counter() - agn_template_start
        agn_matrix = agn_bundle.matrix
        in_template_range &= agn_bundle.support_mask
        polynomial_degree = int(agn_cfg.additive_polynomial_degree)
        multiplicative_polynomial_degree = int(
            agn_cfg.multiplicative_polynomial_degree
        )
        active_slopes = np.asarray(agn_cfg.powerlaw_slopes, dtype=float)
    else:
        agn_matrix = _build_agn_basis(preprocessed.wave_log, agn_powerlaw_slopes)
        active_slopes = np.asarray(agn_powerlaw_slopes, dtype=float)
    fit_templates = np.column_stack([stellar_matrix, agn_matrix])

    base_good = (
        np.isfinite(preprocessed.flux_log)
        & np.isfinite(preprocessed.noise_log)
        & (preprocessed.noise_log > 0)
        & preprocessed.validity_mask_log
        & in_template_range
    )
    initial_emission = np.asarray(
        preprocessed.emission_mask_log, dtype=bool
    ).copy()

    def fit_once(
        emission_mask: np.ndarray,
        clip_mask: np.ndarray,
        noise: np.ndarray,
    ):
        good = base_good & (~emission_mask) & (~clip_mask)
        goodpixels = np.flatnonzero(good)
        if goodpixels.size < max(20, fit_templates.shape[1] // 2):
            raise ValueError(
                f"Too few pPXF good pixels after masking: {goodpixels.size}"
            )
        fit_kwargs = {
            "goodpixels": goodpixels,
            "degree": int(polynomial_degree),
            "mdegree": int(multiplicative_polynomial_degree),
            "quiet": quiet,
        }
        start: Any = [0.0, 150.0]
        if strategy == "agn_pseudocontinuum_masked" and agn_matrix.shape[1]:
            fit_kwargs.update(
                {
                    "component": np.r_[
                        np.zeros(stellar_matrix.shape[1], dtype=int),
                        np.ones(agn_matrix.shape[1], dtype=int),
                    ],
                    "moments": [2, -2],
                    "linear_method": "lsq_box",
                }
            )
            start = [[0.0, 150.0], [0.0, 0.01]]
        fitted = ppxf(
            fit_templates,
            preprocessed.flux_log,
            noise,
            preprocessed.velscale,
            start=start,
            **fit_kwargs,
        )
        return fitted, good

    empty_clip = np.zeros_like(base_good)
    noise_work = np.asarray(preprocessed.noise_log, dtype=float).copy()
    ppxf_fit_start = perf_counter()
    result, _ = fit_once(initial_emission, empty_clip, noise_work)

    standardized = (
        preprocessed.flux_log
        - np.asarray(result.bestfit, dtype=float)
    ) / noise_work
    smooth_positive = gaussian_filter1d(
        np.where(np.isfinite(standardized), standardized, 0.0),
        sigma=2.0,
        mode="nearest",
    )
    significant = smooth_positive > float(adaptive_line_residual_sigma)
    expanded_emission = initial_emission.copy()
    for name in ("MgII", "Hdelta", "Hgamma", "Hbeta", "Halpha"):
        center = float(DEFAULT_LINE_CENTERS[name])
        delta = (
            center * float(adaptive_broad_line_max_velocity) / _C_KMS
        )
        cap = (
            (preprocessed.wave_log >= center - delta)
            & (preprocessed.wave_log <= center + delta)
        )
        seed = initial_emission & cap
        if not np.any(seed):
            continue
        connected_domain = (
            binary_dilation(significant, iterations=2) | seed
        ) & cap
        expanded_emission |= binary_propagation(
            seed,
            mask=connected_domain,
        )

    result, preclip_good = fit_once(
        expanded_emission, empty_clip, noise_work
    )

    residual = (
        preprocessed.flux_log
        - np.asarray(result.bestfit, dtype=float)
    )
    noise_rescale_factors: Dict[str, float] = {}
    observed_wave_log = preprocessed.wave_log * (1.0 + preprocessed.redshift)
    arm_ranges = {
        "b": (3600.0, 5800.0),
        "r": (5760.0, 7620.0),
        "z": (7520.0, 9824.0),
    }
    for arm, (lower, upper) in arm_ranges.items():
        selected = (
            preclip_good
            & (observed_wave_log >= lower)
            & (observed_wave_log <= upper)
        )
        if np.count_nonzero(selected) < 20:
            noise_rescale_factors[arm] = 1.0
            continue
        normalized = residual[selected] / noise_work[selected]
        center = float(np.nanmedian(normalized))
        scatter = 1.4826 * float(
            np.nanmedian(np.abs(normalized - center))
        )
        factor = float(
            np.clip(scatter if np.isfinite(scatter) else 1.0, 1.0, max_noise_rescale)
        )
        noise_rescale_factors[arm] = factor
        noise_work[selected] *= factor

    result, _ = fit_once(expanded_emission, empty_clip, noise_work)
    residual_clip = np.zeros_like(base_good)
    for _ in range(int(residual_clip_iterations)):
        current_residual = (
            preprocessed.flux_log
            - np.asarray(result.bestfit, dtype=float)
        ) / noise_work
        candidate = (
            base_good
            & (~expanded_emission)
            & np.isfinite(current_residual)
            & (np.abs(current_residual) > float(residual_clip_sigma))
        )
        if int(residual_clip_dilation_pixels) > 0:
            candidate = binary_dilation(
                candidate,
                iterations=int(residual_clip_dilation_pixels),
            )
        updated = residual_clip | (
            candidate & base_good & (~expanded_emission)
        )
        if np.array_equal(updated, residual_clip):
            break
        residual_clip = updated
        result, _ = fit_once(
            expanded_emission, residual_clip, noise_work
        )

    result, final_good = fit_once(
        expanded_emission, residual_clip, noise_work
    )
    preprocessed.noise_log = noise_work
    preprocessed.mask_provenance.update(
        {
            "initial_emission_mask_log": initial_emission,
            "expanded_emission_mask_log": expanded_emission,
            "residual_clip_mask_log": residual_clip,
            "final_goodpixels_mask_log": final_good,
        }
    )

    available_count = int(np.count_nonzero(base_good))
    clean_count = int(np.count_nonzero(final_good))
    clean_fraction = (
        float(clean_count / available_count) if available_count else 0.0
    )
    preclip_count = int(np.count_nonzero(preclip_good))
    clipped_count = int(np.count_nonzero(residual_clip & preclip_good))
    clipped_fraction = (
        float(clipped_count / preclip_count) if preclip_count else 1.0
    )
    continuum_snr = float(
        np.nanmedian(
            np.abs(preprocessed.flux_log[final_good])
            / noise_work[final_good]
        )
    ) if clean_count else 0.0
    coverage = classify_host_coverage(
        preprocessed.wave_log,
        base_good,
        coverage_config,
    )
    reliability_reasons: List[str] = []
    resolution_status = str(preprocessed.metadata.get("host_fit_resolution_status", "missing"))
    if resolution_status != "valid":
        reliability_reasons.append("resolution_approximate_or_missing")
    if clean_fraction < float(minimum_clean_fraction):
        reliability_reasons.append("clean_fraction_below_threshold")
    if clean_count < int(minimum_clean_pixels):
        reliability_reasons.append("too_few_clean_pixels")
    if continuum_snr < float(minimum_continuum_snr):
        reliability_reasons.append("continuum_snr_below_threshold")
    if clipped_fraction > float(maximum_clipped_fraction):
        reliability_reasons.append("clipped_fraction_above_threshold")
    if coverage.coverage_class == "blue_optical":
        reliability_reasons.append("limited_wavelength_leverage")
    elif coverage.coverage_class == "insufficient":
        reliability_reasons.append("insufficient_host_wavelength_coverage")
    ppxf_fit_seconds = perf_counter() - ppxf_fit_start
    quality_metrics: Dict[str, Any] = {
        "available_pixel_count": available_count,
        "clean_pixel_count": clean_count,
        "clean_fraction": clean_fraction,
        "median_continuum_snr": continuum_snr,
        "clipped_pixel_count": clipped_count,
        "clipped_fraction": clipped_fraction,
        "initial_emission_mask_count": int(
            np.count_nonzero(initial_emission)
        ),
        "expanded_emission_mask_count": int(
            np.count_nonzero(expanded_emission)
        ),
        "final_goodpixel_count": clean_count,
        "resolution_status": resolution_status,
        "template_coverage_fraction": float(np.count_nonzero(in_template_range) / len(in_template_range)),
        "host_coverage_class": coverage.coverage_class,
        "host_feature_coverage": dict(coverage.feature_coverage),
        "host_coverage_reasons": list(coverage.reasons),
        "ppxf_version": ppxf_version,
        "stellar_template_source": templates.source_path,
        "stellar_template_sha256": templates.metadata.get("source_sha256"),
        "stellar_template_prepare_seconds": float(
            stellar_template_seconds
        ),
        "agn_template_build_seconds": float(agn_template_seconds),
        "ppxf_fit_seconds": float(ppxf_fit_seconds),
    }
    resolution_diagnostic_keys = (
        "stellar_template_profile",
        "stellar_template_family",
        "stellar_template_product_kind",
        "stellar_template_fit_file",
        "stellar_template_fit_sha256",
        "stellar_template_source_file",
        "stellar_template_source_sha256",
        "native_data_preserved",
        "template_resolution_matching_mode",
        "template_resolution_status",
        "template_sharper_than_data_fraction",
        "template_equal_to_data_fraction",
        "template_coarser_than_data_fraction",
        "template_coarser_wave_min",
        "template_coarser_wave_max",
        "median_template_minus_data_sigma_angstrom",
        "p95_template_minus_data_sigma_angstrom",
        "maximum_template_minus_data_sigma_angstrom",
        "median_template_minus_data_sigma_kms",
        "p95_template_minus_data_sigma_kms",
        "maximum_template_minus_data_sigma_kms",
        "additional_template_sigma_nonzero_fraction",
        "additional_template_sigma_median_angstrom",
        "additional_template_sigma_p95_angstrom",
        "preconvolution_cache_key",
        "preconvolution_validation_status",
        "preconvolution_residual_match_status",
        "preconvolved_validation_seconds",
        "source_template_load_seconds",
        "preconvolved_cache_read_seconds",
        "lsf_interpolation_seconds",
        "runtime_convolution_seconds",
    )
    quality_metrics.update(
        {
            key: preprocessed.metadata.get(key)
            for key in resolution_diagnostic_keys
            if key in preprocessed.metadata
        }
    )
    for timing_name in (
        "source_template_load_seconds",
        "preconvolved_cache_read_seconds",
    ):
        if templates.metadata.get(timing_name) is not None:
            quality_metrics[timing_name] = templates.metadata[timing_name]
    coarser_mask = np.asarray(
        preprocessed.metadata.get(
            "template_coarser_than_data_mask",
            np.zeros_like(final_good),
        ),
        dtype=bool,
    )
    quality_metrics["template_coarser_than_data_fraction_goodpixels"] = (
        float(np.count_nonzero(coarser_mask & final_good) / clean_count)
        if clean_count and coarser_mask.shape == final_good.shape
        else np.nan
    )

    weights = np.asarray(getattr(result, "weights", np.zeros(fit_templates.shape[1])), dtype=float)
    n_stellar = stellar_matrix.shape[1]
    stellar_weights = weights[:n_stellar]
    agn_weights = weights[n_stellar:]
    ppxf_matrix = np.asarray(getattr(result, "matrix", fit_templates), dtype=float)
    transformed_templates = ppxf_matrix[:, -len(weights):]
    transformed_stellar = transformed_templates[:, :n_stellar]
    transformed_agn = transformed_templates[:, n_stellar:]
    host_log_norm = transformed_stellar @ stellar_weights
    agn_log_norm = (
        transformed_agn @ agn_weights
        if transformed_agn.size
        else np.zeros_like(host_log_norm)
    )
    total_log_norm = np.asarray(
        getattr(result, "bestfit", host_log_norm + agn_log_norm),
        dtype=float,
    )
    residual_log_norm = preprocessed.flux_log - total_log_norm

    component_models_log_norm: Dict[str, np.ndarray] = {
        "stellar": host_log_norm,
    }
    component_weights: Dict[str, float] = {}
    component_metadata: Dict[str, Dict[str, Any]] = {
        "stellar": {
            "template_profile": templates.profile_id,
            "template_family": templates.family,
            "fit_template_resolution_product": templates.product_kind,
            "source_template_resolution_product": "native",
            "fit_template_sha256": templates.fit_source_sha256,
            "source_template_sha256": templates.source_library_sha256,
            "native_data_preserved": True,
        }
    }
    if agn_bundle is None:
        component_models_log_norm["powerlaw"] = agn_log_norm
        component_weights.update(
            {
                f"powerlaw_slope_{slope:+.1f}": float(weight)
                for slope, weight in zip(active_slopes, agn_weights)
            }
        )
    else:
        aggregates: Dict[str, np.ndarray] = {}
        category_names = {
            "agn_powerlaw": "powerlaw",
            "agn_feii_optical": "feii_optical",
            "agn_feii_uv": "feii_uv",
            "agn_balmer_continuum": "balmer_continuum",
            "agn_balmer_high_order": "balmer_high_order",
        }
        grouped_components: Dict[str, list] = {}
        for component_item in agn_bundle.components:
            grouped_components.setdefault(
                component_item.linear_group, []
            ).append(component_item)
        for group, items in grouped_components.items():
            column = agn_bundle.group_column_indices[group]
            weight = float(agn_weights[column])
            group_model = transformed_agn[:, column] * weight
            raw_group = np.sum([item.values for item in items], axis=0)
            for component_item in items:
                fraction = np.divide(
                    component_item.values,
                    raw_group,
                    out=np.zeros_like(raw_group),
                    where=np.abs(raw_group) > np.finfo(float).eps,
                )
                model = group_model * fraction
                key = category_names[component_item.category]
                aggregates[key] = aggregates.get(key, np.zeros_like(model)) + model
                component_weights[component_item.name] = weight
                component_metadata[component_item.name] = {
                    **component_item.metadata,
                    "category": component_item.category,
                    "source_id": component_item.source_id,
                    "source_reference": component_item.source_reference,
                    "selected_fwhm_kms": component_item.selected_fwhm_kms,
                    "normalization": component_item.normalization,
                    "wavelength_coverage": list(component_item.wavelength_coverage),
                }
        component_models_log_norm.update(aggregates)
        component_metadata["bundle"] = dict(agn_bundle.metadata)
    component_models_log_norm["agn_total"] = agn_log_norm

    additive = np.zeros_like(total_log_norm)
    multiplicative_effect = np.zeros_like(total_log_norm)
    if strategy == "agn_pseudocontinuum_masked":
        if int(polynomial_degree) >= 0:
            candidate = np.asarray(
                getattr(result, "apoly", np.zeros_like(total_log_norm)),
                dtype=float,
            )
            if candidate.shape == additive.shape:
                additive = candidate
        if int(multiplicative_polynomial_degree) > 0:
            multiplicative_effect = total_log_norm - (
                host_log_norm + agn_log_norm + additive
            )
    physical_total = (
        host_log_norm + agn_log_norm + additive + multiplicative_effect
    )
    closure_residual = total_log_norm - physical_total
    component_models_log_norm.update(
        {
            "polynomial_additive": additive,
            "polynomial_multiplicative_effect": multiplicative_effect,
            "physical_component_total": physical_total,
            "ppxf_bestfit": total_log_norm,
            "closure_residual": closure_residual,
        }
    )
    closure_metrics = _closure_diagnostics(
        total_log_norm,
        physical_total,
        1.0,
        explained_by_legacy_polynomial=(
            strategy == "masked_simple" and int(polynomial_degree) >= 0
        ),
    )
    if closure_metrics["closure_status"] == "mismatch":
        reliability_reasons.append("unexplained_component_closure_mismatch")
    if np.any(agn_weights < -1.0e-10):
        reliability_reasons.append("negative_agn_template_weight")

    agn_fraction, agn_fraction_metadata = _global_agn_fraction(
        preprocessed.wave_log,
        host_log_norm,
        agn_log_norm,
        final_good,
    )
    threshold = (
        (agn_pseudocontinuum_config or HostAgnPseudoContinuumConfig())
        .global_fagn_warning_threshold
        if strategy == "agn_pseudocontinuum_masked"
        else 0.8
    )
    high_agn_fraction = bool(
        np.isfinite(agn_fraction) and agn_fraction > float(threshold)
    )
    quality_metrics.update(
        {
            **closure_metrics,
            **agn_fraction_metadata,
            "ppxf_agn_fraction_flux_global": float(agn_fraction),
            "ppxf_agn_fraction_components": [
                key
                for key in component_models_log_norm
                if key
                in {
                    "powerlaw",
                    "feii_optical",
                    "feii_uv",
                    "balmer_continuum",
                    "balmer_high_order",
                }
            ],
            "ppxf_agn_fraction_weight_aydar": np.nan,
            "ppxf_agn_fraction_weight_aydar_status": (
                "exact_definition_not_reproduced"
            ),
            "ppxf_high_agn_fraction_warning": high_agn_fraction,
            "ppxf_high_agn_fraction_warning_threshold": float(threshold),
        }
    )

    host_model = _interpolate_component(preprocessed, host_log_norm)
    agn_model = _interpolate_component(preprocessed, agn_log_norm)
    total_model = _interpolate_component(preprocessed, total_log_norm)
    component_models_log = {
        key: value * preprocessed.normalization
        for key, value in component_models_log_norm.items()
    }
    component_models = {
        key: _interpolate_component(preprocessed, value)
        for key, value in component_models_log_norm.items()
    }
    residual = preprocessed.flux - total_model

    template_resolution_status = str(
        quality_metrics.get("template_resolution_status", "template_resolution_unknown")
    )
    has_coarser = bool(np.any(coarser_mask & in_template_range))
    if has_coarser:
        host_absorption_status = "template_resolution_limited"
        stellar_kinematics_status = "template_resolution_mismatch_not_corrected"
        stellar_population_status = "template_resolution_limited"
    elif template_resolution_status in {
        "matched_by_runtime_convolution",
        "approximately_equal",
        "preconvolved_exact",
        "preconvolved_exact_plus_residual_match",
    }:
        host_absorption_status = "available"
        stellar_kinematics_status = "resolution_matched_candidate"
        stellar_population_status = "resolution_matched_candidate"
    else:
        host_absorption_status = "resolution_unknown"
        stellar_kinematics_status = "resolution_unavailable"
        stellar_population_status = "resolution_unavailable"
    overall_reliable = not reliability_reasons
    quality_metrics.update(
        {
            "host_continuum_reliable": overall_reliable,
            "host_fraction_reliable": overall_reliable,
            "host_absorption_subtraction_status": host_absorption_status,
            "stellar_kinematics_resolution_status": stellar_kinematics_status,
            "stellar_population_resolution_status": stellar_population_status,
            "host_sed_prediction_reliable": overall_reliable,
        }
    )

    sol = np.ravel(np.asarray(getattr(result, "sol", [np.nan, np.nan]), dtype=float))
    return PPXFHostFitResult(
        preprocessed=preprocessed,
        templates=templates,
        host_model_log=host_log_norm * preprocessed.normalization,
        agn_model_log=agn_log_norm * preprocessed.normalization,
        total_model_log=total_log_norm * preprocessed.normalization,
        residual_log=residual_log_norm * preprocessed.normalization,
        host_model=host_model,
        agn_model=agn_model,
        total_model=total_model,
        residual=residual,
        stellar_weights=stellar_weights,
        agn_weights=agn_weights,
        stellar_template_scales=stellar_scales,
        agn_slopes=active_slopes,
        stellar_velocity=float(sol[0]) if sol.size else np.nan,
        stellar_sigma=float(sol[1]) if sol.size > 1 else np.nan,
        chi2=float(getattr(result, "chi2", np.nan)),
        reduced_chi2=float(getattr(result, "chi2", np.nan)),
        status="success",
        initial_emission_mask_log=initial_emission,
        expanded_emission_mask_log=expanded_emission,
        residual_clip_mask_log=residual_clip,
        final_goodpixels_mask_log=final_good,
        noise_rescale_factors=noise_rescale_factors,
        quality_metrics=quality_metrics,
        host_fit_reliable=overall_reliable,
        host_fit_reliability_reasons=reliability_reasons,
        host_continuum_reliable=overall_reliable,
        host_fraction_reliable=overall_reliable,
        host_absorption_subtraction_status=host_absorption_status,
        stellar_kinematics_resolution_status=stellar_kinematics_status,
        stellar_population_resolution_status=stellar_population_status,
        host_sed_prediction_reliable=overall_reliable,
        warnings=warnings_out,
        strategy_requested=strategy_requested or strategy,
        strategy_used=strategy,
        strategy_fallback=bool(strategy_fallback),
        strategy_fallback_reason=strategy_fallback_reason,
        component_models_log=component_models_log,
        component_models=component_models,
        component_weights=component_weights,
        component_metadata=component_metadata,
        coverage=coverage,
        ppxf_agn_fraction_flux_global=float(agn_fraction),
        ppxf_high_agn_fraction_warning=high_agn_fraction,
        closure_metrics=closure_metrics,
        ppxf_result=result,
    )


def build_host_reconstruction_state(
    fit: PPXFHostFitResult,
) -> HostReconstructionState:
    """Build the path-neutral compact state for the stellar template mixture."""

    templates = fit.templates
    metadata = templates.metadata
    required = (
        "source_sha256",
        "template_wave_sha256",
        "template_matrix_sha256",
        "fit_template_wave_sha256",
        "fit_template_matrix_sha256",
    )
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise HostSEDReconstructionError(
            "template_identity_unavailable",
            f"template loader did not provide {missing}",
        )
    return HostReconstructionState(
        host_reconstruction_state_version=HOST_RECONSTRUCTION_STATE_VERSION,
        stellar_weights=np.asarray(fit.stellar_weights, dtype=float).tolist(),
        stellar_template_scales=np.asarray(
            fit.stellar_template_scales, dtype=float
        ).tolist(),
        preprocessing_normalization=float(fit.preprocessed.normalization),
        template_family=str(templates.family),
        template_file_name=str(
            metadata.get("source_file_name") or Path(templates.source_path).name
        ),
        template_file_sha256=str(metadata["source_sha256"]),
        template_profile=str(templates.profile_id),
        template_product_kind=str(templates.product_kind),
        fit_template_file_name=Path(
            str(templates.fit_source_path)
        ).name,
        fit_template_file_sha256=str(templates.fit_source_sha256),
        fit_template_wave_sha256=str(metadata["fit_template_wave_sha256"]),
        fit_template_matrix_sha256=str(
            metadata["fit_template_matrix_sha256"]
        ),
        source_template_file_name=Path(
            str(templates.source_library_path)
        ).name,
        source_template_file_sha256=str(templates.source_library_sha256),
        source_template_wave_sha256=str(metadata["template_wave_sha256"]),
        source_template_matrix_sha256=str(metadata["template_matrix_sha256"]),
        template_axis_metadata=dict(templates.template_axis_metadata),
        template_grid_id=f"sha256:{metadata['template_wave_sha256']}",
        template_wave_sha256=str(metadata["template_wave_sha256"]),
        template_matrix_sha256=str(metadata["template_matrix_sha256"]),
        template_loader_version=str(
            metadata.get("loader_version", PPXF_NPZ_LOADER_VERSION)
        ),
        template_original_shape=list(templates.original_shape),
        template_flattening_order_convention=str(
            metadata.get(
                "flattening_convention", TEMPLATE_FLATTENING_CONVENTION
            )
        ),
        selected_template_indices=None,
        host_sed_method=HOST_SED_METHOD,
        host_sed_method_version=HOST_SED_METHOD_VERSION,
        host_sed_wavelength_min=float(templates.source_wavelength_coverage[0]),
        host_sed_wavelength_max=float(templates.source_wavelength_coverage[1]),
        strategy_used=str(fit.strategy_used),
        fit_redshift=float(fit.preprocessed.redshift),
        fit_normalization_convention=(
            "templates_divided_by_stellar_template_scales_then_weighted_"
            "and_multiplied_by_preprocessing_normalization"
        ),
        fit_warnings=[str(value) for value in fit.warnings],
    )


def _host_sed_from_arrays(
    wave: np.ndarray,
    host_flux: np.ndarray,
    *,
    warnings_in: Sequence[str],
    provenance: Mapping[str, Any],
) -> HostSED:
    wave = np.asarray(wave, dtype=float)
    host_flux = np.asarray(host_flux, dtype=float)
    wave_min = float(np.nanmin(wave))
    wave_max = float(np.nanmax(wave))
    samples: Dict[str, float] = {}
    warnings_out = [str(value) for value in warnings_in]
    for name, wave in SAMPLE_WAVELENGTHS.items():
        if wave_min <= wave <= wave_max:
            samples[name] = float(np.interp(wave, provenance["template_wave"], host_flux))
        else:
            samples[name] = float("nan")
            warnings_out.append(f"{name}_outside_template_coverage")
    flags = {
        "template_covers_1um": bool(wave_min <= 10000.0 <= wave_max),
        "template_covers_1p6um": bool(wave_min <= 16000.0 <= wave_max),
        "template_covers_2p2um": bool(wave_min <= 22000.0 <= wave_max),
    }
    flags["nir_extrapolation_reliable"] = all(flags.values())
    flags["nir_extrapolation_not_available"] = not flags["nir_extrapolation_reliable"]
    if flags["nir_extrapolation_not_available"]:
        warnings_out.append("nir_extrapolation_not_available")
    return HostSED(
        wave_rest=np.asarray(provenance["template_wave"], dtype=float).copy(),
        host_flux=host_flux,
        samples=samples,
        flags=flags,
        warnings=warnings_out,
        provenance={
            key: value
            for key, value in provenance.items()
            if key != "template_wave"
        },
    )


def predict_host_sed(fit: PPXFHostFitResult) -> HostSED:
    """Evaluate the fitted host model on the full template wavelength grid."""

    templates = fit.templates
    scaled_templates = np.asarray(templates.source_flux, dtype=float) / (
        fit.stellar_template_scales[np.newaxis, :]
    )
    host_flux = scaled_templates @ fit.stellar_weights * fit.preprocessed.normalization
    try:
        state = build_host_reconstruction_state(fit)
    except HostSEDReconstructionError:
        state = None
    if state is not None:
        fit.host_reconstruction_state = state.to_dict()
    return _host_sed_from_arrays(
        np.asarray(templates.source_wave, dtype=float),
        host_flux,
        warnings_in=fit.warnings,
        provenance={
            "template_wave": np.asarray(templates.source_wave, dtype=float),
            "host_sed_method": HOST_SED_METHOD,
            "host_sed_method_version": HOST_SED_METHOD_VERSION,
            "host_reconstruction_state_version": (
                HOST_RECONSTRUCTION_STATE_VERSION if state is not None else None
            ),
            "template_file_name": (
                state.template_file_name if state is not None else None
            ),
            "template_file_sha256": (
                state.template_file_sha256 if state is not None else None
            ),
            "template_wave_sha256": (
                state.template_wave_sha256 if state is not None else None
            ),
            "template_matrix_sha256": (
                state.template_matrix_sha256 if state is not None else None
            ),
            "host_sed_template_family": templates.family,
            "host_sed_source_template_sha256": templates.source_library_sha256,
            "host_sed_uses_native_source_library": True,
            "template_profile": templates.profile_id,
            "template_product_kind": templates.product_kind,
            "strategy_used": getattr(fit, "strategy_used", None),
        },
    )


def _state_mapping(
    state: Mapping[str, Any] | HostReconstructionState,
) -> Dict[str, Any]:
    if isinstance(state, HostReconstructionState):
        return state.to_dict()
    if not isinstance(state, Mapping):
        raise HostSEDReconstructionError(
            "invalid_reconstruction_state", "state must be a mapping"
        )
    return dict(state)


def reconstruct_host_sed_from_state(
    state: Mapping[str, Any] | HostReconstructionState,
    *,
    template_root: str,
    template_file: Optional[str] = None,
    verify_hash: bool = True,
) -> HostSED:
    """Reconstruct a stellar-only HostSED without rerunning pPXF."""

    from .templates import load_ppxf_npz_templates

    value = _state_mapping(state)
    state_version = str(value.get("host_reconstruction_state_version"))
    if state_version not in {"1", HOST_RECONSTRUCTION_STATE_VERSION}:
        raise HostSEDReconstructionError(
            "unsupported_reconstruction_state_version",
            f"expected '1' or {HOST_RECONSTRUCTION_STATE_VERSION!r}",
        )
    if value.get("host_sed_method") != HOST_SED_METHOD or value.get(
        "host_sed_method_version"
    ) != HOST_SED_METHOD_VERSION:
        raise HostSEDReconstructionError(
            "incompatible_host_sed_method",
            "HostSED method/version does not match the installed implementation",
        )
    if value.get("template_loader_version") != PPXF_NPZ_LOADER_VERSION:
        raise HostSEDReconstructionError(
            "incompatible_template_loader",
            "template loader/order contract differs from the fitted state",
        )
    if value.get("template_flattening_order_convention") != (
        TEMPLATE_FLATTENING_CONVENTION
    ):
        raise HostSEDReconstructionError(
            "incompatible_template_order",
            "template flattening convention differs from the fitted state",
        )
    selected = value.get("selected_template_indices")
    if selected not in (None, []):
        raise HostSEDReconstructionError(
            "unsupported_template_subset",
            "this state requires an unsupported selected-template subset",
        )
    filename = str(
        template_file
        or value.get("source_template_file_name")
        or value.get("template_file_name")
        or ""
    )
    if not filename:
        raise HostSEDReconstructionError(
            "template_file_unavailable", "state does not name its template file"
        )
    templates = load_ppxf_npz_templates(
        template_root=template_root,
        template_file=filename,
        template_family=str(value.get("template_family", "emiles")),
        write_report=False,
    )
    if state_version == "1":
        identities = {
            "template_file_sha256": templates.metadata.get("source_sha256"),
            "template_wave_sha256": templates.metadata.get("template_wave_sha256"),
            "template_matrix_sha256": templates.metadata.get(
                "template_matrix_sha256"
            ),
        }
    else:
        identities = {
            "template_file_sha256": templates.source_library_sha256,
            "template_wave_sha256": templates.metadata.get(
                "template_wave_sha256"
            ),
            "template_matrix_sha256": templates.metadata.get(
                "template_matrix_sha256"
            ),
            "source_template_file_sha256": templates.source_library_sha256,
            "source_template_wave_sha256": templates.metadata.get(
                "template_wave_sha256"
            ),
            "source_template_matrix_sha256": templates.metadata.get(
                "template_matrix_sha256"
            ),
        }
    if verify_hash:
        for name, actual in identities.items():
            expected = value.get(name)
            if not expected or str(actual) != str(expected):
                raise HostSEDReconstructionError(
                    "template_hash_mismatch",
                    f"{name}: expected {expected!r}, found {actual!r}",
                )
    if list(templates.original_shape) != list(
        value.get("template_original_shape", [])
    ):
        raise HostSEDReconstructionError(
            "template_shape_mismatch",
            "template original shape differs from the fitted state",
        )
    weights = np.asarray(value.get("stellar_weights", []), dtype=float)
    scales = np.asarray(value.get("stellar_template_scales", []), dtype=float)
    if weights.shape != (templates.n_templates,) or scales.shape != (
        templates.n_templates,
    ):
        raise HostSEDReconstructionError(
            "template_weight_shape_mismatch",
            "stellar weights/scales do not match the verified template matrix",
        )
    normalization = float(value.get("preprocessing_normalization", np.nan))
    if not np.isfinite(normalization) or normalization <= 0:
        raise HostSEDReconstructionError(
            "invalid_preprocessing_normalization",
            "normalization must be finite and positive",
        )
    host_flux = (
        np.asarray(templates.source_flux, dtype=float)
        / scales[np.newaxis, :]
    ) @ weights
    host_flux *= normalization
    return _host_sed_from_arrays(
        np.asarray(templates.source_wave, dtype=float),
        host_flux,
        warnings_in=value.get("fit_warnings", []),
        provenance={
            "template_wave": np.asarray(templates.source_wave, dtype=float),
            "host_sed_method": HOST_SED_METHOD,
            "host_sed_method_version": HOST_SED_METHOD_VERSION,
            "host_reconstruction_state_version": (
                HOST_RECONSTRUCTION_STATE_VERSION
            ),
            **identities,
            "template_file_name": filename,
            "host_sed_template_family": templates.family,
            "host_sed_source_template_sha256": templates.source_library_sha256,
            "host_sed_uses_native_source_library": True,
            "template_profile": value.get("template_profile", templates.profile_id),
            "template_product_kind": value.get(
                "template_product_kind", templates.product_kind
            ),
            "strategy_used": value.get("strategy_used"),
            "reconstructed_without_ppxf": True,
        },
    )


def predict_host_sed_on_grid(sed: HostSED, wave_rest: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Evaluate a template-weighted host SED on a target rest-frame grid.

    No extrapolation is performed. Pixels outside template coverage are returned
    as NaN so callers can exclude them from host-subtracted fitting.
    """

    wave = np.asarray(wave_rest, dtype=float)
    host = np.interp(wave, sed.wave_rest, sed.host_flux, left=np.nan, right=np.nan)
    warnings_out: List[str] = []
    if np.any(~np.isfinite(host)):
        warnings_out.append("host_sed_grid_outside_template_coverage")
    return host, warnings_out


def _write_csv(path: Path, data: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


def _interp_no_extrapolate(wave: float, grid: np.ndarray, values: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(grid) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    finite_grid = grid[finite]
    finite_values = values[finite]
    order = np.argsort(finite_grid)
    finite_grid = finite_grid[order]
    finite_values = finite_values[order]
    if wave < finite_grid[0] or wave > finite_grid[-1]:
        return float("nan")
    return float(np.interp(wave, finite_grid, finite_values))


def fitted_host_fraction_samples(fit: PPXFHostFitResult) -> Dict[str, float]:
    samples: Dict[str, float] = {}
    for host_name, wave in SAMPLE_WAVELENGTHS.items():
        suffix = host_name.removeprefix("fHost_")
        host = _interp_no_extrapolate(wave, fit.preprocessed.wave_rest, fit.host_model)
        agn = _interp_no_extrapolate(wave, fit.preprocessed.wave_rest, fit.agn_model)
        total = _interp_no_extrapolate(wave, fit.preprocessed.wave_rest, fit.total_model)
        samples[f"fHostFit_{suffix}"] = host
        samples[f"fAGNFit_{suffix}"] = agn
        samples[f"fTotalFit_{suffix}"] = total
        if np.isfinite(host) and np.isfinite(total) and total != 0:
            samples[f"fracHost_{suffix}"] = float(host / total)
        else:
            samples[f"fracHost_{suffix}"] = float("nan")
    return samples


def _summary_dict(
    spectrum: SpectrumData,
    fit: PPXFHostFitResult,
    sed: HostSED,
    input_file: str,
    output_dir: Path,
    qsospec_status: str,
    qsospec_result_path: Optional[str],
) -> Dict[str, Any]:
    summary = {
        "object_id": spectrum.object_id or spectrum.targetid,
        "targetid": spectrum.targetid,
        "redshift": fit.preprocessed.redshift,
        "ra": spectrum.ra,
        "dec": spectrum.dec,
        "input_file": input_file,
        "flux_unit": spectrum.metadata.get("flux_unit", "cgs"),
        "flux_scale": spectrum.metadata.get("flux_scale", 1e-17),
        "template_file_used": fit.templates.source_path,
        "stellar_template_profile": fit.templates.profile_id,
        "stellar_template_family": fit.templates.family,
        "stellar_template_product_kind": fit.templates.product_kind,
        "stellar_template_fit_file": fit.templates.fit_source_path,
        "stellar_template_fit_sha256": fit.templates.fit_source_sha256,
        "stellar_template_source_file": fit.templates.source_library_path,
        "stellar_template_source_sha256": fit.templates.source_library_sha256,
        "native_data_preserved": True,
        "template_wavelength_min": fit.templates.wavelength_coverage[0],
        "template_wavelength_max": fit.templates.wavelength_coverage[1],
        "host_fit_range_min": float(np.nanmin(fit.preprocessed.wave_log)),
        "host_fit_range_max": float(np.nanmax(fit.preprocessed.wave_log)),
        "ppxf_status": fit.status,
        "host_strategy_requested": fit.strategy_requested,
        "host_strategy_used": fit.strategy_used,
        "host_strategy_fallback": fit.strategy_fallback,
        "host_strategy_fallback_reason": fit.strategy_fallback_reason,
        "host_method_reference": (
            "Aydar et al. 2026, A&A, 710, A141"
            if fit.strategy_requested == "agn_pseudocontinuum_masked"
            else None
        ),
        "host_exact_replication": False,
        "host_coverage_class": (
            fit.coverage.coverage_class if fit.coverage is not None else None
        ),
        "host_feature_coverage": (
            dict(fit.coverage.feature_coverage)
            if fit.coverage is not None
            else {}
        ),
        "ppxf_agn_fraction_flux_global": fit.ppxf_agn_fraction_flux_global,
        "ppxf_high_agn_fraction_warning": (
            fit.ppxf_high_agn_fraction_warning
        ),
        "host_component_weights": dict(fit.component_weights),
        "host_component_metadata": dict(fit.component_metadata),
        "host_closure": dict(fit.closure_metrics),
        "qsospec_status": qsospec_status,
        "qsospec_result_path": qsospec_result_path,
        "stellar_velocity": fit.stellar_velocity,
        "stellar_velocity_dispersion": fit.stellar_sigma,
        "ppxf_reduced_chi2": fit.reduced_chi2,
        "host_fit_reliable": fit.host_fit_reliable,
        "host_continuum_reliable": fit.host_continuum_reliable,
        "host_fraction_reliable": fit.host_fraction_reliable,
        "host_absorption_subtraction_status": (
            fit.host_absorption_subtraction_status
        ),
        "stellar_kinematics_resolution_status": (
            fit.stellar_kinematics_resolution_status
        ),
        "stellar_population_resolution_status": (
            fit.stellar_population_resolution_status
        ),
        "host_sed_prediction_reliable": fit.host_sed_prediction_reliable,
        "host_fit_reliability_reasons": list(
            fit.host_fit_reliability_reasons
        ),
        "host_fit_quality": dict(fit.quality_metrics),
        "host_noise_rescale_factors": dict(
            fit.noise_rescale_factors
        ),
        "host_mask_component_counts": {
            key: int(np.count_nonzero(value))
            for key, value in getattr(
                fit.preprocessed, "mask_provenance", {}
            ).items()
        },
        "qsospec_reduced_chi2": np.nan,
        "fAGN_5100": _interp_no_extrapolate(5100.0, fit.preprocessed.wave_rest, fit.agn_model),
        "broad_Halpha_detected": False,
        "broad_Hbeta_detected": False,
        "host_model_reliability": "template_weighted_ppxf_fit",
        "nir_extrapolation_reliability": "template_limited" if sed.flags["nir_extrapolation_reliable"] else "unavailable",
        "over_subtraction_risk": "inspect_residuals",
        "warnings": ";".join(sorted(set(fit.warnings + sed.warnings))),
    }
    summary.update(sed.samples)
    summary.update(fitted_host_fraction_samples(fit))
    summary.update(sed.flags)
    return summary


def write_host_decomp_outputs(
    output_dir: str,
    spectrum: SpectrumData,
    fit: PPXFHostFitResult,
    sed: HostSED,
    host_subtracted_flux: np.ndarray,
    qsospec_status: str = "not_run",
    qsospec_result_path: Optional[str] = None,
    *,
    write_legacy_products: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Write standard host-decomposition products."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    files["input_total_spectrum"] = _write_csv(
        out / "input_total_spectrum.csv",
        {"wave_obs": fit.preprocessed.wave_obs, "wave_rest": fit.preprocessed.wave_rest, "flux": fit.preprocessed.flux, "error": fit.preprocessed.error},
    )
    files["ppxf_host_model"] = _write_csv(
        out / "ppxf_host_model.csv",
        {"wave_rest": fit.preprocessed.wave_rest, "host_flux": fit.host_model},
    )
    files["ppxf_agn_continuum_model"] = _write_csv(
        out / "ppxf_agn_continuum_model.csv",
        {"wave_rest": fit.preprocessed.wave_rest, "agn_flux": fit.agn_model},
    )
    component_columns = {"wave_rest": fit.preprocessed.wave_rest}
    component_columns.update(
        {
            f"ppxf_{name}": values
            for name, values in fit.component_models.items()
        }
    )
    files["ppxf_component_models"] = _write_csv(
        out / "ppxf_component_models.csv", component_columns
    )
    files["host_subtracted_spectrum"] = _write_csv(
        out / "host_subtracted_spectrum.csv",
        {"wave_obs": fit.preprocessed.wave_obs, "wave_rest": fit.preprocessed.wave_rest, "flux": host_subtracted_flux, "error": fit.preprocessed.error},
    )
    if write_legacy_products:
        files["desi_total_spectrum"] = _write_csv(
            out / "desi_total_spectrum.csv",
            {"wave_obs": fit.preprocessed.wave_obs, "wave_rest": fit.preprocessed.wave_rest, "flux": fit.preprocessed.flux, "error": fit.preprocessed.error},
        )
        files["desi_ppxf_host_model"] = _write_csv(
            out / "desi_ppxf_host_model.csv",
            {"wave_rest": fit.preprocessed.wave_rest, "host_flux": fit.host_model},
        )
        files["desi_ppxf_agn_continuum_model"] = _write_csv(
            out / "desi_ppxf_agn_continuum_model.csv",
            {"wave_rest": fit.preprocessed.wave_rest, "agn_flux": fit.agn_model},
        )
        files["desi_host_subtracted"] = _write_csv(
            out / "desi_host_subtracted.csv",
            {"wave_obs": fit.preprocessed.wave_obs, "wave_rest": fit.preprocessed.wave_rest, "flux": host_subtracted_flux, "error": fit.preprocessed.error},
        )
    files["host_sed_prediction"] = _write_csv(
        out / "host_sed_prediction.csv",
        {"wave_rest": sed.wave_rest, "host_flux": sed.host_flux},
    )
    npz_path = out / "host_decomp_result.npz"
    component_npz = {
        f"ppxf_component_{name}": values
        for name, values in fit.component_models.items()
    }
    np.savez(
        npz_path,
        wave_obs=fit.preprocessed.wave_obs,
        wave_rest=fit.preprocessed.wave_rest,
        flux=fit.preprocessed.flux,
        host_model=fit.host_model,
        agn_model=fit.agn_model,
        total_model=fit.total_model,
        host_subtracted_flux=host_subtracted_flux,
        host_sed_wave=sed.wave_rest,
        host_sed_flux=sed.host_flux,
        stellar_weights=fit.stellar_weights,
        agn_weights=fit.agn_weights,
        host_log_wavelength_rest=fit.preprocessed.wave_log,
        host_initial_emission_mask_log=fit.initial_emission_mask_log,
        host_expanded_emission_mask_log=fit.expanded_emission_mask_log,
        host_residual_clip_mask_log=fit.residual_clip_mask_log,
        host_final_goodpixels_mask_log=fit.final_goodpixels_mask_log,
        **component_npz,
    )
    files["host_decomp_result"] = str(npz_path)
    summary = _summary_dict(
        spectrum,
        fit,
        sed,
        spectrum.metadata.get("input_file", ""),
        out,
        qsospec_status,
        qsospec_result_path,
    )
    summary_json = out / "host_decomp_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    files["host_decomp_summary_json"] = str(summary_json)
    summary_csv = out / "host_decomp_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    files["host_decomp_summary_csv"] = str(summary_csv)
    qsospec_model = out / "qsospec_model.csv"
    if qsospec_model.exists():
        files["qsospec_model"] = str(qsospec_model)
    return files, summary
