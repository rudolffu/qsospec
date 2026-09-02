"""Optional pPXF host subtraction before qsospec fitting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..fitting.local import fit_local
from ..config import (
    GalacticExtinctionConfig,
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    LyaNVComplexConfig,
    LocalFitConfig,
    MgIIComplexConfig,
    UncertaintyConfig,
)
from ..extinction import correct_spectrum_data
from ..fitting.global_fit import fit_global_lines
from ..complex_recipes import ComplexRecipe
from ..global_result import WorkflowResult
from ..measurement_vocabulary import MEASUREMENT_VOCABULARY_VERSION
from ..result import LocalFitResult
from ..spectrum import Spectrum
from ..warnings import FitWarning


def _host_decomp_decision(requested: bool, redshift: Optional[float]) -> Tuple[bool, Optional[str]]:
    """Resolve the object-level pPXF redshift gate."""

    if not requested:
        return False, None
    try:
        value = float(redshift)
    except (TypeError, ValueError):
        return False, "missing_redshift"
    if not np.isfinite(value):
        return False, "missing_redshift"
    if value >= 1.2:
        return False, "redshift_at_or_above_1.2"
    return True, None


@dataclass
class HostWorkflowResult:
    """Result of optional host subtraction followed by a qsospec fit."""

    total_spectrum: Spectrum
    fit_spectrum: Spectrum
    local_result: LocalFitResult
    host_decomp_enabled: bool
    host_fit: Optional[Any] = None
    host_sed: Optional[Any] = None
    host_model_on_quasar_grid: Optional[np.ndarray] = None
    host_subtracted_flux: Optional[np.ndarray] = None
    host_warnings: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


def _good_mask_from_spectrum_data(spectrum_data: Any, extra_mask: Optional[np.ndarray] = None) -> np.ndarray:
    wave = np.asarray(spectrum_data.wave_obs, dtype=float)
    flux = np.asarray(spectrum_data.flux, dtype=float)
    err = np.asarray(spectrum_data.uncertainty(), dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (wave > 0) & (err > 0)
    if spectrum_data.ivar is not None:
        ivar = np.asarray(spectrum_data.ivar, dtype=float)
        good &= np.isfinite(ivar) & (ivar > 0)
    if spectrum_data.mask is not None:
        good &= np.asarray(spectrum_data.mask) == 0
    if extra_mask is not None:
        good &= np.asarray(extra_mask, dtype=bool)
    return good


def _spectrum_from_arrays(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    redshift: float,
    mask: Optional[np.ndarray],
    source: str,
    spectrum_data: Optional[Any] = None,
) -> Spectrum:
    source_metadata = (
        spectrum_data.metadata.get("spectrum_metadata")
        if spectrum_data is not None else None
    )
    extinction = (
        dict(spectrum_data.metadata.get("galactic_extinction", {}))
        if spectrum_data is not None else {}
    )
    if spectrum_data is not None:
        base_metadata = dict(source_metadata or {})
        for key in (
            "flux_unit",
            "flux_scale",
            "flux_frame",
            "rest_frame_conversion",
        ):
            if key in spectrum_data.metadata:
                base_metadata[key] = spectrum_data.metadata[key]
        base_metadata.update(
            {
                "source": source,
                "ra": spectrum_data.ra,
                "dec": spectrum_data.dec,
                "galactic_extinction_corrected": extinction.get("status")
                in (
                    "applied",
                    "declared_corrected",
                    "caller_preprocessed",
                ),
                "galactic_extinction": extinction,
            }
        )
    else:
        base_metadata = source_metadata
    return Spectrum.from_arrays(
        wave_obs,
        flux,
        err=err,
        z=float(redshift),
        wave_frame="observed",
        mask=mask,
        survey=None if base_metadata is not None else "desi",
        source=source,
        ra=None if spectrum_data is None else spectrum_data.ra,
        dec=None if spectrum_data is None else spectrum_data.dec,
        galactic_extinction_corrected=extinction.get("status") in (
            "applied",
            "declared_corrected",
            "caller_preprocessed",
        ),
        galactic_extinction=extinction,
        metadata=base_metadata,
    )


def _spectrum_from_spectrum_data(spectrum_data: Any, source: str) -> Spectrum:
    good = _good_mask_from_spectrum_data(spectrum_data)
    return _spectrum_from_arrays(
        np.asarray(spectrum_data.wave_obs, dtype=float),
        np.asarray(spectrum_data.flux, dtype=float),
        np.asarray(spectrum_data.uncertainty(), dtype=float),
        float(spectrum_data.redshift),
        good,
        source=source,
        spectrum_data=spectrum_data,
    )


def _full_host_grid_masks(
    spectrum_data: Any,
    *,
    redshift: float,
    fit_range: Tuple[float, float],
    host_config: Any,
    finite_host: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pPXF host-fit and emission masks aligned to the full input grid."""

    from .host.ppxf_host import make_emission_line_mask

    wave_obs = np.asarray(spectrum_data.wave_obs, dtype=float)
    wave_rest = wave_obs / (1.0 + float(redshift))
    good = _good_mask_from_spectrum_data(spectrum_data)
    fit_mask = (
        good
        & np.asarray(finite_host, dtype=bool)
        & (wave_rest >= float(fit_range[0]))
        & (wave_rest <= float(fit_range[1]))
    )
    emission_mask = make_emission_line_mask(
        wave_rest,
        line_mask_widths=host_config.line_mask_widths,
        broad_line_mask_widths=host_config.broad_line_mask_widths,
        use_broad_masks=True,
    )
    emission_mask &= good
    return fit_mask, emission_mask


def _host_subtracted_spectrum(
    spectrum_data: Any,
    *,
    redshift: Optional[float],
    template_root: str,
    template_file: str,
    fit_range: Tuple[float, float],
    host_config: Optional[Any],
    source: str,
    pseudocontinuum_width_override_kms: Optional[float] = None,
) -> Tuple[
    Spectrum,
    Spectrum,
    Any,
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list,
]:
    from .host.config import default_config
    from .host.broad_line_prefit import run_host_broad_line_prefit
    from .host.ppxf_host import (
        prepare_spectrum_for_host_decomp,
        predict_host_sed,
        predict_host_sed_on_grid,
        run_ppxf_host_fit,
    )
    from .host.templates import load_ppxf_npz_templates

    cfg = host_config or default_config()
    total_start = perf_counter()
    default_root = "~/tools/ppxf_data"
    default_file = "spectra_emiles_9.0.npz"
    default_range = (3600.0, 7000.0)
    effective_template_root = (
        cfg.template_root
        if template_root == default_root and cfg.template_root != default_root
        else template_root
    )
    effective_template_file = (
        cfg.template_file
        if template_file == default_file and cfg.template_file != default_file
        else template_file
    )
    effective_fit_range = (
        cfg.fit_range
        if tuple(fit_range) == default_range and tuple(cfg.fit_range) != default_range
        else fit_range
    )
    templates = load_ppxf_npz_templates(
        template_root=effective_template_root,
        template_file=effective_template_file,
        report_dir=cfg.output_dir,
        template_family=cfg.template_family,
        template_profile=cfg.template_profile,
        template_product_kind=cfg.template_product_kind,
        source_template_file=cfg.source_template_file,
        template_coarser_action=cfg.template_coarser_action,
        preserve_native_data=cfg.preserve_native_data,
    )
    prep = prepare_spectrum_for_host_decomp(
        spectrum_data,
        redshift=redshift,
        fit_range=effective_fit_range,
        line_mask_widths=cfg.line_mask_widths,
        broad_line_mask_widths=cfg.broad_line_mask_widths,
        observed_artifact_windows=cfg.observed_artifact_windows,
        max_native_gap_pixels=cfg.max_native_gap_pixels,
        systematic_error_floor_fraction=cfg.systematic_error_floor_fraction,
    )
    prep.metadata["spectral_resolution"] = getattr(spectrum_data, "resolution", None)
    strategy_requested = cfg.strategy
    strategy_used = strategy_requested
    strategy_fallback = False
    strategy_fallback_reason = None
    broad_prefit = None
    selected_width = pseudocontinuum_width_override_kms
    broad_prefit_seconds = 0.0
    if strategy_requested == "agn_pseudocontinuum_masked":
        if not cfg.agn_pseudocontinuum.enabled:
            strategy_used = "masked_simple"
            strategy_fallback = True
            strategy_fallback_reason = "agn_pseudocontinuum_disabled"
        if selected_width is None:
            if strategy_used != "agn_pseudocontinuum_masked":
                selected_width = None
            else:
                prefit_start = perf_counter()
                broad_prefit = run_host_broad_line_prefit(
                    _spectrum_from_spectrum_data(spectrum_data, source=source),
                    config=cfg.broad_line_prefit,
                    width_grid_kms=cfg.agn_pseudocontinuum.width_grid_kms,
                )
                broad_prefit_seconds = perf_counter() - prefit_start
                selected_width = broad_prefit.selected_width_grid_kms
                if broad_prefit.fallback_used:
                    strategy_fallback = True
                    strategy_fallback_reason = broad_prefit.fallback_reason
        if (
            strategy_used == "agn_pseudocontinuum_masked"
            and selected_width is None
        ):
            strategy_used = "masked_simple"
            strategy_fallback = True
            strategy_fallback_reason = (
                broad_prefit.fallback_reason
                if broad_prefit is not None
                else "missing_selected_width"
            )
    ppxf_start = perf_counter()
    host_fit = run_ppxf_host_fit(
        prep,
        templates,
        agn_powerlaw_slopes=cfg.agn_powerlaw_slopes,
        polynomial_degree=cfg.polynomial_degree,
        multiplicative_polynomial_degree=cfg.multiplicative_polynomial_degree,
        adaptive_broad_line_max_velocity=cfg.adaptive_broad_line_max_velocity,
        adaptive_line_residual_sigma=cfg.adaptive_line_residual_sigma,
        residual_clip_sigma=cfg.residual_clip_sigma,
        residual_clip_iterations=cfg.residual_clip_iterations,
        residual_clip_dilation_pixels=cfg.residual_clip_dilation_pixels,
        max_noise_rescale=cfg.max_noise_rescale,
        minimum_clean_fraction=cfg.minimum_clean_fraction,
        minimum_clean_pixels=cfg.minimum_clean_pixels,
        minimum_continuum_snr=cfg.minimum_continuum_snr,
        maximum_clipped_fraction=cfg.maximum_clipped_fraction,
        strategy=strategy_used,
        strategy_requested=strategy_requested,
        strategy_fallback=strategy_fallback,
        strategy_fallback_reason=strategy_fallback_reason,
        agn_pseudocontinuum_config=cfg.agn_pseudocontinuum,
        selected_pseudocontinuum_fwhm_kms=selected_width,
        coverage_config=cfg.coverage,
    )
    ppxf_seconds = perf_counter() - ppxf_start
    if broad_prefit is not None:
        host_fit.quality_metrics.update(
            {
                "broad_prefit_status": broad_prefit.status,
                "broad_prefit_line": broad_prefit.selected_line,
                "broad_prefit_fwhm_kms": broad_prefit.fwhm_kms,
                "broad_prefit_fwhm_error_kms": broad_prefit.fwhm_error_kms,
                "broad_prefit_flux_snr": broad_prefit.flux_snr,
                "broad_prefit_fwhm_snr": broad_prefit.fwhm_snr,
                "broad_prefit_velocity_kms": broad_prefit.velocity_kms,
                "broad_prefit_diagnostics": broad_prefit.diagnostics,
            }
        )
    host_fit.quality_metrics.update(
        {
            "pseudocontinuum_width_initial_kms": selected_width,
            "pseudocontinuum_width_final_kms": selected_width,
            "pseudocontinuum_width_iterations": (
                1 if strategy_used == "agn_pseudocontinuum_masked" else 0
            ),
            "pseudocontinuum_width_converged": None,
            "pseudocontinuum_width_change_kms": 0.0,
            "pseudocontinuum_width_status": (
                "initial_selection"
                if strategy_used == "agn_pseudocontinuum_masked"
                else "not_used"
            ),
            "broad_line_prefit_seconds": float(broad_prefit_seconds),
            "host_ppxf_total_seconds": float(ppxf_seconds),
        }
    )
    host_fit.preprocessed.metadata.update(
        {
            "host_strategy_requested": strategy_requested,
            "host_strategy_used": strategy_used,
            "host_strategy_fallback": strategy_fallback,
            "host_strategy_fallback_reason": strategy_fallback_reason,
            "host_method_reference": (
                "Aydar et al. 2026, A&A, 710, A141"
                if strategy_requested == "agn_pseudocontinuum_masked"
                else None
            ),
            "host_exact_replication": (
                False
                if strategy_requested == "agn_pseudocontinuum_masked"
                else None
            ),
        }
    )
    sed_start = perf_counter()
    host_sed = predict_host_sed(host_fit)
    host_fit.quality_metrics["host_sed_prediction_seconds"] = float(
        perf_counter() - sed_start
    )
    host_fit.quality_metrics["host_sed_reconstruction_seconds"] = (
        host_fit.quality_metrics["host_sed_prediction_seconds"]
    )
    full_wave_obs = np.asarray(spectrum_data.wave_obs, dtype=float)
    full_wave_rest = full_wave_obs / (1.0 + float(redshift))
    full_flux = np.asarray(spectrum_data.flux, dtype=float)
    full_error = np.asarray(spectrum_data.uncertainty(), dtype=float)
    full_good = _good_mask_from_spectrum_data(spectrum_data)
    host_on_grid, grid_warnings = predict_host_sed_on_grid(
        host_sed, full_wave_rest
    )
    fitted_host_finite = np.isfinite(host_fit.host_model)
    if np.count_nonzero(fitted_host_finite) >= 2:
        fitted_wave = host_fit.preprocessed.wave_rest[fitted_host_finite]
        fitted_values = host_fit.host_model[fitted_host_finite]
        order = np.argsort(fitted_wave)
        fitted_host_on_grid = np.interp(
            full_wave_rest,
            fitted_wave[order],
            fitted_values[order],
            left=np.nan,
            right=np.nan,
        )
        constrained = np.isfinite(fitted_host_on_grid)
        host_on_grid[constrained] = fitted_host_on_grid[constrained]
        host_fit.preprocessed.metadata[
            "host_model_grid_policy"
        ] = "ppxf_convolved_within_fit_range_stellar_sed_elsewhere"
    host_warnings = list(host_fit.warnings) + list(host_sed.warnings) + list(grid_warnings)
    if host_fit.ppxf_high_agn_fraction_warning:
        host_warnings.append("ppxf_high_agn_fraction_above_0.8")
    finite_host = np.isfinite(host_on_grid)
    host_subtracted_flux = full_flux - np.where(finite_host, host_on_grid, 0.0)
    host_fit_mask, host_emission_mask = _full_host_grid_masks(
        spectrum_data,
        redshift=float(redshift),
        fit_range=effective_fit_range,
        host_config=cfg,
        finite_host=finite_host,
    )

    total_spectrum = _spectrum_from_arrays(
        full_wave_obs,
        full_flux,
        full_error,
        prep.redshift,
        full_good,
        source=source,
        spectrum_data=spectrum_data,
    )
    fit_spectrum = _spectrum_from_arrays(
        full_wave_obs,
        host_subtracted_flux,
        full_error,
        prep.redshift,
        full_good & np.isfinite(host_subtracted_flux) & finite_host,
        source=f"{source}; host_subtracted=ppxf_sed_grid",
        spectrum_data=spectrum_data,
    )
    host_fit.quality_metrics["host_decomposition_seconds"] = float(
        perf_counter() - total_start
    )
    return (
        total_spectrum,
        fit_spectrum,
        host_fit,
        host_sed,
        host_on_grid,
        host_subtracted_flux,
        host_fit_mask,
        host_emission_mask,
        host_warnings,
    )


def fit_with_optional_host_decomp(
    input_path: str,
    local_config: Optional[LocalFitConfig] = None,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    fit_kind: str = "local",
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    galactic_extinction_config: Optional[GalacticExtinctionConfig] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    lya_nv_config: Optional[LyaNVComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
):
    """Read a spectrum, optionally subtract a pPXF host, then run qsospec.

    ``fit_kind`` may be ``"local"`` or ``"global"``.
    """

    if fit_kind == "global":
        return fit_global_lines_workflow(
            input_path,
            row_index=row_index,
            redshift=redshift,
            object_id=object_id,
            run_host_decomp=run_host_decomp,
            template_root=template_root,
            template_file=template_file,
            host_fit_range=host_fit_range,
            host_config=host_config,
            galactic_extinction_config=galactic_extinction_config,
            global_config=global_config,
            hbeta_config=hbeta_config,
            mgii_config=mgii_config,
            halpha_config=halpha_config,
            lya_nv_config=lya_nv_config,
            uncertainty_config=uncertainty_config,
            complexes=complexes,
        )
    if fit_kind != "local":
        raise ValueError("fit_kind must be 'local' or 'global'.")
    if local_config is None:
        raise ValueError("local_config is required when fit_kind='local'.")

    from .host.io import read_sparcli_spectrum

    spectrum_data = read_sparcli_spectrum(
        input_path,
        row_index=row_index,
        redshift=redshift,
        object_id=object_id,
    )
    spectrum_data = correct_spectrum_data(
        spectrum_data, galactic_extinction_config
    )
    source = f"{input_path}:row_index={row_index}"
    host_decomp_enabled, host_skip_reason = _host_decomp_decision(
        run_host_decomp, spectrum_data.redshift
    )
    if host_decomp_enabled:
        (
            total_spectrum,
            fit_spectrum,
            host_fit,
            host_sed,
            host_on_grid,
            host_subtracted_flux,
            _,
            _,
            host_warnings,
        ) = (
            _host_subtracted_spectrum(
                spectrum_data,
                redshift=float(spectrum_data.redshift),
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=host_config,
                source=source,
            )
        )
    else:
        total_spectrum = _spectrum_from_spectrum_data(spectrum_data, source=source)
        fit_spectrum = total_spectrum
        host_fit = None
        host_sed = None
        host_on_grid = None
        host_subtracted_flux = None
        host_warnings = []

    local_result = fit_local(fit_spectrum, local_config)
    metadata = {
        "input_path": input_path,
        "row_index": row_index,
        "object_id": object_id or spectrum_data.object_id or spectrum_data.targetid,
        "targetid": spectrum_data.targetid,
        "ra": spectrum_data.ra,
        "dec": spectrum_data.dec,
        "redshift": fit_spectrum.z,
        "fit_kind": fit_kind,
        "host_decomp_requested": bool(run_host_decomp),
        "host_decomp_enabled": host_decomp_enabled,
        "host_decomp_skip_reason": host_skip_reason,
        "host_model_source": "template_weighted_sed_on_quasar_grid" if host_decomp_enabled else None,
        "host_strategy_requested": (
            host_fit.strategy_requested if host_fit is not None else None
        ),
        "host_strategy_used": (
            host_fit.strategy_used if host_fit is not None else None
        ),
        "host_strategy_fallback": (
            bool(host_fit.strategy_fallback) if host_fit is not None else False
        ),
        "host_strategy_fallback_reason": (
            host_fit.strategy_fallback_reason if host_fit is not None else None
        ),
        "galactic_extinction": dict(
            spectrum_data.metadata.get("galactic_extinction", {})
        ),
    }
    return HostWorkflowResult(
        total_spectrum=total_spectrum,
        fit_spectrum=fit_spectrum,
        local_result=local_result,
        host_decomp_enabled=host_decomp_enabled,
        host_fit=host_fit,
        host_sed=host_sed,
        host_model_on_quasar_grid=host_on_grid,
        host_subtracted_flux=host_subtracted_flux,
        host_warnings=host_warnings,
        metadata=metadata,
    )


def _summarize_mc_results(
    samples: Dict[str, list],
    n_requested: int,
    continuum_success_count: int,
    complex_success_counts: Dict[str, int],
) -> Dict[str, Any]:
    percentiles = {}
    for name, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            p16, p50, p84 = np.percentile(finite, [16.0, 50.0, 84.0])
            percentiles[name] = {"p16": float(p16), "p50": float(p50), "p84": float(p84)}
    return {
        "n_requested": int(n_requested),
        "continuum_success_count": int(continuum_success_count),
        "complex_success_counts": dict(complex_success_counts),
        "percentiles": percentiles,
    }


def _final_broad_width_selection(workflow: WorkflowResult, host_config: Any):
    """Select a reliable final broad width for one bounded host refit."""

    from .host.broad_line_prefit import nearest_width_grid_value

    cfg = host_config.broad_line_prefit
    for line in cfg.preferred_lines:
        recipe = "halpha_nii_sii" if line == "halpha" else "hbeta_oiii"
        prefix = "Ha" if line == "halpha" else "Hb"
        result = workflow.line_complexes.get(recipe)
        if result is None or not result.success:
            continue
        flux = float(result.metrics.get(f"{prefix}_broad_flux_input", np.nan))
        flux_error = float(
            result.metric_errors.get(f"{prefix}_broad_flux_input", np.nan)
        )
        width = float(result.metrics.get(f"{prefix}_broad_fwhm_kms", np.nan))
        width_error = float(
            result.metric_errors.get(f"{prefix}_broad_fwhm_kms", np.nan)
        )
        flux_snr = flux / flux_error if np.isfinite(flux_error) and flux_error > 0 else np.nan
        width_snr = width / width_error if np.isfinite(width_error) and width_error > 0 else np.nan
        at_bound = any(
            warning.code == "parameter_at_bound"
            and "broad" in str(warning.context.get("parameter", "")).lower()
            and any(
                token in str(warning.context.get("parameter", "")).lower()
                for token in ("fwhm", "velocity")
            )
            for warning in result.warnings
        )
        if (
            np.isfinite(flux)
            and flux > 0
            and np.isfinite(width)
            and width > 0
            and np.isfinite(flux_snr)
            and flux_snr >= cfg.minimum_flux_snr
            and np.isfinite(width_snr)
            and width_snr >= cfg.minimum_fwhm_snr
            and (not cfg.reject_parameter_bounds or not at_bound)
        ):
            selected = nearest_width_grid_value(
                width,
                host_config.agn_pseudocontinuum.width_grid_kms,
            )
            return {
                "status": "reliable",
                "line": line,
                "fwhm_kms": width,
                "fwhm_error_kms": width_error,
                "flux_snr": float(flux_snr),
                "fwhm_snr": float(width_snr),
                "selected_width_grid_kms": float(selected),
            }
    return {"status": "no_reliable_final_broad_width"}


def _run_host_refit_mc(
    spectrum_data: Any,
    *,
    n_trials: int,
    seed: Optional[int],
    redshift: Optional[float],
    template_root: str,
    template_file: str,
    host_fit_range: Tuple[float, float],
    host_config: Optional[Any],
    source: str,
    global_config: Optional[GlobalContinuumConfig],
    hbeta_config: Optional[HbetaComplexConfig],
    mgii_config: Optional[MgIIComplexConfig],
    halpha_config: Optional[HalphaComplexConfig],
    lya_nv_config: Optional[LyaNVComplexConfig] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples: Dict[str, list] = {}
    continuum_successes = 0
    complex_successes: Dict[str, int] = {}
    error = np.asarray(spectrum_data.uncertainty(), dtype=float)
    for _ in range(int(n_trials)):
        noisy_data = replace(
            spectrum_data,
            flux=np.asarray(spectrum_data.flux, dtype=float) + rng.normal(0.0, error),
        )
        try:
            _, fit_spectrum, _, _, host_on_grid, _, _, _, _ = _host_subtracted_spectrum(
                noisy_data,
                redshift=redshift,
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=host_config,
                source=source,
            )
            trial = fit_global_lines(
                fit_spectrum,
                global_config,
                hbeta_config,
                mgii_config,
                halpha_config,
                UncertaintyConfig(monte_carlo_trials=0),
                lya_nv_config=lya_nv_config,
                host_model_on_grid=host_on_grid,
                complexes=complexes,
            )
            values = {}
            if trial.continuum_success:
                continuum_successes += 1
                values.update(trial.continuum.param_values)
            for recipe_id, complex_result in trial.line_complexes.items():
                if complex_result.success:
                    complex_successes[recipe_id] = complex_successes.get(recipe_id, 0) + 1
                    values.update(complex_result.metrics)
            for name, value in values.items():
                if np.isfinite(value):
                    samples.setdefault(name, []).append(float(value))
        except Exception:
            continue
    return _summarize_mc_results(
        samples, n_trials, continuum_successes, complex_successes
    )


def _run_global_fit_with_optional_host(
    spectrum_data: Any,
    *,
    source: str,
    input_path: str,
    row_index: Optional[int] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    galactic_extinction_config: Optional[GalacticExtinctionConfig] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    lya_nv_config: Optional[LyaNVComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
) -> WorkflowResult:
    """Authoritative SpectrumData-based host plus global-fit orchestration."""

    from .host.config import default_config

    workflow_start = perf_counter()
    uncertainty = uncertainty_config or UncertaintyConfig()
    resolved_host_config = host_config or default_config()
    spectrum_data = correct_spectrum_data(
        spectrum_data, galactic_extinction_config
    )
    host_decomp_enabled, host_skip_reason = _host_decomp_decision(
        run_host_decomp, spectrum_data.redshift
    )
    if host_decomp_enabled:
        (
            total_spectrum,
            fit_spectrum,
            host_fit,
            host_sed,
            host_on_grid,
            _,
            host_fit_mask,
            host_emission_mask,
            host_warnings,
        ) = (
            _host_subtracted_spectrum(
                spectrum_data,
                redshift=float(spectrum_data.redshift),
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=resolved_host_config,
                source=source,
            )
        )
        primary_uncertainty = (
            replace(uncertainty, monte_carlo_trials=0)
            if uncertainty.monte_carlo_trials > 0 and uncertainty.refit_host_in_mc
            else uncertainty
        )
    else:
        total_spectrum = _spectrum_from_spectrum_data(spectrum_data, source=source)
        fit_spectrum = total_spectrum
        host_fit = None
        host_sed = None
        host_on_grid = None
        host_warnings = []
        primary_uncertainty = uncertainty

    final_fit_start = perf_counter()
    workflow = fit_global_lines(
        fit_spectrum,
        global_config,
        hbeta_config,
        mgii_config,
        halpha_config,
        primary_uncertainty,
        lya_nv_config=lya_nv_config,
        host_model_on_grid=host_on_grid,
        complexes=complexes,
    )
    final_qsospec_seconds = perf_counter() - final_fit_start
    if (
        host_fit is not None
        and host_fit.strategy_used == "agn_pseudocontinuum_masked"
        and resolved_host_config.agn_pseudocontinuum.maximum_width_iterations > 1
    ):
        final_width = _final_broad_width_selection(
            workflow,
            resolved_host_config,
        )
        initial_width = host_fit.quality_metrics.get(
            "pseudocontinuum_width_initial_kms"
        )
        candidate_width = final_width.get("selected_width_grid_kms")
        if candidate_width is None:
            host_fit.quality_metrics.update(
                {
                    "pseudocontinuum_width_converged": None,
                    "pseudocontinuum_width_status": final_width["status"],
                    "final_broad_width_diagnostics": final_width,
                }
            )
        elif (
            abs(float(candidate_width) - float(initial_width))
            <= resolved_host_config.agn_pseudocontinuum.width_convergence_tolerance_kms
        ):
            host_fit.quality_metrics.update(
                {
                    "pseudocontinuum_width_converged": True,
                    "pseudocontinuum_width_status": "stable_after_final_fit",
                    "final_broad_width_diagnostics": final_width,
                }
            )
        else:
            refit_start = perf_counter()
            (
                total_spectrum,
                fit_spectrum,
                updated_host_fit,
                host_sed,
                host_on_grid,
                _,
                host_fit_mask,
                host_emission_mask,
                host_warnings,
            ) = _host_subtracted_spectrum(
                spectrum_data,
                redshift=float(spectrum_data.redshift),
                template_root=template_root,
                template_file=template_file,
                fit_range=host_fit_range,
                host_config=resolved_host_config,
                source=source,
                pseudocontinuum_width_override_kms=float(candidate_width),
            )
            updated_host_fit.quality_metrics.update(
                {
                    "pseudocontinuum_width_initial_kms": initial_width,
                    "pseudocontinuum_width_final_kms": float(candidate_width),
                    "pseudocontinuum_width_iterations": 2,
                    "pseudocontinuum_width_change_kms": float(candidate_width)
                    - float(initial_width),
                    "pseudowidth_refit_seconds": float(
                        perf_counter() - refit_start
                    ),
                    "final_broad_width_diagnostics": final_width,
                }
            )
            for key, value in host_fit.quality_metrics.items():
                if key.startswith("broad_prefit"):
                    updated_host_fit.quality_metrics.setdefault(key, value)
            second_fit_start = perf_counter()
            workflow = fit_global_lines(
                fit_spectrum,
                global_config,
                hbeta_config,
                mgii_config,
                halpha_config,
                primary_uncertainty,
                lya_nv_config=lya_nv_config,
                host_model_on_grid=host_on_grid,
                complexes=complexes,
            )
            final_qsospec_seconds += perf_counter() - second_fit_start
            confirmation = _final_broad_width_selection(
                workflow,
                resolved_host_config,
            )
            updated_host_fit.quality_metrics.update(
                {
                    "pseudocontinuum_width_converged": bool(
                        confirmation.get("selected_width_grid_kms")
                        == float(candidate_width)
                    ),
                    "pseudocontinuum_width_status": (
                        "converged_after_one_update"
                        if confirmation.get("selected_width_grid_kms")
                        == float(candidate_width)
                        else "maximum_iterations_reached"
                    ),
                    "final_broad_width_confirmation": confirmation,
                }
            )
            host_fit = updated_host_fit
    if host_fit is not None:
        host_fit.quality_metrics["final_qsospec_seconds"] = float(
            final_qsospec_seconds
        )
        host_fit.quality_metrics["total_host_workflow_seconds"] = float(
            perf_counter() - workflow_start
        )
    workflow.host_decomp_enabled = host_decomp_enabled
    workflow.total_spectrum = total_spectrum
    workflow.host_fit = host_fit
    workflow.host_sed = host_sed
    workflow.host_reconstruction_state = (
        dict(host_fit.host_reconstruction_state)
        if host_fit is not None and host_fit.host_reconstruction_state
        else None
    )
    workflow.host_model_on_quasar_grid = host_on_grid
    if host_fit is not None:
        host_component_models = {}
        host_wave = np.asarray(host_fit.preprocessed.wave_rest, dtype=float)
        target_wave = np.asarray(fit_spectrum.wave_rest, dtype=float)
        for name, values in host_fit.component_models.items():
            component = np.asarray(values, dtype=float)
            finite = np.isfinite(host_wave) & np.isfinite(component)
            if np.count_nonzero(finite) < 2:
                aligned = np.full_like(target_wave, np.nan, dtype=float)
            else:
                order = np.argsort(host_wave[finite])
                aligned = np.interp(
                    target_wave,
                    host_wave[finite][order],
                    component[finite][order],
                    left=np.nan,
                    right=np.nan,
                )
            host_component_models[name] = aligned
        host_component_models["host_subtracted_flux"] = np.asarray(
            fit_spectrum.flux, dtype=float
        ).copy()
        workflow.host_component_models = host_component_models
    workflow.host_fit_mask = (
        np.asarray(host_fit_mask, dtype=bool).copy()
        if host_fit is not None else None
    )
    workflow.host_emission_mask = (
        np.asarray(host_emission_mask, dtype=bool).copy()
        if host_fit is not None else None
    )
    workflow.host_warnings = [str(item) for item in host_warnings]
    if host_fit is not None:
        from .host.ppxf_host import fitted_host_fraction_samples

        host_fit_samples = fitted_host_fraction_samples(host_fit)
    else:
        host_fit_samples = {}
    workflow.metadata.update(
        {
            "input_path": input_path,
            "row_index": row_index,
            "object_id": object_id or spectrum_data.object_id or spectrum_data.targetid,
            "targetid": spectrum_data.targetid,
            "ra": spectrum_data.ra,
            "dec": spectrum_data.dec,
            "redshift": fit_spectrum.z,
            "fit_kind": "global",
            "measurement_vocabulary_version": (
                MEASUREMENT_VOCABULARY_VERSION
            ),
            "flux_frame": fit_spectrum.flux_frame,
            "rest_frame_conversion": dict(
                fit_spectrum.metadata.rest_frame_conversion
            ),
            "host_decomp_requested": bool(run_host_decomp),
            "host_decomp_enabled": host_decomp_enabled,
            "host_decomp_skip_reason": host_skip_reason,
            "host_model_source": "template_weighted_sed_on_quasar_grid" if host_decomp_enabled else None,
            "host_fit_range": list(host_fit_range),
            "host_mask_provenance": "exact" if host_decomp_enabled else "unavailable",
            "host_ppxf_status": (
                host_fit.status if host_fit is not None else None
            ),
            "host_ppxf_reduced_chi2": (
                float(host_fit.reduced_chi2)
                if host_fit is not None else None
            ),
            "host_fit_reliable": (
                bool(host_fit.host_fit_reliable)
                if host_fit is not None else None
            ),
            "host_fit_reliability_reasons": (
                list(host_fit.host_fit_reliability_reasons)
                if host_fit is not None else []
            ),
            "host_continuum_reliable": (
                bool(getattr(host_fit, "host_continuum_reliable", host_fit.host_fit_reliable))
                if host_fit is not None else None
            ),
            "host_fraction_reliable": (
                bool(getattr(host_fit, "host_fraction_reliable", host_fit.host_fit_reliable))
                if host_fit is not None else None
            ),
            "host_absorption_subtraction_status": (
                getattr(host_fit, "host_absorption_subtraction_status", "unavailable")
                if host_fit is not None else None
            ),
            "stellar_kinematics_resolution_status": (
                getattr(host_fit, "stellar_kinematics_resolution_status", "unavailable")
                if host_fit is not None else None
            ),
            "stellar_population_resolution_status": (
                getattr(host_fit, "stellar_population_resolution_status", "unavailable")
                if host_fit is not None else None
            ),
            "host_sed_prediction_reliable": (
                bool(getattr(host_fit, "host_sed_prediction_reliable", host_fit.host_fit_reliable))
                if host_fit is not None else None
            ),
            "host_fit_quality": (
                dict(host_fit.quality_metrics)
                if host_fit is not None else {}
            ),
            "host_noise_rescale_factors": (
                dict(host_fit.noise_rescale_factors)
                if host_fit is not None else {}
            ),
            "host_mask_components_log": (
                {
                    key: np.asarray(value, dtype=bool).tolist()
                    for key, value in host_fit.preprocessed.mask_provenance.items()
                    if str(key).endswith("_log")
                    or str(key) == "log_grid_valid"
                }
                if host_fit is not None else {}
            ),
            "host_mask_component_counts": (
                {
                    key: int(np.count_nonzero(value))
                    for key, value in host_fit.preprocessed.mask_provenance.items()
                }
                if host_fit is not None else {}
            ),
            "host_template_file": (
                host_fit.templates.source_path
                if host_fit is not None else None
            ),
            "host_template_wavelength_coverage": (
                list(host_fit.templates.wavelength_coverage)
                if host_fit is not None else None
            ),
            "host_template_file_sha256": (
                host_fit.templates.metadata.get("source_sha256")
                if host_fit is not None else None
            ),
            "host_template_profile": (
                getattr(host_fit.templates, "profile_id", "custom_native")
                if host_fit is not None else None
            ),
            "host_template_product_kind": (
                getattr(host_fit.templates, "product_kind", "native")
                if host_fit is not None else None
            ),
            "host_fit_template_file": (
                getattr(host_fit.templates, "fit_source_path", host_fit.templates.source_path)
                if host_fit is not None else None
            ),
            "host_fit_template_sha256": (
                getattr(
                    host_fit.templates,
                    "fit_source_sha256",
                    host_fit.templates.metadata.get("source_sha256"),
                )
                if host_fit is not None else None
            ),
            "host_source_template_file": (
                getattr(host_fit.templates, "source_library_path", host_fit.templates.source_path)
                if host_fit is not None else None
            ),
            "host_source_template_sha256": (
                getattr(
                    host_fit.templates,
                    "source_library_sha256",
                    host_fit.templates.metadata.get("source_sha256"),
                )
                if host_fit is not None else None
            ),
            "host_source_template_wavelength_coverage": (
                list(
                    getattr(
                        host_fit.templates,
                        "source_wavelength_coverage",
                        host_fit.templates.wavelength_coverage,
                    )
                )
                if host_fit is not None else None
            ),
            "host_template_wave_sha256": (
                host_fit.templates.metadata.get("template_wave_sha256")
                if host_fit is not None else None
            ),
            "host_template_matrix_sha256": (
                host_fit.templates.metadata.get("template_matrix_sha256")
                if host_fit is not None else None
            ),
            "host_fit_normalization": (
                float(host_fit.preprocessed.normalization)
                if host_fit is not None else None
            ),
            "host_strategy_requested": (
                host_fit.strategy_requested if host_fit is not None else None
            ),
            "host_strategy_used": (
                host_fit.strategy_used if host_fit is not None else None
            ),
            "host_strategy_fallback": (
                bool(host_fit.strategy_fallback)
                if host_fit is not None else False
            ),
            "host_strategy_fallback_reason": (
                host_fit.strategy_fallback_reason
                if host_fit is not None else None
            ),
            "host_method_reference": (
                "Aydar et al. 2026, A&A, 710, A141"
                if host_fit is not None
                and host_fit.strategy_requested
                == "agn_pseudocontinuum_masked"
                else None
            ),
            "host_exact_replication": (
                False
                if host_fit is not None
                and host_fit.strategy_requested
                == "agn_pseudocontinuum_masked"
                else None
            ),
            "host_coverage_class": (
                host_fit.coverage.coverage_class
                if host_fit is not None and host_fit.coverage is not None
                else None
            ),
            "host_feature_coverage": (
                dict(host_fit.coverage.feature_coverage)
                if host_fit is not None and host_fit.coverage is not None
                else {}
            ),
            "ppxf_agn_fraction_flux_global": (
                float(host_fit.ppxf_agn_fraction_flux_global)
                if host_fit is not None else np.nan
            ),
            "ppxf_agn_fraction_definition": (
                "integrated_positive_model_flux_agn_over_agn_plus_stellar"
                if host_fit is not None else None
            ),
            "ppxf_agn_fraction_wavelength_support": (
                [
                    float(np.nanmin(host_fit.preprocessed.wave_rest)),
                    float(np.nanmax(host_fit.preprocessed.wave_rest)),
                ]
                if host_fit is not None else None
            ),
            "ppxf_high_agn_fraction_warning": (
                bool(host_fit.ppxf_high_agn_fraction_warning)
                if host_fit is not None else False
            ),
            "host_component_weights": (
                dict(host_fit.component_weights)
                if host_fit is not None else {}
            ),
            "host_component_metadata": (
                dict(host_fit.component_metadata)
                if host_fit is not None else {}
            ),
            "host_fit_samples": host_fit_samples,
            "host_closure": (
                dict(host_fit.closure_metrics)
                if host_fit is not None else {}
            ),
            "host_sed_reconstruction_status": (
                "available"
                if workflow.host_reconstruction_state is not None
                else (
                    "not_available_host_not_fit"
                    if run_host_decomp else "not_requested"
                )
            ),
            "host_reconstruction_state_version": (
                workflow.host_reconstruction_state.get(
                    "host_reconstruction_state_version"
                )
                if workflow.host_reconstruction_state is not None else None
            ),
            "resolution_status": spectrum_data.metadata.get(
                "resolution_status",
                (
                    spectrum_data.resolution.status
                    if spectrum_data.resolution is not None else "missing"
                ),
            ),
            "resolution_source": spectrum_data.metadata.get(
                "resolution_source",
                (
                    spectrum_data.resolution.source
                    if spectrum_data.resolution is not None else None
                ),
            ),
            "resolution_is_object_specific": bool(
                spectrum_data.metadata.get(
                    "resolution_is_object_specific",
                    (
                        spectrum_data.resolution.is_object_specific
                        if spectrum_data.resolution is not None else False
                    ),
                )
            ),
            "source_backend": spectrum_data.metadata.get("source_backend"),
            "source_backend_version": spectrum_data.metadata.get(
                "source_backend_version"
            ),
            "galactic_extinction": dict(
                spectrum_data.metadata.get("galactic_extinction", {})
            ),
        }
    )
    if run_host_decomp and not host_decomp_enabled:
        workflow.warnings.append(
            FitWarning(
                code="host_decomp_skipped_redshift",
                message="Host decomposition was requested but skipped by the redshift gate.",
                severity="info",
                context={
                    "redshift": spectrum_data.redshift,
                    "threshold": 1.2,
                    "reason": host_skip_reason,
                },
            )
        )
    if host_decomp_enabled and uncertainty.monte_carlo_trials > 0 and uncertainty.refit_host_in_mc:
        workflow.monte_carlo = _run_host_refit_mc(
            spectrum_data,
            n_trials=uncertainty.monte_carlo_trials,
            seed=uncertainty.random_seed,
            redshift=spectrum_data.redshift,
            template_root=template_root,
            template_file=template_file,
            host_fit_range=host_fit_range,
            host_config=resolved_host_config,
            source=source,
            global_config=global_config,
            hbeta_config=hbeta_config,
            mgii_config=mgii_config,
            halpha_config=halpha_config,
            lya_nv_config=lya_nv_config,
            complexes=complexes,
        )
        workflow.metadata["uncertainty_mode"] = "covariance+monte_carlo_host_refit"
    return workflow


def fit_global_lines_workflow(
    input_path: str,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    galactic_extinction_config: Optional[GalacticExtinctionConfig] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    lya_nv_config: Optional[LyaNVComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
) -> WorkflowResult:
    """Read one spectrum and use the shared host/global-fit orchestration."""

    from .host.io import read_sparcli_spectrum

    spectrum_data = read_sparcli_spectrum(
        input_path,
        row_index=row_index,
        redshift=redshift,
        object_id=object_id,
    )
    return _run_global_fit_with_optional_host(
        spectrum_data,
        source=f"{input_path}:row_index={row_index}",
        input_path=input_path,
        row_index=row_index,
        object_id=object_id,
        run_host_decomp=run_host_decomp,
        template_root=template_root,
        template_file=template_file,
        host_fit_range=host_fit_range,
        host_config=host_config,
        galactic_extinction_config=galactic_extinction_config,
        global_config=global_config,
        hbeta_config=hbeta_config,
        mgii_config=mgii_config,
        halpha_config=halpha_config,
        lya_nv_config=lya_nv_config,
        uncertainty_config=uncertainty_config,
        complexes=complexes,
    )


def fit_global_hbeta_workflow(
    input_path: str,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range: Tuple[float, float] = (3600.0, 7000.0),
    host_config: Optional[Any] = None,
    galactic_extinction_config: Optional[GalacticExtinctionConfig] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
) -> WorkflowResult:
    """Compatibility wrapper for :func:`fit_global_lines_workflow`."""

    result = fit_global_lines_workflow(
        input_path,
        row_index=row_index,
        redshift=redshift,
        object_id=object_id,
        run_host_decomp=run_host_decomp,
        template_root=template_root,
        template_file=template_file,
        host_fit_range=host_fit_range,
        host_config=host_config,
        galactic_extinction_config=galactic_extinction_config,
        global_config=global_config,
        hbeta_config=hbeta_config,
        uncertainty_config=uncertainty_config,
        complexes=("hbeta_oiii",),
    )
    result.metadata["compatibility_hbeta_mode"] = True
    return result
