"""Local Balmer model pairs with explicit instrumental response operators.

The native bounded variable-projection solver fits an affine continuum and
the line bases jointly.  There is no global, host, Balmer-pseudocontinuum or
linked-line workflow here.  Model selection and detection calibration belong
to the caller: this module deliberately returns each requested hypothesis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Mapping, Optional, Sequence, Tuple
import hashlib
import json

import numpy as np
from scipy import sparse

from .. import lines
from ..complex_recipes import ComplexRecipe, ComponentRecipe
from ..global_result import EmissionComplexResult, GlobalContinuumResult
from ..spectrum import Spectrum, require_rest_frame_flux
from ..warnings import FitWarning
from .complexes import GenericComplexContext
from .global_fit import (
    C_KMS, _active_bound_warnings, _covariance_from_jacobian,
    _metric_errors, _solve_once_with_fallback,
)

# The original DESI inverse-variance coaddition is stored and divided in
# float32.  Permit its roundoff, while preserving the actual weights in every
# forward-model evaluation.  This is not a missing-support repair tolerance.
OUTPUT_WEIGHT_SUM_ATOL = 4.0 * float(np.finfo(np.float32).eps)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _vector(value, length, name, *, positive=False):
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        raw = np.full(length, float(raw))
    if raw.shape != (length,) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite and scalar or have the declared length")
    if positive and np.any(raw <= 0):
        raise ValueError(f"{name} must be positive")
    if not positive and np.any(raw < 0):
        raise ValueError(f"{name} must be non-negative")
    raw = raw.copy()
    raw.setflags(write=False)
    return raw


@dataclass(frozen=True)
class BandResolutionOperator:
    """Map one native observed-wavelength band onto Spectrum output rows.

    ``matrix`` has shape (len(output_indices), len(input_wave_obs)).  Use a
    dense matrix or a SciPy sparse matrix, not DESI's diagonal storage array;
    convert diagonal storage with its recorded offsets before construction.
    Operators add ``output_weights * output_factor * matrix @
    (input_factor * model)``.  Output weights must sum to one on every fitted
    row.  Input/output factors permit the exact foreground-correction
    conjugation; they are not inferred or normalized here.  A constant
    observed-to-rest F_lambda factor commutes with the matrix.
    """

    name: str
    input_wave_obs: np.ndarray
    output_indices: np.ndarray
    matrix: Any
    output_weights: Any = 1.0
    input_factor: Any = 1.0
    output_factor: Any = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("Resolution operator name must not be empty")
        wave = np.asarray(self.input_wave_obs, dtype=float)
        if wave.ndim != 1 or wave.size < 2 or not np.isfinite(wave).all() or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError("input_wave_obs must be finite, positive and strictly increasing")
        raw_indices = np.asarray(self.output_indices)
        if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu" or raw_indices.size == 0:
            raise ValueError("output_indices must be a nonempty integer vector")
        indices = raw_indices.astype(np.int64, copy=True)
        if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError("output_indices must be non-negative and unique within a band")
        matrix = self.matrix.tocsr(copy=True) if sparse.issparse(self.matrix) else np.asarray(self.matrix, dtype=float).copy()
        if matrix.shape != (len(indices), len(wave)):
            raise ValueError("Resolution matrix shape must match output rows and native input wavelengths")
        values = matrix.data if sparse.issparse(matrix) else matrix
        if not np.isfinite(values).all():
            raise ValueError("Resolution matrix must contain finite response values")
        weights = _vector(self.output_weights, len(indices), "output_weights")
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        if np.any((weights > 0) & (~np.isfinite(row_sums) | (row_sums <= 0))):
            raise ValueError("Every band row with positive output weight requires a nonzero response")
        wave, indices = wave.copy(), indices.copy()
        wave.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "input_wave_obs", wave)
        object.__setattr__(self, "output_indices", indices)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "output_weights", weights)
        object.__setattr__(self, "input_factor", _vector(self.input_factor, len(wave), "input_factor", positive=True))
        object.__setattr__(self, "output_factor", _vector(self.output_factor, len(indices), "output_factor", positive=True))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def verified_native(self):
        return bool(self.provenance.get("is_object_specific") is True
                    and self.provenance.get("is_approximate") is False
                    and self.provenance.get("source"))

    def card(self):
        if sparse.issparse(self.matrix):
            matrix_hashes = {key: _array_hash(getattr(self.matrix, key)) for key in ("data", "indices", "indptr")}
        else:
            matrix_hashes = {"dense": _array_hash(self.matrix)}
        result = dict(name=self.name, shape=list(self.matrix.shape), matrix_hashes=matrix_hashes,
                      negative_element_count=int(np.count_nonzero((self.matrix.data if sparse.issparse(self.matrix) else self.matrix) < 0)),
                      row_sum_min=float(np.asarray(self.matrix.sum(axis=1)).min()),
                      row_sum_max=float(np.asarray(self.matrix.sum(axis=1)).max()),
                      input_wave_obs_sha256=_array_hash(self.input_wave_obs),
                      output_indices_sha256=_array_hash(self.output_indices),
                      output_weights_sha256=_array_hash(self.output_weights),
                      input_factor_sha256=_array_hash(self.input_factor),
                      output_factor_sha256=_array_hash(self.output_factor),
                      provenance=dict(self.provenance), verified_native=self.verified_native,
                      application="sum(output_weights * output_factor * matrix @ (input_factor * intrinsic_rest_f_lambda))")
        # Fail before fitting if provenance is not serializable, rather than
        # silently dropping information from scientific output fingerprints.
        json.dumps(result, sort_keys=True, allow_nan=False)
        return result


@dataclass(frozen=True)
class BalmerLocalConfig:
    """One independently returned local null or one-broad hypothesis."""

    line: str = "hbeta"
    broad_count: int = 1
    nuisance_family: str = "core"
    velocity_half_window_kms: float = 20000.0
    broad_fwhm_bounds_kms: Tuple[float, float] = (900.0, 20000.0)
    broad_velocity_bounds_kms: Tuple[float, float] = (-2000.0, 2000.0)
    narrow_fwhm_bounds_kms: Tuple[float, float] = (70.0, 1200.0)
    narrow_velocity_bounds_kms: Tuple[float, float] = (-1000.0, 1000.0)
    outflow_fwhm_bounds_kms: Tuple[float, float] = (300.0, 3500.0)
    outflow_velocity_bounds_kms: Tuple[float, float] = (-2000.0, 1000.0)
    broad_width_starts_kms: Tuple[float, ...] = (1200.0, 3000.0, 8000.0)
    broad_velocity_starts_kms: Tuple[float, ...] = (-500.0, 0.0, 500.0)
    outflow_width_starts_kms: Tuple[float, ...] = (600.0, 1500.0)
    outflow_velocity_starts_kms: Tuple[float, ...] = (0.0,)
    mask_windows: Tuple[Tuple[float, float], ...] = ()
    mask_heii: bool = True
    min_coverage_fraction: float = 0.8
    min_valid_pixels: int = 30
    min_core_pixels: int = 3
    core_half_window_kms: float = 500.0
    require_native_response: bool = False
    optimizer_method: str = "auto"
    jacobian_method: str = "semi_analytic"
    max_nfev: Optional[int] = 1500

    def __post_init__(self):
        if self.line not in ("hbeta", "halpha"):
            raise ValueError("line must be 'hbeta' or 'halpha'")
        if isinstance(self.broad_count, bool) or not isinstance(self.broad_count, (int, np.integer)) or self.broad_count not in (0, 1):
            raise ValueError("broad_count must be zero or one")
        if self.nuisance_family not in ("core", "shared_outflow"):
            raise ValueError("nuisance_family must be 'core' or 'shared_outflow'")
        for name in ("velocity_half_window_kms", "core_half_window_kms"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.min_coverage_fraction <= 1 or self.min_valid_pixels < 1 or self.min_core_pixels < 1:
            raise ValueError("Coverage requirements must be positive")
        for stem in ("broad", "narrow", "outflow"):
            width = getattr(self, stem + "_fwhm_bounds_kms")
            velocity = getattr(self, stem + "_velocity_bounds_kms")
            if not np.isfinite(width).all() or not 0 < width[0] < width[1]:
                raise ValueError(f"{stem} FWHM bounds must be finite, positive and increasing")
            if not np.isfinite(velocity).all() or velocity[0] >= velocity[1]:
                raise ValueError(f"{stem} velocity bounds must be finite and increasing")
        for stem in ("broad", "outflow"):
            for kind, bounds_name in (("width", "fwhm"), ("velocity", "velocity")):
                starts = getattr(self, f"{stem}_{kind}_starts_kms")
                bounds = getattr(self, f"{stem}_{bounds_name}_bounds_kms")
                if not starts or not np.isfinite(starts).all() or any(not bounds[0] <= value <= bounds[1] for value in starts):
                    raise ValueError(f"{stem} {kind} starts must lie within finite bounds")
        for window in self.mask_windows:
            if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
                raise ValueError("mask_windows must contain finite increasing windows")
        if self.optimizer_method not in ("auto", "variable_projection", "legacy_joint") or self.jacobian_method not in ("semi_analytic", "2-point"):
            raise ValueError("Unsupported native optimizer/Jacobian method")

    @property
    def center(self):
        return lines.get(self.line).vacuum_wavelength

    @property
    def window(self):
        return (self.center * np.exp(-self.velocity_half_window_kms / C_KMS),
                self.center * np.exp(self.velocity_half_window_kms / C_KMS))

    @property
    def masks(self):
        return self.mask_windows + (((4660.0, 4715.0),) if self.line == "hbeta" and self.mask_heii else ())


def _recipe(config, active_sii=()):
    prefix = "Hb" if config.line == "hbeta" else "Ha"
    members = [(f"{prefix}_narrow", config.line, None, None)]
    if config.line == "hbeta":
        members += [("OIII5007_core", "oiii_5008", None, None),
                    ("OIII4959_core", "oiii_4960", "OIII5007_core", 2.98)]
    else:
        members += [("NII6585", "nii_6585", None, None),
                    ("NII6549", "nii_6550", "NII6585", 2.96)]
        members += [("SII6718" if line == "sii_6718" else "SII6733", line, None, None) for line in active_sii]
    components = []
    for name, line, tied, ratio in members:
        components.append(ComponentRecipe(name, (line,), "narrow", kinematic_group="narrow",
            velocity_bounds_kms=config.narrow_velocity_bounds_kms,
            fwhm_bands_kms=(config.narrow_fwhm_bounds_kms,), fixed_ratio_to=tied, fixed_ratio=ratio))
    if config.nuisance_family == "shared_outflow":
        for name, line, tied, ratio in members:
            out_name = f"{prefix}_outflow" if line == config.line else name + "_outflow"
            out_tied = None if tied is None else tied + "_outflow"
            components.append(ComponentRecipe(out_name, (line,), "wing", kinematic_group="outflow",
                velocity_bounds_kms=config.outflow_velocity_bounds_kms,
                fwhm_bands_kms=(config.outflow_fwhm_bounds_kms,), fixed_ratio_to=out_tied, fixed_ratio=ratio))
    if config.broad_count:
        name = f"{prefix}_broad1"
        components.append(ComponentRecipe(name, (config.line,), "broad", kinematic_group=name,
            velocity_bounds_kms=config.broad_velocity_bounds_kms,
            fwhm_bands_kms=(config.broad_fwhm_bounds_kms,)))
    return ComplexRecipe(id=f"{config.line}_local_{config.nuisance_family}_broad{config.broad_count}",
        aliases=(), label=f"local {config.line}", fit_window=config.window,
        fit_windows=(config.window,), mask_windows=config.masks, components=tuple(components),
        required_line_ids=tuple(line for _, line, _, _ in members if not line.startswith("sii_")),
        min_coverage_fraction=config.min_coverage_fraction, min_valid_pixels=config.min_valid_pixels,
        continuum_mode="linear", backend="generic")


def _coverage(spectrum, config):
    wave = spectrum.wave_rest
    in_window = (wave >= config.window[0]) & (wave <= config.window[1])
    unmasked_window = in_window.copy()
    for lo, hi in config.masks:
        unmasked_window &= ~((wave >= lo) & (wave <= hi))
    mask = spectrum.valid_mask & unmasked_window
    valid_wave = wave[mask]
    span = (max(0.0, min(config.window[1], valid_wave.max()) - max(config.window[0], valid_wave.min()))
            / (config.window[1] - config.window[0])) if len(valid_wave) else 0.0
    fraction = float(mask.sum() / unmasked_window.sum()) if unmasked_window.any() else 0.0
    required = ("hbeta", "oiii_4960", "oiii_5008") if config.line == "hbeta" else ("halpha", "nii_6550", "nii_6585")
    core_counts = {}
    for line in (*required, *(("sii_6718", "sii_6733") if config.line == "halpha" else ())):
        center = lines.get(line).vacuum_wavelength
        core_counts[line] = int(np.count_nonzero(mask & (wave >= center * np.exp(-config.core_half_window_kms / C_KMS))
                                               & (wave <= center * np.exp(config.core_half_window_kms / C_KMS))))
    active_sii = tuple(line for line in ("sii_6718", "sii_6733") if core_counts.get(line, 0) >= config.min_core_pixels)
    ok = (mask.sum() >= config.min_valid_pixels and min(span, fraction) >= config.min_coverage_fraction
          and all(core_counts[line] >= config.min_core_pixels for line in required))
    return mask, active_sii, dict(covered=bool(ok), window=tuple(config.window),
        n_valid_pixels=int(mask.sum()), window_span_fraction=float(span), valid_pixel_fraction=fraction,
        core_valid_pixels=core_counts, active_sii_line_ids=active_sii, mask_windows=config.masks)


class _ResponseContext(GenericComplexContext):
    def __init__(self, recipe, scale, spectrum, mask, operators):
        super().__init__(recipe, [component.id for component in recipe.components], scale)
        self.spectrum = spectrum
        self.fit_indices = np.flatnonzero(mask)
        self.operators = tuple(operators or ())
        self.output_coverage = np.zeros(len(spectrum.flux))
        for operator in self.operators:
            if operator.output_indices.max() >= len(spectrum.flux):
                raise ValueError("Resolution output index is outside the Spectrum grid")
            self.output_coverage[operator.output_indices] += operator.output_weights
        fitted_sums = self.output_coverage[mask]
        self.weight_validation = dict(
            required=bool(self.operators),
            absolute_tolerance=OUTPUT_WEIGHT_SUM_ATOL,
            relative_tolerance=0.0,
            precision_basis="four float32 epsilons for recorded inverse-variance coadd weights",
            fitted_row_sum_min=float(fitted_sums.min()) if len(fitted_sums) else None,
            fitted_row_sum_max=float(fitted_sums.max()) if len(fitted_sums) else None,
            fitted_row_max_abs_deviation=float(np.max(np.abs(fitted_sums-1))) if len(fitted_sums) else None,
            normalization_applied=False,
        )
        if self.operators and not np.allclose(fitted_sums, 1.0, rtol=0, atol=OUTPUT_WEIGHT_SUM_ATOL):
            raise ValueError("Resolution output weights must sum to one on every fitted Spectrum row")

    def _rows(self, wave):
        if np.array_equal(wave, self.spectrum.wave_rest):
            return np.arange(len(wave))
        if np.array_equal(wave, self.spectrum.wave_rest[self.fit_indices]):
            return self.fit_indices
        raise ValueError("Response model evaluation requires the full Spectrum grid or its frozen fitting rows")

    def _apply(self, evaluate):
        output = None
        for operator in self.operators:
            values = evaluate(operator.input_wave_obs / (1.0 + self.spectrum.z))
            factor_shape = (len(operator.input_factor),) + (1,) * (values.ndim - 1)
            projected = np.asarray(operator.matrix.dot(values * operator.input_factor.reshape(factor_shape)))
            out_shape = (len(operator.output_factor),) + (1,) * (values.ndim - 1)
            projected *= (operator.output_factor * operator.output_weights).reshape(out_shape)
            if output is None:
                output = np.zeros((len(self.spectrum.flux),) + values.shape[1:])
            output[operator.output_indices] += projected
        output[self.output_coverage == 0] = np.nan
        return output

    def separable_design(self, nonlinear, wave, need_derivatives):
        if not self.operators:
            return super().separable_design(nonlinear, wave, need_derivatives)
        rows = self._rows(wave)
        def native(native_wave):
            design, derivatives = GenericComplexContext.separable_design(self, nonlinear, native_wave, need_derivatives)
            return np.concatenate((design, *derivatives), axis=1) if need_derivatives else design
        projected = self._apply(native)[rows]
        nlinear = len(self.linear_names)
        design = projected[:, :nlinear]
        derivatives = None
        if need_derivatives:
            derivatives = tuple(projected[:, nlinear*(i+1):nlinear*(i+2)]
                                for i in range(len(self.nonlinear_names)))
        return design, derivatives

    def components(self, theta, wave):
        if not self.operators:
            return super().components(theta, wave)
        rows = self._rows(wave)
        names = list(GenericComplexContext.components(self, theta, self.operators[0].input_wave_obs / (1+self.spectrum.z)))
        def native(native_wave):
            components = GenericComplexContext.components(self, theta, native_wave)
            return np.column_stack([components[name] for name in names])
        projected = self._apply(native)[rows]
        return {name: projected[:, index] for index, name in enumerate(names)}


def _starts(context, config):
    widths = config.broad_width_starts_kms if config.broad_count else (None,)
    velocities = config.broad_velocity_starts_kms if config.broad_count else (None,)
    out_widths = config.outflow_width_starts_kms if config.nuisance_family == "shared_outflow" else (None,)
    out_velocities = config.outflow_velocity_starts_kms if config.nuisance_family == "shared_outflow" else (None,)
    prefix = "Hb" if config.line == "hbeta" else "Ha"
    seen = set()
    for width, velocity, out_width, out_velocity in product(widths, velocities, out_widths, out_velocities):
        start = context.initial.copy()
        for name, value in ((f"{prefix}_broad1.fwhm_kms", width), (f"{prefix}_broad1.velocity_kms", velocity),
                            ("outflow.fwhm_kms", out_width), ("outflow.velocity_kms", out_velocity)):
            if value is not None:
                start[context.index[name]] = value
        signature = tuple(start)
        if signature not in seen:
            seen.add(signature)
            yield start


def _metrics(context, theta, spectrum, config):
    values = dict(zip(context.names, map(float, theta)))
    prefix = "Hb" if config.line == "hbeta" else "Ha"
    flux = values.get(f"{prefix}_broad1.flux", 0.0)
    width = values.get(f"{prefix}_broad1.fwhm_kms", np.nan) if flux > 0 else np.nan
    velocity = values.get(f"{prefix}_broad1.velocity_kms", np.nan) if flux > 0 else np.nan
    continuum = values["continuum.constant"] + values["continuum.slope"] * (config.center-context.pivot)
    scale = spectrum.flux_density_scale_to_cgs
    result = {f"{prefix}_broad_flux_input": flux, f"{prefix}_broad_fwhm_kms": width,
              f"{prefix}_broad_velocity_kms": velocity, f"{prefix}_broad_sigma_kms": width/2.354820045,
              f"{prefix}_broad_ew_rest": flux/continuum if continuum > 0 else np.nan,
              f"{prefix}_broad_flux_cgs": flux*scale if scale is not None else np.nan,
              "continuum_at_line": continuum}
    for role in ("narrow", "outflow"):
        flux_value = values.get(f"{prefix}_{role}.flux", np.nan)
        result[f"{prefix}_{role}_flux_input"] = flux_value
        result[f"{prefix}_{role}_flux_cgs"] = flux_value*scale if scale is not None else np.nan
    # Report every independent full-Gaussian amplitude, preserving doublet
    # sums through the declared fixed ratios in the rendered components.
    for name in context.linear_names:
        if name.endswith(".flux"):
            result[name[:-5]+"_flux_input"] = values[name]
    return result


def _result_pair(spectrum, config, mask, coverage, components, parameters, errors,
                 covariance, metrics, metric_errors, chi2, dof, bic, warnings,
                 metadata, optimizer_result, success, message, status):
    zeros = np.zeros_like(spectrum.flux)
    continuum_components = {name: curve for name, curve in components.items() if name.startswith("local_continuum_")}
    line_components = {name: curve for name, curve in components.items() if not name.startswith("local_continuum_")}
    continuum_model = sum(continuum_components.values(), zeros.copy())
    line_model = sum(line_components.values(), zeros.copy())
    names = list(parameters)
    continuum_names = ["continuum.constant", "continuum.slope"]
    available = all(name in parameters for name in continuum_names)
    continuum_covariance = None
    if covariance is not None and available:
        indices = [names.index(name) for name in continuum_names]
        continuum_covariance = covariance[np.ix_(indices, indices)]
    reduced = chi2/dof if dof else np.nan
    continuum = GlobalContinuumResult(success=bool(success and available), status=status, message=message,
        param_values={name: parameters[name] for name in continuum_names if available},
        param_errors={name: errors.get(name, np.nan) for name in continuum_names if available},
        covariance=continuum_covariance, chi2=chi2, dof=dof, reduced_chi2=reduced,
        wave_rest=spectrum.wave_rest.copy(), model=continuum_model, component_models=continuum_components,
        fit_mask=mask.copy(), clip_mask=mask.copy(), warnings=list(warnings),
        metadata={**metadata, "covariance_parameter_names": continuum_names if available else []},
        optimizer_result=optimizer_result)
    fit = EmissionComplexResult(success=success, status=status, message=message,
        selected_model=f"{config.nuisance_family}_broad{config.broad_count}", param_values=parameters,
        param_errors=errors, covariance=covariance, metrics=metrics, metric_errors=metric_errors,
        chi2=chi2, dof=dof, reduced_chi2=reduced, bic=bic, wave_rest=spectrum.wave_rest.copy(),
        flux_continuum_subtracted=spectrum.flux-continuum_model, err=spectrum.err.copy(),
        model=line_model, component_models=line_components, fit_mask=mask.copy(), warnings=list(warnings),
        metadata=metadata, optimizer_result=optimizer_result)
    return continuum, fit


def fit_balmer_local(spectrum: Spectrum, config: Optional[BalmerLocalConfig] = None, *,
                     instrumental_response: Optional[Sequence[BandResolutionOperator]] = None,
                     compute_covariance: bool = True):
    """Fit one explicit local hypothesis and return (continuum, line_fit).

    Supply a *rest-frame F_lambda* Spectrum; wavelengths in each response
    operator remain observed-frame.  ``broad_count=0`` contains no broad flux,
    width or velocity parameters.  Core/outflow/continuum options are supplied
    identically under both hypotheses, including centered shared outflows.
    Returned display components are forward-convolved; all free coefficients
    and their joint covariance remain on ``line_fit``.  No detection or model
    preference is assigned by this function.
    """
    require_rest_frame_flux(spectrum)
    config = config or BalmerLocalConfig()
    if not isinstance(config, BalmerLocalConfig):
        raise TypeError("config must be BalmerLocalConfig")
    operators = tuple(instrumental_response or ())
    if any(not isinstance(operator, BandResolutionOperator) for operator in operators):
        raise TypeError("instrumental_response must contain BandResolutionOperator objects")
    if config.require_native_response and (not operators or not all(operator.verified_native for operator in operators)):
        raise ValueError("A verified object-specific native response is required; no sigma-only fallback is supported")
    if len({operator.name for operator in operators}) != len(operators):
        raise ValueError("Resolution operator band names must be unique")
    cards = [operator.card() for operator in operators]
    mask, active_sii, coverage = _coverage(spectrum, config)
    recipe = _recipe(config, active_sii)
    metadata = {**spectrum.metadata.to_dict(), "fit_kind": "joint_local_linear_balmer",
        "config": asdict(config), "line": config.line, "broad_count": config.broad_count,
        "n_broad_components": config.broad_count, "nuisance_family": config.nuisance_family,
        "fit_window": config.window, "coverage": coverage, "active_sii_line_ids": active_sii,
        "active_component_ids": tuple(component.id for component in recipe.components),
        "continuum_mode": "linear", "continuum_pivot_angstrom": 0.5*sum(config.window),
        "host_subtraction": False, "balmer_pseudocontinuum": False, "hgamma_linked_fit": False,
        "absorption_model": False, "feii_model": False,
        "uncertainty_conditioning": "joint_local_affine_continuum_and_lines",
        "instrumental_response": cards, "forward_resolution": bool(operators),
        "resolution_status": ("verified_native_operator" if operators and all(operator.verified_native for operator in operators)
                              else "unverified_operator" if operators else "missing_not_intrinsic"),
        "width_definition": ("intrinsic_single_gaussian_conditional_on_supplied_operator" if operators
                             else "unconvolved_observed_profile_single_gaussian"),
        "flux_definition": "full_intrinsic_Gaussian_area_in_input_flux_density_times_rest_Angstrom",
        "broad_detection_assigned": False,
        "chi2_scope": "joint_local_continuum_and_emission_lines_on_frozen_valid_pixels",
        "likelihood": "diagonal_supplied_pixel_errors; no MC",
        "limitations": "Covariance conditions on the affine continuum family, nuisance family and supplied response; absorption, FeII and response uncertainty are not modeled. No broad-line detection probability is assigned.",
        "fit_mask_sha256": _array_hash(mask), "data_sha256": _array_hash(spectrum.flux[mask]),
        "error_sha256": _array_hash(spectrum.err[mask]), "wavelength_sha256": _array_hash(spectrum.wave_obs[mask])}
    if not coverage["covered"]:
        metadata["covariance_parameter_names"] = []
        warning = FitWarning("line_complex_not_covered", "Local Balmer core, continuum or fitting window has insufficient valid support.", context=coverage)
        return _result_pair(spectrum, config, mask, coverage, {}, {}, {}, None, {}, {},
            np.nan, 0, np.nan, [warning], metadata, None, False, warning.message, -1)
    scale = float(np.trapezoid(np.clip(spectrum.flux[mask], 0, None), spectrum.wave_rest[mask]))
    context = _ResponseContext(recipe, scale, spectrum, mask, operators)
    metadata["resolution_weight_validation"] = context.weight_validation
    attempts, candidates = [], []
    for order, start in enumerate(_starts(context, config)):
        result, optimizer, fallback = _solve_once_with_fallback(context, spectrum.wave_rest[mask],
            spectrum.flux[mask], spectrum.err[mask], start, config)
        residual = (spectrum.flux[mask]-context.model(result.x, spectrum.wave_rest[mask]))/spectrum.err[mask]
        chi2 = float(residual @ residual)
        attempts.append(dict(order=order, success=bool(result.success), chi2=chi2,
            start_parameters=dict(zip(context.names, map(float, start))),
            optimizer_used=optimizer, fallback_reason=fallback, nfev=int(result.nfev)))
        candidates.append((not bool(result.success), chi2, order, result, optimizer, fallback))
    _, chi2, order, result, optimizer, fallback = min(candidates, key=lambda row: row[:3])
    dof = max(int(mask.sum()-len(context.names)), 0)
    reduced = chi2/dof if dof else np.nan
    covariance, errors, warnings = _covariance_from_jacobian(result.jac, reduced, context.names) if compute_covariance else (None, dict.fromkeys(context.names, np.nan), [])
    warnings.extend(_active_bound_warnings(result, context.names))
    if fallback:
        warnings.append(FitWarning("optimizer_fallback_legacy", "Native variable projection fell back to the native joint optimizer.", context={"reason": fallback}))
    if not operators:
        warnings.append(FitWarning("instrumental_response_missing", "Widths condition on an unconvolved observed-profile model; they are not intrinsic widths.", severity="info"))
    metric_function = lambda theta: _metrics(context, theta, spectrum, config)
    metrics = metric_function(result.x)
    metric_errors = _metric_errors(result.x, covariance, metric_function)
    if config.broad_count == 0:
        prefix = "Hb" if config.line == "hbeta" else "Ha"
        for name in metric_errors:
            if name.startswith(prefix+"_broad_"):
                metric_errors[name] = np.nan
    metadata.update(covariance_parameter_names=list(context.names), joint_covariance_parameter_names=list(context.names), n_free_parameters=len(context.names),
        deterministic_multistarts=attempts, selected_start=order, optimizer_used=optimizer,
        continuum_at_line=metrics["continuum_at_line"], n_pixels=int(mask.sum()))
    components = context.components(result.x, spectrum.wave_rest)
    return _result_pair(spectrum, config, mask, coverage, components,
        dict(zip(context.names, map(float, result.x))), errors, covariance, metrics, metric_errors,
        chi2, dof, chi2+len(context.names)*np.log(mask.sum()), warnings, metadata, result,
        bool(result.success), str(result.message), int(result.status))


def fit_halpha_local(spectrum: Spectrum, config: Optional[BalmerLocalConfig] = None, **kwargs):
    """Halpha convenience wrapper; defaults to the new +/-20000 km/s recipe."""
    config = config or BalmerLocalConfig(line="halpha")
    if config.line != "halpha":
        raise ValueError("fit_halpha_local requires a halpha configuration")
    return fit_balmer_local(spectrum, config, **kwargs)


def evaluate_balmer_local_model(wave_rest, fit, config: BalmerLocalConfig, *,
                               include_continuum=True, return_components=False):
    """Evaluate the fitted *intrinsic* model on any rest-wavelength grid.

    This does not apply an instrumental response.  It permits a caller to
    construct a model-conditioned native-band null and deterministic signal
    injections before applying the exact native response/foreground operator.
    """
    if (not fit.success or json.dumps(fit.metadata.get("config"), sort_keys=True)
            != json.dumps(asdict(config), sort_keys=True)):
        raise ValueError("A successful fit with the exact recorded configuration is required")
    wave = np.asarray(wave_rest, dtype=float)
    if wave.ndim != 1 or not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError("wave_rest must be a finite positive one-dimensional vector")
    recipe = _recipe(config, tuple(fit.metadata["active_sii_line_ids"]))
    context = GenericComplexContext(recipe, fit.metadata["active_component_ids"], 1.0)
    if set(context.names) != set(fit.param_values):
        raise ValueError("Recorded local-fit parameter contract differs from the recipe")
    theta = np.asarray([fit.param_values[name] for name in context.names])
    components = context.components(theta, wave)
    if not include_continuum:
        components = {name: curve for name, curve in components.items() if not name.startswith("local_continuum_")}
    return components if return_components else sum(components.values(), np.zeros_like(wave))
