"""Exact object/grid/LSF-specific preconvolved XSL template products."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

import numpy as np

from ...resolution import match_template_resolution_toward_data
from .templates import PPXFTemplateLibrary, array_sha256

PRECONVOLUTION_ALGORITHM_VERSION = "qsospec_xsl_one_sided_variable_gaussian_v1"
PRECONVOLUTION_NORMALIZATION_VERSION = "unnormalized_source_flux_v1"


@dataclass(frozen=True)
class PreconvolvedTemplateProduct:
    """Description of one validated derived fitting product."""

    path: str
    cache_key: str
    metadata: Mapping[str, Any]
    fit_wave: np.ndarray
    fit_flux: np.ndarray
    timings: Mapping[str, float] = field(default_factory=dict)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _target_resolution_arrays(
    source_library: PPXFTemplateLibrary,
    preprocessed_spectrum: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wave = np.asarray(preprocessed_spectrum.wave_log, dtype=float)
    resolution = preprocessed_spectrum.metadata.get("spectral_resolution")
    if resolution is None or resolution.status != "valid":
        raise ValueError(
            "Exact XSL preconvolution requires a valid object-specific data LSF."
        )
    if resolution.mode == "banded_matrix":
        raise ValueError(
            "Exact XSL preconvolution does not yet support a banded resolution matrix."
        )
    observed_wave = wave * (1.0 + float(preprocessed_spectrum.redshift))
    data_sigma = resolution.sigma_lambda(observed_wave) / (
        1.0 + float(preprocessed_spectrum.redshift)
    )
    template_fwhm = source_library.source_resolution_metadata.get("fwhm")
    if template_fwhm is None:
        raise ValueError("Native XSL does not provide intrinsic FWHM metadata.")
    template_sigma = np.interp(
        wave,
        np.asarray(source_library.source_wave, dtype=float),
        np.asarray(template_fwhm, dtype=float),
        left=np.nan,
        right=np.nan,
    ) / 2.354820045
    match = match_template_resolution_toward_data(
        data_sigma, template_sigma, wave
    )
    support = (
        (wave >= source_library.source_wavelength_coverage[0])
        & (wave <= source_library.source_wavelength_coverage[1])
    )
    valid = match.comparable & support
    if not np.any(valid):
        raise ValueError(
            "No valid object-specific resolution values overlap the XSL "
            "preconvolution domain."
        )
    return data_sigma, template_sigma, match.additional_sigma_lambda, valid


def preconvolution_contract(
    source_library: PPXFTemplateLibrary,
    preprocessed_spectrum: Any,
    *,
    object_key: str,
    fit_range: tuple[float, float],
) -> dict[str, Any]:
    """Build the exact provenance fields whose digest identifies the cache."""

    data_sigma, _, _, valid = _target_resolution_arrays(
        source_library, preprocessed_spectrum
    )
    wave = np.asarray(preprocessed_spectrum.wave_log, dtype=float)
    return {
        "format_version": 1,
        "source_template_sha256": str(source_library.source_library_sha256),
        "source_template_matrix_sha256": array_sha256(
            np.asarray(source_library.source_flux)
        ),
        "source_template_wave_sha256": array_sha256(
            np.asarray(source_library.source_wave)
        ),
        "source_template_original_shape": list(source_library.original_shape),
        "source_template_count": int(source_library.n_source_templates),
        "source_template_axis_metadata": dict(
            source_library.template_axis_metadata
        ),
        "target_object_key": str(object_key),
        "redshift": float(preprocessed_spectrum.redshift),
        "fit_range": [float(fit_range[0]), float(fit_range[1])],
        "ppxf_log_grid_sha256": array_sha256(wave),
        "rest_target_sigma_lambda_sha256": array_sha256(data_sigma),
        "valid_resolution_mask_sha256": array_sha256(valid.astype(np.uint8)),
        "convolution_algorithm_version": PRECONVOLUTION_ALGORITHM_VERSION,
        "template_normalization_version": PRECONVOLUTION_NORMALIZATION_VERSION,
    }


def preconvolution_cache_key(contract: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 key for a preconvolution contract."""

    return sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def _runtime_convolved_source_matrix(
    source_library: PPXFTemplateLibrary,
    preprocessed_spectrum: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    start = perf_counter()
    wave = np.asarray(preprocessed_spectrum.wave_log, dtype=float)
    matrix = np.empty((wave.size, source_library.n_source_templates), dtype=float)
    for index in range(source_library.n_source_templates):
        matrix[:, index] = np.interp(
            wave,
            np.asarray(source_library.source_wave, dtype=float),
            np.asarray(source_library.source_flux, dtype=float)[:, index],
            left=0.0,
            right=0.0,
        )
    interpolation_seconds = perf_counter() - start
    lsf_start = perf_counter()
    data_sigma, template_sigma, additional_sigma, valid = (
        _target_resolution_arrays(source_library, preprocessed_spectrum)
    )
    lsf_seconds = perf_counter() - lsf_start
    pixel_width = np.gradient(wave)
    sigma_pixels = np.divide(
        np.nan_to_num(additional_sigma, nan=0.0),
        pixel_width,
        out=np.zeros_like(additional_sigma),
        where=pixel_width > 0,
    )
    convolution_start = perf_counter()
    from ppxf.ppxf_util import gaussian_filter1d as variable_gaussian_filter1d

    for index in range(matrix.shape[1]):
        matrix[:, index] = variable_gaussian_filter1d(matrix[:, index], sigma_pixels)
    convolution_seconds = perf_counter() - convolution_start
    diagnostics = {
        "data_sigma_lambda_sha256": array_sha256(data_sigma),
        "template_sigma_lambda_sha256": array_sha256(template_sigma),
        "additional_sigma_lambda_sha256": array_sha256(additional_sigma),
        "valid_resolution_mask_sha256": array_sha256(valid.astype(np.uint8)),
    }
    timings = {
        "source_template_interpolation_seconds": float(interpolation_seconds),
        "lsf_interpolation_seconds": float(lsf_seconds),
        "runtime_convolution_seconds": float(convolution_seconds),
    }
    return matrix, diagnostics, timings


def build_preconvolved_xsl_product(
    source_library: PPXFTemplateLibrary,
    preprocessed_spectrum: Any,
    *,
    output_path: str | Path,
    object_key: str,
    fit_range: tuple[float, float],
    overwrite: bool = False,
) -> PreconvolvedTemplateProduct:
    """Build and atomically write one exact object-specific XSL product."""

    if source_library.profile_id != "xsl_native" or source_library.product_kind != "native":
        raise ValueError("Preconvolution requires a native XSL source library.")
    path = Path(output_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Preconvolved XSL product already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = preconvolution_contract(
        source_library,
        preprocessed_spectrum,
        object_key=object_key,
        fit_range=fit_range,
    )
    cache_key = preconvolution_cache_key(contract)
    fit_flux, diagnostics, timings = _runtime_convolved_source_matrix(
        source_library, preprocessed_spectrum
    )
    metadata = {
        **contract,
        **diagnostics,
        "builder_timings": dict(timings),
        "cache_key": cache_key,
        "fit_template_matrix_sha256": array_sha256(fit_flux),
        "fit_template_wave_sha256": array_sha256(
            np.asarray(preprocessed_spectrum.wave_log, dtype=float)
        ),
    }
    with NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(
            stream,
            fit_flux=fit_flux,
            fit_wave=np.asarray(preprocessed_spectrum.wave_log, dtype=float),
            source_template_indices=np.arange(
                source_library.n_source_templates, dtype=np.int64
            ),
            preconvolution_metadata_json=np.asarray(_canonical_json(metadata)),
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        with np.load(temporary, allow_pickle=False) as check:
            if array_sha256(np.asarray(check["fit_flux"])) != metadata[
                "fit_template_matrix_sha256"
            ]:
                raise RuntimeError("Preconvolved XSL read-back matrix validation failed.")
            decoded = json.loads(str(np.asarray(check["preconvolution_metadata_json"]).item()))
            if decoded.get("cache_key") != cache_key:
                raise RuntimeError("Preconvolved XSL read-back provenance validation failed.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return PreconvolvedTemplateProduct(
        path=str(path),
        cache_key=cache_key,
        metadata=metadata,
        fit_wave=np.asarray(preprocessed_spectrum.wave_log, dtype=float).copy(),
        fit_flux=fit_flux,
        timings=timings,
    )


def validate_preconvolved_xsl_product(
    library: PPXFTemplateLibrary,
    preprocessed_spectrum: Any,
    *,
    fit_range: tuple[float, float] | None = None,
    object_key: str | None = None,
) -> dict[str, Any]:
    """Validate a loaded derived XSL product against the current fit state."""

    if library.profile_id != "xsl_preconvolved":
        return {"preconvolution_validation_status": "not_applicable"}
    metadata = dict(library.preconvolution_metadata)
    required = {
        "cache_key",
        "source_template_sha256",
        "ppxf_log_grid_sha256",
        "rest_target_sigma_lambda_sha256",
        "valid_resolution_mask_sha256",
        "redshift",
        "fit_range",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(
            f"Preconvolved XSL product lacks exact provenance fields: {missing}"
        )
    expected_object = object_key or preprocessed_spectrum.metadata.get("object_id")
    product_object = metadata.get("target_object_key")
    if expected_object is None:
        raise ValueError(
            "Exact preconvolved XSL validation requires the current object key."
        )
    if str(product_object) != str(expected_object):
        raise ValueError("Preconvolved XSL product belongs to another object.")
    if not np.isclose(float(metadata["redshift"]), float(preprocessed_spectrum.redshift), rtol=0, atol=1e-12):
        raise ValueError("Preconvolved XSL product redshift does not match this spectrum.")
    if fit_range is not None and not np.allclose(
        np.asarray(metadata["fit_range"], dtype=float),
        np.asarray(fit_range, dtype=float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Preconvolved XSL product fit range does not match.")
    wave = np.asarray(preprocessed_spectrum.wave_log, dtype=float)
    if metadata["ppxf_log_grid_sha256"] != array_sha256(wave):
        raise ValueError("Preconvolved XSL pPXF grid does not match this spectrum.")
    data_sigma, _, _, valid = _target_resolution_arrays(
        library, preprocessed_spectrum
    )
    if metadata["rest_target_sigma_lambda_sha256"] != array_sha256(data_sigma):
        raise ValueError("Preconvolved XSL target LSF does not match this spectrum.")
    if metadata["valid_resolution_mask_sha256"] != array_sha256(valid.astype(np.uint8)):
        raise ValueError("Preconvolved XSL valid-resolution mask does not match.")
    contract = {
        key: metadata[key]
        for key in (
            "format_version",
            "source_template_sha256",
            "source_template_matrix_sha256",
            "source_template_wave_sha256",
            "source_template_original_shape",
            "source_template_count",
            "source_template_axis_metadata",
            "target_object_key",
            "redshift",
            "fit_range",
            "ppxf_log_grid_sha256",
            "rest_target_sigma_lambda_sha256",
            "valid_resolution_mask_sha256",
            "convolution_algorithm_version",
            "template_normalization_version",
        )
    }
    if preconvolution_cache_key(contract) != metadata["cache_key"]:
        raise ValueError("Preconvolved XSL cache key does not match its provenance.")
    return {
        "preconvolution_cache_key": metadata["cache_key"],
        "preconvolution_validation_status": "preconvolved_exact",
        "preconvolution_residual_match_status": "not_required",
    }
