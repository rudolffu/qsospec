"""pPXF template loading for externally installed SPS template bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import json

import numpy as np


SAMPLE_WAVELENGTHS = {
    "fHost_4000": 4000.0,
    "fHost_5100": 5100.0,
    "fHost_8000": 8000.0,
    "fHost_1um": 10000.0,
    "fHost_1p6um": 16000.0,
    "fHost_2p2um": 22000.0,
}

PPXF_NPZ_LOADER_VERSION = "qsospec_ppxf_npz_v1"
TEMPLATE_FLATTENING_CONVENTION = (
    "move_wavelength_axis_to_front_then_c_order_flatten"
)

EMILES_NATIVE_FILE = "spectra_emiles_9.0.npz"
XSL_NATIVE_FILE = "spectra_xsl_9.0.npz"


@dataclass(frozen=True)
class ResolvedHostTemplateProfile:
    """Resolved fit/source products for one stellar-template profile."""

    profile_id: str
    family: str
    product_kind: str
    fit_template_path: str
    fit_template_sha256: str
    source_template_path: str
    source_template_sha256: str
    resolution_matching_mode: str
    template_coarser_action: str
    preserve_native_data: bool
    exact_preconvolution_required: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PPXFTemplateLibrary:
    """Stellar templates loaded from a local pPXF-compatible NPZ file."""

    flux: np.ndarray
    wave: np.ndarray
    log_wave: np.ndarray
    family: str
    source_path: str
    wavelength_coverage: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_shape: Tuple[int, ...] = field(default_factory=tuple)
    warnings: List[str] = field(default_factory=list)
    source_flux: Optional[np.ndarray] = None
    source_wave: Optional[np.ndarray] = None
    profile_id: str = "custom_native"
    product_kind: str = "native"
    fit_source_path: Optional[str] = None
    fit_source_sha256: Optional[str] = None
    source_library_path: Optional[str] = None
    source_library_sha256: Optional[str] = None
    fit_resolution_metadata: Dict[str, Any] = field(default_factory=dict)
    source_resolution_metadata: Dict[str, Any] = field(default_factory=dict)
    preconvolution_metadata: Dict[str, Any] = field(default_factory=dict)
    template_axis_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.flux = np.asarray(self.flux, dtype=float)
        self.wave = np.asarray(self.wave, dtype=float)
        if self.source_flux is None:
            self.source_flux = self.flux
        else:
            self.source_flux = np.asarray(self.source_flux, dtype=float)
        if self.source_wave is None:
            self.source_wave = self.wave
        else:
            self.source_wave = np.asarray(self.source_wave, dtype=float)
        if self.fit_source_path is None:
            self.fit_source_path = self.source_path
        if self.source_library_path is None:
            self.source_library_path = self.source_path
        if self.fit_source_sha256 is None:
            self.fit_source_sha256 = self.metadata.get("source_sha256")
        if self.source_library_sha256 is None:
            self.source_library_sha256 = self.metadata.get("source_sha256")

    @property
    def fit_flux(self) -> np.ndarray:
        return self.flux

    @property
    def fit_wave(self) -> np.ndarray:
        return self.wave

    @property
    def n_templates(self) -> int:
        return int(self.flux.shape[1])

    @property
    def n_source_templates(self) -> int:
        return int(np.asarray(self.source_flux).shape[1])

    @property
    def source_wavelength_coverage(self) -> Tuple[float, float]:
        wave = np.asarray(self.source_wave, dtype=float)
        return float(np.nanmin(wave)), float(np.nanmax(wave))


def _is_increasing_wave(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.size < 10 or not np.issubdtype(arr.dtype, np.number):
        return False
    finite = np.isfinite(arr)
    if np.sum(finite) < 10:
        return False
    vals = arr[finite]
    return bool(np.nanmin(vals) > 0 and np.all(np.diff(vals) > 0))


def _find_wave_key(npz: Any) -> str:
    preferred = ("lam", "lambda", "wave", "wavelength", "wavelengths")
    for key in preferred:
        if key in npz.files and _is_increasing_wave(npz[key]):
            return key
    for key in npz.files:
        if _is_increasing_wave(npz[key]):
            return key
    raise ValueError(f"Could not identify a wavelength grid in NPZ keys: {npz.files}")


def _find_template_key(npz: Any, n_wave: int) -> str:
    preferred = ("templates", "fit_flux", "spectra", "flux", "ssp", "models")
    for key in preferred:
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.ndim >= 2 and n_wave in arr.shape:
                return key
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number) and n_wave in arr.shape:
            return key
    raise ValueError(f"Could not identify template spectra in NPZ keys: {npz.files}")


def _flatten_templates(templates: np.ndarray, wave: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    arr = np.asarray(templates, dtype=float)
    n_wave = len(wave)
    if arr.shape[0] == n_wave:
        original_shape = tuple(arr.shape)
        return arr.reshape(n_wave, -1), original_shape
    axis = [i for i, size in enumerate(arr.shape) if size == n_wave]
    if not axis:
        raise ValueError(f"Template array shape {arr.shape} is incompatible with wavelength length {n_wave}")
    arr = np.moveaxis(arr, axis[0], 0)
    original_shape = tuple(arr.shape)
    return arr.reshape(n_wave, -1), original_shape


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)) if np.issubdtype(value.dtype, np.number) else None,
            "max": float(np.nanmax(value)) if np.issubdtype(value.dtype, np.number) else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _coverage_warnings(wave_min: float, wave_max: float) -> List[str]:
    warnings = []
    if wave_max < 10000.0:
        warnings.append("template_does_not_cover_nir")
    if wave_max < 22000.0:
        warnings.append("template_wavelength_coverage_insufficient")
    if wave_max < 10000.0:
        warnings.append("nir_extrapolation_not_available")
    return warnings


@lru_cache(maxsize=16)
def _cached_file_sha256(
    path_string: str, size: int, modified_ns: int
) -> str:
    digest = sha256()
    with Path(path_string).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_file_sha256(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    )


def resolve_host_template_profile(
    *,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = EMILES_NATIVE_FILE,
    template_family: str = "emiles",
    template_profile: str | None = None,
    template_product_kind: str = "native",
    source_template_file: str | None = None,
    template_coarser_action: str = "warn",
    preserve_native_data: bool = True,
) -> ResolvedHostTemplateProfile:
    """Resolve legacy template fields into one explicit profile contract."""

    if not preserve_native_data:
        raise ValueError(
            "Host decomposition requires preserve_native_data=True; the input "
            "science spectrum is never degraded to the stellar template."
        )
    root = Path(template_root).expanduser()
    fit_path = root / template_file
    family = str(template_family).lower()
    product_kind = str(template_product_kind).lower()

    if template_profile is None:
        if product_kind == "preconvolved":
            profile_id = "xsl_preconvolved"
        elif Path(template_file).name == EMILES_NATIVE_FILE:
            profile_id = "emiles_native"
        elif Path(template_file).name == XSL_NATIVE_FILE:
            profile_id = "xsl_native"
        else:
            profile_id = "custom_native"
    else:
        profile_id = str(template_profile).lower()

    expected: Dict[str, Tuple[str, str]] = {
        "emiles_native": ("emiles", "native"),
        "xsl_native": ("xsl", "native"),
        "xsl_preconvolved": ("xsl", "preconvolved"),
    }
    if profile_id not in {*expected, "custom_native"}:
        raise ValueError(f"Unknown host template profile: {profile_id!r}")
    if profile_id in expected:
        expected_family, expected_kind = expected[profile_id]
        # The historical family default was always ``emiles``. Permit the
        # canonical XSL filename to supply the missing intent when no explicit
        # profile was provided, but reject explicit profile conflicts.
        inferred_xsl_from_filename = (
            template_profile is None
            and Path(template_file).name == XSL_NATIVE_FILE
            and family == "emiles"
        )
        if family != expected_family and not inferred_xsl_from_filename:
            raise ValueError(
                f"Template profile {profile_id!r} requires family "
                f"{expected_family!r}, not {family!r}."
            )
        if product_kind != expected_kind:
            raise ValueError(
                f"Template profile {profile_id!r} requires product kind "
                f"{expected_kind!r}, not {product_kind!r}."
            )
        family = expected_family
    elif product_kind != "native":
        raise ValueError("custom_native requires template_product_kind='native'.")

    if profile_id == "emiles_native" and Path(template_file).name != EMILES_NATIVE_FILE:
        raise ValueError(
            f"emiles_native requires {EMILES_NATIVE_FILE!r}; use custom_native "
            "for another E-MILES-like product."
        )
    if profile_id == "xsl_native" and Path(template_file).name != XSL_NATIVE_FILE:
        raise ValueError(
            f"xsl_native requires {XSL_NATIVE_FILE!r}; use custom_native for another product."
        )
    if not fit_path.exists():
        raise FileNotFoundError(f"pPXF template file not found: {fit_path}")

    if profile_id == "xsl_preconvolved":
        if not source_template_file:
            raise ValueError(
                "xsl_preconvolved requires source_template_file naming native XSL."
            )
        source_path = root / source_template_file
        if Path(source_template_file).name != XSL_NATIVE_FILE:
            raise ValueError(
                f"xsl_preconvolved source_template_file must be {XSL_NATIVE_FILE!r}."
            )
    else:
        if source_template_file and Path(source_template_file) != Path(template_file):
            raise ValueError(
                "Native profiles use the fit file as their source library; do not "
                "supply a different source_template_file."
            )
        source_path = fit_path
    if not source_path.exists():
        raise FileNotFoundError(f"Native source template file not found: {source_path}")

    if profile_id == "xsl_native":
        matching_mode = "object_specific_runtime"
    elif profile_id == "xsl_preconvolved":
        matching_mode = "preconvolved_exact"
    else:
        matching_mode = "convolve_template_toward_data_only"
    return ResolvedHostTemplateProfile(
        profile_id=profile_id,
        family=family,
        product_kind=product_kind,
        fit_template_path=str(fit_path.resolve()),
        fit_template_sha256=_file_sha256(fit_path),
        source_template_path=str(source_path.resolve()),
        source_template_sha256=_file_sha256(source_path),
        resolution_matching_mode=matching_mode,
        template_coarser_action=str(template_coarser_action),
        preserve_native_data=True,
        exact_preconvolution_required=(profile_id == "xsl_preconvolved"),
        metadata={
            "fit_template_file": fit_path.name,
            "source_template_file": source_path.name,
        },
    )


def array_sha256(array: np.ndarray) -> str:
    """Return a stable identity for one numeric template array."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_reports(report_dir: Path, report_name: str, payload: Dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_name}.json"
    md_path = report_dir / f"{report_name}.md"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# pPXF Template Inspection: {payload['template_family']}",
        "",
        f"- Source file: `{payload['source_path']}`",
        f"- Keys: {', '.join(payload['keys'])}",
        f"- Wavelength key: `{payload['wavelength_key']}`",
        f"- Template key: `{payload['template_key']}`",
        f"- Coverage: {payload['wavelength_coverage'][0]:.2f} - {payload['wavelength_coverage'][1]:.2f} Angstrom",
        f"- Template matrix shape: {payload['template_shape']}",
        f"- Flattened template count: {payload['n_templates']}",
        f"- Warnings: {', '.join(payload['warnings']) if payload['warnings'] else 'none'}",
        "",
        "## Metadata",
    ]
    for key, value in payload["metadata"].items():
        lines.append(f"- `{key}`: {_json_safe(value)}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@lru_cache(maxsize=8)
def _load_npz_payload(
    path_string: str,
    size: int,
    modified_ns: int,
) -> Dict[str, Any]:
    del size, modified_ns
    source = Path(path_string)
    with np.load(source, allow_pickle=True) as npz:
        wave_key = _find_wave_key(npz)
        wave = np.asarray(npz[wave_key], dtype=float)
        template_key = _find_template_key(npz, len(wave))
        flux, original_shape = _flatten_templates(npz[template_key], wave)
        metadata = {
            key: np.asarray(npz[key])
            for key in npz.files
            if key not in (wave_key, template_key)
        }
    return {
        "wave": wave,
        "flux": flux,
        "original_shape": original_shape,
        "metadata": metadata,
        "wavelength_key": wave_key,
        "template_key": template_key,
    }


def _read_npz_payload(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return _load_npz_payload(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    )


def _template_axis_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "flattening_convention": TEMPLATE_FLATTENING_CONVENTION,
    }
    for key in ("ages", "metals", "masses", "imf", "isochrone"):
        if key not in metadata:
            continue
        value = np.asarray(metadata[key])
        result[f"{key}_shape"] = list(value.shape)
        result[f"{key}_sha256"] = array_sha256(value)
        if value.size <= 256:
            result[key] = value.tolist()
    return result


def _decode_preconvolution_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("preconvolution_metadata_json")
    if raw is None:
        return {}
    value = np.asarray(raw)
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid preconvolution_metadata_json in template product.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("preconvolution_metadata_json must encode an object.")
    return decoded


def load_ppxf_npz_templates(
    template_root: str = "~/tools/ppxf_data",
    template_file: str = EMILES_NATIVE_FILE,
    report_dir: str = "outputs/ppxf_qsospec",
    template_family: str = "emiles",
    write_report: bool = True,
    *,
    template_profile: str | None = None,
    template_product_kind: str = "native",
    source_template_file: str | None = None,
    template_coarser_action: str = "warn",
    preserve_native_data: bool = True,
) -> PPXFTemplateLibrary:
    """Load externally installed pPXF NPZ templates without vendoring data."""

    profile = resolve_host_template_profile(
        template_root=template_root,
        template_file=template_file,
        template_family=template_family,
        template_profile=template_profile,
        template_product_kind=template_product_kind,
        source_template_file=source_template_file,
        template_coarser_action=template_coarser_action,
        preserve_native_data=preserve_native_data,
    )
    fit_path = Path(profile.fit_template_path)
    source_path = Path(profile.source_template_path)
    load_start = perf_counter()
    fit_payload = _read_npz_payload(fit_path)
    fit_load_seconds = perf_counter() - load_start
    source_load_start = perf_counter()
    source_payload = (
        fit_payload if fit_path == source_path else _read_npz_payload(source_path)
    )
    source_load_seconds = (
        fit_load_seconds
        if fit_path == source_path
        else perf_counter() - source_load_start
    )
    fit_wave = np.asarray(fit_payload["wave"], dtype=float)
    fit_flux = np.asarray(fit_payload["flux"], dtype=float)
    source_wave = np.asarray(source_payload["wave"], dtype=float)
    source_flux = np.asarray(source_payload["flux"], dtype=float)
    source_metadata = dict(source_payload["metadata"])
    fit_metadata = dict(fit_payload["metadata"])
    preconvolution_metadata = _decode_preconvolution_metadata(fit_metadata)
    if profile.product_kind == "preconvolved":
        indices = np.asarray(
            fit_metadata.get("source_template_indices", []), dtype=int
        )
        if fit_flux.shape[1] != source_flux.shape[1] or not np.array_equal(
            indices, np.arange(source_flux.shape[1], dtype=int)
        ):
            raise ValueError(
                "Preconvolved XSL template order does not map one-to-one to native XSL."
            )
        if preconvolution_metadata.get("source_template_sha256") != (
            profile.source_template_sha256
        ):
            raise ValueError(
                "Preconvolved XSL source hash does not match the native XSL library."
            )
        expected_matrix_hash = preconvolution_metadata.get(
            "source_template_matrix_sha256"
        )
        actual_matrix_hash = array_sha256(source_flux)
        if expected_matrix_hash and expected_matrix_hash != actual_matrix_hash:
            raise ValueError(
                "Preconvolved XSL source-template matrix identity does not match."
            )
        if preconvolution_metadata.get("source_template_wave_sha256") != array_sha256(
            source_wave
        ):
            raise ValueError(
                "Preconvolved XSL source-template wavelength identity does not match."
            )
        if list(
            preconvolution_metadata.get("source_template_original_shape", [])
        ) != list(source_payload["original_shape"]):
            raise ValueError(
                "Preconvolved XSL source-template original shape does not match."
            )
        if preconvolution_metadata.get("source_template_axis_metadata") != (
            _template_axis_metadata(source_metadata)
        ):
            raise ValueError(
                "Preconvolved XSL source-template axis metadata/order does not match."
            )

    source_wave_hash = array_sha256(source_wave)
    source_matrix_hash = array_sha256(source_flux)
    fit_wave_hash = array_sha256(fit_wave)
    fit_matrix_hash = array_sha256(fit_flux)
    warnings_out = _coverage_warnings(
        float(np.nanmin(source_wave)), float(np.nanmax(source_wave))
    )
    payload = {
        "template_family": profile.family,
        "source_path": str(source_path),
        "source_sha256": profile.source_template_sha256,
        "keys": [
            fit_payload["template_key"],
            fit_payload["wavelength_key"],
            *fit_metadata.keys(),
        ],
        "wavelength_key": fit_payload["wavelength_key"],
        "template_key": fit_payload["template_key"],
        "loader_version": PPXF_NPZ_LOADER_VERSION,
        "flattening_convention": TEMPLATE_FLATTENING_CONVENTION,
        "template_wave_sha256": source_wave_hash,
        "template_matrix_sha256": source_matrix_hash,
        "fit_template_wave_sha256": fit_wave_hash,
        "fit_template_matrix_sha256": fit_matrix_hash,
        "wavelength_coverage": [
            float(np.nanmin(source_wave)),
            float(np.nanmax(source_wave)),
        ],
        "template_shape": list(source_payload["original_shape"]),
        "n_templates": int(source_flux.shape[1]),
        "metadata": source_metadata,
        "warnings": warnings_out,
    }

    if write_report:
        _write_reports(
            Path(report_dir),
            f"template_inspection_{profile.profile_id}",
            payload,
        )

    return PPXFTemplateLibrary(
        flux=fit_flux,
        wave=fit_wave,
        log_wave=np.log(fit_wave),
        family=profile.family,
        source_path=str(fit_path),
        wavelength_coverage=(
            float(np.nanmin(fit_wave)),
            float(np.nanmax(fit_wave)),
        ),
        metadata={
            **source_metadata,
            "source_sha256": payload["source_sha256"],
            "source_file_name": source_path.name,
            "wavelength_key": source_payload["wavelength_key"],
            "template_key": source_payload["template_key"],
            "loader_version": PPXF_NPZ_LOADER_VERSION,
            "flattening_convention": TEMPLATE_FLATTENING_CONVENTION,
            "template_wave_sha256": payload["template_wave_sha256"],
            "template_matrix_sha256": payload["template_matrix_sha256"],
            "fit_template_wave_sha256": fit_wave_hash,
            "fit_template_matrix_sha256": fit_matrix_hash,
            "fit_source_sha256": profile.fit_template_sha256,
            "fit_source_file_name": fit_path.name,
            "template_profile": profile.profile_id,
            "template_product_kind": profile.product_kind,
            "resolution_matching_mode": profile.resolution_matching_mode,
            "template_coarser_action": template_coarser_action,
            "native_data_preserved": True,
            "source_template_load_seconds": float(source_load_seconds),
            "preconvolved_cache_read_seconds": (
                float(fit_load_seconds)
                if profile.product_kind == "preconvolved"
                else None
            ),
        },
        original_shape=source_payload["original_shape"],
        warnings=list(warnings_out),
        source_flux=source_flux,
        source_wave=source_wave,
        profile_id=profile.profile_id,
        product_kind=profile.product_kind,
        fit_source_path=str(fit_path),
        fit_source_sha256=profile.fit_template_sha256,
        source_library_path=str(source_path),
        source_library_sha256=profile.source_template_sha256,
        fit_resolution_metadata={
            "fwhm": fit_metadata.get("fwhm"),
            "resolution_matching_mode": profile.resolution_matching_mode,
        },
        source_resolution_metadata={"fwhm": source_metadata.get("fwhm")},
        preconvolution_metadata=preconvolution_metadata,
        template_axis_metadata=_template_axis_metadata(source_metadata),
    )
