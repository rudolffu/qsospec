"""Generic first-pass Euclid RGS configuration and portable input validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .config import (
    BalmerPseudoContinuumConfig,
    GalacticExtinctionConfig,
    GlobalContinuumConfig,
    PowerLawConfig,
    UncertaintyConfig,
)

RGS_OUTPUT_BINS = 500
RGS_SOURCE_BINS = 531
RGS_SOURCE_SLICE = slice(11, 511)
RGS_WAVELENGTH_MIN = 12047.400390625
RGS_WAVELENGTH_MAX = 18734.0
SAMPLE_MANIFEST_FIELDS = {
    "sample_id",
    "sample_scope",
    "input_rows",
    "input_sha256",
    "schema_sha256",
    "ordered_object_id_sha256",
    "ordered_object_key_sha256",
    "wavelength_grid_sha256",
    "flux_scale",
    "flux_density_unit",
    "wavelength_frame",
    "flux_frame",
    "galactic_extinction_corrected",
    "shard_id",
    "n_shards",
}


def scientific_configuration(dustmaps_data_dir: str) -> dict[str, Any]:
    """Return the immutable no-host/no-Balmer-pseudocontinuum first pass."""

    global_config = GlobalContinuumConfig(
        power_law=PowerLawConfig(mode="single"),
        balmer_pseudocontinuum=BalmerPseudoContinuumConfig(
            enabled=False,
            fit_fwhm=False,
            sync_with_hbeta="never",
            sync_with_hgamma="never",
        ),
    )
    extinction = GalacticExtinctionConfig(
        enabled=True,
        map_name="planck",
        law="f99",
        wavelength_out_of_range="raise",
        rv=3.1,
        dustmaps_data_dir=str(Path(dustmaps_data_dir).expanduser()),
    )
    uncertainty = UncertaintyConfig(
        covariance=True,
        monte_carlo_trials=0,
        random_seed=12345,
        refit_host_in_mc=False,
    )
    return {
        "global_config": global_config,
        "galactic_extinction_config": extinction,
        "uncertainty_config": uncertainty,
        "run_host_decomp": False,
        "complexes": None,
    }


def config_manifest(dustmaps_data_dir: str) -> dict[str, Any]:
    config = scientific_configuration(dustmaps_data_dir)
    return {
        "global_config": asdict(config["global_config"]),
        "galactic_extinction_config": asdict(config["galactic_extinction_config"]),
        "uncertainty_config": asdict(config["uncertainty_config"]),
        "run_host_decomp": False,
        "complexes": None,
    }


def normalize_rgs_grid(values: Any) -> np.ndarray:
    """Validate a 500-bin grid or apply only the established 531-bin slice."""

    wavelength = np.asarray(values, dtype=float)
    if wavelength.ndim != 1:
        raise ValueError("Euclid RGS wavelength must be one-dimensional")
    if wavelength.size == RGS_SOURCE_BINS:
        wavelength = wavelength[RGS_SOURCE_SLICE]
    elif wavelength.size != RGS_OUTPUT_BINS:
        raise ValueError(
            f"Euclid RGS rows require {RGS_OUTPUT_BINS} bins, or the exact {RGS_SOURCE_BINS}-bin source contract"
        )
    if wavelength.size != RGS_OUTPUT_BINS or not np.all(np.isfinite(wavelength)):
        raise ValueError("Euclid RGS wavelength grid must contain 500 finite bins")
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Euclid RGS wavelength grid must be strictly increasing")
    if not np.isclose(wavelength[0], RGS_WAVELENGTH_MIN, atol=1e-6, rtol=0):
        raise ValueError(f"Unexpected Euclid RGS grid start: {wavelength[0]}")
    if not np.isclose(wavelength[-1], RGS_WAVELENGTH_MAX, atol=1e-6, rtol=0):
        raise ValueError(f"Unexpected Euclid RGS grid end: {wavelength[-1]}")
    return wavelength


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_sample_manifest(
    input_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate exact portable sample identity before starting a run."""

    input_path = Path(input_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest: Mapping[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = SAMPLE_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise KeyError(f"Sample manifest is missing fields: {sorted(missing)}")
    parquet = pq.ParquetFile(input_path)
    required = {"object_id", "qsospec_object_key", "wavelength", "flux", "ivar", "mask", "redshift"}
    missing_columns = required - set(parquet.schema_arrow.names)
    if missing_columns:
        raise KeyError(f"Euclid RGS input is missing columns: {sorted(missing_columns)}")
    table = pq.read_table(
        input_path,
        columns=["object_id", "qsospec_object_key", "wavelength"],
        use_threads=False,
    )
    ids = [str(value) for value in table["object_id"].to_pylist()]
    keys = [str(value).strip() for value in table["qsospec_object_key"].to_pylist()]
    if any(not value for value in keys) or len(set(keys)) != len(keys):
        raise ValueError("Explicit qsospec_object_key values must be non-empty and unique")
    if len(set(ids)) != len(ids):
        raise ValueError("Euclid RGS input contains duplicate object_id rows")
    if table.num_rows != int(manifest["input_rows"]):
        raise ValueError("Sample input row count differs from manifest")
    grid = normalize_rgs_grid(table["wavelength"][0].as_py())
    for chunk in table["wavelength"].chunks:
        for scalar in chunk:
            if not np.array_equal(normalize_rgs_grid(scalar.as_py()), grid):
                raise ValueError("Euclid RGS input contains inconsistent wavelength grids")
    checks = {
        "input_sha256": _sha256_file(input_path),
        "schema_sha256": hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest(),
        "ordered_object_id_sha256": _sha256_lines(ids),
        "ordered_object_key_sha256": _sha256_lines(keys),
        "wavelength_grid_sha256": hashlib.sha256(np.asarray(grid, dtype="<f8").tobytes()).hexdigest(),
    }
    mismatch = {
        name: {"manifest": manifest[name], "actual": value} for name, value in checks.items() if manifest[name] != value
    }
    if mismatch:
        raise ValueError(f"Euclid RGS sample-manifest mismatch: {mismatch}")
    return {
        "sample_id": str(manifest["sample_id"]),
        "sample_scope": str(manifest["sample_scope"]),
        "input_rows": int(table.num_rows),
        "manifest_sha256": _sha256_file(manifest_path),
        **checks,
    }
