#!/usr/bin/env python3
"""Measure uniform broad+narrow RGS line decompositions from a qsospec archive.

The command is measurement-first: it writes continuous flux, width, broad
fraction, and residual diagnostics and never assigns a physical class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qsospec import (
    BROAD_NARROW_COMPLEX_ORDER,
    BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
    BroadNarrowMeasurementConfig,
    LineSpreadFunctionConfig,
    Spectrum,
    load_model_by_key,
    measure_broad_narrow_complex,
    open_run,
    signed_to_uint64_string,
)
from qsospec.euclid_rgs import validate_sample_manifest

EXPECTED_GOLD_ROWS = 8_530
PRODUCT_NAME = "broad_narrow_r480_v1"
_WORKER_STORE = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _default_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--measurement-directory", type=Path)
    parser.add_argument(
        "--sample-manifest", type=Path,
        help="Portable sample manifest; enables generic per-shard production mode.",
    )
    parser.add_argument(
        "--product-name",
        help="Versioned product name required with --sample-manifest.",
    )
    parser.add_argument(
        "--complexes", nargs="+", choices=BROAD_NARROW_COMPLEX_ORDER,
        default=list(BROAD_NARROW_COMPLEX_ORDER),
    )
    parser.add_argument("--resolving-power", type=float, default=480.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--smoke-count", type=int, default=64)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--redshift-override-table", type=Path,
        help=(
            "Explicit reviewed override table. Only accepted_vi/accepted_manual "
            "rows are refitted, and --measurement-directory is required."
        ),
    )
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-covariance", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--allow-active-source-run", action="store_true",
        help="Development only: allow an active/partial source archive.",
    )
    parser.add_argument(
        "--allow-noncanonical-gold-count", action="store_true",
        help="Synthetic/integration tests only.",
    )
    return parser.parse_args()


def _read_input(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "object_id" not in frame:
        raise KeyError("Input is missing object_id")
    if frame["object_id"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate object_id values")
    frame = frame.copy()
    frame["object_id"] = frame["object_id"].map(lambda value: int(str(value))).astype("int64")
    if "membership_order" not in frame:
        frame["membership_order"] = np.arange(len(frame), dtype=np.int64)
    if frame["membership_order"].duplicated().any():
        raise ValueError("membership_order must be unique")
    return frame.sort_values("membership_order", kind="stable").reset_index(drop=True)


def _read_selection(path: Path) -> set[str]:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "object_id" not in frame:
        raise KeyError("Selection is missing object_id")
    keys = frame["object_id"].map(lambda value: str(int(str(value))))
    if keys.duplicated().any():
        raise ValueError("Selection contains duplicate object IDs")
    return set(keys)


def _read_redshift_overrides(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {
        "object_id", "z_original", "z_revised", "redshift_revision_reason",
        "redshift_revision_status", "redshift_revision_source", "alias_rule_version",
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"Redshift override table is missing {sorted(missing)}")
    frame = frame.copy()
    frame["object_id"] = frame["object_id"].map(lambda value: int(str(value))).astype("int64")
    if frame["object_id"].duplicated().any():
        raise ValueError("Redshift override table contains duplicate object IDs")
    accepted = frame["redshift_revision_status"].isin(["accepted_vi", "accepted_manual"])
    frame = frame.loc[accepted].copy()
    revised = pd.to_numeric(frame["z_revised"], errors="coerce")
    original = pd.to_numeric(frame["z_original"], errors="coerce")
    if not (np.isfinite(revised) & (revised > -1.0) & np.isfinite(original)).all():
        raise ValueError("Accepted redshift overrides require finite original/revised redshifts")
    if not len(frame):
        raise ValueError("Redshift override table contains no accepted rows")
    return frame


def _validate_source_configuration(manifest: dict[str, Any]) -> None:
    """Require the archived no-host/no-Balmer-continuum scientific contract."""

    configuration = dict(manifest.get("configuration") or {})
    if configuration.get("run_host_decomp", False):
        raise ValueError("Source run enabled host decomposition and is incompatible")
    global_config = dict(configuration.get("global_config") or {})
    balmer = dict(global_config.get("balmer_pseudocontinuum") or {})
    if balmer.get("enabled", True):
        raise ValueError(
            "Source run does not record the required disabled Balmer pseudo-continuum"
        )


def _evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values(["_z", "object_id"], kind="stable")
    if len(ordered) <= count:
        return ordered.copy()
    return ordered.iloc[np.linspace(0, len(ordered) - 1, count, dtype=int)].copy()


def select_smoke_objects(
    frame: pd.DataFrame,
    available_ids: set[str],
    complexes: tuple[str, ...],
    count: int,
) -> pd.DataFrame:
    """Choose <=100 deterministic archived objects spanning every complex/redshift."""

    if count > 100:
        raise ValueError("Development smoke selections are capped at 100 objects")
    pool = frame[frame["object_id"].astype(str).isin(available_ids)].copy()
    z_column = "redshift" if "redshift" in pool else "z_final"
    pool["_z"] = pd.to_numeric(pool.get(z_column), errors="coerce")
    observed_lo, observed_hi = 12047.0, 18734.0
    windows = {
        "halpha": (6400.0, 6800.0),
        "hbeta": (4640.0, 5100.0),
        "mgii": (2700.0, 2900.0),
        "hei_pgamma": (10550.0, 11150.0),
    }
    selected = []
    per_complex = max(1, count // len(complexes))
    for name in complexes:
        lo, hi = windows[name]
        z = pool["_z"]
        covered = pool[
            np.isfinite(z)
            & (lo * (1.0 + z) >= observed_lo)
            & (hi * (1.0 + z) <= observed_hi)
        ]
        selected.append(_evenly_spaced(covered, per_complex))
    combined = pd.concat(selected, ignore_index=True).drop_duplicates("object_id")
    if len(combined) < count:
        remainder = pool[~pool["object_id"].isin(combined["object_id"])]
        combined = pd.concat(
            [combined, _evenly_spaced(remainder, count - len(combined))],
            ignore_index=True,
        )
    combined = combined.head(count)
    if len(combined) != count:
        raise ValueError(f"Only {len(combined)} archived objects are available for a {count}-row smoke run")
    order = dict(zip(frame["object_id"].astype(str), frame["membership_order"]))
    combined["_order"] = combined["object_id"].astype(str).map(order)
    return combined.sort_values("_order", kind="stable").drop(columns=["_order", "_z"])


def _initialize_worker(run_directory: str) -> None:
    global _WORKER_STORE
    _WORKER_STORE = open_run(run_directory)


def _status_record(object_id: int, complex_name: str, status: str, message: str) -> dict[str, Any]:
    return {
        "object_id": int(object_id),
        "object_id_uint64": signed_to_uint64_string(object_id),
        "measurement_schema_version": BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
        "complex_name": complex_name,
        "fit_status": status,
        "fit_success": False,
        "fit_message": message,
    }


def _reconcile_finalized_source_failures(
    selected: pd.DataFrame,
    store: Any,
    *,
    require_expected_keys: bool,
) -> pd.DataFrame:
    """Annotate selected rows that have a recorded, finalized source-fit failure.

    A finalized run may contain both successful object archives and terminal
    failure rows.  ``build_object_index`` intentionally indexes only successful
    archives, so missing keys are acceptable only when the same object is
    represented uniquely in the run's failure table.
    """

    missing = selected["object_key"].isna()
    if not missing.any():
        return selected

    failure_rows = store.read_table(
        "failures",
        columns=["object_id", "object_key", "exception_type", "message"],
    ).to_pylist()
    failures: dict[str, dict[str, Any]] = {}
    for row in failure_rows:
        object_id = str(row["object_id"])
        if object_id in failures:
            raise ValueError(
                "Finalized source run has duplicate/ambiguous failure rows for "
                f"object_id {object_id}"
            )
        failures[object_id] = row

    missing_rows = selected.loc[missing]
    missing_ids = missing_rows["object_id"].astype(str)
    unexplained = [object_id for object_id in missing_ids if object_id not in failures]
    if unexplained:
        raise ValueError(
            "Finalized source run has selected objects that are neither archived "
            "nor recorded as failures; examples: "
            f"{unexplained[:5]}"
        )

    if require_expected_keys:
        if "qsospec_object_key" not in selected:
            raise KeyError("Generic sample input is missing qsospec_object_key")
        mismatched = []
        for row in missing_rows.itertuples(index=False):
            failure_key = str(failures[str(row.object_id)]["object_key"])
            if failure_key != str(row.qsospec_object_key):
                mismatched.append(str(row.object_id))
        if mismatched:
            raise ValueError(
                "Source failure object keys differ from sample manifest input: "
                f"{mismatched[:5]}"
            )

    result = selected.copy()
    failure_by_id = {
        object_id: {
            "source_failure_object_key": str(row["object_key"]),
            "source_failure_exception_type": str(row["exception_type"]),
            "source_failure_message": str(row["message"]),
        }
        for object_id, row in failures.items()
    }
    for column in (
        "source_failure_object_key",
        "source_failure_exception_type",
        "source_failure_message",
    ):
        result[column] = result["object_id"].astype(str).map(
            lambda object_id, name=column: failure_by_id.get(object_id, {}).get(name)
        )
    return result


def _fit_one(
    store: Any,
    payload: dict[str, Any],
    complexes: tuple[str, ...],
    config: BroadNarrowMeasurementConfig,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    object_id = int(payload["object_id"])
    raw_object_key = payload.get("object_key")
    object_key = None if pd.isna(raw_object_key) else str(raw_object_key)
    context = {
        "resolving_power": config.lsf.resolving_power,
        "instrumental_fwhm_kms": config.instrumental_fwhm_kms,
        "local_continuum_mode": config.local_continuum_mode,
        "narrow_fwhm_observed_lower_bound_kms": config.observed_bounds(broad=False)[0],
        "narrow_fwhm_observed_upper_bound_kms": config.observed_bounds(broad=False)[1],
    }
    row_provenance = {
        **provenance,
        "object_id": object_id,
        "object_id_uint64": signed_to_uint64_string(object_id),
        "object_key": object_key,
        "membership_order": int(payload["membership_order"]),
        "adopted_redshift": payload.get(
            "z_revised", payload.get("redshift", payload.get("z_final"))
        ),
        "redshift_source": (
            "reviewed_redshift_override"
            if payload.get("z_revised") is not None and np.isfinite(float(payload["z_revised"]))
            else payload.get("qsospec_redshift_source")
        ),
        "source_failure_object_key": payload.get("source_failure_object_key"),
        "source_failure_exception_type": payload.get("source_failure_exception_type"),
        "source_failure_message": payload.get("source_failure_message"),
    }
    if not object_key:
        source_error = payload.get("source_failure_message")
        source_exception = payload.get("source_failure_exception_type")
        message = (
            f"Source fit failed ({source_exception}): {source_error}"
            if source_error
            else "Object is not yet archived."
        )
        return [
            {
                **_status_record(object_id, name, "not_available", message),
                **context,
                "broad_fwhm_observed_lower_bound_kms": config.observed_bounds(
                    broad=True, hei=name == "hei_pgamma"
                )[0],
                "broad_fwhm_observed_upper_bound_kms": config.observed_bounds(
                    broad=True, hei=name == "hei_pgamma"
                )[1],
                **row_provenance,
            }
            for name in complexes
        ]
    load_start = time.perf_counter()
    try:
        loaded = load_model_by_key(store, str(object_key))
    except Exception as error:  # noqa: BLE001 - malformed source becomes a row product
        message = "".join(traceback.format_exception_only(type(error), error)).strip()
        return [
            {
                **_status_record(object_id, name, "fit_failed", message),
                **context,
                "broad_fwhm_observed_lower_bound_kms": config.observed_bounds(
                    broad=True, hei=name == "hei_pgamma"
                )[0],
                "broad_fwhm_observed_upper_bound_kms": config.observed_bounds(
                    broad=True, hei=name == "hei_pgamma"
                )[1],
                **row_provenance,
            }
            for name in complexes
        ]
    load_seconds = time.perf_counter() - load_start
    revised = payload.get("z_revised")
    if revised is not None and np.isfinite(float(revised)):
        revised = float(revised)
        reframed = Spectrum.from_arrays(
            loaded.spectrum.wave_obs,
            loaded.spectrum.flux,
            err=loaded.spectrum.err,
            z=revised,
            wave_frame="observed",
            mask=loaded.spectrum.mask,
            metadata=loaded.spectrum.metadata,
        )
        loaded.spectrum = reframed
        loaded.continuum = replace(
            loaded.continuum,
            wave_rest=reframed.wave_rest.copy(),
            metadata={
                **loaded.continuum.metadata,
                "redshift_override_applied": True,
                "z_original": float(payload["z_original"]),
                "z_revised": revised,
                "redshift_revision_status": str(payload["redshift_revision_status"]),
                "redshift_revision_source": str(payload["redshift_revision_source"]),
            },
        )
    records = []
    for complex_name in complexes:
        fit_start = time.perf_counter()
        try:
            record, _ = measure_broad_narrow_complex(
                loaded.spectrum, loaded.continuum, complex_name, config
            )
        except Exception as error:  # noqa: BLE001 - one local fit cannot abort the catalogue
            message = "".join(traceback.format_exception_only(type(error), error)).strip()
            record = _status_record(object_id, complex_name, "fit_failed", message)
        record["local_fit_seconds"] = time.perf_counter() - fit_start
        records.append(record)
    fit_seconds = sum(float(record["local_fit_seconds"]) for record in records)
    for record in records:
        record.setdefault("measurement_schema_version", BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION)
        for name, value in context.items():
            record.setdefault(name, value)
        record.setdefault(
            "broad_fwhm_observed_lower_bound_kms",
            config.observed_bounds(broad=True, hei=record["complex_name"] == "hei_pgamma")[0],
        )
        record.setdefault(
            "broad_fwhm_observed_upper_bound_kms",
            config.observed_bounds(broad=True, hei=record["complex_name"] == "hei_pgamma")[1],
        )
        record.update(row_provenance)
        record.update({
            "archived_object_load_seconds": load_seconds,
            "local_fit_total_seconds": fit_seconds,
        })
        for name in (
            "z_original", "z_revised", "redshift_revision_reason",
            "redshift_revision_status", "redshift_revision_source",
            "alias_rule_version", "review_timestamp",
        ):
            if name in payload:
                record[name] = payload[name]
    return records


def _worker(payload: tuple[dict[str, Any], tuple[str, ...], dict[str, Any], dict[str, Any]]):
    row, complexes, config_payload, provenance = payload
    config = BroadNarrowMeasurementConfig(
        lsf=LineSpreadFunctionConfig(resolving_power=float(config_payload["resolving_power"])),
        compute_covariance=bool(config_payload["compute_covariance"]),
    )
    return _fit_one(_WORKER_STORE, row, complexes, config, provenance)


def _expected_keys(rows: pd.DataFrame, complexes: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(str(value), name) for value in rows["object_id"] for name in complexes]


def _part_matches(path: Path, expected: list[tuple[str, str]], fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(
            path,
            columns=["object_id", "complex_name", "part_fingerprint", "measurement_schema_version"],
        )
        found = list(zip(frame["object_id"].astype(str), frame["complex_name"].astype(str)))
        return (
            found == expected
            and frame["part_fingerprint"].eq(fingerprint).all()
            and frame["measurement_schema_version"].eq(BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION).all()
        )
    except (OSError, KeyError, ValueError):
        return False


def _write_atomic(frame: pd.DataFrame, path: Path) -> float:
    start = time.perf_counter()
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return time.perf_counter() - start


def _make_qa_selection(long: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    failed = long[long["fit_status"].isin(
        ["fit_failed", "continuum_unavailable", "not_available"]
    )].head(25).copy()
    failed["qa_reason"] = "fit_failure_or_unavailable"
    selected.append(failed)
    complete = long[long["fit_status"].eq("complete")].copy()
    if "residual_abs_max_sigma" in complete:
        sample = complete.sort_values(
            "residual_abs_max_sigma", ascending=False, kind="stable"
        ).head(25).copy()
        sample["qa_reason"] = "high_residual"
        selected.append(sample)
    if "active_bound_parameters" in complete:
        bound = complete[complete["active_bound_parameters"].map(
            lambda value: value is not None and len(value) > 0
        )].head(25).copy()
        bound["qa_reason"] = "active_kinematic_bound"
        selected.append(bound)
    for target, reason in (
        (0.05, "broad_fraction_near_zero"),
        (0.50, "broad_fraction_intermediate"),
        (0.95, "broad_fraction_near_one"),
    ):
        values = pd.to_numeric(complete.get("broad_fraction"), errors="coerce")
        sample = complete.assign(_distance=(values - target).abs()).sort_values(
            "_distance", kind="stable"
        ).head(10).drop(columns="_distance")
        sample["qa_reason"] = reason
        selected.append(sample)
    widths = pd.to_numeric(
        complete.get("total_profile_fwhm_observed_kms"), errors="coerce"
    )
    finite_widths = widths[np.isfinite(widths)]
    if len(finite_widths):
        for quantile, reason in (
            (0.1, "profile_width_low"),
            (0.5, "profile_width_middle"),
            (0.9, "profile_width_high"),
        ):
            target = float(finite_widths.quantile(quantile))
            sample = complete.assign(_distance=(widths - target).abs()).sort_values(
                "_distance", kind="stable"
            ).head(10).drop(columns="_distance")
            sample["qa_reason"] = reason
            selected.append(sample)
    if {"narrow_flux_snr", "broad_flux_snr"}.issubset(complete):
        signal = np.fmax(
            pd.to_numeric(complete["narrow_flux_snr"], errors="coerce"),
            pd.to_numeric(complete["broad_flux_snr"], errors="coerce"),
        )
        controls = complete.assign(_signal=signal)
        redshift = (
            pd.to_numeric(controls["adopted_redshift"], errors="coerce")
            if "adopted_redshift" in controls
            else pd.Series(np.nan, index=controls.index)
        )
        finite = controls[np.isfinite(redshift)].copy()
        if len(finite) >= 4:
            finite["_redshift_bin"] = pd.qcut(
                pd.to_numeric(finite["adopted_redshift"], errors="coerce"),
                4, labels=False, duplicates="drop",
            )
            controls = (
                finite.sort_values(
                    ["_redshift_bin", "_signal", "membership_order"],
                    ascending=[True, False, True], kind="stable",
                )
                .groupby("_redshift_bin", sort=True, group_keys=False)
                .head(5)
                .drop(columns="_redshift_bin")
            )
        else:
            controls = controls.sort_values(
                ["_signal", "membership_order"], ascending=[False, True], kind="stable"
            ).head(20)
        controls = controls.drop(columns="_signal")
        controls["qa_reason"] = "high_snr_control"
        selected.append(controls)
    if complete["complex_name"].nunique() > 1:
        pivot = complete.pivot(index="object_id", columns="complex_name", values="broad_fraction")
        disagreement = (pivot.max(axis=1, skipna=True) - pivot.min(axis=1, skipna=True)).sort_values(
            ascending=False
        )
        ids = disagreement[disagreement.notna()].head(20).index
        cross_line = complete[complete["object_id"].isin(ids)].copy()
        cross_line["qa_reason"] = "cross_line_disagreement"
        selected.append(cross_line)
    if not selected:
        return pd.DataFrame(columns=["object_id", "complex_name", "qa_reason"])
    return pd.concat(selected, ignore_index=True).drop_duplicates(["object_id", "complex_name"])


def _finalize(
    directory: Path,
    selected: pd.DataFrame,
    input_frame: pd.DataFrame,
    complexes: tuple[str, ...],
    config_payload: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    files = sorted((directory / "parts").glob("part-*.parquet"))
    if not files:
        raise ValueError("No part files are available to finalize")
    long = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    expected = _expected_keys(selected, complexes)
    found = list(zip(long["object_id"].astype(str), long["complex_name"].astype(str)))
    if found != expected:
        raise ValueError("Part rows do not match the exact selected object/complex order")
    if long.duplicated(["object_id", "complex_name"]).any():
        raise ValueError("Duplicate object/complex rows in measurement parts")
    long.to_parquet(directory / "broad_narrow_line_measurements.parquet", index=False)

    audit = input_frame.copy()
    audit["_id"] = audit["object_id"].astype(str)
    audit = audit.drop(columns=[
        "object_id",
        *[name for name in audit if name in long and name not in {"object_id", "_id"}],
    ])
    ledger = long.assign(_id=long["object_id"].astype(str)).merge(
        audit, on="_id", how="left", validate="many_to_one", sort=False,
    ).drop(columns="_id")
    if list(zip(ledger["object_id"].astype(str), ledger["complex_name"])) != expected:
        raise ValueError("Ledger order changed during audit metadata join")
    ledger.to_parquet(directory / "broad_narrow_measurement_ledger.parquet", index=False)

    useful = [
        "fit_status", "broad_fraction", "broad_fraction_error", "narrow_flux",
        "broad_flux", "narrow_flux_snr", "broad_flux_snr",
        "total_profile_fwhm_observed_kms", "narrow_fwhm_observed_kms",
        "broad_fwhm_observed_kms", "reduced_chi2", "residual_abs_max_sigma",
        "hei_broad_fraction", "pagamma_broad_fraction", "joint_broad_fraction",
    ]
    index = selected[["object_id", "membership_order"]].copy()
    wide = index
    for name in complexes:
        subset = long[long["complex_name"].eq(name)].set_index("object_id")
        columns = [column for column in useful if column in subset]
        renamed = subset[columns].rename(columns={column: f"{name}_{column}" for column in columns})
        wide = wide.join(renamed, on="object_id", validate="one_to_one")
    wide.to_parquet(directory / "broad_narrow_measurements_wide.parquet", index=False)
    qa = _make_qa_selection(long)
    qa.to_parquet(directory / "qa_selection.parquet", index=False)

    status_counts = (
        long.groupby(["complex_name", "fit_status"], dropna=False).size()
        .rename("count").reset_index().to_dict("records")
    )
    timing_by_complex = []
    for complex_name, group in long.groupby("complex_name", sort=False):
        values = (
            pd.to_numeric(group["local_fit_seconds"], errors="coerce").dropna()
            if "local_fit_seconds" in group else pd.Series(dtype=float)
        )
        timing_by_complex.append({
            "complex_name": complex_name,
            "count": len(values),
            "sum_seconds": float(values.sum()),
            "median_seconds": float(values.median()) if len(values) else np.nan,
            "p95_seconds": float(values.quantile(0.95)) if len(values) else np.nan,
        })
    load_values = (
        pd.to_numeric(
            long.drop_duplicates("object_id")["archived_object_load_seconds"],
            errors="coerce",
        ).dropna()
        if "archived_object_load_seconds" in long else pd.Series(dtype=float)
    )
    part_write_seconds = []
    for path in sorted((directory / "parts").glob("part-*.timing.json")):
        part_write_seconds.append(float(json.loads(path.read_text())["part_write_seconds"]))
    summary = {
        "measurement_schema_version": BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
        "n_selected_objects": len(selected),
        "n_long_rows": len(long),
        "complexes": list(complexes),
        "status_counts": status_counts,
        "finite_broad_fraction_rows": int(np.isfinite(pd.to_numeric(long.get("broad_fraction"), errors="coerce")).sum()),
        "active_bound_rows": int(long.get(
            "active_bound_parameters", pd.Series(dtype=object)
        ).map(lambda value: value is not None and len(value) > 0).sum()),
        "physical_class_counts": None,
        "timings": {
            "local_fit_by_complex": timing_by_complex,
            "archived_object_load_median_seconds": (
                float(load_values.median()) if len(load_values) else np.nan
            ),
            "archived_object_load_p95_seconds": (
                float(load_values.quantile(0.95)) if len(load_values) else np.nan
            ),
            "part_write_total_seconds": float(sum(part_write_seconds)),
            "part_write_count": len(part_write_seconds),
        },
        "created_at": _now(),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    known_descriptions = {
        "object_id": "Signed int64 Euclid spectrum identifier.",
        "object_id_uint64": "Unsigned uint64 decimal representation of object_id.",
        "object_key": "Immutable source RunStore key used for exact shard lookup.",
        "membership_order": "Zero-based order in the gold input ledger.",
        "fit_status": "Explicit complete/not_covered/continuum_unavailable/fit_failed/not_available status.",
        "narrow_flux": "Integrated narrow flux of the primary permitted line.",
        "broad_flux": "Integrated broad flux of the primary permitted line.",
        "total_flux": "Sum of primary-line narrow and broad integrated flux.",
        "broad_fraction": "Broad/(broad+narrow) integrated flux for the primary permitted line only.",
        "broad_fraction_error": "Full-covariance delta-method uncertainty of broad_fraction.",
        "total_profile_fwhm_observed_kms": "Outer half-maximum span of a dense summed primary-line narrow+broad Gaussian profile.",
        "total_profile_sigma_kms": "Flux-weighted second-moment velocity sigma of the summed primary-line profile.",
        "total_profile_fwhm_ambiguous": "True when the half-maximum set has multiple disjoint intervals.",
        "resolving_power": "Fixed Gaussian constant-R line-spread assumption.",
        "instrumental_fwhm_kms": "c/resolving_power.",
    }
    column_descriptions = {}
    fitted_columns = set(long.columns)
    for column in ledger.columns:
        if column in known_descriptions:
            description = known_descriptions[column]
        elif column.endswith("_intrinsic_approx_kms"):
            description = "Approximate quadrature-deconvolved width using the fixed Gaussian LSF."
        elif column.startswith("component_"):
            description = "Complex-specific fitted component metric; not used in the generic broad-fraction denominator."
        elif column in fitted_columns:
            description = "Broad+narrow spectral-fit measurement, diagnostic, status, or provenance field."
        else:
            description = "Gold-input audit metadata copied after fitting; not used by the spectral model."
        column_descriptions[column] = description
    dictionary = {
        "measurement_schema_version": BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
        "columns": column_descriptions,
        "wildcard_note": "Columns ending _intrinsic_approx_kms are secondary fixed-LSF deconvolutions.",
    }
    (directory / "column_dictionary.json").write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    product_status = "partial_source" if long["fit_status"].eq("not_available").any() else "complete"
    manifest.update({
        "status": product_status,
        "completed_at": _now(),
        "configuration": config_payload,
        "summary": summary,
        "outputs": [
            "broad_narrow_line_measurements.parquet",
            "broad_narrow_measurement_ledger.parquet",
            "broad_narrow_measurements_wide.parquet",
            "qa_selection.parquet", "summary.json", "column_dictionary.json",
            "parts/part-*.parquet", "parts/part-*.timing.json",
        ],
    })
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    requested_product = args.product_name.strip() if args.product_name else PRODUCT_NAME
    root = args.output_root
    if root is None and (
        args.input is None or args.run_directory is None
    ):
        root = _default_root()
    if args.sample_manifest and args.measurement_directory is None and root is None:
        raise ValueError(
            "Generic sample-manifest mode requires --measurement-directory or --output-root"
        )
    input_path = args.input or root / "input/spectra.parquet"
    run_directory = args.run_directory or root / "runs/production_no_balmer_no_host_v1"
    base = args.measurement_directory or root / f"measurements/{requested_product}"
    if args.redshift_override_table and args.measurement_directory is None:
        raise ValueError(
            "--redshift-override-table requires a separate explicit --measurement-directory"
        )
    directory = base / "smoke" if args.mode == "smoke" and args.measurement_directory is None else base
    directory.mkdir(parents=True, exist_ok=True)
    parts = directory / "parts"
    parts.mkdir(exist_ok=True)
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive")
    if args.smoke_count > 100:
        raise ValueError("Development smoke runs are capped at 100 objects")

    input_frame = _read_input(input_path)
    sample_validation = None
    product_name = PRODUCT_NAME
    if args.sample_manifest:
        if not args.product_name or not args.product_name.strip():
            raise ValueError("--product-name is required with --sample-manifest")
        if args.allow_noncanonical_gold_count:
            raise ValueError("--allow-noncanonical-gold-count is not used in sample-manifest mode")
        sample_validation = validate_sample_manifest(input_path, args.sample_manifest)
        product_name = args.product_name.strip()
    elif args.product_name:
        raise ValueError("--product-name requires --sample-manifest")
    elif len(input_frame) != EXPECTED_GOLD_ROWS and not args.allow_noncanonical_gold_count:
        raise ValueError(f"Expected {EXPECTED_GOLD_ROWS} gold rows; found {len(input_frame)}")
    store = open_run(str(run_directory))
    _validate_source_configuration(store.manifest)
    source_status = str(store.manifest.get("status", "unknown"))
    if source_status != "complete" and not args.allow_active_source_run:
        raise ValueError("Source run is not finalized; pass --allow-active-source-run only for bounded development")
    object_index = store.build_object_index()
    input_frame["object_key"] = input_frame["object_id"].astype(str).map(object_index)
    if sample_validation is not None:
        if "qsospec_object_key" not in input_frame:
            raise KeyError("Generic sample input is missing qsospec_object_key")
        expected_keys = input_frame["qsospec_object_key"].astype(str)
        mismatched = input_frame["object_key"].notna() & input_frame["object_key"].astype(str).ne(expected_keys)
        if mismatched.any():
            examples = input_frame.loc[mismatched, "object_id"].head(5).tolist()
            raise ValueError(f"Source run object keys differ from sample manifest input: {examples}")
        source_sample = store.manifest.get("sample_manifest") or {}
        for name in ("ordered_object_id_sha256", "ordered_object_key_sha256", "input_sha256"):
            if source_sample.get(name) != sample_validation.get(name):
                raise ValueError(f"Source run provenance does not match sample manifest field {name}")

    overrides = None
    if args.redshift_override_table:
        overrides = _read_redshift_overrides(args.redshift_override_table)
        audit_columns = [column for column in overrides if column != "object_id"]
        input_frame = input_frame.merge(
            overrides[["object_id", *audit_columns]], on="object_id", how="left",
            validate="one_to_one", sort=False,
        )

    complexes = tuple(args.complexes)
    if len(set(complexes)) != len(complexes):
        raise ValueError("--complexes contains duplicates")
    if args.mode == "production" and not np.isclose(
        args.resolving_power, 480.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("The broad_narrow_r480_v1 production product is locked to R=480")
    if overrides is not None:
        accepted_ids = set(overrides["object_id"].astype(str))
        if args.selection:
            requested_ids = _read_selection(args.selection)
            unaccepted = requested_ids - accepted_ids
            if unaccepted:
                raise ValueError(
                    "Selection contains objects without accepted redshift overrides: "
                    f"{sorted(unaccepted)[:5]}"
                )
            accepted_ids = requested_ids
        selected = input_frame[input_frame["object_id"].astype(str).isin(accepted_ids)].copy()
    elif args.selection:
        selected_ids = _read_selection(args.selection)
        missing = selected_ids - set(input_frame["object_id"].astype(str))
        if missing:
            raise ValueError(f"Selection IDs absent from gold input: {sorted(missing)[:5]}")
        selected = input_frame[input_frame["object_id"].astype(str).isin(selected_ids)].copy()
    elif args.mode == "smoke":
        selected = select_smoke_objects(
            input_frame, set(object_index), complexes, args.smoke_count
        )
    else:
        selected = input_frame.copy()
    selected = selected.sort_values("membership_order", kind="stable").reset_index(drop=True)
    if args.mode == "production" and source_status == "complete":
        selected = _reconcile_finalized_source_failures(
            selected,
            store,
            require_expected_keys=sample_validation is not None,
        )

    config = BroadNarrowMeasurementConfig(
        lsf=LineSpreadFunctionConfig(resolving_power=args.resolving_power),
        compute_covariance=not args.no_covariance,
    )
    measurement_commit = _git_commit()
    measurement_module = (
        Path(__file__).resolve().parents[1]
        / "src/qsospec/broad_narrow_measurements.py"
    )
    config_payload = {
        "measurement_schema_version": BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
        "mode": args.mode,
        "complexes": complexes,
        "resolving_power": args.resolving_power,
        "compute_covariance": not args.no_covariance,
        "chunk_size": args.chunk_size,
        "source_run_id": store.manifest.get("run_id"),
        "source_configuration_hash": store.manifest.get("configuration_hash"),
        "source_manifest_sha256": _sha256(run_directory / "manifest.json"),
        "input_sha256": _sha256(input_path),
        "measurement_code_git_commit": measurement_commit,
        "measurement_script_sha256": _sha256(Path(__file__).resolve()),
        "measurement_module_sha256": _sha256(measurement_module),
        "redshift_override_table": (
            str(args.redshift_override_table.resolve()) if args.redshift_override_table else None
        ),
        "redshift_override_sha256": (
            _sha256(args.redshift_override_table) if args.redshift_override_table else None
        ),
        "sample_manifest": str(args.sample_manifest.resolve()) if args.sample_manifest else None,
        "sample_manifest_sha256": _sha256(args.sample_manifest) if args.sample_manifest else None,
        "sample_validation": sample_validation,
    }
    config_fingerprint = _json_hash({
        key: value for key, value in config_payload.items()
        if key != "source_manifest_sha256"
    })
    config_path = directory / "measurement_config.json"
    if config_path.exists() and json.loads(config_path.read_text()).get("fingerprint") != config_fingerprint:
        if not args.force:
            raise ValueError("Existing measurement directory has an incompatible configuration")
        for path in parts.glob("part-*.parquet"):
            path.unlink()
    config_path.write_text(json.dumps({**config_payload, "fingerprint": config_fingerprint}, indent=2), encoding="utf-8")
    provenance = {
        "source_run_id": store.manifest.get("run_id"),
        "source_configuration_hash": store.manifest.get("configuration_hash"),
        "source_run_git_commit": store.manifest.get("git_commit"),
        "measurement_code_git_commit": measurement_commit,
    }
    manifest = {
        "product": product_name,
        "measurement_schema_version": BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
        "status": "active",
        "created_at": _now(),
        "invocation": " ".join(os.sys.argv),
        "input": str(input_path.resolve()),
        "run_directory": str(run_directory.resolve()),
        "source_run_status": source_status,
        "configuration_fingerprint": config_fingerprint,
        "n_input_objects": len(input_frame),
        "n_selected_objects": len(selected),
        "no_physical_classification": True,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.finalize_only:
        progress = None
        if not args.no_progress:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(selected), desc="broad+narrow", unit="object", dynamic_ncols=True)
        executor = (
            ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_initialize_worker,
                initargs=(str(run_directory),),
            )
            if args.workers > 1 else None
        )
        try:
            completed_chunks = 0
            for start in range(0, len(selected), args.chunk_size):
                if args.max_chunks is not None and completed_chunks >= args.max_chunks:
                    break
                stop = min(start + args.chunk_size, len(selected))
                rows = selected.iloc[start:stop].copy()
                expected = _expected_keys(rows, complexes)
                availability = list(zip(rows["object_id"].astype(str), rows["object_key"].fillna("not_available")))
                part_fingerprint = _json_hash({
                    "config": config_fingerprint, "expected": expected, "availability": availability,
                })
                part_path = parts / f"part-{start:06d}-{stop - 1:06d}.parquet"
                if not args.force and _part_matches(part_path, expected, part_fingerprint):
                    completed_chunks += 1
                    if progress is not None:
                        progress.update(len(rows))
                    continue
                projected = rows.to_dict("records")
                if args.workers == 1:
                    nested = [_fit_one(store, row, complexes, config, provenance) for row in projected]
                else:
                    payloads = [(row, complexes, config_payload, provenance) for row in projected]
                    nested = list(executor.map(_worker, payloads))
                records = [record for group in nested for record in group]
                frame = pd.DataFrame(records)
                frame["part_fingerprint"] = part_fingerprint
                write_seconds = _write_atomic(frame, part_path)
                part_path.with_suffix(".timing.json").write_text(
                    json.dumps({
                        "part": part_path.name,
                        "n_objects": len(rows),
                        "n_rows": len(frame),
                        "part_write_seconds": write_seconds,
                    }, indent=2),
                    encoding="utf-8",
                )
                completed_chunks += 1
                if progress is not None:
                    progress.update(len(rows))
        finally:
            if progress is not None:
                progress.close()
            if executor is not None:
                executor.shutdown(wait=True)

    expected_part_count = int(np.ceil(len(selected) / args.chunk_size))
    actual_part_count = len(list(parts.glob("part-*.parquet")))
    if actual_part_count != expected_part_count:
        manifest.update({
            "status": "partial", "completed_parts": actual_part_count,
            "expected_parts": expected_part_count,
        })
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return
    summary = _finalize(directory, selected, input_frame, complexes, config_payload, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
