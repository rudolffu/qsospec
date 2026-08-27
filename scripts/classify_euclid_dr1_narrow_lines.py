"""Classify provisional Euclid DR1 RGS narrow-line evidence.

The command compares only N0 and B1 for H-alpha and for the dedicated local
He I 10833 + Pa-gamma complex.  It writes a complete 8,530-object evidence
ledger in gold-membership order, but never assigns a physical class.  VI and
PCF metadata are copied only after the fit-derived decision is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep numerical backends single-threaded so ``--workers`` controls local CPU
# use during bounded development runs.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np
import pandas as pd

from qsospec import (
    HalphaModelSelectionConfig,
    HeIPagammaModelSelectionConfig,
    LineSpreadFunctionConfig,
    fit_halpha_model_grid,
    fit_hei_pgamma_model_pair,
    load_model,
    open_run,
)
from qsospec.narrow_line_evidence import (
    BROAD_DELTA_BIC_THRESHOLD,
    BROAD_SNR_THRESHOLD,
    NARROW_SNR_THRESHOLD,
    PURITY_TARGET,
)

RUN_NAME = "production_no_balmer_no_host_v1"
CLASSIFICATION_NAME = "narrow_line_r480_v2"
EXPECTED_GOLD_ROWS = 8_530
CONSERVATIVE_RESOLVING_POWER = 480.0
SCHEMA_VERSION = "narrow_line_fixed_r480_v3.1-development"
COMPLEXES = ("halpha", "hei_pgamma")
FIT_COLUMNS = (
    "object_id",
    "complex_name",
    "fit_status",
    "evidence_status",
    "provisional_narrow_evidence",
    "broad_evidence",
    "narrow_line_snr",
    "narrow_width_secure_below_boundary",
    "narrow_kinematic_bound_hit",
    "delta_bic_broad",
    "maximum_broad_line_snr",
)
AUDIT_COLUMNS = (
    "membership_order",
    "selection_tier",
    "field",
    "redshift",
    "qsospec_redshift_source",
    "med_snr",
    "n_invalid",
    "p_uncertain",
    "flag_uncertain",
    "class_final",
    "qsospec_vi_class",
    "vi_review_group",
    "catalog_prediction_source",
    "pcf_template_best",
)
_WORKER_STORE = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        help=(
            "Optional Parquet/CSV table containing exact object_id values for "
            "a bounded smoke run."
        ),
    )
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--classification-directory", type=Path)
    parser.add_argument(
        "--complexes", nargs="+", choices=COMPLEXES, default=list(COMPLEXES)
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "production"),
        required=True,
        help="Require an explicit choice so development cannot start all archived fits accidentally.",
    )
    parser.add_argument(
        "--resolving-power",
        type=float,
        default=CONSERVATIVE_RESOLVING_POWER,
        help=(
            "One constant resolving power for every object. Production is "
            "locked to conservative R=480; morphology never defines the LSF."
        ),
    )
    parser.add_argument("--smoke-count", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-covariance", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--allow-noncanonical-gold-count",
        action="store_true",
        help="Permit a non-8,530-row input for synthetic integration tests only.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _decode_key_values(values: Any) -> dict[str, object]:
    output: dict[str, object] = {}
    if values is None:
        return output
    for item in values:
        key = str(item["key"])
        value = item["value"]
        try:
            output[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            output[key] = value
    return output


def _as_signed_int64(value: object) -> int:
    parsed = int(str(value))
    if not -(2**63) <= parsed < 2**63:
        raise ValueError(f"object_id is outside signed int64: {value}")
    return parsed


def _as_uint64_decimal(signed: int) -> str:
    return str(signed if signed >= 0 else signed + 2**64)


def _validate_resolution_policy(mode: str, resolving_power: float) -> None:
    if mode == "production" and not np.isclose(
        resolving_power,
        CONSERVATIVE_RESOLVING_POWER,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Production narrow-line classification is locked to the fixed "
            f"conservative R={CONSERVATIVE_RESOLVING_POWER:g} LSF; do not "
            "derive or pass an object-specific morphology-based resolution."
        )


def _read_selection(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        selection = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        selection = pd.read_csv(path)
    else:
        raise ValueError("--selection must be a Parquet or CSV table")
    if "object_id" not in selection.columns:
        raise KeyError("--selection table is missing object_id")
    selection = selection[["object_id"]].copy()
    selection["_key"] = selection["object_id"].map(
        lambda value: str(_as_signed_int64(value))
    )
    if selection["_key"].duplicated().any():
        raise ValueError("--selection contains duplicate object IDs")
    return selection


def _status_record(
    object_id: object,
    complex_name: str,
    fit_status: str,
    *,
    error: str | None = None,
) -> dict[str, object]:
    signed = _as_signed_int64(object_id)
    if fit_status not in {"fit_failed", "not_covered", "not_available"}:
        raise ValueError(f"Unsupported status-only record: {fit_status}")
    return {
        "object_id": signed,
        "object_id_uint64": _as_uint64_decimal(signed),
        "classification_schema_version": SCHEMA_VERSION,
        "complex_name": complex_name,
        "fit_status": fit_status,
        "evidence_status": fit_status,
        "n0_success": False,
        "b1_success": False,
        "provisional_narrow_evidence": False,
        "broad_evidence": False,
        "physical_class": None,
        "promotion_status": "not_calibrated",
        "error": error,
    }


def _production_statuses(row: pd.Series) -> dict[str, object]:
    return _decode_key_values(row.get("complex_statuses"))


def _fit_one_complex(
    loaded: Any,
    row: pd.Series,
    complex_name: str,
    lsf: LineSpreadFunctionConfig,
    *,
    compute_covariance: bool,
) -> dict[str, object]:
    signed = _as_signed_int64(row["object_id"])
    if complex_name == "halpha":
        production_status = _production_statuses(row).get("halpha_nii_sii")
        if production_status == "failed":
            return _status_record(
                signed,
                complex_name,
                "fit_failed",
                error="The archived production H-alpha complex failed; not retried.",
            )
        if production_status != "fit":
            return _status_record(signed, complex_name, "not_covered")
        result = fit_halpha_model_grid(
            loaded.spectrum,
            loaded.continuum,
            selection_config=HalphaModelSelectionConfig(lsf=lsf),
            compute_covariance=compute_covariance,
        )
        record = result.to_record(signed)
    elif complex_name == "hei_pgamma":
        pair = fit_hei_pgamma_model_pair(
            loaded.spectrum,
            loaded.continuum,
            selection_config=HeIPagammaModelSelectionConfig(lsf=lsf),
            compute_covariance=compute_covariance,
        )
        if pair is None:
            return _status_record(signed, complex_name, "not_covered")
        record = pair.to_record(signed)
    else:
        raise ValueError(f"Unknown complex: {complex_name}")
    record["object_id"] = signed
    record["object_id_uint64"] = _as_uint64_decimal(signed)
    record["classification_schema_version"] = SCHEMA_VERSION
    record["error"] = None
    return record


def _fit_object(
    store: Any,
    row: pd.Series,
    complexes: Iterable[str],
    lsf: LineSpreadFunctionConfig,
    *,
    compute_covariance: bool,
) -> list[dict[str, object]]:
    signed = _as_signed_int64(row["object_id"])
    try:
        loaded = load_model(store, str(signed))
    except Exception as error:  # noqa: BLE001 - one bad archive row must be recorded
        message = "".join(
            traceback.format_exception_only(type(error), error)
        ).strip()
        return [
            _status_record(signed, name, "fit_failed", error=message)
            for name in complexes
        ]
    records: list[dict[str, object]] = []
    for name in complexes:
        try:
            records.append(
                _fit_one_complex(
                    loaded,
                    row,
                    name,
                    lsf,
                    compute_covariance=compute_covariance,
                )
            )
        except Exception as error:  # noqa: BLE001 - fit failures are row products
            records.append(
                _status_record(
                    signed,
                    name,
                    "fit_failed",
                    error="".join(
                        traceback.format_exception_only(type(error), error)
                    ).strip(),
                )
            )
    return records


def _initialize_worker(run_directory: str) -> None:
    global _WORKER_STORE
    _WORKER_STORE = open_run(run_directory)


def _worker_fit(payload: tuple[dict[str, object], tuple[str, ...], float, bool]):
    row_record, complexes, resolving_power, compute_covariance = payload
    return _fit_object(
        _WORKER_STORE,
        pd.Series(row_record),
        complexes,
        LineSpreadFunctionConfig(resolving_power=resolving_power),
        compute_covariance=compute_covariance,
    )


def _fit_rows(
    rows: pd.DataFrame,
    store: Any,
    complexes: tuple[str, ...],
    lsf: LineSpreadFunctionConfig,
    *,
    compute_covariance: bool,
    workers: int,
    executor: ProcessPoolExecutor | None,
) -> list[dict[str, object]]:
    projected = rows[["object_id", "complex_statuses"]].to_dict("records")
    if workers == 1:
        nested = [
            _fit_object(
                store,
                pd.Series(row),
                complexes,
                lsf,
                compute_covariance=compute_covariance,
            )
            for row in projected
        ]
    else:
        if executor is None:
            raise RuntimeError("A process executor is required when workers > 1")
        payloads = [
            (row, complexes, lsf.resolving_power, compute_covariance)
            for row in projected
        ]
        nested = list(executor.map(_worker_fit, payloads))
    return [record for records in nested for record in records]


def _expected_part_keys(
    rows: pd.DataFrame, complexes: tuple[str, ...]
) -> list[tuple[str, str]]:
    return [
        (str(object_id), complex_name)
        for object_id in rows["object_id"]
        for complex_name in complexes
    ]


def _part_matches(path: Path, expected: list[tuple[str, str]]) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(
            path,
            columns=[
                "object_id",
                "complex_name",
                "classification_schema_version",
            ],
        )
        if not frame["classification_schema_version"].eq(SCHEMA_VERSION).all():
            return False
        found = list(
            zip(frame["object_id"].astype(str), frame["complex_name"].astype(str))
        )
        return found == expected
    except (OSError, KeyError, ValueError):
        return False


def _write_part(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _evenly_spaced(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    ordered = frame.sort_values(["redshift", "object_id"], kind="stable")
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return ordered.iloc[indices].copy()


def select_smoke_objects(
    objects: pd.DataFrame,
    input_frame: pd.DataFrame,
    complexes: tuple[str, ...],
    count: int = 64,
) -> pd.DataFrame:
    """Select a deterministic, redshift-spanning smoke set without VI labels."""

    frame = objects.copy()
    input_audit = input_frame[["object_id", "redshift"]].copy()
    input_audit = input_audit.rename(columns={"redshift": "_adopted_redshift"})
    input_audit["_key"] = input_audit["object_id"].astype(str)
    frame["_key"] = frame["object_id"].astype(str)
    frame = frame.merge(
        input_audit[["_key", "_adopted_redshift"]],
        on="_key",
        how="left",
        validate="one_to_one",
    )
    frame["redshift"] = pd.to_numeric(frame["_adopted_redshift"], errors="coerce")
    selected: list[pd.DataFrame] = []
    if "halpha" in complexes:
        halpha = frame[
            frame["complex_statuses"].map(
                lambda value: _decode_key_values(value).get("halpha_nii_sii") == "fit"
            )
        ]
        selected.append(_evenly_spaced(halpha, count // len(complexes)))
    if "hei_pgamma" in complexes:
        z = pd.to_numeric(frame["redshift"], errors="coerce")
        hei = frame[
            np.isfinite(z)
            & (10550.0 * (1.0 + z) >= 12047.0)
            & (11150.0 * (1.0 + z) <= 18734.0)
        ]
        selected.append(_evenly_spaced(hei, count // len(complexes)))
    combined = pd.concat(selected, ignore_index=True).drop_duplicates("_key")
    if len(combined) < count:
        remainder = frame[~frame["_key"].isin(combined["_key"])]
        combined = pd.concat(
            [combined, _evenly_spaced(remainder, count - len(combined))],
            ignore_index=True,
        )
    if len(combined) != count:
        raise ValueError(f"Could not construct the requested {count}-object smoke sample")
    order = {key: index for index, key in enumerate(input_frame["object_id"].astype(str))}
    combined["_order"] = combined["_key"].map(order)
    return combined.sort_values("_order", kind="stable").drop(columns=["_order"])


def combine_object_evidence(records: pd.DataFrame) -> dict[str, object]:
    """Combine complex-level evidence with deterministic broad veto precedence."""

    broad = records.get("broad_evidence", pd.Series(False, index=records.index))
    broad = broad.fillna(False).astype(bool)
    provisional = records.get(
        "provisional_narrow_evidence", pd.Series(False, index=records.index)
    ).fillna(False).astype(bool)
    statuses = set(records["evidence_status"].astype(str))
    passing = records.loc[provisional, "complex_name"].astype(str).tolist()
    broad_complexes = records.loc[broad, "complex_name"].astype(str).tolist()
    if broad.any():
        status = "broad_evidence"
    elif provisional.any():
        status = "provisional_narrow"
    elif "indeterminate" in statuses:
        status = "indeterminate"
    elif "fit_failed" in statuses:
        status = "fit_failed"
    elif "poor_fit" in statuses:
        status = "poor_fit"
    elif statuses and statuses.issubset({"not_covered"}):
        status = "not_covered"
    else:
        status = "not_available"
    return {
        "narrow_line_status": status,
        "provisional_narrow": status == "provisional_narrow",
        "broad_evidence": status == "broad_evidence",
        "passing_complexes": ",".join(passing) if passing else None,
        "broad_evidence_complexes": ",".join(broad_complexes) if broad_complexes else None,
        "physical_class": None,
        "promotion_status": "not_calibrated",
    }


def build_evidence_ledger(
    input_frame: pd.DataFrame,
    complex_records: pd.DataFrame,
    complexes: tuple[str, ...],
) -> pd.DataFrame:
    """Return one row per gold object in exact membership order."""

    rows: list[dict[str, object]] = []
    grouped = {
        str(key): group.copy()
        for key, group in complex_records.groupby(
            complex_records["object_id"].astype(str), sort=False
        )
    }
    for source in input_frame.to_dict("records"):
        signed = _as_signed_int64(source["object_id"])
        available = grouped.get(str(signed))
        if available is None:
            status_rows = pd.DataFrame(
                [
                    _status_record(signed, name, "not_available")
                    for name in complexes
                ]
            )
            evidence = combine_object_evidence(status_rows)
        else:
            evidence = combine_object_evidence(available)
        row = {
            "object_id": signed,
            "object_id_uint64": _as_uint64_decimal(signed),
            "classification_schema_version": SCHEMA_VERSION,
            **evidence,
        }
        for name in AUDIT_COLUMNS:
            if name in source:
                row[name] = source[name]
        rows.append(row)
    ledger = pd.DataFrame(rows)
    if ledger["object_id"].astype(str).tolist() != input_frame["object_id"].astype(str).tolist():
        raise ValueError("Evidence ledger does not preserve exact gold input order")
    return ledger


def diagnostic_threshold_sweep(records: pd.DataFrame) -> pd.DataFrame:
    """Sweep nearby decision thresholds without changing the locked rule."""

    fitted = records[records["fit_status"].eq("complete")].copy()
    rows: list[dict[str, object]] = []
    for snr_threshold in (3.0, 4.0, 5.0, 6.0, 7.0):
        for bic_threshold in (6.0, 10.0, 14.0):
            for width_sigma in (1.0, 2.0, 3.0):
                observed = pd.to_numeric(
                    fitted["narrow_fwhm_observed_kms"], errors="coerce"
                )
                error = pd.to_numeric(
                    fitted["narrow_fwhm_observed_error_kms"], errors="coerce"
                )
                instrument = pd.to_numeric(
                    fitted["instrumental_fwhm_kms"], errors="coerce"
                )
                intrinsic_upper = np.sqrt(
                    np.maximum((observed + width_sigma * error) ** 2 - instrument**2, 0.0)
                )
                narrow = (
                    pd.to_numeric(fitted["narrow_line_snr"], errors="coerce")
                    >= snr_threshold
                )
                secure = intrinsic_upper < 1200.0
                no_bound = ~fitted["narrow_kinematic_bound_hit"].astype(
                    "boolean"
                ).fillna(False)
                broad = (
                    (pd.to_numeric(fitted["delta_bic_broad"], errors="coerce") >= bic_threshold)
                    & (
                        pd.to_numeric(
                            fitted["maximum_broad_line_snr"], errors="coerce"
                        )
                        >= BROAD_SNR_THRESHOLD
                    )
                    & fitted["broad_residual_support"].astype(
                        "boolean"
                    ).fillna(False)
                )
                residual_ok = fitted["residual_quality_pass"].astype(
                    "boolean"
                ).fillna(False)
                selected = narrow & secure & no_bound & residual_ok & ~broad
                rows.append(
                    {
                        "narrow_snr_threshold": snr_threshold,
                        "broad_delta_bic_threshold": bic_threshold,
                        "width_upper_sigma": width_sigma,
                        "n_fitted_complexes": len(fitted),
                        "n_provisional_complex_passes": int(selected.sum()),
                        "n_broad_evidence_complexes": int(broad.sum()),
                        "is_preregistered_rule": bool(
                            snr_threshold == NARROW_SNR_THRESHOLD
                            and bic_threshold == BROAD_DELTA_BIC_THRESHOLD
                            and width_sigma == 2.0
                        ),
                        "selection_status": "diagnostic_not_calibrated",
                    }
                )
    return pd.DataFrame(rows)


def select_qa(records: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Select deterministic post-fit QA; VI and PCF do not enter the selection."""

    selected: dict[tuple[str, str], dict[str, object]] = {}

    def add(frame: pd.DataFrame, reason: str, cap: int | None = limit) -> None:
        if cap is not None:
            frame = frame.head(cap)
        for row in frame.to_dict("records"):
            key = (str(row["object_id"]), str(row["complex_name"]))
            if key not in selected:
                selected[key] = {
                    "object_id": row["object_id"],
                    "complex_name": row["complex_name"],
                    "qa_reason": reason,
                }

    failures = records[records["fit_status"].eq("fit_failed")].sort_values(
        ["complex_name", "object_id"], kind="stable"
    )
    add(failures, "fit_failure", cap=None)
    broad = records[records["broad_evidence"].fillna(False).astype(bool)].sort_values(
        ["complex_name", "object_id"], kind="stable"
    )
    add(broad, "broad_evidence", cap=None)
    fitted = records[records["fit_status"].eq("complete")].copy()
    fitted["_width_distance"] = (
        pd.to_numeric(fitted["narrow_fwhm_intrinsic_upper_2sigma_kms"], errors="coerce")
        - 1200.0
    ).abs()
    add(
        fitted.sort_values(["_width_distance", "object_id"], kind="stable"),
        "width_boundary",
    )
    fitted["_bic_distance"] = pd.to_numeric(
        fitted["delta_bic_broad"], errors="coerce"
    ).abs()
    add(
        fitted.sort_values(["_bic_distance", "object_id"], kind="stable"),
        "bic_boundary",
    )
    fitted["_residual"] = pd.to_numeric(
        fitted.get("n0_residual_abs_p95_sigma"), errors="coerce"
    )
    add(
        fitted.sort_values(["_residual", "object_id"], ascending=[False, True], kind="stable"),
        "high_residual",
    )
    passes = fitted[
        fitted["provisional_narrow_evidence"].fillna(False).astype(bool)
    ].sort_values(["complex_name", "object_id"], kind="stable")
    if len(passes):
        indices = np.linspace(0, len(passes) - 1, min(20, len(passes)), dtype=int)
        add(passes.iloc[indices], "provisional_control", cap=None)
    output = pd.DataFrame(selected.values())
    if len(output):
        output["selection_status"] = "post_fit_qa_only"
    return output


def _validate_part_order(
    combined: pd.DataFrame,
    expected: list[tuple[str, str]],
) -> None:
    found = list(
        zip(combined["object_id"].astype(str), combined["complex_name"].astype(str))
    )
    if found != expected:
        raise ValueError("Part products do not reproduce exact object/complex order")
    if combined.duplicated(["object_id", "complex_name"]).any():
        raise ValueError("Duplicate object/complex rows in finalized model comparison")


def finalize_outputs(
    parts: list[Path],
    expected: list[tuple[str, str]],
    input_frame: pd.DataFrame,
    output_dir: Path,
    complexes: tuple[str, ...],
    provenance: dict[str, object],
    *,
    full_ledger: bool,
) -> dict[str, object]:
    missing = [str(path) for path in parts if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Cannot finalize: {len(missing)} expected part files are missing; "
            f"first={missing[:3]}"
        )
    frames = [pd.read_parquet(path) for path in parts]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _validate_part_order(combined, expected)

    comparison = combined[~combined["fit_status"].eq("not_covered")].copy()
    comparison.to_parquet(output_dir / "line_model_comparison.parquet", index=False)
    diagnostic_threshold_sweep(comparison).to_parquet(
        output_dir / "threshold_sweep.parquet", index=False
    )
    select_qa(comparison).to_parquet(output_dir / "qa_selection.parquet", index=False)

    ledger_input = input_frame if full_ledger else input_frame[
        input_frame["object_id"].astype(str).isin(combined["object_id"].astype(str))
    ]
    ledger = build_evidence_ledger(ledger_input, combined, complexes)
    ledger.to_parquet(output_dir / "narrow_line_evidence.parquet", index=False)
    candidates = ledger[ledger["provisional_narrow"]].copy()
    candidates.to_parquet(
        output_dir / "provisional_narrow_candidates.parquet", index=False
    )
    status_counts = {
        str(key): int(value)
        for key, value in ledger["narrow_line_status"].value_counts(dropna=False).items()
    }
    complex_status_counts = {
        str(complex_name): {
            str(key): int(value)
            for key, value in group["evidence_status"].value_counts(dropna=False).items()
        }
        for complex_name, group in combined.groupby("complex_name", sort=False)
    }
    summary = {
        **provenance,
        "created_at": utc_now(),
        "n_evidence_rows": len(ledger),
        "n_archived_objects_processed": int(combined["object_id"].nunique()),
        "n_complex_rows_in_parts": len(combined),
        "n_fitted_or_failed_complex_rows": len(comparison),
        "n_provisional_narrow": int(ledger["provisional_narrow"].sum()),
        "n_broad_evidence": int(ledger["broad_evidence"].sum()),
        "evidence_status_counts": status_counts,
        "complex_evidence_status_counts": complex_status_counts,
        "physical_classes_assigned": 0,
        "promotion_status": "not_calibrated",
        "purity_target": PURITY_TARGET,
        "independent_calibration_required": True,
        "vi_pcf_role": "post_hoc_reference_only",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and "parts" not in path.parts
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({**summary, "outputs": outputs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = f"""# Provisional RGS narrow-line evidence

The current run contains **{len(ledger):,}** gold-ledger rows and model results for **{combined['object_id'].nunique():,}** archived completed fits. It identifies **{int(ledger['provisional_narrow'].sum()):,}** provisional narrow-line objects and **{int(ledger['broad_evidence'].sum()):,}** objects with secure broad evidence.

The two local comparisons are H-alpha N0/B1 and He I 10833 + Pa-gamma N0/B1. He I alone must have narrow-component S/N at least {NARROW_SNR_THRESHOLD:g}; Pa-gamma is a tied, non-negative nuisance line and is never required. A complex passes only when both fits succeed, the quadrature-deconvolved intrinsic narrow-width {provenance['width_upper_sigma']:g}-sigma upper bound is below 1200 km/s, the narrow kinematics have no relevant active-bound warning, and the N0 residual-quality gates pass. BIC preference for N0 is diagnostic only and is no longer required. Broad evidence requires delta-BIC at least {BROAD_DELTA_BIC_THRESHOLD:g}, broad-line S/N at least {BROAD_SNR_THRESHOLD:g}, and coherent improvement in the 95th-percentile standardized residual.

An object passes provisionally when either covered complex passes and no covered complex has broad evidence. Every object uses the same fixed conservative Gaussian LSF at R={provenance['resolving_power']:g}: instrumental FWHM {provenance['instrumental_fwhm_kms']:.1f} km/s and an observed 1200 km/s boundary of {provenance['observed_narrow_boundary_kms']:.1f} km/s. No effective resolution is derived from morphology. For extended sources this point-source instrumental width under-corrects morphological broadening, which makes the inferred intrinsic width larger and the narrow-width gate conservative.

## Interpretation boundary

These are provisional evidence flags, not physical classes. `physical_class` is deliberately unset. VI and PCF columns are copied only after the fit-derived decision and never enter a fit or selection expression. Promotion requires an estimated purity of at least 95%, plus independent higher-resolution decompositions that quantify the broad contribution; injection/recovery alone is insufficient.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.workers <= 0:
        raise ValueError("--chunk-size and --workers must be positive")
    _validate_resolution_policy(args.mode, args.resolving_power)
    complexes = tuple(dict.fromkeys(args.complexes))
    root = args.output_root or default_root()
    input_path = args.input or root / "input/spectra.parquet"
    run_dir = args.run_directory or root / "runs" / RUN_NAME
    base_output = (
        args.classification_directory
        or root / "classification" / CLASSIFICATION_NAME
    )
    output_dir = base_output / "smoke" if args.mode == "smoke" else base_output
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    input_frame = pd.read_parquet(input_path).reset_index(drop=True)
    if input_frame["object_id"].astype(str).duplicated().any():
        raise ValueError("Gold input contains duplicate object IDs")
    if (
        not args.allow_noncanonical_gold_count
        and len(input_frame) != EXPECTED_GOLD_ROWS
    ):
        raise ValueError(
            f"Expected {EXPECTED_GOLD_ROWS} gold rows, found {len(input_frame)}"
        )
    store = open_run(str(run_dir))
    objects = store.read_table("objects").to_pandas().reset_index(drop=True)
    if objects["object_id"].astype(str).duplicated().any():
        raise ValueError("Run store contains duplicate object IDs")
    gold_ids = set(input_frame["object_id"].astype(str))
    outside = sorted(set(objects["object_id"].astype(str)) - gold_ids)
    if outside:
        raise ValueError(f"Run store contains IDs outside the gold input: {outside[:10]}")
    selection_sha256 = None
    if args.selection is not None:
        if args.mode != "smoke":
            raise ValueError("--selection is restricted to --mode smoke")
        selection = _read_selection(args.selection)
        keyed_objects = objects.copy()
        keyed_objects["_key"] = keyed_objects["object_id"].astype(str)
        target = selection.merge(
            keyed_objects,
            on="_key",
            how="left",
            validate="one_to_one",
            suffixes=("_selection", ""),
        )
        missing = target.loc[target["complex_statuses"].isna(), "_key"].tolist()
        if missing:
            raise ValueError(
                f"--selection contains IDs absent from the run store: {missing[:10]}"
            )
        target = target.drop(columns=["object_id_selection", "_key"])
        selection_sha256 = _sha256(args.selection)
    else:
        target = (
            select_smoke_objects(
                objects, input_frame, complexes, count=args.smoke_count
            )
            if args.mode == "smoke"
            else objects.copy()
        )
    order = {key: index for index, key in enumerate(input_frame["object_id"].astype(str))}
    target["_membership_order"] = target["object_id"].astype(str).map(order)
    target = target.sort_values("_membership_order", kind="stable").reset_index(drop=True)

    lsf = LineSpreadFunctionConfig(resolving_power=args.resolving_power)
    part_paths: list[Path] = []
    expected: list[tuple[str, str]] = []
    executor = (
        ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(str(run_dir),),
        )
        if args.workers > 1 and not args.finalize_only
        else None
    )
    processed = 0
    try:
        for start in range(0, len(target), args.chunk_size):
            stop = min(start + args.chunk_size, len(target))
            rows = target.iloc[start:stop]
            keys = _expected_part_keys(rows, complexes)
            expected.extend(keys)
            part_path = parts_dir / f"rows-{start:05d}-{stop:05d}.parquet"
            part_paths.append(part_path)
            if _part_matches(part_path, keys) and not args.force:
                continue
            if args.finalize_only:
                continue
            if args.max_chunks is not None and processed >= args.max_chunks:
                continue
            records = _fit_rows(
                rows,
                store,
                complexes,
                lsf,
                compute_covariance=not args.no_covariance,
                workers=args.workers,
                executor=executor,
            )
            _write_part(pd.DataFrame(records), part_path)
            processed += 1
            print(f"wrote {part_path} ({len(records)} rows)", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    provenance = {
        "material_passport_version": "dr1_narrow_fixed_r480_v3_development",
        "classification_schema_version": SCHEMA_VERSION,
        "verification_status": "UNVERIFIED",
        "input": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "run_directory": str(run_dir.resolve()),
        "classification_directory": str(output_dir.resolve()),
        "qsospec_commit": _git_commit(),
        "mode": args.mode,
        "complexes": list(complexes),
        "n_gold_input": len(input_frame),
        "n_run_store_completed": len(objects),
        "n_target_objects": len(target),
        "smoke_count": args.smoke_count if args.mode == "smoke" else None,
        "selection": (
            str(args.selection.resolve()) if args.selection is not None else None
        ),
        "selection_sha256": selection_sha256,
        "resolving_power": float(args.resolving_power),
        "resolution_policy": "fixed_conservative_r480_for_all_objects",
        "object_specific_lsf": False,
        "morphology_used_for_lsf": False,
        "instrumental_fwhm_kms": lsf.instrumental_fwhm_kms,
        "intrinsic_narrow_boundary_kms": 1200.0,
        "observed_narrow_boundary_kms": float(np.hypot(1200.0, lsf.instrumental_fwhm_kms)),
        "width_upper_sigma": 2.0,
        "narrow_snr_threshold": NARROW_SNR_THRESHOLD,
        "broad_delta_bic_threshold": BROAD_DELTA_BIC_THRESHOLD,
        "broad_snr_threshold": BROAD_SNR_THRESHOLD,
        "pagamma_detection_required": False,
        "broad_fraction_threshold": None,
        "compute_covariance": not args.no_covariance,
    }
    invocation = {
        **provenance,
        "started_at": utc_now(),
        "chunk_size": args.chunk_size,
        "workers": args.workers,
        "force": args.force,
        "finalize_only": args.finalize_only,
        "max_chunks": args.max_chunks,
    }
    (output_dir / "invocation.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.max_chunks is not None:
        print(
            json.dumps(
                {
                    "status": "partial",
                    "processed_chunks": processed,
                    "n_target_objects": len(target),
                    "classification_directory": str(output_dir),
                },
                indent=2,
            )
        )
        return
    summary = finalize_outputs(
        part_paths,
        expected,
        input_frame,
        output_dir,
        complexes,
        provenance,
        full_ledger=args.mode == "production",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
