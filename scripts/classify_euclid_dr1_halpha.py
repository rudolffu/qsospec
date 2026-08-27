"""Measure RGS-aware H-alpha model diagnostics for the DR1 gold sample.

This command compares a tied narrow-only model (N0) with one flexible broad
H-alpha component (B1).  More complex broad models are opt-in diagnostics for
selected objects, not default fits for the full sample.  The command does not
assign a physical type or publish a narrow-line candidate catalogue: the broad
H-alpha fraction and model-selection thresholds still require calibration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np
import pandas as pd

from qsospec import (
    HalphaModelSelectionConfig,
    LineSpreadFunctionConfig,
    diagnostic_bic_sweep,
    fit_halpha_model_grid,
    load_model,
    open_run,
)


RUN_NAME = "production_no_balmer_no_host_v1"
CLASSIFICATION_NAME = "halpha_narrow_r480_v1"
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
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--classification-directory", type=Path)
    parser.add_argument("--resolving-power", type=float, default=480.0)
    parser.add_argument(
        "--broad-component-counts",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=(1,),
        help=(
            "Broad models to compare with N0. The full-sample default is B1; "
            "use additional counts only for targeted diagnostics."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Process at most this many non-empty chunks and leave outputs partial.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-covariance", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
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
        ).strip()
    except Exception:
        return None


def _decode_key_values(values) -> dict[str, object]:
    output: dict[str, object] = {}
    if values is None:
        return output
    for item in values:
        key = str(item["key"])
        value = item["value"]
        try:
            output[key] = json.loads(value)
        except Exception:
            output[key] = value
    return output


def _as_signed_int64(value: object) -> int:
    parsed = int(str(value))
    if not -(2**63) <= parsed < 2**63:
        raise ValueError(f"object_id is outside signed int64: {value}")
    return parsed


def _as_uint64_decimal(signed: int) -> str:
    return str(signed if signed >= 0 else signed + 2**64)


def _eligible_ids(objects: pd.DataFrame) -> set[str]:
    eligible = set()
    for row in objects[["object_id", "complex_statuses"]].to_dict("records"):
        statuses = _decode_key_values(row["complex_statuses"])
        if statuses.get("halpha_nii_sii") == "fit":
            eligible.add(str(row["object_id"]))
    return eligible


def _part_matches(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        found = pd.read_parquet(path, columns=["object_id"])["object_id"]
        return found.astype(str).tolist() == expected_ids
    except Exception:
        return False


def _write_part(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _failure_record(row: pd.Series, message: str) -> dict[str, object]:
    signed = _as_signed_int64(row["object_id"])
    return {
        "object_id": signed,
        "object_id_uint64": _as_uint64_decimal(signed),
        "fit_status": "failed",
        "error": message,
        "selection_status": "not_calibrated",
        "physical_class": None,
        "broad_fraction_status": "not_calibrated_profile_limit_not_run",
    }


def _fit_row(
    store,
    row: pd.Series,
    selection_config: HalphaModelSelectionConfig,
    *,
    compute_covariance: bool,
) -> dict[str, object]:
    signed = _as_signed_int64(row["object_id"])
    loaded = load_model(store, str(signed))
    result = fit_halpha_model_grid(
        loaded.spectrum,
        loaded.continuum,
        selection_config=selection_config,
        compute_covariance=compute_covariance,
    )
    record = result.to_record(signed)
    record["object_id"] = signed
    record["object_id_uint64"] = _as_uint64_decimal(signed)
    record["error"] = None
    for name in AUDIT_COLUMNS:
        if name in row.index:
            record[name] = row[name]
    return record


def _initialize_worker(run_directory: str) -> None:
    global _WORKER_STORE
    _WORKER_STORE = open_run(run_directory)


def _worker_fit(payload) -> dict[str, object]:
    row_record, selection_config, compute_covariance = payload
    row = pd.Series(row_record)
    try:
        return _fit_row(
            _WORKER_STORE,
            row,
            selection_config,
            compute_covariance=compute_covariance,
        )
    except Exception as error:
        return _failure_record(
            row,
            "".join(traceback.format_exception_only(type(error), error)).strip(),
        )


def _fit_rows(
    rows: pd.DataFrame,
    store,
    selection_config: HalphaModelSelectionConfig,
    *,
    compute_covariance: bool,
    workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, object]]:
    projected = [
        {name: row[name] for name in ("object_id", *AUDIT_COLUMNS) if name in row}
        for row in rows.to_dict("records")
    ]
    if workers == 1:
        records = []
        for row_record in projected:
            row = pd.Series(row_record)
            try:
                records.append(
                    _fit_row(
                        store,
                        row,
                        selection_config,
                        compute_covariance=compute_covariance,
                    )
                )
            except Exception as error:
                records.append(
                    _failure_record(
                        row,
                        "".join(
                            traceback.format_exception_only(type(error), error)
                        ).strip(),
                    )
                )
        return records

    payloads = [
        (row_record, selection_config, compute_covariance)
        for row_record in projected
    ]
    if executor is None:
        raise RuntimeError("A process executor is required when workers > 1")
    return list(executor.map(_worker_fit, payloads))


def _select_qa(frame: pd.DataFrame, limit_per_reason: int = 25) -> pd.DataFrame:
    """Choose deterministic diagnostic QA without using VI as selection truth."""

    chosen: dict[str, dict[str, object]] = {}

    def add(rows: pd.DataFrame, reason: str) -> None:
        for row in rows.head(limit_per_reason).to_dict("records"):
            key = str(row["object_id"])
            if key not in chosen:
                chosen[key] = {
                    "object_id": row["object_id"],
                    "qa_reason": reason,
                    "selection_status": "diagnostic_not_calibrated",
                }

    failed = frame[frame["fit_status"] != "complete"].sort_values(
        "object_id", kind="stable"
    )
    add(failed, "fit_failure")

    completed = frame[frame["fit_status"] == "complete"].copy()
    width_distance = (
        pd.to_numeric(completed["narrow_fwhm_intrinsic_upper_2sigma_kms"], errors="coerce")
        - 1200.0
    ).abs()
    add(completed.assign(_distance=width_distance).sort_values(
        ["_distance", "object_id"], kind="stable"
    ), "near_intrinsic_width_boundary")

    bic_distance = pd.to_numeric(
        completed["delta_bic_broad"], errors="coerce"
    ).abs()
    add(completed.assign(_distance=bic_distance).sort_values(
        ["_distance", "object_id"], kind="stable"
    ), "near_equal_bic")

    disagreement = completed[
        completed["minimum_bic_model"].eq("N0")
        != completed["narrow_width_secure_below_boundary"].fillna(False)
    ].copy()
    add(disagreement.sort_values("object_id", kind="stable"), "width_model_disagreement")
    return pd.DataFrame(chosen.values())


def _finalize(
    parts: list[Path],
    expected_ids: list[str],
    output_dir: Path,
    provenance: dict[str, object],
) -> dict[str, object]:
    missing = [str(path) for path in parts if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Cannot finalize: {len(missing)} expected chunk parts are missing; "
            f"first={missing[:3]}"
        )
    frames = [pd.read_parquet(path) for path in parts]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    found_ids = combined["object_id"].astype(str).tolist() if len(combined) else []
    if found_ids != expected_ids:
        raise ValueError("Chunk products do not reproduce exact eligible input order")
    if combined["object_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate object IDs in finalized H-alpha measurements")

    comparison = output_dir / "halpha_model_comparison.parquet"
    combined.to_parquet(comparison, index=False)
    diagnostic_bic_sweep(combined).to_parquet(
        output_dir / "threshold_sweep.parquet", index=False
    )
    _select_qa(combined).to_parquet(output_dir / "qa_selection.parquet", index=False)

    complete = combined["fit_status"].eq("complete")
    summary = {
        **provenance,
        "created_at": utc_now(),
        "n_eligible_halpha": int(len(combined)),
        "n_complete_model_grids": int(complete.sum()),
        "n_failed_or_partial": int((~complete).sum()),
        "minimum_bic_model_counts": {
            str(key): int(value)
            for key, value in combined.loc[complete, "minimum_bic_model"]
            .value_counts(dropna=False)
            .items()
        },
        "n_physical_classes_assigned": int(combined["physical_class"].notna().sum()),
        "selection_status": "not_calibrated",
        "broad_fraction_threshold": None,
        "publication_candidate_catalog_written": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({**summary, "outputs": sorted(path.name for path in output_dir.iterdir())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model_names = "/".join(
        ["N0", *(f"B{count}" for count in provenance["broad_component_counts"])]
    )
    report = f"""# RGS-aware H-alpha model comparison

This run measured **{len(combined):,}** H-alpha-covered DR1 gold spectra. Complete {model_names} comparisons were obtained for **{int(complete.sum()):,}** objects; **{int((~complete).sum()):,}** were partial or failed.

The instrumental approximation is a Gaussian, constant resolving power of **R={provenance['resolving_power']:g}** for a point source, corresponding to **{provenance['instrumental_fwhm_kms']:.1f} km s^-1**. Fitted widths remain observed widths; the table also records quadrature-deconvolved intrinsic widths. The intrinsic narrow/broad boundary is 1200 km s^-1, which becomes {provenance['observed_narrow_boundary_kms']:.1f} km s^-1 in observed fitted width.

## Interpretation boundary

No physical type has been assigned and no narrow-line candidate catalogue has been written. `delta_bic_broad`, the intrinsic narrow-width upper bound, and the same-line broad H-alpha fraction are diagnostics. The broad-fraction upper limit and its purity threshold remain uncalibrated. VI classes are copied only as post-hoc audit metadata and never enter model fitting or candidate selection.

The next step is injection/recovery across redshift, S/N, source extent, and broad fraction, followed by comparison with independent higher-resolution decompositions. Objects with conspicuous asymmetric residuals can receive targeted B2/B3 refits; those extra models are not run across the full sample. Only calibration should freeze a purity-oriented selection.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    root = args.output_root or default_root()
    input_path = args.input or root / "input/spectra.parquet"
    run_dir = args.run_directory or root / "runs" / RUN_NAME
    output_dir = args.classification_directory or root / "classification" / CLASSIFICATION_NAME
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    input_frame = pd.read_parquet(input_path).reset_index(drop=True)
    if input_frame["object_id"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate object IDs")
    input_frame["_object_id_text"] = input_frame["object_id"].astype(str)
    store = open_run(str(run_dir))
    objects = store.read_table("objects").to_pandas()
    eligible = _eligible_ids(objects)
    target = input_frame[input_frame["_object_id_text"].isin(eligible)].copy()
    expected_ids = target["_object_id_text"].tolist()
    if len(expected_ids) != len(eligible):
        missing = sorted(eligible - set(expected_ids))
        raise ValueError(f"Run-store IDs absent from gold input: {missing[:10]}")

    selection_config = HalphaModelSelectionConfig(
        lsf=LineSpreadFunctionConfig(resolving_power=args.resolving_power),
        intrinsic_broad_component_counts=tuple(args.broad_component_counts),
    )
    part_paths: list[Path] = []
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
        for start in range(0, len(input_frame), args.chunk_size):
            stop = min(start + args.chunk_size, len(input_frame))
            rows = input_frame.iloc[start:stop]
            rows = rows[rows["_object_id_text"].isin(eligible)]
            if rows.empty:
                continue
            part_path = parts_dir / f"rows-{start:05d}-{stop:05d}.parquet"
            part_paths.append(part_path)
            ids = rows["_object_id_text"].tolist()
            if _part_matches(part_path, ids) and not args.force:
                continue
            if args.finalize_only:
                continue
            if args.max_chunks is not None and processed >= args.max_chunks:
                continue
            records = _fit_rows(
                rows,
                store,
                selection_config,
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
        "input": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "run_directory": str(run_dir.resolve()),
        "classification_directory": str(output_dir.resolve()),
        "qsospec_commit": _git_commit(),
        "n_gold_input": int(len(input_frame)),
        "n_run_store_completed": int(len(objects)),
        "resolving_power": float(args.resolving_power),
        "instrumental_fwhm_kms": selection_config.lsf.instrumental_fwhm_kms,
        "intrinsic_narrow_boundary_kms": 1200.0,
        "broad_component_counts": list(args.broad_component_counts),
        "observed_narrow_boundary_kms": float(
            np.hypot(1200.0, selection_config.lsf.instrumental_fwhm_kms)
        ),
        "width_parameterization": "observed",
        "vi_role": "post_hoc_audit_only",
        "compute_covariance": not args.no_covariance,
    }
    invocation = {
        **provenance,
        "started_at": utc_now(),
        "chunk_size": args.chunk_size,
        "max_chunks": args.max_chunks,
        "workers": args.workers,
        "force": args.force,
        "finalize_only": args.finalize_only,
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
                    "n_eligible_halpha": len(expected_ids),
                    "classification_directory": str(output_dir),
                },
                indent=2,
            )
        )
        return
    summary = _finalize(part_paths, expected_ids, output_dir, provenance)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
