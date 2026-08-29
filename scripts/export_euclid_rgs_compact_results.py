#!/usr/bin/env python3
"""Export compact, analysis-ready products from sharded Euclid RGS run stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qsospec import open_run

SOURCE_PROJECTIONS = {
    "objects": [
        "run_id",
        "object_key",
        "object_id",
        "redshift",
        "ra",
        "dec",
        "continuum_success",
        "continuum_reduced_chi2",
        "host_decomp_enabled",
        "complex_statuses",
        "warning_codes",
        "completed_at",
    ],
    "measurements": [
        "run_id",
        "object_key",
        "object_id",
        "section",
        "recipe_id",
        "feature_id",
        "role",
        "quantity",
        "value",
        "error",
        "unit",
        "method",
    ],
    "warnings": [
        "run_id",
        "object_key",
        "object_id",
        "section",
        "recipe_id",
        "code",
        "severity",
        "message",
    ],
    "failures": [
        "run_id",
        "object_key",
        "object_id",
        "source",
        "row_index",
        "exception_type",
        "message",
        "failed_at",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--measurement-root", type=Path)
    parser.add_argument(
        "--measurement-product",
        default="dr1_identified_full_rgs_broad_narrow_r480_v1",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--expected-shards", type=int, default=32)
    parser.add_argument("--allow-incomplete-source-runs", action="store_true")
    parser.add_argument("--allow-partial-measurements", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
        write_statistics=True,
    )


class _TableSink:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, table: pa.Table, shard_id: int) -> None:
        shard = pa.array(np.full(table.num_rows, shard_id, dtype=np.int16))
        table = table.append_column("source_shard_id", shard)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd", write_statistics=True)
        elif table.schema != self.writer.schema:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += table.num_rows

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _require_membership(frame: pd.DataFrame, expected_shards: int) -> pd.DataFrame:
    required = {
        "object_id",
        "object_id_uint64",
        "membership_order",
        "qsospec_object_key",
        "shard_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Membership is missing columns: {sorted(missing)}")
    if frame["object_id"].duplicated().any():
        raise ValueError("Membership contains duplicate object IDs")
    if frame["qsospec_object_key"].astype(str).duplicated().any():
        raise ValueError("Membership contains duplicate qsospec object keys")
    ordered = frame.sort_values("membership_order", kind="stable").reset_index(drop=True)
    if not np.array_equal(ordered["membership_order"], np.arange(len(ordered))):
        raise ValueError("Membership order is not contiguous from zero")
    found = set(pd.to_numeric(ordered["shard_id"], errors="raise").astype(int))
    expected = set(range(expected_shards))
    if found != expected:
        raise ValueError(f"Membership shard IDs differ: found={sorted(found)}, expected={sorted(expected)}")
    return ordered


def _unique_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame["object_id"].astype(str).duplicated().any():
        examples = frame.loc[frame["object_id"].astype(str).duplicated(), "object_id"].head(5).tolist()
        raise ValueError(f"{label} contains duplicate object IDs: {examples}")
    indexed = frame.copy()
    indexed["object_id"] = indexed["object_id"].astype(str)
    return indexed.set_index("object_id", drop=False)


def _status_for_shard(
    membership: pd.DataFrame,
    store: Any,
    shard_id: int,
    *,
    allow_incomplete: bool,
    objects_table: pa.Table | None = None,
    failures_table: pa.Table | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = store.manifest
    status = str(manifest.get("status", "unknown"))
    if status != "complete" and not allow_incomplete:
        raise ValueError(f"Source shard {shard_id:03d} is not complete: {status}")

    objects = _unique_index(
        (
            objects_table.select(["object_id", "object_key"])
            if objects_table is not None
            else store.read_table("objects", columns=["object_id", "object_key"])
        ).to_pandas(),
        f"source objects shard {shard_id:03d}",
    )
    failures = _unique_index(
        (
            failures_table.select(["object_id", "object_key", "exception_type", "message"])
            if failures_table is not None
            else store.read_table(
                "failures",
                columns=["object_id", "object_key", "exception_type", "message"],
            )
        ).to_pandas(),
        f"source failures shard {shard_id:03d}",
    )
    overlap = set(objects.index) & set(failures.index)
    if overlap:
        raise ValueError(f"Source shard {shard_id:03d} has completed/failed overlap: {sorted(overlap)[:5]}")

    selected = membership[membership["shard_id"].astype(int).eq(shard_id)].copy()
    selected_ids = set(selected["object_id"].astype(str))
    accounted = set(objects.index) | set(failures.index)
    unexplained = selected_ids - accounted
    extras = accounted - selected_ids
    if (unexplained or extras) and not allow_incomplete:
        raise ValueError(
            f"Source shard {shard_id:03d} reconciliation failed: "
            f"missing={sorted(unexplained)[:5]}, extras={sorted(extras)[:5]}"
        )

    selected_id = selected["object_id"].astype(str)
    completed_key = selected_id.map(objects["object_key"] if len(objects) else {})
    failed_key = selected_id.map(failures["object_key"] if len(failures) else {})
    source_key = completed_key.fillna(failed_key)
    expected_key = selected["qsospec_object_key"].astype(str)
    mismatched = source_key.notna() & source_key.astype(str).ne(expected_key)
    if mismatched.any():
        examples = selected.loc[mismatched, "object_id"].head(5).tolist()
        raise ValueError(f"Source shard {shard_id:03d} object-key mismatch: {examples}")

    result = selected[
        [
            "object_id",
            "object_id_uint64",
            "membership_order",
            "shard_id",
            "qsospec_object_key",
        ]
    ].copy()
    result = result.rename(columns={"shard_id": "source_shard_id"})
    result["source_run_status"] = status
    result["source_run_id"] = manifest.get("run_id")
    result["source_fit_status"] = np.where(
        selected_id.isin(objects.index),
        "complete",
        np.where(selected_id.isin(failures.index), "failed", "missing"),
    )
    result["source_object_key"] = source_key.to_numpy()
    result["source_failure_exception_type"] = selected_id.map(
        failures["exception_type"] if len(failures) else {}
    ).to_numpy()
    result["source_failure_message"] = selected_id.map(failures["message"] if len(failures) else {}).to_numpy()
    summary = {
        "shard_id": shard_id,
        "status": status,
        "membership_rows": len(selected),
        "completed_rows": len(objects),
        "failed_rows": len(failures),
        "missing_rows": len(unexplained),
        "extra_rows": len(extras),
        "run_id": manifest.get("run_id"),
        "source_git_commit": manifest.get("git_commit"),
    }
    return result, summary


def _append_measurement_product(
    measurement_root: Path,
    product: str,
    membership: pd.DataFrame,
    expected_shards: int,
    temporary: Path,
    *,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    long_sink: _TableSink | None = None
    wide_sink: _TableSink | None = None
    coverage: list[dict[str, Any]] = []
    for shard_id in range(expected_shards):
        directory = measurement_root / "full" / f"shard-{shard_id:03d}-of-{expected_shards:03d}" / product
        long_path = directory / "broad_narrow_line_measurements.parquet"
        wide_path = directory / "broad_narrow_measurements_wide.parquet"
        manifest_path = directory / "manifest.json"
        available = long_path.is_file() and wide_path.is_file() and manifest_path.is_file()
        record: dict[str, Any] = {
            "shard_id": shard_id,
            "available": available,
            "directory": str(directory.resolve()),
        }
        if not available:
            coverage.append(record)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"complete", "partial_source"}:
            raise ValueError(f"Broad/narrow shard {shard_id:03d} is not finalized")
        long = pq.read_table(long_path)
        wide = pq.read_table(wide_path)
        expected_ids = set(membership.loc[membership["shard_id"].astype(int).eq(shard_id), "object_id"].astype(str))
        long_ids = set(map(str, long.column("object_id").to_pylist()))
        wide_ids = list(map(str, wide.column("object_id").to_pylist()))
        if not long_ids.issubset(expected_ids) or not set(wide_ids).issubset(expected_ids):
            raise ValueError(f"Broad/narrow shard {shard_id:03d} contains foreign object IDs")
        if len(wide_ids) != len(set(wide_ids)):
            raise ValueError(f"Broad/narrow shard {shard_id:03d} wide table has duplicate IDs")
        if long_sink is None:
            long_sink = _TableSink(temporary / "broad_narrow_line_measurements.parquet")
            wide_sink = _TableSink(temporary / "broad_narrow_measurements_wide.parquet")
        long_sink.write(long, shard_id)
        assert wide_sink is not None
        wide_sink.write(wide, shard_id)
        record.update(
            {
                "status": manifest.get("status"),
                "long_rows": long.num_rows,
                "wide_rows": wide.num_rows,
                "manifest_sha256": _sha256(manifest_path),
                "long_sha256": _sha256(long_path),
                "wide_sha256": _sha256(wide_path),
            }
        )
        coverage.append(record)
    available_count = sum(bool(row["available"]) for row in coverage)
    if available_count != expected_shards and not allow_partial:
        raise ValueError(
            f"Broad/narrow measurements are available for {available_count}/{expected_shards} shards; "
            "pass --allow-partial-measurements for a diagnostic export"
        )
    if long_sink is not None:
        long_sink.close()
        assert wide_sink is not None
        wide_sink.close()
    return coverage, {
        "available_shards": available_count,
        "expected_shards": expected_shards,
        "long_rows": 0 if long_sink is None else long_sink.rows,
        "wide_rows": 0 if wide_sink is None else wide_sink.rows,
    }


def main() -> None:
    args = parse_args()
    sample_root = args.sample_root.expanduser().resolve()
    membership_path = (args.membership or sample_root / "input/plan/full_membership.parquet").resolve()
    run_root = (args.run_root or sample_root / "runs/full").resolve()
    measurement_root = (args.measurement_root or sample_root / "measurements").resolve()
    output = (args.output_directory or sample_root / "analysis/compact_v1").resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory exists: {output}; pass --overwrite to replace it")
    membership = _require_membership(pd.read_parquet(membership_path), args.expected_shards)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    sinks = {name: _TableSink(temporary / f"source_{name}.parquet") for name in SOURCE_PROJECTIONS}
    status_frames: list[pd.DataFrame] = []
    shard_summaries: list[dict[str, Any]] = []
    run_manifest_hashes: list[dict[str, Any]] = []
    try:
        for shard_id in range(args.expected_shards):
            run = run_root / f"shard-{shard_id:03d}-of-{args.expected_shards:03d}"
            manifest_path = run / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing source shard manifest: {manifest_path}")
            store = open_run(str(run))
            projected = {name: store.read_table(name, columns=columns) for name, columns in SOURCE_PROJECTIONS.items()}
            status, shard_summary = _status_for_shard(
                membership,
                store,
                shard_id,
                allow_incomplete=args.allow_incomplete_source_runs,
                objects_table=projected["objects"],
                failures_table=projected["failures"],
            )
            status_frames.append(status)
            shard_summaries.append(shard_summary)
            run_manifest_hashes.append(
                {
                    "shard_id": shard_id,
                    "path": str(manifest_path.resolve()),
                    "sha256": _sha256(manifest_path),
                }
            )
            for name, table in projected.items():
                sinks[name].write(table, shard_id)
        for sink in sinks.values():
            sink.close()

        fit_status = (
            pd.concat(status_frames, ignore_index=True)
            .sort_values("membership_order", kind="stable")
            .reset_index(drop=True)
        )
        if not np.array_equal(fit_status["object_id"].to_numpy(), membership["object_id"].to_numpy()):
            raise ValueError("Compact source-fit status does not preserve membership order")
        _write_frame(temporary / "source_fit_status.parquet", fit_status)
        _write_frame(temporary / "source_shard_summary.parquet", pd.DataFrame(shard_summaries))

        warnings = pq.read_table(
            temporary / "source_warnings.parquet",
            columns=["section", "recipe_id", "code", "severity"],
        ).to_pandas()
        warning_counts = (
            warnings.value_counts(["section", "recipe_id", "code", "severity"], dropna=False)
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False, kind="stable")
        )
        _write_frame(temporary / "source_warning_counts.parquet", warning_counts)

        measurement_coverage, measurement_summary = _append_measurement_product(
            measurement_root,
            args.measurement_product,
            membership,
            args.expected_shards,
            temporary,
            allow_partial=args.allow_partial_measurements,
        )
        _write_frame(
            temporary / "broad_narrow_shard_coverage.parquet",
            pd.DataFrame(measurement_coverage),
        )

        fit_counts = fit_status["source_fit_status"].value_counts(dropna=False).to_dict()
        summary = {
            "status": "complete",
            "membership_rows": len(membership),
            "expected_shards": args.expected_shards,
            "source_fit_counts": {str(key): int(value) for key, value in fit_counts.items()},
            "source_table_rows": {name: sink.rows for name, sink in sinks.items()},
            "warning_code_rows": len(warning_counts),
            "broad_narrow": measurement_summary,
        }
        _write_json(temporary / "summary.json", summary)
        outputs = []
        for path in sorted(temporary.iterdir()):
            if not path.is_file():
                continue
            row_count = pq.ParquetFile(path).metadata.num_rows if path.suffix == ".parquet" else None
            outputs.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "rows": row_count,
                }
            )
        manifest = {
            "export_type": "euclid_rgs_compact_results",
            "status": "complete",
            "sample_root": str(sample_root),
            "membership": str(membership_path),
            "membership_sha256": _sha256(membership_path),
            "run_root": str(run_root),
            "measurement_root": str(measurement_root),
            "measurement_product": args.measurement_product,
            "allow_incomplete_source_runs": args.allow_incomplete_source_runs,
            "allow_partial_measurements": args.allow_partial_measurements,
            "run_manifests": run_manifest_hashes,
            "summary": summary,
            "outputs": outputs,
        }
        _write_json(temporary / "manifest.json", manifest)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        print(json.dumps(summary, indent=2))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
