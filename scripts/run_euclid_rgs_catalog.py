#!/usr/bin/env python3
"""Run or resume one manifest-validated portable Euclid RGS sample shard."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from qsospec import finalize_run, fit_batch, open_run
from qsospec.euclid_rgs import (
    config_manifest,
    scientific_configuration,
    validate_sample_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--dustmaps-data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument("--task-size", type=int, default=8)
    parser.add_argument("--manifest-update-interval", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--reference-bytes-per-object", type=float)
    parser.add_argument("--disk-headroom", type=float, default=1.3)
    return parser.parse_args()


def _smoke_rows(path: Path, count: int = 64) -> list[int]:
    frame = pd.read_parquet(path, columns=["object_id", "redshift"])
    if len(frame) < count:
        raise ValueError(f"Smoke mode requires at least {count} rows")
    frame["_input_row"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["redshift", "object_id"], kind="stable")
    positions = np.rint(np.linspace(0, len(frame) - 1, count)).astype(int)
    return frame.iloc[positions]["_input_row"].astype(int).tolist()


def _disk_preflight(args: argparse.Namespace, input_rows: int) -> dict:
    args.run_directory.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.run_directory.parent).free
    input_bytes = args.input.stat().st_size
    estimate = input_bytes
    if args.reference_bytes_per_object is not None:
        estimate += int(float(args.reference_bytes_per_object) * input_rows)
    required = int(float(args.disk_headroom) * estimate)
    if free < required:
        raise RuntimeError(
            f"Insufficient free space: require {required / 1024**3:.2f} GiB with "
            f"headroom, found {free / 1024**3:.2f} GiB"
        )
    return {
        "free_bytes": int(free),
        "estimated_input_and_archive_bytes": int(estimate),
        "required_with_headroom_bytes": int(required),
        "reference_bytes_per_object": args.reference_bytes_per_object,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.parquet_batch_size < 1 or args.task_size < 1:
        raise ValueError("workers, parquet-batch-size, and task-size must be positive")
    if not args.dustmaps_data_dir.is_dir():
        raise FileNotFoundError(args.dustmaps_data_dir)
    validation = validate_sample_manifest(args.input, args.sample_manifest)
    disk = _disk_preflight(args, validation["input_rows"])
    row_indices = _smoke_rows(args.input) if args.mode == "smoke" else None
    progress_total = len(row_indices) if row_indices is not None else validation["input_rows"]
    args.run_directory.mkdir(parents=True, exist_ok=True)
    invocation = {
        "mode": args.mode,
        "input": str(args.input.resolve()),
        "sample_manifest": str(args.sample_manifest.resolve()),
        "sample_validation": validation,
        "disk_preflight": disk,
        "workers": args.workers,
        "parquet_batch_size": args.parquet_batch_size,
        "task_size": args.task_size,
        "resume": args.resume,
        "retry_failures": args.retry_failures,
        "scientific_configuration": config_manifest(str(args.dustmaps_data_dir)),
    }
    (args.run_directory / "invocation.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.finalize_only:
        datasets = finalize_run(open_run(str(args.run_directory)))
        print(json.dumps({"finalize_only": True, "datasets": datasets}, indent=2, default=str))
        return
    config = scientific_configuration(str(args.dustmaps_data_dir))
    result = fit_batch(
        str(args.input),
        str(args.run_directory),
        row_indices=row_indices,
        parquet_batch_size=args.parquet_batch_size,
        task_size=args.task_size,
        n_workers=args.workers,
        run_host_decomp=False,
        galactic_extinction_config=config["galactic_extinction_config"],
        global_config=config["global_config"],
        uncertainty_config=config["uncertainty_config"],
        complexes=None,
        resume=args.resume,
        retry_failures=args.retry_failures,
        finalize=True,
        compact_models=False,
        write_legacy_products=False,
        manifest_update_interval=args.manifest_update_interval,
        show_progress=not args.no_progress,
        progress_total=progress_total,
    )
    store = open_run(str(args.run_directory))
    store.manifest["sample_manifest"] = validation
    store.manifest["sample_manifest_path"] = str(args.sample_manifest.resolve())
    store._write_manifest(reconcile=False)
    payload = vars(result)
    payload["sample_validation"] = validation
    (args.run_directory / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
