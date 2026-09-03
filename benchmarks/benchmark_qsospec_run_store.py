#!/usr/bin/env python3
"""Read-only benchmark for schema-v5 qsospec run-store lookup overhead."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

from qsospec import load_model_by_key, open_run

_WORKER_STORE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker-counts", nargs="+", type=int, default=(),
        help="Optional read-only parallel direct-load throughput trials, e.g. 8 16 24 32.",
    )
    parser.add_argument(
        "--include-full-scan-comparison", action="store_true",
        help="Time legacy dataset-filter reads without modifying the run.",
    )
    return parser.parse_args()


def _initialize_worker(run_directory: str) -> None:
    global _WORKER_STORE
    _WORKER_STORE = open_run(run_directory)


def _load_in_worker(object_key: str) -> str:
    load_model_by_key(_WORKER_STORE, object_key)
    return object_key


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median_seconds": float(np.median(array)) if array.size else np.nan,
        "p95_seconds": float(np.percentile(array, 95.0)) if array.size else np.nan,
        "total_seconds": float(np.sum(array)),
    }


def main() -> None:
    args = parse_args()
    if args.sample_size < 1 or args.sample_size > 100:
        raise ValueError("sample-size must be in [1, 100]")
    store = open_run(str(args.run_directory))

    start = time.perf_counter()
    authoritative = store._authoritative_state()
    completed_scan_seconds = time.perf_counter() - start
    start = time.perf_counter()
    index = store.build_object_index()
    index_seconds = time.perf_counter() - start
    keys = sorted(authoritative["completed_keys"])
    indices = np.linspace(0, len(keys) - 1, min(args.sample_size, len(keys)), dtype=int)
    selected = [keys[index] for index in indices]

    direct = []
    for object_key in selected:
        start = time.perf_counter()
        load_model_by_key(store, object_key)
        direct.append(time.perf_counter() - start)

    legacy = []
    if args.include_full_scan_comparison:
        for object_key in selected:
            start = time.perf_counter()
            for table in ("objects", "models", "measurements", "warnings"):
                store.read_table(
                    table,
                    filter_expression=pc.field("object_key") == object_key,
                )
            legacy.append(time.perf_counter() - start)

    parallel_trials = []
    for workers in args.worker_counts:
        if workers < 1 or workers > 64:
            raise ValueError("worker-counts must be in [1, 64]")
        start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(args.run_directory),),
        ) as executor:
            loaded = list(executor.map(_load_in_worker, selected))
        elapsed = time.perf_counter() - start
        if loaded != selected:
            raise ValueError("Parallel direct-load trial changed object order")
        parallel_trials.append({
            "workers": workers,
            "objects": len(selected),
            "wall_seconds": elapsed,
            "objects_per_second": len(selected) / elapsed if elapsed > 0 else np.nan,
        })

    output = {
        "run_directory": str(args.run_directory.resolve()),
        "read_only": True,
        "completed_key_scan_seconds": completed_scan_seconds,
        "object_index_seconds": index_seconds,
        "object_index_size": len(index),
        "completed_objects": len(keys),
        "shards_by_table": authoritative["shard_state"],
        "direct_object_load": _summary(direct),
        "legacy_filtered_dataset_reads": _summary(legacy),
        "parallel_direct_load_trials": parallel_trials,
        "sample_object_keys": selected,
    }
    encoded = json.dumps(output, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
