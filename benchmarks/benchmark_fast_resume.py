"""Synthetic, no-fit benchmark for scalar resume planning and sparse reads."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qsospec
from qsospec.io.run_store import RunStore


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="qsospec-fast-resume-") as temporary:
        root = Path(temporary)
        source = root / "spectra.parquet"
        wave = np.linspace(3600.0, 9800.0, 4000)
        rows = [{
            "qsospec_object_key": f"benchmark-{index:03d}",
            "object_id": f"object-{index:03d}",
            "redshift": index / 224.0,
            "input_row_index": index,
            "qsospec_shard_id": 0,
            "wavelength": wave.tolist(),
            "flux": np.ones_like(wave).tolist(),
            "ivar": np.ones_like(wave).tolist(),
            "mask": np.zeros_like(wave, dtype=np.int16).tolist(),
            "lsf_sigma_angstrom": np.ones_like(wave).tolist(),
        } for index in range(224)]
        pd.DataFrame(rows).to_parquet(source, index=False, row_group_size=32)
        run = root / "run"
        store = RunStore.create(str(run), configuration={"benchmark": True})
        descriptors = list(qsospec.scan_parquet_spectrum_inputs(str(source)))

        def set_completed(count: int) -> None:
            objects = run / "data" / "objects"
            objects.mkdir(parents=True, exist_ok=True)
            for path in objects.glob("*.parquet"):
                path.unlink()
            for descriptor in descriptors[:count]:
                store.object_shard_path("objects", descriptor.object_key).touch()

        set_completed(224)
        started = time.perf_counter()
        complete = qsospec.plan_batch_resume(str(source), str(run))
        complete_seconds = time.perf_counter() - started

        set_completed(221)
        started = time.perf_counter()
        partial = qsospec.plan_batch_resume(str(source), str(run))
        partial_plan_seconds = time.perf_counter() - started
        started = time.perf_counter()
        loaded = list(qsospec.scan_parquet_spectra(
            str(source), row_indices=partial.unfinished_row_indices
        ))
        sparse_read_seconds = time.perf_counter() - started

        set_completed(0)
        started = time.perf_counter()
        fresh = qsospec.plan_batch_resume(str(source), str(run))
        fresh_seconds = time.perf_counter() - started

        output = {
            "synthetic_rows": 224,
            "pixels_per_row": 4000,
            "numerical_fits_run": 0,
            "fully_complete": {
                "wall_seconds": complete_seconds,
                "identity_scan_seconds": complete.timings["resume_identity_scan_seconds"],
                "manifest_seconds": complete.timings["resume_manifest_seconds"],
                "vector_rows_loaded": 0,
                "vector_rows_avoided": complete.skipped_count,
                "workers_started": 0,
            },
            "partial_221_of_224": {
                "plan_wall_seconds": partial_plan_seconds,
                "sparse_vector_read_seconds": sparse_read_seconds,
                "vector_rows_loaded": len(loaded),
                "vector_rows_avoided": partial.skipped_count,
                "unfinished_row_indices": list(
                    partial.unfinished_row_indices[str(source)]
                ),
            },
            "fresh": {
                "plan_wall_seconds": fresh_seconds,
                "unfinished_count": fresh.unfinished_count,
            },
            "identity_equivalence": descriptors == [
                descriptor
                for descriptor, _ in qsospec.scan_parquet_spectra(str(source))
            ],
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
