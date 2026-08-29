"""Tests for the compact Euclid RGS run-store exporter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/export_euclid_rgs_compact_results.py"
    spec = importlib.util.spec_from_file_location("compact_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_id": [-1, 2, 3],
            "object_id_uint64": [str(2**64 - 1), "2", "3"],
            "membership_order": [0, 1, 2],
            "qsospec_object_key": ["sample:-1", "sample:2", "sample:3"],
            "shard_id": [0, 0, 1],
        }
    )


def test_membership_validation_requires_exact_contiguous_order():
    module = _load_script()
    validated = module._require_membership(_membership(), 2)
    assert validated["object_id"].tolist() == [-1, 2, 3]
    broken = _membership()
    broken.loc[2, "membership_order"] = 4
    with pytest.raises(ValueError, match="not contiguous"):
        module._require_membership(broken, 2)


def test_status_reconciliation_preserves_known_failures():
    module = _load_script()
    objects = pa.table({"object_id": ["-1"], "object_key": ["sample:-1"]})
    failures = pa.table(
        {
            "object_id": ["2"],
            "object_key": ["sample:2"],
            "exception_type": ["ValueError"],
            "message": ["Too few valid pixels"],
        }
    )

    def read_table(name, columns):
        table = objects if name == "objects" else failures
        return table.select(columns)

    store = SimpleNamespace(
        manifest={"status": "complete", "run_id": "run-0", "git_commit": "abc"},
        read_table=read_table,
    )
    status, summary = module._status_for_shard(_membership(), store, 0, allow_incomplete=False)
    assert status["source_fit_status"].tolist() == ["complete", "failed"]
    assert status.loc[1, "source_failure_exception_type"] == "ValueError"
    assert summary["completed_rows"] == 1
    assert summary["failed_rows"] == 1


def test_status_reconciliation_rejects_unexplained_missing_member():
    module = _load_script()
    objects = pa.table({"object_id": ["-1"], "object_key": ["sample:-1"]})
    failures = pa.table(
        {
            "object_id": pa.array([], type=pa.string()),
            "object_key": pa.array([], type=pa.string()),
            "exception_type": pa.array([], type=pa.string()),
            "message": pa.array([], type=pa.string()),
        }
    )
    store = SimpleNamespace(
        manifest={"status": "complete"},
        read_table=lambda name, columns: (objects if name == "objects" else failures).select(columns),
    )
    with pytest.raises(ValueError, match="reconciliation failed"):
        module._status_for_shard(_membership(), store, 0, allow_incomplete=False)


def test_partial_broad_narrow_merge_is_explicit_and_ordered(tmp_path):
    module = _load_script()
    root = tmp_path / "measurements"
    product = "test_product"
    directory = root / "full/shard-000-of-002" / product
    directory.mkdir(parents=True)
    long = pa.table(
        {
            "object_id": [-1, -1, 2, 2],
            "complex_name": ["halpha", "hbeta", "halpha", "hbeta"],
            "fit_status": ["complete"] * 4,
        }
    )
    wide = pa.table({"object_id": [-1, 2], "halpha_broad_fraction": [0.1, 0.8]})
    pq.write_table(long, directory / "broad_narrow_line_measurements.parquet")
    pq.write_table(wide, directory / "broad_narrow_measurements_wide.parquet")
    (directory / "manifest.json").write_text(json.dumps({"status": "partial_source"}), encoding="utf-8")
    output = tmp_path / "compact"
    output.mkdir()

    coverage, summary = module._append_measurement_product(root, product, _membership(), 2, output, allow_partial=True)
    assert summary == {
        "available_shards": 1,
        "expected_shards": 2,
        "long_rows": 4,
        "wide_rows": 2,
    }
    assert coverage[0]["available"] is True
    assert coverage[1]["available"] is False
    merged = pq.read_table(output / "broad_narrow_line_measurements.parquet")
    assert merged.column("source_shard_id").to_pylist() == [0, 0, 0, 0]
