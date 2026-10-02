"""Resumability and publication-gate tests for the DR1 H-alpha command."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/classify_euclid_dr1_halpha.py"
SPEC = importlib.util.spec_from_file_location("classify_euclid_dr1_halpha", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(object_id, model="N0"):
    return {
        "object_id": object_id,
        "object_id_uint64": MODULE._as_uint64_decimal(object_id),
        "fit_status": "complete",
        "minimum_bic_model": model,
        "best_broad_model": "B1",
        "delta_bic_broad": -10.0 if model == "N0" else 10.0,
        "narrow_width_secure_below_boundary": model == "N0",
        "narrow_fwhm_intrinsic_upper_2sigma_kms": 900.0 if model == "N0" else 1500.0,
        "selection_status": "not_calibrated",
        "physical_class": None,
    }


def test_signed_to_unsigned_decimal_preserves_bit_pattern():
    assert MODULE._as_uint64_decimal(7) == "7"
    assert MODULE._as_uint64_decimal(-1) == str(2**64 - 1)
    assert MODULE._as_signed_int64(str(-(2**63))) == -(2**63)
    with pytest.raises(ValueError, match="outside signed int64"):
        MODULE._as_signed_int64(str(2**63))


def test_part_alignment_is_exact_and_order_sensitive(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame({"object_id": [-2, 3]}).to_parquet(path, index=False)
    assert MODULE._part_matches(path, ["-2", "3"])
    assert not MODULE._part_matches(path, ["3", "-2"])


def test_finalize_preserves_order_and_does_not_publish_class_catalog(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame([_record(-2)]).to_parquet(first, index=False)
    pd.DataFrame([_record(3, "B1")]).to_parquet(second, index=False)
    provenance = {
        "resolving_power": 480.0,
        "broad_component_counts": [1],
        "instrumental_fwhm_kms": 624.5676208333333,
        "observed_narrow_boundary_kms": 1352.80623630785,
    }
    summary = MODULE._finalize(
        [first, second], ["-2", "3"], tmp_path, provenance
    )
    output = pd.read_parquet(tmp_path / "halpha_model_comparison.parquet")
    assert output["object_id"].tolist() == [-2, 3]
    assert summary["n_physical_classes_assigned"] == 0
    assert not summary["publication_candidate_catalog_written"]
    assert not (tmp_path / "narrow_line_candidates.parquet").exists()
    with pytest.raises(ValueError, match="exact eligible input order"):
        MODULE._finalize(
            [first, second], ["3", "-2"], tmp_path / "bad", provenance
        )
