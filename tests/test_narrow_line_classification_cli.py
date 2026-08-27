"""Ledger, precedence, and resumability tests for the unified classifier."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/classify_euclid_dr1_narrow_lines.py"
)
SPEC = importlib.util.spec_from_file_location(
    "classify_euclid_dr1_narrow_lines", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(object_id, complex_name, status, *, provisional=False, broad=False):
    return {
        "object_id": object_id,
        "complex_name": complex_name,
        "fit_status": "complete" if status in {"provisional_narrow", "broad_evidence", "indeterminate"} else status,
        "evidence_status": status,
        "provisional_narrow_evidence": provisional,
        "broad_evidence": broad,
    }


def test_signed_to_unsigned_export():
    assert MODULE._as_uint64_decimal(3) == "3"
    assert MODULE._as_uint64_decimal(-1) == str(2**64 - 1)


def test_production_lsf_is_fixed_r480_and_not_object_specific():
    MODULE._validate_resolution_policy("production", 480.0)
    MODULE._validate_resolution_policy("smoke", 400.0)
    with pytest.raises(ValueError, match="fixed conservative R=480"):
        MODULE._validate_resolution_policy("production", 400.0)


def test_part_alignment_requires_exact_object_complex_order(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "object_id": [-2, -2, 3, 3],
            "complex_name": ["halpha", "hei_pgamma"] * 2,
            "classification_schema_version": [MODULE.SCHEMA_VERSION] * 4,
        }
    ).to_parquet(path, index=False)
    expected = [
        ("-2", "halpha"),
        ("-2", "hei_pgamma"),
        ("3", "halpha"),
        ("3", "hei_pgamma"),
    ]
    assert MODULE._part_matches(path, expected)
    assert not MODULE._part_matches(path, list(reversed(expected)))

    legacy = pd.read_parquet(path)
    legacy["classification_schema_version"] = "legacy"
    legacy.to_parquet(path, index=False)
    assert not MODULE._part_matches(path, expected)


def test_broad_evidence_vetoes_a_pass_from_the_other_complex():
    frame = pd.DataFrame(
        [
            _record(-4, "halpha", "provisional_narrow", provisional=True),
            _record(-4, "hei_pgamma", "broad_evidence", broad=True),
        ]
    )
    evidence = MODULE.combine_object_evidence(frame)
    assert evidence["narrow_line_status"] == "broad_evidence"
    assert not evidence["provisional_narrow"]


def test_only_narrow_fit_passes_even_if_other_complex_is_not_covered():
    frame = pd.DataFrame(
        [
            _record(5, "halpha", "not_covered"),
            _record(5, "hei_pgamma", "provisional_narrow", provisional=True),
        ]
    )
    evidence = MODULE.combine_object_evidence(frame)
    assert evidence["narrow_line_status"] == "provisional_narrow"
    assert evidence["passing_complexes"] == "hei_pgamma"


def test_ledger_preserves_exact_gold_order_and_leaves_physical_class_unset():
    gold = pd.DataFrame(
        {
            "object_id": [7, -2, 11],
            "membership_order": [0, 1, 2],
            "qsospec_vi_class": ["QSO_NARROW", None, "QSO"],
            "pcf_template_best": ["type2", "type1", "type2"],
        }
    )
    records = pd.DataFrame(
        [
            _record(-2, "halpha", "not_covered"),
            _record(-2, "hei_pgamma", "provisional_narrow", provisional=True),
            _record(7, "halpha", "broad_evidence", broad=True),
            _record(7, "hei_pgamma", "not_covered"),
        ]
    )
    ledger = MODULE.build_evidence_ledger(
        gold, records, ("halpha", "hei_pgamma")
    )
    assert ledger["object_id"].tolist() == [7, -2, 11]
    assert ledger["narrow_line_status"].tolist() == [
        "broad_evidence",
        "provisional_narrow",
        "not_available",
    ]
    assert ledger["physical_class"].isna().all()
    assert ledger["classification_schema_version"].eq(MODULE.SCHEMA_VERSION).all()
    assert ledger.loc[0, "qsospec_vi_class"] == "QSO_NARROW"
