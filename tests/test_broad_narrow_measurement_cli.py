"""Integration-level table and resume tests for the measurement CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/measure_euclid_dr1_broad_narrow_lines.py"
    spec = importlib.util.spec_from_file_location("measure_broad_narrow_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(module):
    rows = []
    for order, object_id in enumerate((-1, 2)):
        for complex_name in ("halpha", "mgii"):
            rows.append({
                "object_id": object_id,
                "object_id_uint64": str(2**64 - 1) if object_id < 0 else str(object_id),
                "object_key": f"source#{order}",
                "membership_order": order,
                "measurement_schema_version": module.BROAD_NARROW_MEASUREMENT_SCHEMA_VERSION,
                "complex_name": complex_name,
                "fit_status": "complete",
                "fit_success": True,
                "broad_fraction": 0.25 + 0.1 * order,
                "broad_fraction_error": 0.02,
                "narrow_flux": 3.0,
                "broad_flux": 1.0,
                "narrow_flux_snr": 6.0,
                "broad_flux_snr": 3.0,
                "total_profile_fwhm_observed_kms": 1500.0,
                "narrow_fwhm_observed_kms": 800.0,
                "broad_fwhm_observed_kms": 3500.0,
                "reduced_chi2": 1.1,
                "residual_abs_max_sigma": 2.5,
                "active_bound_parameters": [],
                "part_fingerprint": "fingerprint",
            })
    return pd.DataFrame(rows)


def test_part_resume_requires_exact_order_schema_and_fingerprint(tmp_path):
    module = _load_script()
    frame = _rows(module)
    part = tmp_path / "part.parquet"
    frame.to_parquet(part, index=False)
    expected = [
        ("-1", "halpha"), ("-1", "mgii"),
        ("2", "halpha"), ("2", "mgii"),
    ]
    assert module._part_matches(part, expected, "fingerprint")
    assert not module._part_matches(part, list(reversed(expected)), "fingerprint")
    assert not module._part_matches(part, expected, "changed")

    changed = frame.copy()
    changed["measurement_schema_version"] = "future"
    changed.to_parquet(part, index=False)
    assert not module._part_matches(part, expected, "fingerprint")


def test_finalize_preserves_ledger_order_and_writes_unique_long_and_wide(tmp_path):
    module = _load_script()
    directory = tmp_path / "measurement"
    parts = directory / "parts"
    parts.mkdir(parents=True)
    frame = _rows(module)
    frame.to_parquet(parts / "part-000000-000001.parquet", index=False)
    selected = pd.DataFrame({"object_id": [-1, 2], "membership_order": [0, 1]})
    input_frame = selected.assign(field=["EDFN", "EDFS"], vi_reference=["QSO", "QSO_NARROW"])

    summary = module._finalize(
        directory,
        selected,
        input_frame,
        ("halpha", "mgii"),
        {"resolving_power": 480.0},
        {"status": "active"},
    )
    long = pd.read_parquet(directory / "broad_narrow_line_measurements.parquet")
    ledger = pd.read_parquet(directory / "broad_narrow_measurement_ledger.parquet")
    wide = pd.read_parquet(directory / "broad_narrow_measurements_wide.parquet")
    assert summary["n_long_rows"] == 4
    assert not long.duplicated(["object_id", "complex_name"]).any()
    assert list(zip(ledger["object_id"].astype(str), ledger["complex_name"])) == [
        ("-1", "halpha"), ("-1", "mgii"),
        ("2", "halpha"), ("2", "mgii"),
    ]
    assert ledger["field"].tolist() == ["EDFN", "EDFN", "EDFS", "EDFS"]
    assert wide["object_id"].tolist() == [-1, 2]
    assert "halpha_broad_fraction" in wide
    assert "mgii_broad_fraction" in wide


def test_smoke_selection_hard_cap_is_100():
    module = _load_script()
    frame = pd.DataFrame({
        "object_id": range(101),
        "membership_order": range(101),
        "redshift": [1.0] * 101,
    })
    try:
        module.select_smoke_objects(frame, {str(value) for value in range(101)}, ("halpha",), 101)
    except ValueError as error:
        assert "capped at 100" in str(error)
    else:
        raise AssertionError("101-object development smoke selection was accepted")
