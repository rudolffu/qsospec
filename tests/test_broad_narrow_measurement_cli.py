"""Integration-level table and resume tests for the measurement CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import qsospec
from qsospec.global_result import GlobalContinuumResult


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


def test_finalized_source_failures_are_retained_as_unavailable_rows():
    module = _load_script()
    selected = pd.DataFrame({
        "object_id": [-1, 2],
        "object_key": [None, "source#1"],
        "qsospec_object_key": ["source#0", "source#1"],
        "membership_order": [0, 1],
    })
    failure_table = pa.Table.from_pylist([{
        "object_id": "-1",
        "object_key": "source#0",
        "exception_type": "ValueError",
        "message": "Too few valid continuum-window pixels.",
    }])
    store = SimpleNamespace(read_table=lambda name, columns: failure_table.select(columns))

    reconciled = module._reconcile_finalized_source_failures(
        selected, store, require_expected_keys=True
    )
    assert reconciled.loc[0, "source_failure_object_key"] == "source#0"
    assert reconciled.loc[0, "source_failure_exception_type"] == "ValueError"
    records = module._fit_one(
        object(), reconciled.iloc[0].to_dict(), ("halpha",),
        qsospec.BroadNarrowMeasurementConfig(), {},
    )
    assert records[0]["fit_status"] == "not_available"
    assert "Too few valid continuum-window pixels" in records[0]["fit_message"]


def test_finalized_source_reconciliation_rejects_unexplained_missing_object():
    module = _load_script()
    selected = pd.DataFrame({
        "object_id": [-1],
        "object_key": [None],
        "qsospec_object_key": ["source#0"],
        "membership_order": [0],
    })
    empty = pa.table({
        "object_id": pa.array([], type=pa.string()),
        "object_key": pa.array([], type=pa.string()),
        "exception_type": pa.array([], type=pa.string()),
        "message": pa.array([], type=pa.string()),
    })
    store = SimpleNamespace(read_table=lambda name, columns: empty.select(columns))

    with pytest.raises(ValueError, match="neither archived nor recorded as failures"):
        module._reconcile_finalized_source_failures(
            selected, store, require_expected_keys=True
        )


def test_override_reader_keeps_only_explicitly_accepted_rows(tmp_path):
    module = _load_script()
    path = tmp_path / "overrides.parquet"
    pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "z_original": [1.0, 1.0, 1.0],
            "z_revised": [0.5, 0.6, 0.7],
            "redshift_revision_reason": ["alias"] * 3,
            "redshift_revision_status": ["candidate_only", "accepted_vi", "accepted_manual"],
            "redshift_revision_source": ["rule", "vi", "review"],
            "alias_rule_version": ["v1"] * 3,
        }
    ).to_parquet(path, index=False)
    accepted = module._read_redshift_overrides(path)
    assert accepted["object_id"].tolist() == [2, 3]


def test_selective_fit_reframes_archived_pixels_and_records_provenance(monkeypatch):
    module = _load_script()
    wave = np.linspace(12000.0, 18000.0, 100)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        np.ones(100),
        err=np.ones(100),
        z=1.0,
        wave_frame="observed",
        flux_unit="relative",
    )
    continuum = GlobalContinuumResult(
        True, 1, "archived", {}, {}, None, 0.0, 1, 0.0,
        spectrum.wave_rest.copy(), np.ones(100), {}, np.ones(100, dtype=bool),
        np.ones(100, dtype=bool),
    )
    loaded = SimpleNamespace(spectrum=spectrum, continuum=continuum)
    monkeypatch.setattr(module, "load_model_by_key", lambda store, key: loaded)
    seen = []

    def fake_measure(reframed, reframed_continuum, complex_name, config):
        seen.append((reframed.z, reframed_continuum.wave_rest.copy()))
        return {"complex_name": complex_name, "fit_status": "complete"}, None

    monkeypatch.setattr(module, "measure_broad_narrow_complex", fake_measure)
    payload = {
        "object_id": -1,
        "object_key": "source#0",
        "membership_order": 0,
        "z_original": 1.0,
        "z_revised": 0.5,
        "redshift_revision_reason": "reviewed alias",
        "redshift_revision_status": "accepted_manual",
        "redshift_revision_source": "review",
        "alias_rule_version": "v1",
    }
    records = module._fit_one(
        object(), payload, ("halpha",), qsospec.BroadNarrowMeasurementConfig(), {}
    )
    assert seen[0][0] == 0.5
    assert np.allclose(seen[0][1], wave / 1.5)
    assert records[0]["adopted_redshift"] == 0.5
    assert records[0]["redshift_source"] == "reviewed_redshift_override"
    assert records[0]["redshift_revision_status"] == "accepted_manual"
