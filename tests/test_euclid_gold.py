from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsospec import fit_batch, open_run
from qsospec.euclid_gold import (
    build_fit_status,
    build_gold_input,
    scientific_configuration,
    select_qa_objects,
    select_smoke_rows,
    strict_gold_mask,
)


def _arrays(offset=0.0):
    wave = np.linspace(12047.4, 18734.0, 500)
    flux = 5.0 + offset + 0.2 * np.sin(wave / 200.0)
    ivar = np.full(500, 4.0)
    valid = np.ones(500, dtype=bool)
    valid[-2:] = False
    ivar[-2:] = np.nan
    return wave, flux, ivar, valid


def _frames():
    rows = []
    members = []
    for index, object_id in enumerate([-4, -3, 2, 7]):
        wave, flux, ivar, valid = _arrays(index)
        rows.append({
            "object_id": object_id, "membership_order": index,
            "selection_tier": "primary", "field": "EDFN",
            "z_final": 1.1 + index * 0.1, "redshift_source": "hybrid",
            "med_snr": 4.0, "n_invalid": 2, "p_uncertain": 0.05,
            "flag_uncertain": 0, "class_final": "QSO_DEFAULT",
            "class_vi": None, "vi_checked": False,
            "catalog_prediction_source": "x", "source_spectra_path": "x",
            "wavelength": wave, "flux": flux, "ivar": ivar,
            "variance": np.where(np.isfinite(ivar), 1 / ivar, np.inf),
            "valid_mask": valid,
        })
        members.append({
            "object_id": object_id, "membership_order": index, "ra": 10.0 + index,
            "dec": -2.0 + index, "z_vi": 1.5 if index == 1 else np.nan,
            "class_vi": "QSO_DEFAULT" if index == 1 else None,
            "vi_checked": index == 1, "z_hybrid": 1.1 + index * 0.1,
            "z_fusion": 1.0, "z_phot": 1.0, "z_pcf_best": 1.0,
            "source_bundle": "bundle", "source_tier_file": "tier",
            "tier": "A", "tier_subtype": "x", "sample_group": "g",
            "domain": "deep", "survey_mode": "RGS", "vi_source": None,
            "vi_source_file": None, "resolved_spectrum_path": "bundle",
        })
    latest = pd.DataFrame([{
        "objid": -4, "class_vi": "GALAXY", "z_vi": 0.8,
        "objname": "a", "targetid": "a", "data_release": "dr1", "qa_flag": 0,
    }])
    return pd.DataFrame(rows), pd.DataFrame(members), latest


def test_strict_selection_boundaries():
    stack, _, _ = _frames()
    assert strict_gold_mask(stack).all()
    for column, value in (
        ("med_snr", 3.0), ("n_invalid", 16), ("p_uncertain", 0.1),
        ("flag_uncertain", 1), ("z_final", 0.0), ("class_final", "GALAXY"),
    ):
        changed = stack.copy()
        changed.loc[0, column] = value
        assert not strict_gold_mask(changed).iloc[0]
    changed = stack.copy()
    changed.loc[0, "n_invalid"] = 15
    assert strict_gold_mask(changed).iloc[0]


def test_build_input_vi_precedence_signed_ids_and_mask():
    stack, membership, latest = _frames()
    output, summary = build_gold_input(stack, membership, latest, enforce_snapshot=False)
    assert output["object_id"].dtype == np.dtype("int64")
    assert output["object_id"].tolist() == [-4, -3, 2, 7]
    by_id = output.set_index("object_id")
    assert by_id.loc[-4, "redshift"] == pytest.approx(0.8)
    assert by_id.loc[-4, "qsospec_redshift_source"] == "latest_specbox_vi"
    assert by_id.loc[-3, "redshift"] == pytest.approx(1.5)
    assert by_id.loc[-3, "qsospec_redshift_source"] == "catalog_vi"
    assert by_id.loc[2, "qsospec_redshift_source"] == "hybrid"
    assert np.array_equal(by_id.loc[-4, "mask"][-2:], [1, 1])
    assert np.array_equal(by_id.loc[-4, "ivar"][-2:], [0.0, 0.0])
    assert summary["common_wavelength_grid"]


def test_build_input_rejects_duplicates_and_inconsistent_grid():
    stack, membership, latest = _frames()
    duplicate = pd.concat([stack, stack.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_gold_input(duplicate, membership, latest, enforce_snapshot=False)
    stack = stack.copy()
    wave = stack.at[2, "wavelength"].copy()
    wave[10] += 0.1
    stack.at[2, "wavelength"] = wave
    with pytest.raises(ValueError, match="inconsistent wavelength"):
        build_gold_input(stack, membership, latest, enforce_snapshot=False)


def test_smoke_selection_is_exact_and_redshift_spanning():
    rows = []
    for index in range(80):
        if index < 16:
            source = "latest_specbox_vi"
            klass = "GALAXY" if index < 12 else "LIKELY_Q"
        elif index < 26:
            source, klass = "catalog_vi", "QSO_DEFAULT"
        elif index < 42:
            source, klass = "latest_specbox_vi", "QSO_DEFAULT"
        else:
            source, klass = "hybrid", None
        rows.append({"object_id": index - 50, "redshift": 0.1 + index / 10, "qsospec_redshift_source": source, "qsospec_vi_class": klass})
    selected = select_smoke_rows(pd.DataFrame(rows))
    assert len(selected) == selected["object_id"].nunique() == 64
    assert (selected["smoke_reason"] == "latest_vi_non_qso_or_likely").sum() == 16
    assert (selected["smoke_reason"] == "catalog_vi_redshift_span").sum() == 8
    assert (selected["smoke_reason"] == "latest_vi_qso_redshift_span").sum() == 8
    assert (selected["smoke_reason"] == "hybrid_redshift_span").sum() == 32


def _kv(mapping):
    import json
    return [{"key": key, "value": json.dumps(value)} for key, value in mapping.items()]


def test_report_hard_bad_and_capped_deterministic_qa():
    expected = pd.DataFrame({
        "object_id": [str(index) for index in range(90)],
        "redshift": np.linspace(0.2, 3.0, 90),
        "qsospec_redshift_source": ["hybrid"] * 90,
    })
    objects = []
    for index in range(89):
        warning = ["at_bound"] if index < 40 else []
        objects.append({
            "object_id": str(index), "continuum_success": index != 0,
            "continuum_reduced_chi2": float(index + 1), "host_decomp_enabled": False,
            "complex_statuses": _kv({"mgii": "failed" if index == 1 else "fit"}),
            "warning_codes": np.asarray(warning, dtype=object),
            "metadata": np.asarray(_kv({"galactic_extinction": {"status": "applied"}}), dtype=object),
        })
    failures = pd.DataFrame([{"object_id": "89", "exception_type": "RuntimeError", "message": "x"}])
    status = build_fit_status(expected, pd.DataFrame(objects), failures)
    assert status["hard_bad"].sum() == 3
    qa1 = select_qa_objects(status)
    qa2 = select_qa_objects(status)
    pd.testing.assert_frame_equal(qa1, qa2)
    assert (qa1["qa_category"] == "warning").sum() <= 25
    assert (qa1["qa_category"] == "good_control").sum() <= 20
    assert qa1["object_id"].is_unique


def test_parquet_fit_has_no_balmer_continuum_or_host(tmp_path: Path):
    wave = np.linspace(12047.4, 18734.0, 500)
    z = 1.5
    rest = wave / (1 + z)
    flux = 4.0 + 0.2 * (rest / 5500.0) ** -1.0
    for center in (4861.3, 5006.8, 6562.8):
        flux += 2.0 * np.exp(-0.5 * ((rest - center) / 12.0) ** 2)
    input_path = tmp_path / "spectra.parquet"
    pd.DataFrame([{
        "object_id": -12, "wavelength": wave, "flux": flux,
        "ivar": np.full(500, 25.0), "mask": np.zeros(500, dtype=np.uint8),
        "redshift": z, "ra": 120.0, "dec": 1.0,
    }]).to_parquet(input_path, index=False)
    config = scientific_configuration(str(tmp_path))
    extinction = replace(config["galactic_extinction_config"], ebv_override=0.0)
    result = fit_batch(
        str(input_path), str(tmp_path / "run"), n_workers=1,
        galactic_extinction_config=extinction,
        global_config=config["global_config"], uncertainty_config=config["uncertainty_config"],
        run_host_decomp=False, retry_failures=False, write_legacy_products=False,
    )
    assert result.n_completed == 1
    store = open_run(str(tmp_path / "run"))
    obj = store.read_table("objects").to_pandas().iloc[0]
    assert not obj["host_decomp_enabled"]
    statuses = {item["key"]: item["value"] for item in obj["complex_statuses"]}
    assert "hbeta_oiii" in statuses and "halpha_nii_sii" in statuses
    model = store.read_table("models").to_pandas().iloc[0]
    continuum_names = [item["name"].lower() for item in model["components"] if item["section"] == "continuum"]
    assert not any("balmer" in name for name in continuum_names)
