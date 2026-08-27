"""Reproducible Euclid DR1 identified-gold input and reporting helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    BalmerPseudoContinuumConfig,
    GalacticExtinctionConfig,
    GlobalContinuumConfig,
    PowerLawConfig,
    UncertaintyConfig,
)


GOLD_EXPECTED_ROWS = 8_530
EXPECTED_REDSHIFT_COUNTS = {
    "latest_specbox_vi": 688,
    "catalog_vi": 98,
    "hybrid": 7_744,
}
EXPECTED_LATEST_VI_CLASSES = {
    "confirmed_qso": 672,
    "GALAXY": 12,
    "LIKELY_Q": 4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def strict_gold_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the strict production-QC gold selection with strict bounds."""

    required = {
        "class_final",
        "z_final",
        "med_snr",
        "n_invalid",
        "p_uncertain",
        "flag_uncertain",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Gold input is missing selection columns: {missing}")
    z = pd.to_numeric(frame["z_final"], errors="coerce")
    snr = pd.to_numeric(frame["med_snr"], errors="coerce")
    n_invalid = pd.to_numeric(frame["n_invalid"], errors="coerce")
    p_uncertain = pd.to_numeric(frame["p_uncertain"], errors="coerce")
    flag_uncertain = pd.to_numeric(frame["flag_uncertain"], errors="coerce")
    return (
        frame["class_final"].astype("string").str.startswith("QSO", na=False)
        & np.isfinite(z)
        & (z > 0)
        & np.isfinite(snr)
        & (snr > 3)
        & np.isfinite(n_invalid)
        & (n_invalid <= 15)
        & np.isfinite(p_uncertain)
        & (p_uncertain < 0.1)
        & np.isfinite(flag_uncertain)
        & (flag_uncertain == 0)
    )


def _as_int64_ids(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError(f"{label} contains missing or non-integral object IDs")
    return numeric.astype("int64")


def _assert_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    duplicates = frame.loc[frame[column].duplicated(keep=False), column]
    if len(duplicates):
        raise ValueError(
            f"{label} contains duplicate {column} values: "
            f"{duplicates.astype(str).head(10).tolist()}"
        )


def _validated_arrays(row: Mapping[str, Any], expected_wave: np.ndarray | None):
    wave = np.asarray(row["wavelength"], dtype=float)
    flux = np.asarray(row["flux"], dtype=float)
    ivar = np.asarray(row["ivar"], dtype=float)
    valid = np.asarray(row["valid_mask"], dtype=bool)
    if not (wave.ndim == flux.ndim == ivar.ndim == valid.ndim == 1):
        raise ValueError(f"Object {row['object_id']} has non-vector spectral arrays")
    if not (len(wave) == len(flux) == len(ivar) == len(valid) == 500):
        raise ValueError(f"Object {row['object_id']} does not have 500 aligned bins")
    if expected_wave is not None and not np.array_equal(wave, expected_wave):
        raise ValueError(f"Object {row['object_id']} has an inconsistent wavelength grid")
    if not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError(f"Object {row['object_id']} has an invalid wavelength grid")
    if np.any(valid & (~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0))):
        raise ValueError(
            f"Object {row['object_id']} marks non-finite/non-positive data as valid"
        )
    output_ivar = np.where(valid, ivar, 0.0).astype(float)
    mask = np.where(valid, 0, 1).astype("uint8")
    return wave, flux, output_ivar, mask


def build_gold_input(
    stack: pd.DataFrame,
    membership: pd.DataFrame,
    latest_vi: pd.DataFrame,
    *,
    enforce_snapshot: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the reader-compatible DR1 gold RGS table and audit summary."""

    stack = stack.copy()
    membership = membership.copy()
    latest_vi = latest_vi.copy()
    for frame, column, label in (
        (stack, "object_id", "gold stack"),
        (membership, "object_id", "gold membership"),
        (latest_vi, "objid", "latest SpecBox VI"),
    ):
        frame[column] = _as_int64_ids(frame[column], label)
        _assert_unique(frame, column, label)

    if len(stack) != len(membership):
        raise ValueError("Gold stack and membership row counts differ")
    if set(stack["object_id"]) != set(membership["object_id"]):
        raise ValueError("Gold stack and membership object sets differ")
    if not strict_gold_mask(stack).all():
        rejected = stack.loc[~strict_gold_mask(stack), "object_id"].head(10).tolist()
        raise ValueError(f"Gold stack contains rows outside strict selection: {rejected}")
    if enforce_snapshot and len(stack) != GOLD_EXPECTED_ROWS:
        raise ValueError(f"Expected {GOLD_EXPECTED_ROWS} gold rows, found {len(stack)}")

    member_columns = [
        "object_id", "membership_order", "ra", "dec", "z_vi", "class_vi",
        "vi_checked", "z_hybrid", "z_fusion", "z_phot", "z_pcf_best",
        "source_bundle", "source_tier_file", "tier", "tier_subtype",
        "sample_group", "domain", "survey_mode", "vi_source", "vi_source_file",
        "resolved_spectrum_path",
    ]
    missing_member = [name for name in member_columns if name not in membership]
    if missing_member:
        raise KeyError(f"Membership is missing audit columns: {missing_member}")
    member = membership[member_columns].rename(
        columns={
            "membership_order": "membership_order_member",
            "z_vi": "z_vi_catalog",
            "class_vi": "class_vi_catalog",
            "vi_checked": "vi_checked_catalog",
        }
    )
    joined = stack.merge(member, on="object_id", how="left", validate="one_to_one")
    if joined["ra"].isna().any() or joined["dec"].isna().any():
        raise ValueError("Gold membership is missing coordinates")
    if not np.array_equal(
        joined["membership_order"].to_numpy(),
        joined["membership_order_member"].to_numpy(),
    ):
        raise ValueError("Gold stack and membership order columns disagree")

    vi_columns = ["objid", "class_vi", "z_vi", "objname", "targetid", "data_release", "qa_flag"]
    missing_vi = [name for name in vi_columns if name not in latest_vi]
    if missing_vi:
        raise KeyError(f"Latest SpecBox VI is missing columns: {missing_vi}")
    vi = latest_vi[vi_columns].rename(
        columns={
            "objid": "object_id",
            "class_vi": "class_vi_latest",
            "z_vi": "z_vi_latest",
            "objname": "vi_objname_latest",
            "targetid": "vi_targetid_latest",
            "data_release": "vi_data_release_latest",
            "qa_flag": "vi_qa_flag_latest",
        }
    )
    if not set(vi["object_id"]).issubset(set(joined["object_id"])):
        unknown = sorted(set(vi["object_id"]) - set(joined["object_id"]))[:10]
        raise ValueError(f"Latest SpecBox VI contains objects outside gold: {unknown}")
    joined = joined.merge(vi, on="object_id", how="left", validate="one_to_one")

    new_vi = joined["z_vi_latest"].notna()
    catalog_vi = (
        ~new_vi
        & joined["vi_checked_catalog"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(joined["z_vi_catalog"], errors="coerce"))
    )
    if (new_vi & catalog_vi).any():
        raise ValueError("Latest and catalog VI groups overlap")
    joined["redshift"] = pd.to_numeric(joined["z_final"], errors="coerce")
    joined["qsospec_redshift_source"] = "hybrid"
    joined.loc[catalog_vi, "redshift"] = pd.to_numeric(
        joined.loc[catalog_vi, "z_vi_catalog"], errors="coerce"
    )
    joined.loc[catalog_vi, "qsospec_redshift_source"] = "catalog_vi"
    joined.loc[new_vi, "redshift"] = pd.to_numeric(
        joined.loc[new_vi, "z_vi_latest"], errors="coerce"
    )
    joined.loc[new_vi, "qsospec_redshift_source"] = "latest_specbox_vi"
    if not (np.isfinite(joined["redshift"]) & (joined["redshift"] > 0)).all():
        raise ValueError("Adopted redshifts must all be finite and positive")

    joined["qsospec_vi_class"] = joined["class_vi_latest"].where(
        new_vi, joined["class_vi_catalog"].where(catalog_vi)
    )
    joined["vi_review_group"] = np.select(
        [new_vi, catalog_vi], ["latest_specbox_vi", "catalog_vi"], default="unreviewed"
    )
    joined["vi_class_mismatch"] = new_vi & joined["class_vi_latest"].isin(
        ["GALAXY", "LIKELY_Q"]
    )

    expected_wave: np.ndarray | None = None
    wave_values: list[np.ndarray] = []
    flux_values: list[np.ndarray] = []
    ivar_values: list[np.ndarray] = []
    mask_values: list[np.ndarray] = []
    for row in joined.to_dict("records"):
        wave, flux, ivar, mask = _validated_arrays(row, expected_wave)
        if expected_wave is None:
            expected_wave = wave.copy()
        wave_values.append(wave)
        flux_values.append(flux)
        ivar_values.append(ivar)
        mask_values.append(mask)
    joined["wavelength"] = wave_values
    joined["flux"] = flux_values
    joined["ivar"] = ivar_values
    joined["mask"] = mask_values
    joined["ra"] = pd.to_numeric(joined["ra"], errors="raise").astype(float)
    joined["dec"] = pd.to_numeric(joined["dec"], errors="raise").astype(float)

    drop_columns = ["membership_order_member", "valid_mask", "variance"]
    output = joined.drop(columns=[name for name in drop_columns if name in joined])
    required_first = ["object_id", "wavelength", "flux", "ivar", "mask", "redshift", "ra", "dec"]
    output = output[required_first + [name for name in output if name not in required_first]]
    output = output.sort_values(["membership_order", "object_id"], kind="stable").reset_index(drop=True)
    _assert_unique(output, "object_id", "qsospec input")

    redshift_counts = output["qsospec_redshift_source"].value_counts().sort_index().to_dict()
    new_classes = output.loc[output["qsospec_redshift_source"] == "latest_specbox_vi", "qsospec_vi_class"]
    latest_class_counts = {
        "confirmed_qso": int(new_classes.astype(str).str.startswith("QSO").sum()),
        "GALAXY": int((new_classes == "GALAXY").sum()),
        "LIKELY_Q": int((new_classes == "LIKELY_Q").sum()),
    }
    if enforce_snapshot and redshift_counts != EXPECTED_REDSHIFT_COUNTS:
        raise ValueError(f"Unexpected redshift-source counts: {redshift_counts}")
    if enforce_snapshot and latest_class_counts != EXPECTED_LATEST_VI_CLASSES:
        raise ValueError(f"Unexpected latest-VI class counts: {latest_class_counts}")
    summary = {
        "n_rows": int(len(output)),
        "n_unique_objects": int(output["object_id"].nunique()),
        "redshift_source_counts": redshift_counts,
        "latest_vi_class_counts": latest_class_counts,
        "vi_groups_disjoint": True,
        "n_wavelength_bins": int(len(expected_wave)) if expected_wave is not None else 0,
        "wavelength_min_angstrom": float(expected_wave[0]) if expected_wave is not None else None,
        "wavelength_max_angstrom": float(expected_wave[-1]) if expected_wave is not None else None,
        "common_wavelength_grid": True,
        "mask_convention": "0=usable, 1=invalid; converted from production valid_mask",
        "invalid_ivar_policy": "set to zero wherever mask != 0",
        "selection": (
            "class_final starts with QSO; finite z_final > 0; med_snr > 3; "
            "production n_invalid <= 15; p_uncertain < 0.1; flag_uncertain == 0"
        ),
    }
    return output, summary


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> list[int]:
    """Choose deterministic redshift-spanning row indices."""

    if count < 0 or len(frame) < count:
        raise ValueError(f"Cannot choose {count} rows from a frame of length {len(frame)}")
    if count == 0:
        return []
    ordered = frame.sort_values(["redshift", "object_id", "_input_row"], kind="stable")
    positions = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    if len(np.unique(positions)) != count:
        raise RuntimeError("Evenly spaced selector produced duplicate positions")
    return ordered.iloc[positions]["_input_row"].astype(int).tolist()


def select_smoke_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the fixed 64-object validation mixture from the real input."""

    work = frame.reset_index(drop=True).copy()
    work["_input_row"] = np.arange(len(work), dtype=int)
    latest = work[work["qsospec_redshift_source"] == "latest_specbox_vi"]
    mismatch = latest[latest["qsospec_vi_class"].isin(["GALAXY", "LIKELY_Q"])]
    catalog = work[work["qsospec_redshift_source"] == "catalog_vi"]
    latest_qso = latest[latest["qsospec_vi_class"].astype(str).str.startswith("QSO")]
    hybrid = work[work["qsospec_redshift_source"] == "hybrid"]
    if len(mismatch) != 16:
        raise ValueError(f"Expected 16 newly reviewed non-QSO/likely rows, found {len(mismatch)}")
    selections = [(int(row), "latest_vi_non_qso_or_likely") for row in mismatch["_input_row"]]
    selections += [(row, "catalog_vi_redshift_span") for row in evenly_spaced_rows(catalog, 8)]
    selections += [(row, "latest_vi_qso_redshift_span") for row in evenly_spaced_rows(latest_qso, 8)]
    selections += [(row, "hybrid_redshift_span") for row in evenly_spaced_rows(hybrid, 32)]
    if len(selections) != 64 or len({row for row, _ in selections}) != 64:
        raise RuntimeError("Smoke selection is not 64 unique objects")
    reason = dict(selections)
    result = work.loc[sorted(reason)].copy()
    result["smoke_reason"] = result["_input_row"].map(reason)
    return result[["_input_row", "object_id", "redshift", "qsospec_redshift_source", "qsospec_vi_class", "smoke_reason"]]


def scientific_configuration(dustmaps_data_dir: str) -> dict[str, Any]:
    """Return the immutable first-pass scientific configuration."""

    global_config = GlobalContinuumConfig(
        power_law=PowerLawConfig(mode="single"),
        balmer_pseudocontinuum=BalmerPseudoContinuumConfig(
            enabled=False,
            fit_fwhm=False,
            sync_with_hbeta="never",
            sync_with_hgamma="never",
        ),
    )
    extinction = GalacticExtinctionConfig(
        enabled=True,
        map_name="planck",
        law="f99",
        wavelength_out_of_range="raise",
        rv=3.1,
        dustmaps_data_dir=str(Path(dustmaps_data_dir).expanduser()),
    )
    uncertainty = UncertaintyConfig(
        covariance=True,
        monte_carlo_trials=0,
        random_seed=12345,
        refit_host_in_mc=False,
    )
    return {
        "global_config": global_config,
        "galactic_extinction_config": extinction,
        "uncertainty_config": uncertainty,
        "run_host_decomp": False,
        "complexes": None,
    }


def _decode_key_values(items: Any) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    if items is None:
        items = ()
    for item in items:
        try:
            decoded[str(item["key"])] = json.loads(item["value"])
        except Exception:
            decoded[str(item.get("key"))] = item.get("value")
    return decoded


def _as_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, float) and np.isnan(values):
        return []
    return list(values)


def build_fit_status(
    expected: pd.DataFrame,
    objects: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile one run with its expected input and classify hard failures."""

    expected = expected.reset_index(drop=True).copy()
    expected["object_id"] = expected["object_id"].astype(str)
    objects = objects.copy()
    failures = failures.copy()
    if len(objects):
        objects["object_id"] = objects["object_id"].astype(str)
        _assert_unique(objects, "object_id", "completed run objects")
    if len(failures):
        failures["object_id"] = failures["object_id"].astype(str)
        _assert_unique(failures, "object_id", "run failures")
    overlap = set(objects.get("object_id", ())) & set(failures.get("object_id", ()))
    if overlap:
        raise ValueError(f"Objects occur in both completed and failed tables: {sorted(overlap)[:10]}")

    object_by_id = objects.set_index("object_id").to_dict("index") if len(objects) else {}
    failure_by_id = failures.set_index("object_id").to_dict("index") if len(failures) else {}
    rows: list[dict[str, Any]] = []
    for item in expected.to_dict("records"):
        object_id = str(item["object_id"])
        row = dict(item)
        completed = object_by_id.get(object_id)
        failed = failure_by_id.get(object_id)
        reasons: list[str] = []
        if failed is not None:
            reasons.append("batch_exception")
        elif completed is None:
            reasons.append("not_accounted")
        else:
            chi2 = completed.get("continuum_reduced_chi2")
            if not bool(completed.get("continuum_success")) or not np.isfinite(chi2):
                reasons.append("continuum_failed_or_nonfinite")
            statuses = _decode_key_values(completed.get("complex_statuses"))
            failed_complexes = sorted(key for key, value in statuses.items() if value == "failed")
            reasons.extend(f"complex_failed:{name}" for name in failed_complexes)
            row["complex_statuses_json"] = json.dumps(statuses, sort_keys=True)
            row["warning_codes"] = _as_list(completed.get("warning_codes"))
            row["continuum_success"] = bool(completed.get("continuum_success"))
            row["continuum_reduced_chi2"] = float(chi2)
            row["host_decomp_enabled"] = bool(completed.get("host_decomp_enabled"))
            metadata = _decode_key_values(completed.get("metadata"))
            extinction = metadata.get("galactic_extinction") or {}
            row["galactic_extinction_status"] = extinction.get("status")
        row["fit_accounting_status"] = (
            "failed" if failed is not None else "completed" if completed is not None else "missing"
        )
        row["hard_bad"] = bool(reasons)
        row["hard_bad_reasons"] = reasons
        row["failure_exception_type"] = failed.get("exception_type") if failed else None
        row["failure_message"] = failed.get("message") if failed else None
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(expected) or result["object_id"].duplicated().any():
        raise RuntimeError("Fit status does not preserve one row per expected object")
    return result


def select_qa_objects(status: pd.DataFrame, *, random_seed: int = 12345) -> pd.DataFrame:
    """Apply the deterministic, priority-deduplicated deferred-QA policy."""

    selected: dict[str, dict[str, Any]] = {}

    def add(rows: pd.DataFrame, category: str, detail: str | None = None) -> None:
        for record in rows.to_dict("records"):
            key = str(record["object_id"])
            if key not in selected:
                selected[key] = {
                    "object_id": key,
                    "qa_category": category,
                    "qa_detail": detail,
                    "continuum_reduced_chi2": record.get("continuum_reduced_chi2"),
                    "redshift": record.get("redshift"),
                    "qsospec_redshift_source": record.get("qsospec_redshift_source"),
                    "fit_accounting_status": record.get("fit_accounting_status"),
                }

    add(status[status["hard_bad"]], "hard_bad")
    completed = status[(status["fit_accounting_status"] == "completed") & ~status["hard_bad"]].copy()
    completed["_chi2"] = pd.to_numeric(completed["continuum_reduced_chi2"], errors="coerce").fillna(-np.inf)
    warning_codes = sorted({code for values in completed["warning_codes"] for code in _as_list(values)})
    for code in warning_codes:
        rows = completed[completed["warning_codes"].map(lambda values: code in _as_list(values))]
        rows = rows.sort_values(["_chi2", "object_id"], ascending=[False, True], kind="stable").head(25)
        add(rows, "warning", code)
    clean = completed[completed["warning_codes"].map(lambda values: len(_as_list(values)) == 0)]
    clean = clean.sort_values(["_chi2", "object_id"], ascending=[False, True], kind="stable")
    add(clean.head(25), "chi2_tail")

    remaining = clean[~clean["object_id"].astype(str).isin(selected)].copy()
    controls: list[pd.DataFrame] = []
    if len(remaining):
        remaining["_zbin"] = pd.qcut(
            remaining["redshift"], q=min(5, len(remaining)), labels=False, duplicates="drop"
        )
        groups = list(remaining.groupby(["qsospec_redshift_source", "_zbin"], dropna=False, sort=True))
        rng = np.random.default_rng(random_seed)
        candidates = []
        for group_key, group in groups:
            ordered = group.sort_values("object_id", kind="stable")
            candidates.append(ordered.iloc[int(rng.integers(0, len(ordered)))])
        if candidates:
            controls.append(pd.DataFrame(candidates))
        already = pd.concat(controls, ignore_index=True) if controls else remaining.iloc[0:0]
        needed = 20 - len(already)
        if needed > 0:
            pool = remaining[~remaining["object_id"].astype(str).isin(already["object_id"].astype(str))]
            if len(pool):
                chosen = np.sort(rng.choice(len(pool), size=min(needed, len(pool)), replace=False))
                controls.append(pool.iloc[chosen])
        if controls:
            add(pd.concat(controls, ignore_index=True).head(20), "good_control")
    result = pd.DataFrame(selected.values())
    category_order = {"hard_bad": 0, "warning": 1, "chi2_tail": 2, "good_control": 3}
    if len(result):
        result["_category_order"] = result["qa_category"].map(category_order)
        result = result.sort_values(["_category_order", "object_id"], kind="stable").drop(columns="_category_order").reset_index(drop=True)
    return result


def count_complex_statuses(objects: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[tuple[str, str]] = Counter()
    for values in objects.get("complex_statuses", ()):
        for name, status in _decode_key_values(values).items():
            counter[(str(name), str(status))] += 1
    return pd.DataFrame(
        [{"recipe_id": key[0], "status": key[1], "count": value} for key, value in sorted(counter.items())]
    )


def count_warnings(warnings: pd.DataFrame) -> pd.DataFrame:
    if warnings.empty:
        return pd.DataFrame(columns=["code", "severity", "count"])
    return (
        warnings.groupby(["code", "severity"], dropna=False).size().rename("count").reset_index()
        .sort_values(["count", "code"], ascending=[False, True], kind="stable")
    )


def config_manifest(dustmaps_data_dir: str) -> dict[str, Any]:
    config = scientific_configuration(dustmaps_data_dir)
    return {
        "global_config": asdict(config["global_config"]),
        "galactic_extinction_config": asdict(config["galactic_extinction_config"]),
        "uncertainty_config": asdict(config["uncertainty_config"]),
        "run_host_decomp": False,
        "complexes": None,
    }
