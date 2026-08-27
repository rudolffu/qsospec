"""Narrow Euclid DR1 catalogue bridge for resumable qsospec fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_euclid_dr1_qsospec_input(
    identified: pd.DataFrame,
    candidate_ledger: pd.DataFrame,
    external_spectra: pd.DataFrame,
    *,
    external_survey: str,
    galactic_extinction_corrected: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a one-row-per-external-spectrum qsospec input table."""

    if not external_survey.strip():
        raise ValueError("external_survey must be non-empty")
    for label, frame in (
        ("identified", identified),
        ("candidate ledger", candidate_ledger),
        ("external spectra", external_spectra),
    ):
        if "object_id" not in frame:
            raise KeyError(f"{label} is missing object_id")
        if frame["object_id"].isna().any() or frame["object_id"].duplicated().any():
            raise ValueError(f"{label} object_id values must be non-null and unique")
    required_arrays = {"wavelength", "flux"}
    if not required_arrays <= set(external_spectra):
        raise KeyError(f"external spectra are missing {sorted(required_arrays - set(external_spectra))}")
    uncertainty = next(
        (column for column in ("ivar", "variance", "var") if column in external_spectra), None
    )
    if uncertainty is None:
        raise KeyError("external spectra require ivar, variance, or var")
    working = identified.loc[
        identified["class_final"].astype("string").str.startswith("QSO", na=False)
    ].copy()
    redshift = pd.to_numeric(working["z_final"], errors="coerce")
    working = working.loc[np.isfinite(redshift) & (redshift > 0)].copy()
    ledger_columns = [
        column for column in ("object_id", "selection_tier", "field") if column in candidate_ledger
    ]
    working = working.merge(
        candidate_ledger[ledger_columns], on="object_id", how="inner", validate="one_to_one"
    )
    output = working.merge(external_spectra, on="object_id", how="inner", validate="one_to_one")
    output["object_id"] = output["object_id"].astype("int64")
    output["external_survey"] = external_survey
    if "external_spectrum_id" not in output:
        output["external_spectrum_id"] = output["object_id"].astype("string")
    output["redshift"] = pd.to_numeric(output["z_final"], errors="coerce")
    output["euclid_redshift_provenance"] = output.get("redshift_source", pd.NA)
    output["galactic_extinction_corrected"] = bool(galactic_extinction_corrected)
    if "valid_mask" not in output and "mask" not in output:
        output["valid_mask"] = [
            np.isfinite(np.asarray(flux, dtype=float)) for flux in output["flux"]
        ]
    columns = [
        "object_id", "external_survey", "external_spectrum_id", "ra", "dec",
        "redshift", "wavelength", "flux", uncertainty,
        "galactic_extinction_corrected", "selection_tier", "field",
        "euclid_redshift_provenance",
    ]
    if "valid_mask" in output:
        columns.append("valid_mask")
    elif "mask" in output:
        columns.append("mask")
    result = output.loc[:, [column for column in columns if column in output]].copy()
    summary = {
        "n_identified_input": len(identified),
        "n_qso_positive_redshift_in_candidate_union": len(working),
        "n_external_spectra_input": len(external_spectra),
        "n_output": len(result),
        "n_unmatched_selected": int(len(working) - len(result)),
        "external_survey": external_survey,
        "galactic_extinction_corrected": bool(galactic_extinction_corrected),
        "uncertainty_column": uncertainty,
    }
    return result, summary


def write_bridge_manifest(path: str | Path, summary: dict[str, Any], output: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps({"bridge": "euclid_dr1_qsospec_v1", **summary, "output": str(output)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination
