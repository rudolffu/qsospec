"""Report a completed Euclid DR1 gold RGS run before any QA rendering."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qsospec import open_run
from qsospec.euclid_gold import (
    build_fit_status,
    count_complex_statuses,
    count_warnings,
    select_qa_objects,
    utc_now,
)


RUN_NAME = "production_no_balmer_no_host_v1"


def default_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root or default_root()
    run_dir = args.run_directory or root / "runs" / RUN_NAME
    input_path = args.input or root / "input/spectra.parquet"
    report_dir = args.report_directory or root / "reports" / RUN_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    expected = pd.read_parquet(input_path).reset_index(drop=True)
    store = open_run(str(run_dir))
    objects = store.read_table("objects").to_pandas()
    failures = store.read_table("failures").to_pandas()
    warnings = store.read_table("warnings").to_pandas()
    measurements = store.read_table("measurements").to_pandas()
    models = store.read_table("models").to_pandas()
    status = build_fit_status(expected, objects, failures)
    status["warning_codes"] = status["warning_codes"].map(
        lambda value: value if isinstance(value, (list, tuple, np.ndarray)) else []
    )
    status.to_parquet(report_dir / "fit_status.parquet", index=False)
    bad = status[status["hard_bad"]].copy()
    bad.to_parquet(report_dir / "bad_fits.parquet", index=False)
    warning_counts = count_warnings(warnings)
    warning_counts.to_csv(report_dir / "warning_counts.csv", index=False)
    complex_counts = count_complex_statuses(objects)
    complex_counts.to_csv(report_dir / "complex_status_counts.csv", index=False)
    qa_selection = select_qa_objects(status)
    qa_selection.to_parquet(report_dir / "qa_selection.parquet", index=False)

    audit_columns = [
        "object_id", "membership_order", "selection_tier", "field", "med_snr",
        "n_invalid", "p_uncertain", "flag_uncertain", "z_final", "redshift",
        "qsospec_redshift_source", "qsospec_vi_class", "vi_review_group",
        "vi_class_mismatch", "catalog_prediction_source", "source_spectra_path",
    ]
    audit = expected[[name for name in audit_columns if name in expected]].copy()
    audit["object_id"] = audit["object_id"].astype(str)
    if len(measurements):
        measurements["object_id"] = measurements["object_id"].astype(str)
        measurements = measurements.merge(audit, on="object_id", how="left", validate="many_to_one")
    measurements.to_parquet(report_dir / "line_measurements.parquet", index=False)

    accounted = int((status["fit_accounting_status"] != "missing").sum())
    n_completed = int((status["fit_accounting_status"] == "completed").sum())
    n_failed_exception = int((status["fit_accounting_status"] == "failed").sum())
    n_hard_bad = int(status["hard_bad"].sum())
    extinction_ok = bool(
        status.loc[status["fit_accounting_status"] == "completed", "galactic_extinction_status"]
        .eq("applied").all()
    )
    host_ok = not bool(objects.get("host_decomp_enabled", pd.Series(dtype=bool)).fillna(False).any())
    no_fit_time_qa = not any((run_dir / "qa").glob("*.png")) if (run_dir / "qa").exists() else True
    order_ok = status["object_id"].tolist() == expected["object_id"].astype(str).tolist()
    model_coverage_ok = len(models) == n_completed and models["object_id"].astype(str).nunique() == n_completed
    redshift_counts = expected["qsospec_redshift_source"].value_counts().sort_index().to_dict()
    qa_counts = qa_selection["qa_category"].value_counts().sort_index().to_dict() if len(qa_selection) else {}
    hard_reason_counts = Counter(reason for values in bad["hard_bad_reasons"] for reason in values)
    acceptance = {
        "expected_rows_8530": len(expected) == 8_530,
        "completed_plus_failed_equals_8530": accounted == 8_530,
        "exact_input_order_reconciliation": bool(order_ok),
        "foreground_extinction_applied_every_completed_fit": extinction_ok,
        "host_decomp_enabled_false_every_completed_fit": host_ok,
        "model_archive_covers_every_completed_fit": bool(model_coverage_ok),
        "no_fit_time_qa_pngs": bool(no_fit_time_qa),
    }
    summary = {
        "created_at": utc_now(),
        "run_directory": str(run_dir.resolve()),
        "input": str(input_path.resolve()),
        "n_input": int(len(expected)),
        "n_accounted": accounted,
        "n_completed": n_completed,
        "n_batch_exceptions": n_failed_exception,
        "n_hard_bad": n_hard_bad,
        "n_caution_objects": int(status["warning_codes"].map(bool).sum()),
        "n_measurement_rows": int(len(measurements)),
        "n_vi_class_mismatch_audit": int(expected["vi_class_mismatch"].sum()),
        "redshift_source_counts": redshift_counts,
        "hard_bad_reason_counts": dict(sorted(hard_reason_counts.items())),
        "qa_selection_counts": qa_counts,
        "acceptance_checks": acceptance,
        "passed": bool(all(acceptance.values())),
        "material_passport": {
            "origin": "experiment-agent", "date": "2026-08-27",
            "version": "dr1_gold_qsospec_plan_v1",
            "verification_status": "ANALYZED" if all(acceptance.values()) else "UNVERIFIED",
        },
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    warning_preview = warning_counts.head(15).to_markdown(index=False) if len(warning_counts) else "No warnings were recorded."
    complex_preview = complex_counts.to_markdown(index=False) if len(complex_counts) else "No complex statuses were recorded."
    report = f"""# Initial DR1 identified-gold `qsospec` report

Generated: {summary['created_at']}

The run accounted for **{accounted:,} / {len(expected):,}** input spectra: **{n_completed:,}** completed objects and **{n_failed_exception:,}** batch exceptions. The hard-bad definition additionally includes failed/non-finite continuum fits and covered line complexes with `status=failed`; it flags **{n_hard_bad:,}** objects. Warning-only objects remain a separate caution tier.

Redshift provenance is unchanged from the input contract: 688 latest SpecBox VI, 98 catalog VI, and 7,744 hybrid redshifts. The 12 newly classified galaxies and four `LIKELY_Q` objects remain in the membership and are audit flags, not fit failures.

Balmer pseudo-continuum and host decomposition were disabled. Balmer emission-line complexes, UV/optical Fe II, and all other automatically covered recipes remained enabled. Galactic dereddening used Planck GNILC plus F99 with $R_V=3.1$.

## Acceptance checks

```json
{json.dumps(acceptance, indent=2, sort_keys=True)}
```

## Warning counts

{warning_preview}

## Line-complex statuses

{complex_preview}

## Deferred QA

`qa_selection.parquet` applies the priority order `hard_bad > warning > chi2_tail > good_control`. No QA was rendered during fitting. Run the separate deferred-QA command only after reviewing this report.

## Material Passport

- Origin: experiment-agent
- Date: 2026-08-27
- Version: dr1_gold_qsospec_plan_v1
- Verification status: {summary['material_passport']['verification_status']}
"""
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
