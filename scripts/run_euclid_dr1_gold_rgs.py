"""Run or resume the fixed Euclid DR1 identified-gold RGS qsospec fit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from qsospec import fit_batch, open_run
from qsospec.euclid_gold import (
    build_fit_status,
    config_manifest,
    scientific_configuration,
    select_smoke_rows,
    utc_now,
)


RUN_NAMES = {
    "smoke": "smoke_no_balmer_no_host_v1",
    "production": "production_no_balmer_no_host_v1",
}


def default_output_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(RUN_NAMES), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--dustmaps-data-dir",
        type=Path,
        default=Path(os.environ["DUSTMAPS_DATA_DIR"]) if os.environ.get("DUSTMAPS_DATA_DIR") else None,
        help="Dustmaps data directory (or set DUSTMAPS_DATA_DIR).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument("--task-size", type=int, default=8)
    parser.add_argument("--manifest-update-interval", type=int, default=128)
    return parser.parse_args()


def _metadata(items) -> dict:
    output = {}
    if items is None:
        items = ()
    for item in items:
        try:
            output[item["key"]] = json.loads(item["value"])
        except Exception:
            output[item.get("key")] = item.get("value")
    return output


def validate_smoke(run_dir: Path, expected: pd.DataFrame, input_frame: pd.DataFrame) -> dict:
    store = open_run(str(run_dir))
    objects = store.read_table("objects").to_pandas()
    failures = store.read_table("failures").to_pandas()
    warnings = store.read_table("warnings").to_pandas()
    models = store.read_table("models").to_pandas()
    status = build_fit_status(expected, objects, failures)
    input_by_id = input_frame.assign(object_id=input_frame["object_id"].astype(str)).set_index("object_id")
    mask_ok = True
    balmer_components = []
    extinction_bad = []
    host_bad = []
    for model in models.to_dict("records"):
        object_id = str(model["object_id"])
        archived_good = np.asarray(model["input_mask"], dtype=bool)
        expected_good = np.asarray(input_by_id.loc[object_id, "mask"]) == 0
        mask_ok &= np.array_equal(archived_good, expected_good)
        components = model.get("components")
        if components is None:
            components = ()
        balmer_components.extend(
            str(component["name"])
            for component in components
            if component.get("section") == "continuum" and "balmer" in str(component.get("name", "")).lower()
        )
        metadata = _metadata(model.get("workflow_metadata"))
        if (metadata.get("galactic_extinction") or {}).get("status") != "applied":
            extinction_bad.append(object_id)
        if metadata.get("host_decomp_enabled"):
            host_bad.append(object_id)
    sync_warnings = []
    if len(warnings):
        sync_mask = (
            warnings["code"].astype(str).str.contains("balmer.*sync", case=False, regex=True)
            | warnings["message"].astype(str).str.contains("balmer.*sync", case=False, regex=True)
        )
        sync_warnings = warnings.loc[sync_mask, "object_id"].astype(str).tolist()
    accounted = int((status["fit_accounting_status"] != "missing").sum())
    continuum_success = int(status.get("continuum_success", pd.Series(False, index=status.index)).fillna(False).sum())
    complex_names = set()
    for values in objects.get("complex_statuses", ()):
        complex_names.update(_metadata(values))
    checks = {
        "n_expected": int(len(expected)),
        "n_accounted": accounted,
        "n_completed": int(len(objects)),
        "n_failed": int(len(failures)),
        "n_continuum_success": continuum_success,
        "extinction_applied_all_completed": not extinction_bad,
        "mask_roundtrip_exact": bool(mask_ok),
        "host_decomposition_disabled_all": not host_bad and not objects.get("host_decomp_enabled", pd.Series(dtype=bool)).fillna(False).any(),
        "balmer_pseudocontinuum_components_absent": not balmer_components,
        "balmer_sync_warnings_absent": not sync_warnings,
        "hbeta_recipe_available": "hbeta_oiii" in complex_names,
        "halpha_recipe_available": "halpha_nii_sii" in complex_names,
    }
    passed = (
        len(expected) == 64
        and accounted == 64
        and continuum_success >= 60
        and checks["extinction_applied_all_completed"]
        and checks["mask_roundtrip_exact"]
        and checks["host_decomposition_disabled_all"]
        and checks["balmer_pseudocontinuum_components_absent"]
        and checks["balmer_sync_warnings_absent"]
        and checks["hbeta_recipe_available"]
        and checks["halpha_recipe_available"]
    )
    validation = {
        "created_at": utc_now(),
        "passed": bool(passed),
        "checks": checks,
        "bad_object_samples": {
            "extinction": extinction_bad[:10], "host": host_bad[:10],
            "balmer_sync": sync_warnings[:10], "balmer_components": balmer_components[:10],
        },
    }
    (run_dir / "smoke_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return validation


def main() -> None:
    args = parse_args()
    root = args.output_root or default_output_root()
    input_path = args.input or root / "input/spectra.parquet"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.dustmaps_data_dir is None:
        raise RuntimeError("Pass --dustmaps-data-dir or set DUSTMAPS_DATA_DIR")
    if not args.dustmaps_data_dir.exists():
        raise FileNotFoundError(args.dustmaps_data_dir)
    if args.mode == "production":
        free = shutil.disk_usage(root if root.exists() else root.parent).free
        if free < 12 * 1024**3:
            raise RuntimeError(f"Production requires at least 12 GiB free; found {free / 1024**3:.2f} GiB")
        smoke_validation = root / "runs" / RUN_NAMES["smoke"] / "smoke_validation.json"
        if not smoke_validation.exists() or not json.loads(smoke_validation.read_text())["passed"]:
            raise RuntimeError("Production is gated on a passing 64-object smoke_validation.json")

    input_frame = pd.read_parquet(input_path)
    if len(input_frame) != 8_530 or input_frame["object_id"].nunique() != 8_530:
        raise ValueError("Real gold input must contain 8,530 unique objects")
    smoke = select_smoke_rows(input_frame)
    run_dir = root / "runs" / RUN_NAMES[args.mode]
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke.to_parquet(run_dir / "smoke_selection.parquet", index=False)
    row_indices = smoke["_input_row"].astype(int).tolist() if args.mode == "smoke" else None
    invocation = {
        "started_at": utc_now(), "mode": args.mode, "input": str(input_path.resolve()),
        "run_directory": str(run_dir.resolve()), "workers": args.workers,
        "parquet_batch_size": args.parquet_batch_size, "task_size": args.task_size,
        "manifest_update_interval": args.manifest_update_interval,
        "retry_failures": False, "write_legacy_products": False,
        "write_fit_time_qa": False, "scientific_configuration": config_manifest(str(args.dustmaps_data_dir)),
    }
    (run_dir / "invocation.json").write_text(json.dumps(invocation, indent=2, sort_keys=True), encoding="utf-8")
    config = scientific_configuration(str(args.dustmaps_data_dir))
    result = fit_batch(
        str(input_path), str(run_dir), row_indices=row_indices,
        parquet_batch_size=args.parquet_batch_size, task_size=args.task_size,
        n_workers=args.workers, run_host_decomp=False,
        galactic_extinction_config=config["galactic_extinction_config"],
        global_config=config["global_config"], uncertainty_config=config["uncertainty_config"],
        complexes=None, resume=True, retry_failures=False, finalize=True,
        compact_models=False, write_legacy_products=False,
        manifest_update_interval=args.manifest_update_interval,
        show_progress=True,
        progress_total=64 if args.mode == "smoke" else 8_530,
    )
    result_payload = vars(result)
    result_payload["finished_at"] = utc_now()
    if args.mode == "smoke":
        expected = input_frame.iloc[smoke["_input_row"].astype(int)].copy()
        validation = validate_smoke(run_dir, expected, input_frame)
        result_payload["smoke_validation"] = validation
        if not validation["passed"]:
            (run_dir / "result.json").write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
            raise RuntimeError(f"Smoke validation failed: {validation['checks']}")
    (run_dir / "result.json").write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
