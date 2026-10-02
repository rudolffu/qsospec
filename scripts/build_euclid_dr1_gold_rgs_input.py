"""Build the 8,530-object Euclid DR1 identified-gold RGS qsospec input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from qsospec.euclid_gold import build_gold_input, git_commit, sha256_file, utc_now


def default_output_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    composite_root = os.environ.get("EUCLID_COMPOSITE_ROOT")
    source = (
        Path(composite_root) / "data/mlspecz/dr1_identified_gold_v1"
        if composite_root
        else None
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-stack", type=Path, default=source / "gold_stack_input.parquet" if source else None)
    parser.add_argument("--gold-membership", type=Path, default=source / "gold_membership.parquet" if source else None)
    parser.add_argument(
        "--latest-vi", type=Path,
        default=source / "vi_type2_candidates_unchecked_specbox_results.csv" if source else None,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def schema_hash(path: Path) -> str:
    return hashlib.sha256(str(pq.read_schema(path)).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    composite_root = os.environ.get("EUCLID_COMPOSITE_ROOT")
    missing = [
        name
        for name in ("gold_stack", "gold_membership", "latest_vi")
        if getattr(args, name) is None
    ]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise RuntimeError(
            f"Pass {options}, or set EUCLID_COMPOSITE_ROOT to provide their defaults"
        )
    output_root = args.output_root or default_output_root()
    input_dir = output_root / "input"
    output_path = input_dir / "spectra.parquet"
    manifest_path = input_dir / "manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {input_dir}")
    for path in (args.gold_stack, args.gold_membership, args.latest_vi):
        if not path.exists():
            raise FileNotFoundError(path)

    output, summary = build_gold_input(
        pd.read_parquet(args.gold_stack),
        pd.read_parquet(args.gold_membership),
        pd.read_csv(args.latest_vi),
        enforce_snapshot=True,
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    output.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    inputs = {}
    for label, path in (
        ("gold_stack", args.gold_stack),
        ("gold_membership", args.gold_membership),
        ("latest_specbox_vi", args.latest_vi),
    ):
        inputs[label] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "created_at": utc_now(),
        "workflow": "dr1_identified_gold_rgs_qsospec_input_v1",
        "material_passport": {
            "origin": "experiment-agent",
            "date": "2026-08-27",
            "version": "dr1_gold_qsospec_plan_v1",
            "verification_status": "INPUT_VALIDATED",
        },
        "inputs": inputs,
        "output": {
            "path": str(output_path.resolve()),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "parquet_rows": pq.read_metadata(output_path).num_rows,
            "schema": str(pq.read_schema(output_path)),
            "schema_sha256": schema_hash(output_path),
        },
        "repository_commits": {
            "qsospec": git_commit(Path(__file__).resolve().parents[1]),
            "euclidqso_composite": git_commit(Path(composite_root)) if composite_root else None,
        },
        **summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "manifest": str(manifest_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
