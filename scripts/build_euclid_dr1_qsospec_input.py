"""Prepare matched external spectra for Euclid DR1 qsospec fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qsospec.catalog_bridge import build_euclid_dr1_qsospec_input, write_bridge_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identified", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--external-spectra", type=Path, required=True)
    parser.add_argument("--external-survey", required=True)
    parser.add_argument("--galactic-extinction-corrected", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, summary = build_euclid_dr1_qsospec_input(
        pd.read_parquet(args.identified),
        pd.read_parquet(args.ledger),
        pd.read_parquet(args.external_spectra),
        external_survey=args.external_survey,
        galactic_extinction_corrected=args.galactic_extinction_corrected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = args.output.with_suffix(".manifest.json")
    write_bridge_manifest(manifest, summary, args.output)
    print(json.dumps({"output": str(args.output), "manifest": str(manifest), **summary}, indent=2))


if __name__ == "__main__":
    main()
