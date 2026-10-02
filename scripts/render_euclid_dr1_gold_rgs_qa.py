"""Render only the deferred QA selection from the DR1 gold report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from qsospec import GlobalQAPlotConfig, render_qa
from qsospec.euclid_gold import utc_now


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
    parser.add_argument("--report-directory", type=Path)
    parser.add_argument("--qa-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root or default_root()
    run_dir = args.run_directory or root / "runs" / RUN_NAME
    report_dir = args.report_directory or root / "reports" / RUN_NAME
    qa_dir = args.qa_directory or root / "qa" / RUN_NAME
    selection_path = report_dir / "qa_selection.parquet"
    summary_path = report_dir / "summary.json"
    if not selection_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run report_euclid_dr1_gold_rgs.py before rendering QA")
    selection = pd.read_parquet(selection_path)
    qa_dir.mkdir(parents=True, exist_ok=True)
    config = GlobalQAPlotConfig(output_format="png", write_other_diagnostics=False)
    outputs = {}
    for category in ("hard_bad", "warning", "chi2_tail", "good_control"):
        rows = selection[selection["qa_category"] == category]
        if rows.empty:
            continue
        destination = qa_dir / category
        object_ids = rows["object_id"].astype(str).tolist()
        rendered = render_qa(
            str(run_dir), object_ids=object_ids,
            include_failed=(category == "hard_bad"), plot_config=config,
            output_dir=str(destination),
        )
        outputs[category] = {"n_selected": len(object_ids), "n_rendered": len(rendered), "directory": str(destination)}
    manifest = {
        "created_at": utc_now(), "run_directory": str(run_dir.resolve()),
        "selection": str(selection_path.resolve()), "categories": outputs,
        "policy": "hard_bad > warning > chi2_tail > good_control; pre-deduplicated by report",
    }
    (qa_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
