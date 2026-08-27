"""Render only selected QA for the provisional DR1 narrow-line workflow."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# QA rendering runs many independent nonlinear fits.  Keep numerical backends
# single-threaded so ``--workers`` is also a useful upper bound on CPU use.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qsospec

RUN_NAME = "production_no_balmer_no_host_v1"
CLASSIFICATION_NAME = "narrow_line_r480_v2"
_WORKER_STORE = None


def default_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--classification-directory", type=Path)
    parser.add_argument("--qa-selection", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resolving-power", type=float, default=480.0)
    parser.add_argument(
        "--complexes",
        nargs="+",
        choices=("halpha", "hei_pgamma"),
        help="Optionally render only selected complexes.",
    )
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _initialize_worker(run_directory: str) -> None:
    global _WORKER_STORE
    _WORKER_STORE = qsospec.open_run(run_directory)


def _qa_payload(row: dict[str, object]) -> dict[str, object]:
    object_id = str(row["object_id"])
    complex_name = str(row["complex_name"])
    output = {**row, "object_id": object_id, "complex_name": complex_name}
    try:
        resolving_power = float(row.get("resolving_power", 480.0))
        lsf = qsospec.LineSpreadFunctionConfig(
            resolving_power=resolving_power
        )
        loaded = qsospec.load_model(_WORKER_STORE, object_id)
        if complex_name == "halpha":
            result = qsospec.fit_halpha_model_grid(
                loaded.spectrum,
                loaded.continuum,
                selection_config=qsospec.HalphaModelSelectionConfig(lsf=lsf),
            )
            n0 = result.narrow
            b1 = result.best_broad
            window = (6250.0, 6800.0)
        elif complex_name == "hei_pgamma":
            result = qsospec.fit_hei_pgamma_model_pair(
                loaded.spectrum,
                loaded.continuum,
                selection_config=qsospec.HeIPagammaModelSelectionConfig(
                    lsf=lsf
                ),
            )
            if result is None:
                raise RuntimeError("He I/Pa-gamma is not covered")
            n0, b1 = result.n0, result.b1
            window = qsospec.HEI_PGAMMA_WINDOW
        else:
            raise ValueError(f"Unknown complex: {complex_name}")
        if b1 is None:
            raise RuntimeError("B1 result is unavailable")
        mask = (
            (n0.wave_rest >= window[0])
            & (n0.wave_rest <= window[1])
            & np.isfinite(n0.flux_continuum_subtracted)
            & np.isfinite(n0.err)
            & (n0.err > 0)
        )
        n0_residual_sigma = (
            n0.flux_continuum_subtracted[mask] - n0.model[mask]
        ) / n0.err[mask]
        b1_residual_sigma = (
            n0.flux_continuum_subtracted[mask] - b1.model[mask]
        ) / n0.err[mask]
        n0_residual_abs_max_sigma = float(
            np.max(np.abs(n0_residual_sigma))
        )
        b1_residual_abs_max_sigma = float(
            np.max(np.abs(b1_residual_sigma))
        )
        output.update(
            {
                "success": True,
                "wave": n0.wave_rest[mask],
                "flux": n0.flux_continuum_subtracted[mask],
                "error": n0.err[mask],
                "n0_model": n0.model[mask],
                "b1_model": b1.model[mask],
                "n0_bic": n0.bic,
                "b1_bic": b1.bic,
                "resolving_power": resolving_power,
                "n0_residual_abs_max_sigma": n0_residual_abs_max_sigma,
                "b1_residual_abs_max_sigma": b1_residual_abs_max_sigma,
                "residual_abs_max_improvement_sigma": (
                    n0_residual_abs_max_sigma - b1_residual_abs_max_sigma
                ),
                "error_message": None,
            }
        )
    except Exception as error:  # noqa: BLE001 - render compact failure QA
        output.update(
            {
                "success": False,
                "error_message": "".join(
                    traceback.format_exception_only(type(error), error)
                ).strip(),
            }
        )
    return output


def _plot_payload(payload: dict[str, object], path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.04},
    )
    title = (
        f"{payload['object_id']} — {payload['complex_name']} — "
        f"{payload.get('qa_reason', 'selected QA')}"
    )
    if payload["success"]:
        wave = np.asarray(payload["wave"])
        flux = np.asarray(payload["flux"])
        error = np.asarray(payload["error"])
        n0 = np.asarray(payload["n0_model"])
        b1 = np.asarray(payload["b1_model"])
        axes[0].plot(wave, flux, color="0.35", lw=0.7, label="continuum-subtracted data")
        axes[0].plot(wave, n0, color="#0072B2", lw=1.2, label="N0")
        axes[0].plot(wave, b1, color="#D55E00", lw=1.2, label="B1")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_ylabel(r"Flux density")
        axes[1].plot(wave, (flux - n0) / error, color="#0072B2", lw=0.7)
        axes[1].plot(wave, (flux - b1) / error, color="#D55E00", lw=0.7)
        axes[1].axhline(0.0, color="0.5", lw=0.6)
        axes[1].text(
            0.99,
            0.92,
            (
                r"max $|r/\sigma|$: "
                f"N0={float(payload['n0_residual_abs_max_sigma']):.2f}, "
                f"B1={float(payload['b1_residual_abs_max_sigma']):.2f}"
            ),
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        axes[1].set_ylabel(r"Residual / $\sigma$")
        axes[1].set_xlabel(r"Rest wavelength (Å)")
        title += f" — ΔBIC={float(payload['n0_bic']) - float(payload['b1_bic']):.1f}"
        title += f" — R={float(payload['resolving_power']):.0f}"
    else:
        axes[0].text(
            0.5,
            0.5,
            str(payload["error_message"]),
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            wrap=True,
        )
        axes[1].set_axis_off()
    axes[0].set_title(title, fontsize=9)
    for axis in axes:
        axis.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    root = args.output_root or default_root()
    run_dir = args.run_directory or root / "runs" / RUN_NAME
    classification = (
        args.classification_directory
        or root / "classification" / CLASSIFICATION_NAME
    )
    selection_path = args.qa_selection or classification / "qa_selection.parquet"
    output_dir = args.output_directory or classification / "qa"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_parquet(selection_path)
    if "resolving_power" not in selection.columns:
        selection["resolving_power"] = args.resolving_power
    else:
        selection["resolving_power"] = pd.to_numeric(
            selection["resolving_power"], errors="coerce"
        ).fillna(args.resolving_power)
    if args.complexes:
        selection = selection[
            selection["complex_name"].astype(str).isin(args.complexes)
        ]
    if args.max_objects is not None:
        selection = selection.head(args.max_objects)
    rows = selection.to_dict("records")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(str(run_dir),),
    ) as executor:
        payloads = list(executor.map(_qa_payload, rows))
    statuses = []
    for payload in payloads:
        filename = (
            f"{payload['complex_name']}__{payload['object_id']}__"
            f"{payload.get('qa_reason', 'qa')}.png"
        )
        path = output_dir / filename
        if args.force or not path.exists():
            _plot_payload(payload, path)
        statuses.append(
            {
                "object_id": payload["object_id"],
                "complex_name": payload["complex_name"],
                "qa_reason": payload.get("qa_reason"),
                "render_success": bool(payload["success"]),
                "error": payload.get("error_message"),
                "output": str(path),
                "resolving_power": payload.get("resolving_power"),
                "n0_residual_abs_max_sigma": payload.get(
                    "n0_residual_abs_max_sigma"
                ),
                "b1_residual_abs_max_sigma": payload.get(
                    "b1_residual_abs_max_sigma"
                ),
                "residual_abs_max_improvement_sigma": payload.get(
                    "residual_abs_max_improvement_sigma"
                ),
            }
        )
    status_frame = pd.DataFrame(statuses)
    status_path = output_dir / "qa_render_status.parquet"
    if status_path.exists():
        existing = pd.read_parquet(status_path)
        current_keys = set(
            zip(
                status_frame["object_id"].astype(str),
                status_frame["complex_name"].astype(str),
            )
        )
        keep = [
            (str(row.object_id), str(row.complex_name)) not in current_keys
            for row in existing.itertuples(index=False)
        ]
        status_frame = pd.concat(
            [existing.loc[keep], status_frame], ignore_index=True
        )
    status_frame.to_parquet(status_path, index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": str(selection_path.resolve()),
        "run_directory": str(run_dir.resolve()),
        "n_selected_this_run": len(statuses),
        "n_selected_total": len(status_frame),
        "n_full_model_plots": int(status_frame["render_success"].sum()),
        "n_failure_plots": int((~status_frame["render_success"]).sum()),
        "png_only": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
