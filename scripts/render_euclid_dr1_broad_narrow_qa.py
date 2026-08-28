#!/usr/bin/env python3
"""Render QA only for explicitly selected broad+narrow measurement rows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qsospec import (
    BroadNarrowMeasurementConfig,
    LineSpreadFunctionConfig,
    load_model_by_key,
    measure_broad_narrow_complex,
    open_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resolving-power", type=float, default=480.0)
    parser.add_argument("--max-plots", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _style() -> None:
    try:
        import niceplots

        niceplots.initPlot()
    except (ImportError, RuntimeError):
        plt.rcParams.update({
            "font.family": "sans-serif", "font.size": 10, "axes.grid": False,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
        })


def _render(object_id: int, complex_name: str, loaded, result, record, output: Path) -> None:
    mask = np.asarray(result.fit_mask, dtype=bool)
    wave = result.wave_rest[mask] / 10.0
    data = result.flux_continuum_subtracted[mask]
    error = result.err[mask]
    model = result.model[mask]
    narrow = sum(
        values[mask] for name, values in result.component_models.items()
        if "narrow" in name or name.startswith(("NII", "SII", "OIII"))
    )
    broad = sum(
        values[mask] for name, values in result.component_models.items() if "broad" in name
    )
    local = sum(
        values[mask] for name, values in result.component_models.items() if name.startswith("local_continuum")
    )
    fig, (ax, residual_ax) = plt.subplots(
        2, 1, figsize=(7.2, 5.2), sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
    )
    ax.plot(wave, data, color="0.35", linewidth=0.8, label="Continuum-subtracted data")
    ax.plot(wave, model, color="#0072B2", linewidth=1.5, label="Broad+narrow model")
    ax.plot(wave, narrow + local, color="#009E73", linewidth=1.1, label="Narrow family + local baseline")
    ax.plot(wave, broad, color="#D55E00", linewidth=1.1, label="Broad family")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Flux density")
    fraction = record.get("broad_fraction", np.nan)
    width = record.get("total_profile_fwhm_observed_kms", np.nan)
    ax.set_title(f"{object_id} — {complex_name}")
    ax.text(
        0.02, 0.96,
        f"Broad fraction = {fraction:.3f}\nTotal FWHM = {width:.0f} km/s",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
    )
    residual = (data - model) / error
    residual_ax.axhline(0.0, color="0.6", linewidth=0.8)
    residual_ax.plot(wave, residual, color="#0072B2", linewidth=0.9)
    residual_ax.set(xlabel="Rest wavelength (nm)", ylabel=r"Residual / $\sigma$")
    for axis in (ax, residual_ax):
        axis.grid(False)
        axis.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_plots < 1 or args.max_plots > 100:
        raise ValueError("max-plots must be in [1, 100]")
    selection = pd.read_parquet(args.selection) if args.selection.suffix.lower() == ".parquet" else pd.read_csv(args.selection)
    required = {"object_id", "complex_name"}
    if not required.issubset(selection):
        raise KeyError(f"Selection is missing {sorted(required - set(selection))}")
    selection = selection.drop_duplicates(["object_id", "complex_name"]).head(args.max_plots)
    store = open_run(str(args.run_directory))
    object_index = store.build_object_index()
    config = BroadNarrowMeasurementConfig(
        lsf=LineSpreadFunctionConfig(resolving_power=args.resolving_power)
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    _style()
    for row in selection.to_dict("records"):
        object_id = int(str(row["object_id"]))
        complex_name = str(row["complex_name"])
        object_key = row.get("object_key") or object_index.get(str(object_id))
        if not object_key:
            continue
        output = args.output_directory / f"{complex_name}__{_safe(object_id)}.png"
        if output.exists() and not args.overwrite:
            continue
        loaded = load_model_by_key(store, str(object_key))
        record, result = measure_broad_narrow_complex(
            loaded.spectrum, loaded.continuum, complex_name, config
        )
        if result is None or not result.success:
            continue
        _render(object_id, complex_name, loaded, result, record, output)


if __name__ == "__main__":
    main()
