#!/usr/bin/env python3
"""Explore broad+narrow measurements without imposing a scientific class."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--broad-fraction-thresholds", nargs="+", type=float, default=(0.1, 0.2, 0.5, 0.8))
    parser.add_argument("--width-thresholds", nargs="+", type=float, default=(1200.0, 2000.0, 4000.0))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-column")
    return parser.parse_args()


def _default_measurements() -> Path:
    root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --measurements")
    return Path(root) / (
        "outputs/qsospec/dr1_identified_gold_rgs_v1/measurements/"
        "broad_narrow_r480_v1/broad_narrow_line_measurements.parquet"
    )


def _style() -> None:
    try:
        import niceplots

        niceplots.initPlot()
    except (ImportError, RuntimeError):
        plt.rcParams.update({
            "font.family": "sans-serif", "font.size": 10,
            "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "xtick.minor.visible": True, "ytick.minor.visible": True,
        })


def _finite(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column in columns:
        mask &= np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
    return frame[mask]


def _save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def complex_figure(frame: pd.DataFrame, complex_name: str, output: Path) -> None:
    fitted = frame[frame["fit_status"].eq("complete")].copy()
    if fitted.empty:
        fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
        counts = frame["fit_status"].value_counts()
        ax.bar(counts.index.astype(str), counts.to_numpy(), color="#0072B2")
        ax.set(xlabel="Fit status", ylabel="Number", title=f"{complex_name}: no completed measurements")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(False)
        _save(fig, output / f"{complex_name}_population")
        return
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    good = _finite(fitted, "broad_fraction", "total_profile_fwhm_observed_kms")
    axes[0, 0].scatter(
        good["total_profile_fwhm_observed_kms"], good["broad_fraction"],
        s=8, alpha=0.45, linewidths=0, rasterized=True,
    )
    axes[0, 0].set(xlabel=r"Total-profile FWHM (km s$^{-1}$)", ylabel="Broad flux fraction")
    axes[0, 1].hist(pd.to_numeric(fitted["broad_fraction"], errors="coerce").dropna(), bins=np.linspace(0, 1, 31), histtype="step", linewidth=1.5)
    axes[0, 1].set(xlabel="Broad flux fraction", ylabel="Number")
    for column, label, color in (
        ("total_profile_fwhm_observed_kms", "Total profile", "#0072B2"),
        ("narrow_fwhm_observed_kms", "Narrow component", "#009E73"),
        ("broad_fwhm_observed_kms", "Broad component", "#D55E00"),
    ):
        values = pd.to_numeric(fitted.get(column), errors="coerce").dropna()
        axes[0, 2].hist(values, bins=35, histtype="step", linewidth=1.4, label=label, color=color)
    axes[0, 2].set(xlabel=r"Observed width (km s$^{-1}$)", ylabel="Number")
    axes[0, 2].legend(frameon=False, fontsize=8)
    good = _finite(fitted, "broad_fraction", "broad_flux_snr")
    axes[1, 0].scatter(good["broad_fraction"], good["broad_flux_snr"], s=8, alpha=0.45, linewidths=0, rasterized=True)
    axes[1, 0].set(xlabel="Broad flux fraction", ylabel="Broad-component flux S/N")
    good = _finite(
        fitted, "broad_fraction", "total_profile_fwhm_observed_kms", "reduced_chi2"
    )
    color_max = float(np.percentile(good["reduced_chi2"], 95)) if len(good) else 1.0
    artist = axes[1, 1].scatter(
        good["total_profile_fwhm_observed_kms"], good["broad_fraction"],
        c=np.clip(good["reduced_chi2"], 0, color_max),
        s=9, cmap="viridis", linewidths=0, rasterized=True,
    )
    axes[1, 1].set(xlabel=r"Total-profile FWHM (km s$^{-1}$)", ylabel="Broad flux fraction")
    fig.colorbar(artist, ax=axes[1, 1], label=r"Reduced $\chi^2$")
    good = _finite(fitted, "broad_fraction", "residual_abs_max_sigma")
    axes[1, 2].scatter(good["broad_fraction"], good["residual_abs_max_sigma"], s=8, alpha=0.45, linewidths=0, rasterized=True)
    axes[1, 2].set(xlabel="Broad flux fraction", ylabel=r"Maximum $|$residual$|/\sigma$")
    for ax in axes.flat:
        ax.grid(False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.suptitle(f"{complex_name}: measurement-first broad+narrow diagnostics")
    _save(fig, output / f"{complex_name}_population")


def main() -> None:
    args = parse_args()
    measurements = args.measurements or _default_measurements()
    output = args.output_directory or measurements.parent / "exploration"
    output.mkdir(parents=True, exist_ok=True)
    _style()
    frame = pd.read_parquet(measurements)
    labels = None
    if args.labels:
        if not args.label_column:
            raise ValueError("--label-column is required with --labels")
        labels = (
            pd.read_parquet(args.labels)
            if args.labels.suffix.lower() == ".parquet"
            else pd.read_csv(args.labels)
        )
        if not {"object_id", args.label_column}.issubset(labels):
            raise KeyError("Post-hoc label table is missing object_id or the requested label column")
        if labels["object_id"].astype(str).duplicated().any():
            raise ValueError("Post-hoc label table contains duplicate object IDs")
    for complex_name, group in frame.groupby("complex_name", sort=False):
        complex_figure(group, str(complex_name), output)

    fitted = frame[frame["fit_status"].eq("complete")].copy()
    rows = []
    for complex_name, group in fitted.groupby("complex_name", sort=False):
        fraction = pd.to_numeric(group["broad_fraction"], errors="coerce")
        width = pd.to_numeric(group["total_profile_fwhm_observed_kms"], errors="coerce")
        for fraction_cut in args.broad_fraction_thresholds:
            for width_cut in args.width_thresholds:
                mask = np.isfinite(fraction) & np.isfinite(width) & (fraction <= fraction_cut) & (width <= width_cut)
                row = {
                    "complex_name": complex_name,
                    "broad_fraction_max": fraction_cut,
                    "total_profile_fwhm_max_kms": width_cut,
                    "n_selected": int(mask.sum()),
                    "n_fitted": len(group),
                    "selected_fraction": float(mask.mean()) if len(group) else np.nan,
                }
                if labels is not None:
                    merged = group[["object_id"]].assign(_selected=mask.to_numpy()).merge(
                        labels[["object_id", args.label_column]], on="object_id", how="left", validate="many_to_one"
                    )
                    row["posthoc_label_counts"] = merged.loc[merged["_selected"], args.label_column].value_counts(dropna=False).to_json()
                rows.append(row)
    pd.DataFrame(rows).to_csv(output / "threshold_grid.csv", index=False)

    pivot = fitted.pivot(index="object_id", columns="complex_name", values="broad_fraction")
    pivot = pivot.dropna(axis=1, how="all")
    has_pair = any(
        pivot[[left, right]].dropna().shape[0] > 0
        for index, left in enumerate(pivot.columns)
        for right in pivot.columns[:index]
    )
    if pivot.shape[1] >= 2 and has_pair:
        fig, axes = plt.subplots(pivot.shape[1], pivot.shape[1], figsize=(2.5 * pivot.shape[1], 2.5 * pivot.shape[1]), squeeze=False)
        for i, left in enumerate(pivot.columns):
            for j, right in enumerate(pivot.columns):
                ax = axes[i, j]
                if i == j:
                    ax.hist(pivot[left].dropna(), bins=np.linspace(0, 1, 25), histtype="step")
                elif i > j:
                    values = pivot[[right, left]].dropna()
                    ax.scatter(values[right], values[left], s=7, alpha=0.45, linewidths=0, rasterized=True)
                else:
                    ax.axis("off")
                if i == len(pivot.columns) - 1 and i != j:
                    ax.set_xlabel(str(right))
                if j == 0 and i != j:
                    ax.set_ylabel(str(left))
                ax.grid(False)
                ax.tick_params(which="both", direction="in", top=True, right=True)
        fig.suptitle("Cross-line broad-fraction measurements")
        fig.subplots_adjust(wspace=0.28, hspace=0.28, top=0.92)
        _save(fig, output / "cross_line_broad_fractions")


if __name__ == "__main__":
    main()
