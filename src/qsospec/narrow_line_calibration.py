"""Calibration summaries for provisional narrow-line evidence.

These helpers deliberately separate injection/recovery performance from label
promotion.  Injection results can establish operating characteristics, but an
independent higher-resolution decomposition is still required before a
physical or publication label may be promoted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .narrow_line_evidence import (
    BROAD_DELTA_BIC_THRESHOLD,
    BROAD_RESIDUAL_P95_IMPROVEMENT_MIN_SIGMA,
    BROAD_SNR_THRESHOLD,
    N0_RESIDUAL_ABS_MAX_SIGMA,
    N0_RESIDUAL_ABS_P95_MAX_SIGMA,
    NARROW_SNR_THRESHOLD,
    PURITY_TARGET,
)


def selection_mask(
    frame: pd.DataFrame,
    *,
    narrow_snr_threshold: float = NARROW_SNR_THRESHOLD,
    broad_delta_bic_threshold: float = BROAD_DELTA_BIC_THRESHOLD,
    broad_snr_threshold: float = BROAD_SNR_THRESHOLD,
    width_upper_sigma: float = 2.0,
    intrinsic_boundary_kms: float = 1200.0,
) -> tuple[pd.Series, pd.Series]:
    """Return provisional-narrow and broad-evidence masks for trial thresholds."""

    required = {
        "fit_status",
        "narrow_line_snr",
        "delta_bic_broad",
        "maximum_broad_line_snr",
        "narrow_kinematic_bound_hit",
        "narrow_fwhm_observed_kms",
        "narrow_fwhm_observed_error_kms",
        "instrumental_fwhm_kms",
        "n0_residual_abs_p95_sigma",
        "n0_residual_abs_max_sigma",
        "b1_residual_abs_p95_sigma",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Calibration frame is missing fit columns: {missing}")
    complete = frame["fit_status"].eq("complete")
    observed = pd.to_numeric(frame["narrow_fwhm_observed_kms"], errors="coerce")
    error = pd.to_numeric(
        frame["narrow_fwhm_observed_error_kms"], errors="coerce"
    )
    instrument = pd.to_numeric(frame["instrumental_fwhm_kms"], errors="coerce")
    intrinsic_upper = np.sqrt(
        np.maximum((observed + width_upper_sigma * error) ** 2 - instrument**2, 0.0)
    )
    delta = pd.to_numeric(frame["delta_bic_broad"], errors="coerce")
    broad_snr = pd.to_numeric(frame["maximum_broad_line_snr"], errors="coerce")
    n0_residual_p95 = pd.to_numeric(
        frame["n0_residual_abs_p95_sigma"], errors="coerce"
    )
    n0_residual_max = pd.to_numeric(
        frame["n0_residual_abs_max_sigma"], errors="coerce"
    )
    b1_residual_p95 = pd.to_numeric(
        frame["b1_residual_abs_p95_sigma"], errors="coerce"
    )
    broad_residual_support = (
        n0_residual_p95 - b1_residual_p95
        >= BROAD_RESIDUAL_P95_IMPROVEMENT_MIN_SIGMA
    )
    broad = (
        complete
        & (delta >= broad_delta_bic_threshold)
        & (broad_snr >= broad_snr_threshold)
        & broad_residual_support
    )
    residual_quality = (
        (n0_residual_p95 <= N0_RESIDUAL_ABS_P95_MAX_SIGMA)
        & (n0_residual_max <= N0_RESIDUAL_ABS_MAX_SIGMA)
    )
    provisional = (
        complete
        & (pd.to_numeric(frame["narrow_line_snr"], errors="coerce") >= narrow_snr_threshold)
        & (intrinsic_upper < intrinsic_boundary_kms)
        & ~frame["narrow_kinematic_bound_hit"].fillna(False).astype(bool)
        & residual_quality
        & ~broad
    )
    return provisional, broad


def summarize_recovery(
    frame: pd.DataFrame,
    provisional: pd.Series,
    *,
    truth_broad_column: str = "truth_has_broad",
) -> dict[str, object]:
    """Summarize purity/completeness and broad false-narrow recovery."""

    if truth_broad_column not in frame:
        raise KeyError(f"Missing truth column: {truth_broad_column}")
    truth_broad = frame[truth_broad_column].fillna(False).astype(bool)
    truth_narrow = ~truth_broad
    n_selected = int(provisional.sum())
    true_narrow_selected = int((provisional & truth_narrow).sum())
    broad_selected = int((provisional & truth_broad).sum())
    n_truth_narrow = int(truth_narrow.sum())
    n_truth_broad = int(truth_broad.sum())
    purity = true_narrow_selected / n_selected if n_selected else np.nan
    completeness = (
        true_narrow_selected / n_truth_narrow if n_truth_narrow else np.nan
    )
    false_narrow_rate = broad_selected / n_truth_broad if n_truth_broad else np.nan
    return {
        "n_injections": len(frame),
        "n_selected": n_selected,
        "n_truth_narrow": n_truth_narrow,
        "n_truth_broad": n_truth_broad,
        "n_true_narrow_selected": true_narrow_selected,
        "n_broad_selected_as_narrow": broad_selected,
        "estimated_purity": purity,
        "estimated_completeness": completeness,
        "broad_false_narrow_rate": false_narrow_rate,
        "purity_target": PURITY_TARGET,
        "injection_purity_target_met": bool(
            np.isfinite(purity) and purity >= PURITY_TARGET
        ),
        "promotion_allowed": False,
        "promotion_blocker": "independent_higher_resolution_decomposition_required",
    }


def threshold_sweep(
    frame: pd.DataFrame,
    *,
    snr_thresholds: Sequence[float] = (4.0, 5.0, 6.0),
    bic_thresholds: Sequence[float] = (6.0, 10.0, 14.0),
    width_sigmas: Sequence[float] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Evaluate nearby calibration thresholds and identify the locked row."""

    rows: list[dict[str, object]] = []
    for snr in snr_thresholds:
        for bic in bic_thresholds:
            for width_sigma in width_sigmas:
                provisional, broad = selection_mask(
                    frame,
                    narrow_snr_threshold=float(snr),
                    broad_delta_bic_threshold=float(bic),
                    width_upper_sigma=float(width_sigma),
                )
                rows.append(
                    {
                        "narrow_snr_threshold": float(snr),
                        "broad_delta_bic_threshold": float(bic),
                        "broad_snr_threshold": BROAD_SNR_THRESHOLD,
                        "width_upper_sigma": float(width_sigma),
                        "n_broad_evidence": int(broad.sum()),
                        **summarize_recovery(frame, provisional),
                        "is_preregistered_rule": bool(
                            snr == NARROW_SNR_THRESHOLD
                            and bic == BROAD_DELTA_BIC_THRESHOLD
                            and width_sigma == 2.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def stratified_recovery(
    frame: pd.DataFrame,
    provisional: pd.Series,
    strata: Iterable[str] = ("snr_stratum", "redshift_stratum", "resolution_stratum"),
) -> pd.DataFrame:
    """Report recovery diagnostics by required simulation strata."""

    rows: list[dict[str, object]] = []
    for column in strata:
        if column not in frame:
            continue
        for label, group in frame.groupby(column, dropna=False, sort=False):
            metrics = summarize_recovery(frame.loc[group.index], provisional.loc[group.index])
            rows.append({"stratum": column, "value": str(label), **metrics})
    return pd.DataFrame(rows)
