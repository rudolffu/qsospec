"""Shared provisional-evidence rules for RGS narrow-line classifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

NARROW_SNR_THRESHOLD = 5.0
BROAD_DELTA_BIC_THRESHOLD = 10.0
BROAD_SNR_THRESHOLD = 5.0
N0_RESIDUAL_ABS_P95_MAX_SIGMA = 3.0
N0_RESIDUAL_ABS_MAX_SIGMA = 8.0
BROAD_RESIDUAL_P95_IMPROVEMENT_MIN_SIGMA = 0.25
PURITY_TARGET = 0.95


def evaluate_provisional_narrow_evidence(
    record: Mapping[str, Any],
    *,
    narrow_snr_field: str = "narrow_line_snr",
    broad_snr_fields: Sequence[str] = ("broad_line_snr",),
) -> dict[str, Any]:
    """Evaluate the preregistered provisional narrow-line rule.

    This function deliberately consumes only fit-derived fields.  VI, PCF, and
    photometric metadata are not accepted as arguments and cannot influence the
    result.
    """

    narrow_snr = float(record.get(narrow_snr_field, np.nan))
    broad_snrs = np.asarray(
        [float(record.get(name, np.nan)) for name in broad_snr_fields],
        dtype=float,
    )
    finite_broad = broad_snrs[np.isfinite(broad_snrs)]
    maximum_broad_snr = (
        float(np.max(finite_broad)) if finite_broad.size else np.nan
    )
    delta_bic = float(record.get("delta_bic_broad", np.nan))
    fits_complete = bool(record.get("n0_success", False)) and bool(
        record.get("b1_success", False)
    )
    width_secure = bool(record.get("narrow_width_secure_below_boundary", False))
    bound_hit = bool(record.get("narrow_kinematic_bound_hit", False))
    narrow_detected = bool(
        np.isfinite(narrow_snr) and narrow_snr >= NARROW_SNR_THRESHOLD
    )
    n0_preferred = bool(np.isfinite(delta_bic) and delta_bic <= 0.0)
    n0_residual_p95 = float(
        record.get("n0_residual_abs_p95_sigma", np.nan)
    )
    n0_residual_max = float(
        record.get("n0_residual_abs_max_sigma", np.nan)
    )
    b1_residual_p95 = float(
        record.get("b1_residual_abs_p95_sigma", np.nan)
    )
    residual_quality_available = bool(
        np.isfinite(n0_residual_p95) and np.isfinite(n0_residual_max)
    )
    residual_quality_pass = bool(
        residual_quality_available
        and n0_residual_p95 <= N0_RESIDUAL_ABS_P95_MAX_SIGMA
        and n0_residual_max <= N0_RESIDUAL_ABS_MAX_SIGMA
    )
    broad_residual_p95_improvement = (
        n0_residual_p95 - b1_residual_p95
        if np.isfinite(n0_residual_p95) and np.isfinite(b1_residual_p95)
        else np.nan
    )
    broad_residual_support = bool(
        np.isfinite(broad_residual_p95_improvement)
        and broad_residual_p95_improvement
        >= BROAD_RESIDUAL_P95_IMPROVEMENT_MIN_SIGMA
    )
    broad_evidence = bool(
        fits_complete
        and np.isfinite(delta_bic)
        and delta_bic >= BROAD_DELTA_BIC_THRESHOLD
        and np.isfinite(maximum_broad_snr)
        and maximum_broad_snr >= BROAD_SNR_THRESHOLD
        and broad_residual_support
    )
    provisional = bool(
        fits_complete
        and narrow_detected
        and width_secure
        and not bound_hit
        and residual_quality_pass
        and not broad_evidence
    )
    if not fits_complete:
        status = "fit_failed"
    elif broad_evidence:
        status = "broad_evidence"
    elif provisional:
        status = "provisional_narrow"
    elif residual_quality_available and not residual_quality_pass:
        status = "poor_fit"
    else:
        status = "indeterminate"
    return {
        "narrow_snr_threshold": NARROW_SNR_THRESHOLD,
        "broad_delta_bic_threshold": BROAD_DELTA_BIC_THRESHOLD,
        "broad_snr_threshold": BROAD_SNR_THRESHOLD,
        "n0_residual_abs_p95_max_sigma": N0_RESIDUAL_ABS_P95_MAX_SIGMA,
        "n0_residual_abs_max_sigma_threshold": N0_RESIDUAL_ABS_MAX_SIGMA,
        "broad_residual_p95_improvement_min_sigma": (
            BROAD_RESIDUAL_P95_IMPROVEMENT_MIN_SIGMA
        ),
        "purity_target": PURITY_TARGET,
        "narrow_line_detected": narrow_detected,
        "n0_bic_preferred": n0_preferred,
        "residual_quality_available": residual_quality_available,
        "residual_quality_pass": residual_quality_pass,
        "broad_residual_p95_improvement_sigma": (
            broad_residual_p95_improvement
        ),
        "broad_residual_support": broad_residual_support,
        "maximum_broad_line_snr": maximum_broad_snr,
        "broad_evidence": broad_evidence,
        "provisional_narrow_evidence": provisional,
        "evidence_status": status,
        "promotion_status": "not_calibrated",
        "physical_class": None,
    }
