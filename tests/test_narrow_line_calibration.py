"""Tests for provisional narrow-line injection/recovery summaries."""

import numpy as np
import pandas as pd

import qsospec


def _frame():
    return pd.DataFrame(
        {
            "fit_status": ["complete"] * 4,
            "narrow_line_snr": [6.0, 6.0, 4.9, 6.0],
            "delta_bic_broad": [-2.0, -2.0, -2.0, 12.0],
            "maximum_broad_line_snr": [0.0, 0.0, 0.0, 5.0],
            "n0_residual_abs_p95_sigma": [2.0, 2.0, 2.0, 3.0],
            "n0_residual_abs_max_sigma": [4.0, 4.0, 4.0, 5.0],
            "b1_residual_abs_p95_sigma": [2.0, 2.0, 2.0, 2.0],
            "narrow_kinematic_bound_hit": [False] * 4,
            "narrow_fwhm_observed_kms": [800.0] * 4,
            "narrow_fwhm_observed_error_kms": [30.0] * 4,
            "instrumental_fwhm_kms": [624.5676208333333] * 4,
            "truth_has_broad": [False, True, False, True],
            "snr_stratum": ["high", "high", "low", "high"],
            "redshift_stratum": ["low"] * 4,
            "resolution_stratum": ["point", "extended", "point", "point"],
        }
    )


def test_recovery_metrics_and_independent_promotion_gate():
    frame = _frame()
    provisional, broad = qsospec.narrow_line_calibration_selection_mask(frame)
    assert provisional.tolist() == [True, True, False, False]
    assert broad.tolist() == [False, False, False, True]
    summary = qsospec.summarize_narrow_line_recovery(frame, provisional)
    assert summary["estimated_purity"] == 0.5
    assert summary["estimated_completeness"] == 0.5
    assert summary["broad_false_narrow_rate"] == 0.5
    assert not summary["promotion_allowed"]


def test_threshold_sweep_contains_the_preregistered_rule():
    sweep = qsospec.narrow_line_calibration_threshold_sweep(_frame())
    locked = sweep[sweep["is_preregistered_rule"]]
    assert len(locked) == 1
    assert locked.iloc[0]["narrow_snr_threshold"] == 5.0
    assert locked.iloc[0]["broad_delta_bic_threshold"] == 10.0
    assert locked.iloc[0]["width_upper_sigma"] == 2.0
    assert not locked.iloc[0]["promotion_allowed"]


def test_stratified_metrics_cover_snr_redshift_and_resolution():
    frame = _frame()
    provisional, _ = qsospec.narrow_line_calibration_selection_mask(frame)
    output = qsospec.narrow_line_stratified_recovery(frame, provisional)
    assert set(output["stratum"]) == {
        "snr_stratum",
        "redshift_stratum",
        "resolution_stratum",
    }
    assert np.all(output["promotion_allowed"].eq(False))
