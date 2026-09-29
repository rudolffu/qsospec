# Adaptive [O III] implementation and validation

New fits default to adaptive selection, with explicit `oiii_profile_mode="legacy"` for the previous model and rejection policy. The shared generic engine supplies all numerical fits. Input redshifts, frozen parent statistics, proposal cuts and manuscript files are unchanged.

## Verification

- Regression suite: **480 passed, 1 warning in 382.85s (0:06:22)** using `/Users/yuming/miniforge3/bin/python`.
- Sphinx HTML documentation built successfully with warnings treated as errors and external intersphinx inventories disabled for the offline build.
- Synthetic tests cover centered broad bases, width contrast below two, red and third components, independent narrow Hβ, weak/noise spectra, masked runs, label/covariance permutation, failed starts, absent covariance, constant/varying LSFs, frame conversion, flux conservation, analytic derivatives, unresolved widths, covariance-versus-parameter-draw agreement, and save/load/peak recovery.
- Reload and post-processing tests prohibit optimizer calls. Real-target post-processing made **0** optimizer calls and preserved the adopted redshift exactly.
- Existing uncommitted custom-kinematics functionality remains in place; its regression tests pass.

## Fixed saved host and continuum

Every QSOFEED archive was compared with a new adaptive line fit; J1509 is the clean control. J1242 and J1743 use their specified saved native workflows. Candidate models, parameter/covariance arrays, start histories and decisions are archived in each target's `adaptive.json` and `adaptive.pkl`. Originals are read-only inputs.

| Target   |   Components |   Old 5008 RMS (σ) |   New 5008 RMS (σ) | Adequate   | Core reliable   |
|:---------|-------------:|-------------------:|-------------------:|:-----------|:----------------|
| J1010    |            2 |              7.419 |              1.867 | True       | True            |
| J1100    |            3 |             13.127 |              1.575 | True       | True            |
| J1356    |            3 |              4.941 |              1.860 | True       | True            |
| J1430    |            2 |              9.513 |              1.885 | True       | True            |
| J1509    |            2 |              1.551 |              1.511 | True       | True            |
| J0802+25 |            3 |              4.136 |              1.707 | False      | True            |
| J0945+17 |            3 |              5.486 |              1.911 | False      | True            |
| J1034+60 |            3 |             15.203 |              3.233 | False      | False           |
| J1440+53 |            3 |              6.865 |              1.689 | True       | True            |
| J1455+32 |            3 |              9.252 |              1.221 | True       | True            |
| J1713+57 |            2 |              9.196 |              2.928 | False      | True            |
| J1242    |            2 |              1.673 |              1.637 | False      | True            |
| J1743    |            2 |              1.818 |              1.810 | False      | True            |

**7/11** QSOFEED profiles pass the recorded adequacy checks. The extra same-sign single-line-run and local mean-chi-square diagnostics flag residual problems without adding a component outside the specified trigger. The local residual statistic uses each line's ±1200 km/s neighborhood; comparison tables use the frozen core to keep before/after pixels fixed.

J1034+60 retains a narrow component near +1000 km/s with only about 2% of the 5008 flux. Its gas reference is unreliable by the documented 5% fraction rule. Core-relative quantiles/fractions are withheld; the input-frame quantiles remain available. It also retains residual structure after three components.

J1242 and J1743 do **not** satisfy the required matched-doublet third-component trigger, although the earlier targeted three-component analyses show meaningful structure. They remain two-component fits with explicit residual warnings. The general default does not reproduce those custom, target-specific decompositions automatically. They are unsuitable for unqualified high-velocity claims from this default result alone.

Every real archive used here lacks usable persisted resolution. These are explicitly **observed-profile** fits. The Gaussian forward-LSF implementation is validated synthetically; no survey-average resolution was invented and no real-target intrinsic-width claim follows from this validation.

## Full-workflow and continuum sensitivity

All 13 targets were rerun through host decomposition, global continuum fitting and the adaptive Hβ complex. Input arrays were taken from the archived, already extinction-corrected total spectra and explicitly marked caller-preprocessed. The run uses the saved QSOFEED scientific configuration and the Hβ complex; J1242 keeps the no-Balmer, single-power-law choice and continuum windows below 5500 Å. These runs exercise the complete host/continuum/Hβ path, not a re-release of the original multicomplex catalogue.

The maximum absolute W80 change between fixed-continuum and full-workflow fits is **0.203%**; median absolute change is **0.040%**. Individual host/continuum levels can change substantially (see `full_workflow_comparison.csv`), while the fitted residual linear continuum absorbs much of the local difference. This is an empirical sensitivity check, not a replacement for matched continuum/host bootstrap uncertainty. Full-workflow QA is in `qa/full_workflow_qa.pdf`.

## Deterministic 100-object catalogue control

The manifest samples the frozen 6344-object parent by survey, recorded `class_final` (including QSO_NARROW and GALAXY), and old [O III] S/N strata. Within each stratum, SHA-256 of `1729:object_key` sets a reproducible order; round-robin selection produces exactly 100 objects. Object identity joins use both object ID and object key, preserving duplicates in the source catalogue. Missing S/N remains a separate stratum. Optical classes are catalogue labels, not newly inferred physical types.

All **100** fits completed without exceptions. **81** retain one component and **19** retain two; none requires three by the configured rule. **83** pass profile-adequacy checks, **75** have a reliable fitted gas reference, and **68** pass both. These are validation-sample counts, not parent-population fractions or revised membership. Catalogues must retain these flags and measurement missingness; optimizer success alone is insufficient. No optical-type label suppresses the configured broad-Hβ family.

| survey     | optical_class   | snr_bin   |   N |
|:-----------|:----------------|:----------|----:|
| desi       | GALAXY          | 5_to_20   |   2 |
| desi       | GALAXY          | above_20  |   6 |
| desi       | QSO             | 5_to_20   |   2 |
| desi       | QSO             | above_20  |   3 |
| desi       | QSO             | below_5   |   1 |
| desi       | QSO_AUTO        | 5_to_20   |   6 |
| desi       | QSO_AUTO        | above_20  |   6 |
| desi       | QSO_AUTO        | below_5   |   6 |
| desi       | QSO_DEFAULT     | 5_to_20   |   6 |
| desi       | QSO_DEFAULT     | above_20  |   6 |
| desi       | QSO_DEFAULT     | below_5   |   6 |
| desi       | QSO_NARROW      | 5_to_20   |   6 |
| desi       | QSO_NARROW      | above_20  |   5 |
| desi       | QSO_NARROW      | below_5   |   2 |
| sdss_dr20q | QSO_AUTO        | 5_to_20   |   6 |
| sdss_dr20q | QSO_AUTO        | above_20  |   6 |
| sdss_dr20q | QSO_AUTO        | below_5   |   5 |
| sdss_dr20q | QSO_AUTO        | missing   |   4 |
| sdss_dr20q | QSO_DEFAULT     | 5_to_20   |   5 |
| sdss_dr20q | QSO_DEFAULT     | above_20  |   5 |
| sdss_dr20q | QSO_DEFAULT     | below_5   |   5 |
| sdss_dr20q | QSO_NARROW      | above_20  |   1 |

## Runtime and products

For the 100 controls the median adaptive fit time is **3.44 s**, 95th percentile **22.39 s**, and maximum **52.48 s**. The 13 stress/control targets span **2.24–41.84 s**. Explicit legacy refits take **0.13–0.37 s**; adaptive fitting is substantially more expensive because it searches 6–24 starts for each candidate. These are measured wall times under concurrent validation, not isolated throughput benchmarks. Post-processing alone takes **0.08–0.30 s** and performs no fitting.

- `fixed_continuum_comparison.csv`: baseline/adaptive residuals and kinematics for all 13 targets.
- `control_selection.csv` and `control_comparison.csv`: exact 100-object selection, classifications, S/N strata and results.
- `full_workflow_comparison.csv`: host/continuum sensitivity and runtime.
- `runtime_and_postprocessing.csv`: explicit legacy timing and optimizer-call audit.
- `qa/fixed_continuum_comparisons.pdf`: both doublet lines and residuals for 13 targets.
- `qa/control_comparisons.pdf`: all 100 control panels.
- `fixed_continuum/*/adaptive.json`, `controls/*/adaptive.json`: complete candidate audits (local large products excluded from Git).
- `validation_manifest.json`: source hashes, method version and environment.

The 5008-only and aligned-doublet fractions agree for the fixed-ratio intrinsic model. Wavelength-dependent LSFs can make the observed fractions differ; the implementation uses both convolved line profiles and explicitly records that distinction. Covariance errors are conditional on the selected model and fixed host/continuum; they do not supply intrinsic redshift-calibration scatter.

Reproduce the stages with `control_selection.py`, `fetch_controls.py` (localhost:6000, source reads only), `validate_real.py`, `validate_controls.py`, `validate_workflow.py`, `benchmark_postprocessing.py`, `plot_comparisons.py` and `finalize_report.py`. All outputs are written under this new validation directory. No automatic sample, proposal or manuscript revision is performed.
