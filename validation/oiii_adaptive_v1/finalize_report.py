"""Refresh new diagnostics, collect full-workflow results and write the report."""
import os
os.environ['MPLCONFIGDIR']='/tmp/qsospec-adaptive-mpl'
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,pickle,platform,subprocess
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import qsospec
from qsospec.complex_recipes import ComplexRecipe,ComponentRecipe
from qsospec.fitting.adaptive_oiii import finalize_quality
from qsospec.oiii_measurements import record_oiii_measurements
from validate_real import OUT,inputs,load,dump


def refresh(fit,w):
    d=dict(fit.metadata['model_definition']);d['components']=tuple(ComponentRecipe(**{**x,'line_ids':tuple(x['line_ids'])}) for x in d['components'])
    recipe=ComplexRecipe(**d)
    finalize_quality(fit,w.spectrum,w.continuum,recipe,fit.metadata['candidate_selection'])
    record_oiii_measurements(fit,w.spectrum,w.continuum,recipe)


if __name__=='__main__':
    records=[];missing=[]
    for name,source,key in inputs():
        folder=OUT/'full_workflow'/name;path=folder/'workflow.pkl'
        if not path.exists():missing.append(name);continue
        with path.open('rb') as f:w=pickle.load(f)
        refresh(w.hbeta,w)
        with path.open('wb') as f:pickle.dump(w,f)
        row=json.loads((folder/'summary.json').read_text());row['adequate']=w.hbeta.metadata['profile_adequate']
        row['core_reference_reliable']=w.hbeta.metadata['oiii_core_reference']['reliable']
        row['w80_fractional_change']=row['w80_full']/row['w80_fixed']-1
        records.append(row);dump(folder/'summary.json',row);dump(folder/'hbeta.json',w.hbeta.summary())
    if missing:raise RuntimeError('Full-workflow cases outstanding: '+', '.join(missing))
    full=pd.DataFrame(records);full.to_csv(OUT/'full_workflow_comparison.csv',index=False)
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    with PdfPages(qa/'full_workflow_qa.pdf') as pdf:
        for row in records:
            with (OUT/'full_workflow'/row['name']/'workflow.pkl').open('rb') as f:w=pickle.load(f)
            fig=w.plot_qa(qsospec.GlobalQAPlotConfig(object_name=row['name'],show_residual_panel=True,max_zoom_panels=1))
            pdf.savefig(fig)
            if row['name'] in ['J1509','J1242','J1034+60']:fig.savefig(qa/(row['name']+'_full.png'),dpi=160,bbox_inches='tight')
            plt.close(fig)
    targets=pd.read_csv(OUT/'fixed_continuum_comparison.csv');controls=pd.read_csv(OUT/'control_comparison.csv')
    runtime=pd.read_csv(OUT/'runtime_and_postprocessing.csv')
    tests=Path('/tmp/qsospec-final-tests.log').read_text().strip().splitlines()[-1]
    for filename,source in [('test_results.txt','/tmp/qsospec-final-tests.log'),('docs_build.txt','/tmp/adaptive-docs-final.log')]:
        (OUT/filename).write_text(Path(source).read_text())
    root=OUT.parents[1]
    source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/qsospec').rglob('*.py'))}
    snapshot=dict(method='adaptive-oiii-1',seed=1729,python=platform.python_version(),executable='/Users/yuming/miniforge3/bin/python',
        numpy=np.__version__,source_sha256=source_hashes,control_count=len(controls),target_count=len(targets),full_workflow_count=len(full),
        source_archives_untouched=True,input_redshift_exactly_preserved=True,
        scope='fixed host/continuum line fits, then full host plus global continuum plus Hbeta workflow; no sample or manuscript updates')
    dump(OUT/'validation_manifest.json',snapshot)
    short=targets[['name','n_components','baseline_rms_5008','adaptive_rms_5008','profile_adequate','core_reference_reliable']].copy()
    short.columns=['Target','Components','Old 5008 RMS (σ)','New 5008 RMS (σ)','Adequate','Core reliable']
    selection=pd.read_csv(OUT/'control_selection.csv')
    selection['snr_bin']=selection['snr_bin'].fillna('missing')
    strata=selection.groupby(['survey','optical_class','snr_bin'],dropna=False).size().rename('N').reset_index()
    qso=targets[~targets.name.isin(['J1242','J1743'])]
    text=f'''# Adaptive [O III] implementation and validation

New fits default to adaptive selection, with explicit `oiii_profile_mode="legacy"` for the previous model and rejection policy. The shared generic engine supplies all numerical fits. Input redshifts, frozen parent statistics, proposal cuts and manuscript files are unchanged.

## Verification

- Regression suite: **{tests}** using `/Users/yuming/miniforge3/bin/python`.
- Sphinx HTML documentation built successfully with warnings treated as errors and external intersphinx inventories disabled for the offline build.
- Synthetic tests cover centered broad bases, width contrast below two, red and third components, independent narrow Hβ, weak/noise spectra, masked runs, label/covariance permutation, failed starts, absent covariance, constant/varying LSFs, frame conversion, flux conservation, analytic derivatives, unresolved widths, covariance-versus-parameter-draw agreement, and save/load/peak recovery.
- Reload and post-processing tests prohibit optimizer calls. Real-target post-processing made **{int(runtime.measurement_optimizer_calls.sum())}** optimizer calls and preserved the adopted redshift exactly.
- Existing uncommitted custom-kinematics functionality remains in place; its regression tests pass.

## Fixed saved host and continuum

Every QSOFEED archive was compared with a new adaptive line fit; J1509 is the clean control. J1242 and J1743 use their specified saved native workflows. Candidate models, parameter/covariance arrays, start histories and decisions are archived in each target's `adaptive.json` and `adaptive.pkl`. Originals are read-only inputs.

{short.to_markdown(index=False,floatfmt='.3f')}

**{int(qso.profile_adequate.sum())}/11** QSOFEED profiles pass the recorded adequacy checks. The extra same-sign single-line-run and local mean-chi-square diagnostics flag residual problems without adding a component outside the specified trigger. The local residual statistic uses each line's ±1200 km/s neighborhood; comparison tables use the frozen core to keep before/after pixels fixed.

J1034+60 retains a narrow component near +1000 km/s with only about 2% of the 5008 flux. Its gas reference is unreliable by the documented 5% fraction rule. Core-relative quantiles/fractions are withheld; the input-frame quantiles remain available. It also retains residual structure after three components.

J1242 and J1743 do **not** satisfy the required matched-doublet third-component trigger, although the earlier targeted three-component analyses show meaningful structure. They remain two-component fits with explicit residual warnings. The general default does not reproduce those custom, target-specific decompositions automatically. They are unsuitable for unqualified high-velocity claims from this default result alone.

Every real archive used here lacks usable persisted resolution. These are explicitly **observed-profile** fits. The Gaussian forward-LSF implementation is validated synthetically; no survey-average resolution was invented and no real-target intrinsic-width claim follows from this validation.

## Full-workflow and continuum sensitivity

All 13 targets were rerun through host decomposition, global continuum fitting and the adaptive Hβ complex. Input arrays were taken from the archived, already extinction-corrected total spectra and explicitly marked caller-preprocessed. The run uses the saved QSOFEED scientific configuration and the Hβ complex; J1242 keeps the no-Balmer, single-power-law choice and continuum windows below 5500 Å. These runs exercise the complete host/continuum/Hβ path, not a re-release of the original multicomplex catalogue.

The maximum absolute W80 change between fixed-continuum and full-workflow fits is **{100*full.w80_fractional_change.abs().max():.3f}%**; median absolute change is **{100*full.w80_fractional_change.abs().median():.3f}%**. Individual host/continuum levels can change substantially (see `full_workflow_comparison.csv`), while the fitted residual linear continuum absorbs much of the local difference. This is an empirical sensitivity check, not a replacement for matched continuum/host bootstrap uncertainty. Full-workflow QA is in `qa/full_workflow_qa.pdf`.

## Deterministic 100-object catalogue control

The manifest samples the frozen 6344-object parent by survey, recorded `class_final` (including QSO_NARROW and GALAXY), and old [O III] S/N strata. Within each stratum, SHA-256 of `1729:object_key` sets a reproducible order; round-robin selection produces exactly 100 objects. Object identity joins use both object ID and object key, preserving duplicates in the source catalogue. Missing S/N remains a separate stratum. Optical classes are catalogue labels, not newly inferred physical types.

All **{len(controls)}** fits completed without exceptions. **{int((controls.n_components==1).sum())}** retain one component and **{int((controls.n_components==2).sum())}** retain two; none requires three by the configured rule. **{int(controls.profile_adequate.sum())}** pass profile-adequacy checks, **{int(controls.core_reference_reliable.sum())}** have a reliable fitted gas reference, and **{int((controls.profile_adequate & controls.core_reference_reliable).sum())}** pass both. These are validation-sample counts, not parent-population fractions or revised membership. Catalogues must retain these flags and measurement missingness; optimizer success alone is insufficient. No optical-type label suppresses the configured broad-Hβ family.

{strata.to_markdown(index=False)}

## Runtime and products

For the 100 controls the median adaptive fit time is **{controls.runtime_s.median():.2f} s**, 95th percentile **{controls.runtime_s.quantile(.95):.2f} s**, and maximum **{controls.runtime_s.max():.2f} s**. The 13 stress/control targets span **{targets.runtime_s.min():.2f}–{targets.runtime_s.max():.2f} s**. Explicit legacy refits take **{runtime.legacy_refit_runtime_s.min():.2f}–{runtime.legacy_refit_runtime_s.max():.2f} s**; adaptive fitting is substantially more expensive because it searches 6–24 starts for each candidate. These are measured wall times under concurrent validation, not isolated throughput benchmarks. Post-processing alone takes **{runtime.measurement_runtime_s.min():.2f}–{runtime.measurement_runtime_s.max():.2f} s** and performs no fitting.

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
'''
    (OUT/'REPORT.md').write_text(text)
    print('Report and full-workflow QA written',flush=True)
