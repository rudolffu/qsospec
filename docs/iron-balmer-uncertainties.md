# Regional iron, Hγ refinement, and native uncertainty products

New standard VW01/Park22 global configurations enable a regional Verner09
component when accepted continuum pixels overlap its nonzero weighted basis.
`RegionalIronConfig(enabled=False)` requests the historical split renderer.
Full-range Verner09 remains exclusive. Other empirical template pairs do not
acquire the bridge. Disabling both empirical components disables automatic
bridge activation.

The bridge adds `middle_iron.amp`, constrained nonnegative. It shares the
optical additional convolution kernel, falling back to UV when optical coverage
is absent. It has no independent fitted width. Nominal handoffs are 3300–3450
and 4100–4250 Å; four-sigma guards using configured maximum widths resolve
fixed intervals before optimization. The actual intervals, parent and fixed
reference normalization are in `continuum.metadata['regional_iron']`.
Its normalization is the weighted Verner09 integral on the template grid at a
3000 km/s reference kernel, independent of the object's surviving pixels.
The rendered flux changes with width; the amplitude is not a final band flux.
Outer empirical tapers are preserved. Inner handoffs replace empirical tapers.

`resolve_iron_width` and `evaluate_iron_kernel` in `qsospec.templates.iron`
provide explicit width resolution and kernel-coordinate derivatives.
`IronTemplateConfig(kernel_fwhm_kms=...)` and `target_fwhm_kms=...` accept
explicit requests; `fwhm_kms` remains the legacy compatibility coordinate.
Legacy empirical widths are additional kernels. Legacy Verner09 widths are
target Gaussian-equivalent widths, retaining its bundled 900 km/s native-width
assumption. Unknown/mixed empirical native widths have no reported effective
FWHM. Target sharpening is rejected and equality renders the zero-kernel limit.
These template kernels are separate from instrumental resolution corrections.
`fwhm_bounds` retains its historical coordinate (Verner09 target, empirical
kernel); configure bounds to include the requested starting value.

Independent empirical widths remain the reference default. Set
`iron_width_coupling='soft'` for the log10 kernel-ratio residual with default
center zero and scatter 0.25 dex. Positive lower bounds are required. The
regularized path uses the existing joint least-squares solver and keeps data
χ² separate from prior penalty and total objective. Diagnostics include
profile data curvature after projecting nuisance columns and prior curvature.
These scatters are engineering regularization choices.

The default `BalmerPseudoContinuumConfig(sync_with_hgamma='soft')` jointly
refines the continuum and blue optical complex on the union of their masks.
Broad Hγ flux is the Hβ-equivalent Balmer amplitude times the adopted ratio
times `10**delta_gamma`; the offset has a 0.30 dex model-ratio tolerance.
Narrow Hγ and [O III] 4364.436 Å (vacuum) remain independent. The wavelength
comes from the [SDSS reference line table](https://classic.sdss.org/dr7/algorithms/linestable.php).
The existing bound-free/high-order-series join is unchanged. No Hδ intermediary
is needed. Coverage is checked over the initial fitted Hγ wings; missing or
truncated coverage skips the constraint. `none` disables synchronization;
`hard_legacy` retains historical synchronization. Explicit old `auto`, `never`,
and `require` values retain their historical policy. Hβ is refitted after a
final continuum change.

The fast covariance mode includes amplitude/shape covariance, including
shared bridge derivatives. It is conditional on the host subtraction and,
for sequential line fits, the fitted continuum. The joint Hγ block includes
its continuum correlations and uses absolute pixel errors and prior scatter.
The ordinary continuum covariance retains residual noise scaling, applying
that scale only to data information when a width prior is present. Rank-deficient
parameter errors are unavailable; an identifiable subspace pseudoinverse is
retained, with a warning. Bound flags and curvature diagnostics must be checked
before treating a local symmetric interval as a measurement.

Continuum errors at supported 1350, 3000 and 5100 Å samples use full numerical
gradients of the actual continuum renderer, including broken power laws. They
are in `metadata['continuum_sample_errors']` and native measurement rows.
Host fractions need matched host–AGN information; a fixed host does not acquire
an invented error. `host_agn_covariance(G, A, covariance)` transforms a supplied
joint 2×2 block into total flux and host fraction covariance.

Set `UncertaintyConfig(monte_carlo_trials=20, random_seed=21)` to rerun the
AGN workflow on matched realizations. This uses a final-model parametric
bootstrap with diagonal pixel errors, preserving the Spectrum metadata and
mask through dataclass replacement. Use the existing host workflow with
`refit_host_in_mc=True` to rerun host subtraction for every realization. The
host path currently perturbs the observed input spectrum and labels that
sampling scheme explicitly; it is not posterior sampling. Trial IDs, failures,
valid counts, intervals, parameter draws and cross-measurement covariance are
retained. Cross-measurement covariance uses complete matched trials. One trial
can produce a descriptive percentile but cannot produce a standard error.
Best-fit point estimates remain unchanged. No nested parallelism is introduced.

`measure_selected_profile(fit, component_ids, parameter_draws=None,
continuum=None)` supports native generic complexes, including C IV, without
adopting a pruning rule. It measures flux, centroid, sigma and the FWHM from the
summed selected profile. FWHM uses the nearest half-maximum crossings enclosing
the global peak; missing crossings are unavailable. Optional EW is conditional
on the supplied fixed continuum. Matched parameter draws can replace the local
Gaussian approximation. Selection IDs and definitions are persisted in the
complex metadata. Specialized adapter results are rejected by this API rather
than assigned uncertainties from a different profile.

## Schema and recovery

Run schema 6 stores compact named covariance blocks and matched draws in the
existing model archive's structured metadata. It reads schema 5 bundles;
missing covariance remains unavailable. Free parameter ordering is explicit,
including legacy fixed-parameter exclusions and conditional polynomial blocks.
Joint Hγ covariance remains a joint block; independent fit blocks do not acquire
zero cross-covariance. Per-object summaries also expose covariance and names.

`recover_uncertainties(result, '/new/path/recovery.json')` creates an exclusive
new report. It exposes saved errors, recovers power-law errors where identified
covariance and a saved pivot permit it, handles the pure pivot normalization
special case, and otherwise reports refitting as required. It preserves point
estimates and original bundles. Changing iron/Balmer defaults requires a new
fit; upgrading cannot supply missing historical covariance.

## Reproducible commands

```bash
/Users/yuming/miniforge3/bin/python examples/iron_balmer_uncertainty.py --output /tmp/iron-comparison.json
/Users/yuming/miniforge3/bin/python examples/iron_balmer_uncertainty.py --trials 20 --output /tmp/iron-bootstrap.json
/Users/yuming/miniforge3/bin/python -m pytest tests/test_qsospec_iron_templates.py tests/test_qsospec_global_workflow.py tests/test_qsospec_run_store.py tests/test_qsospec_host_workflow.py
```

The comparison includes a deliberately non-adopted Hγ ratio and an omitted
spectral feature. Residual statistics use common masks and unsmoothed data.
Tiny resampling runs verify plumbing, not accurate tails or model systematics.
No downstream catalogue, bolometric correction, mass recipe or external stellar
library is changed.

A native host-inclusive invocation using an existing local template library is:

```python
import qsospec
result = qsospec.fit_global_lines_workflow(
    "input_spectrum.parquet", row_index=0, run_host_decomp=True,
    template_root="/Users/yuming/tools/ppxf_data",
    template_file="spectra_emiles_9.0.npz",
    uncertainty_config=qsospec.UncertaintyConfig(
        monte_carlo_trials=20, random_seed=21, refit_host_in_mc=True,
    ),
)
```

`UncertaintyConfig(pixel_covariance=matrix)` optionally supplies the original-grid
noise covariance in the input flux units for resampling. It is checked for
symmetry and positive semidefiniteness and factored once. This changes the
bootstrap perturbations; the existing fit objective still uses its pixel errors.
The host path expects this matrix in the host-wrapper input frame.

The broad/narrow measurement table also propagates the summed-profile FWHM,
profile sigma, local-continuum EW and approximate intrinsic-width errors. Widths
near the unresolved boundary have unavailable intrinsic errors. EW remains
conditional on the global continuum in this fast path. Existing flux-sum and
fraction gradients retain covariance terms.

The real-spectrum comparison can be repeated after extracting the documented
JSONL fields from a saved run:

```bash
/Users/yuming/miniforge3/bin/python examples/compare_archived_iron_balmer.py --input /tmp/qsospec-three-spectra.jsonl --output-dir /tmp/new-real-comparison
/Users/yuming/miniforge3/bin/python examples/plot_iron_balmer_comparison.py --source /tmp/new-real-comparison --output /tmp/new-real-comparison-plots
```
