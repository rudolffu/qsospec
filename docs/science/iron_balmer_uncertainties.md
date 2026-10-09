# Regional iron, Hγ refinement, and native uncertainty products

Use the [continuum explanation](continuum_model.rst) for the current model,
defaults and scientific width conventions. This page collects the specialist
diagnostics needed to inspect that model or recover saved uncertainty products.
For repeated fits and error interpretation, see
[Understand and estimate uncertainties](uncertainties.rst).

## Inspect the regional iron bridge

For the supported VW01/Park22 pair, the regional Verner09 bridge activates
when accepted continuum pixels overlap its nonzero weighted basis. It adds
the nonnegative `middle_iron.amp` and shares the optical additional convolution
kernel, falling back to UV when optical support is absent. It has no independent
width. A full-range Verner09 model uses its single component instead. See
[configuration](../reference/configuration.rst) for alternative choices.

`continuum.metadata['regional_iron']` records the resolved intervals, width
parent and reference normalization. Nominal handoffs are 3300–3450 and
4100–4250 Å; four-sigma guards from configured maximum widths establish fixed
intervals before optimization. The weighted Verner09 integral at a 3000 km/s
reference kernel sets its normalization on the template grid, independently
of the surviving data pixels. The rendered band flux varies with width, so
`middle_iron.amp` is not the final band flux. Outer empirical tapers are
preserved and the inner handoffs replace empirical tapers.

## Read width diagnostics

`IronTemplateConfig.width_mode` determines the coordinate of `fwhm_kms` and
`fwhm_bounds`: `kernel` is additional template convolution; `target` requests
an effective Gaussian-equivalent width with an established native width.
These coordinates are separate from instrumental broadening. The
[continuum chapter](continuum_model.rst) explains which width can be reported.

`qsospec.templates.iron.resolve_iron_width` records the requested, kernel,
native and effective widths; `evaluate_iron_kernel` evaluates the basis and
kernel-coordinate derivatives. The Verner09 native-width assumption is
900 km/s. An unknown or mixed empirical native width has no reported
effective FWHM. Target sharpening is rejected; equality evaluates the
zero-kernel limit. Target bounds respect the native-width floor, while kernel
bounds can include zero. Soft log-kernel coupling needs positive kernel bounds.

For `iron_width_coupling='soft'`, the default log10 kernel-ratio residual has
center zero and scatter 0.25 dex. The saved diagnostics separate data $\chi^2$,
prior penalty and total objective. Profile data information is evaluated after
projecting out nuisance columns and compared with prior information. Inspect
whether a width is data-constrained, predominantly regularized, at a bound or
unidentified before using its local error.

## Inspect the Hγ constraint

The current soft refinement is described in the
[continuum chapter](continuum_model.rst). Its saved `hgamma_joint_status`
distinguishes a completed joint fit from unavailable coverage, truncated wings
or optimizer failure. `hgamma_ratio_scatter_dex` records a model-ratio tolerance,
not an observed Hγ error. `joint_covariance` identifies the continuum/blue-line
parameters and absolute noise-scaling convention. Sequential fits of other
complexes do not gain this joint block's cross-covariance.

The narrow Hγ profile and [O III] 4364.436 Å remain independent of the broad
Balmer amplitude. The vacuum [O III] wavelength follows the
[SDSS reference line table](https://classic.sdss.org/dr7/algorithms/linestable.php).
The bound-free continuum joins continuously to the high-order series. A
continuum change is followed by an Hβ refit.

Canonical `sync_with_hgamma` modes are `off`, `soft`, `hard` and `require`.
Compatibility aliases normalize at construction: `none`/`never` to `off`,
and `auto`/`hard_legacy` to `hard`. See
[the migration guide](../getting_started/migration_0_2.rst) when reproducing
a fit made with a different preset.

## Find errors for the quantity you measured

The [measurement dictionary](../reference/measurement_dictionary.rst)
identifies the value/error access paths and profile definitions. Continuum
errors at supported 1350, 3000 and 5100 Å samples use gradients of the actual
renderer, including a broken power law when selected, and appear in
`metadata['continuum_sample_errors']`. Summed broad/narrow products propagate
FWHM, dispersion, local-continuum EW and approximate intrinsic-width errors.
Near an unresolved boundary, intrinsic errors are unavailable. Sequential EW
errors condition on the fitted global continuum.

`qsospec.uncertainties.measure_selected_profile` measures an explicit generic
component selection and records its membership and definitions. Its FWHM
uses the nearest half-maximum crossings enclosing the global peak; absent
crossings are unavailable. Its optional EW uses a ratio integral with the
supplied continuum, which differs from native sampled-continuum EW fields.
See [measurement definitions](measurements.rst) for both conventions and
[uncertainties](uncertainties.rst) for covariance, matched draws and host
fractions.

## Schema and recovery

Schema 8 saves compact model recipes, named covariance blocks and matched
draws. Schemas 5–7 remain readable. Explicit free-parameter ordering excludes
fixed coordinates; the joint Hγ covariance and conditional polynomial blocks
retain their separate definitions. Cross-covariance between independently
fitted blocks is unavailable. See [run bundles](../reference/run_bundles.rst)
for persistence, reconstruction and schema compatibility.

With a fitted or loaded `result`, write a new recovery report:

```python
from qsospec.uncertainties import recover_uncertainties

report = recover_uncertainties(result, "recovery.json")
```

The report path must not exist. Recovery exposes saved errors and can
propagate identified saved power-law covariance when a pivot is available,
including a pure pivot-normalization case. Other entries state that a refit
is required. It preserves point estimates and the original bundle. Changing
an iron/Balmer model requires a new fit; loading or upgrading cannot supply
covariance that was not saved.

Matched workflow products use `measurement_key_schema="qualified_v1"`:
`continuum_sample:<name>`, `continuum_param:<name>`, `derived:<name>` and
`line:<recipe>:<metric>`. Native metric dictionaries keep their existing
names. Older matched draws are read through an explicit namespace conversion;
ambiguous unqualified aliases are omitted in favor of saved recipe-qualified
entries.

## Reproducible commands

The [development guide](../contributing/development.rst) contains the
synthetic iron/Balmer comparison, bootstrap checks, regression commands and
archived-spectrum comparison scripts. These are developer validation tasks.
For a scientific workflow with your own input, use the complete uncertainty
procedure in [Understand and estimate uncertainties](uncertainties.rst).
