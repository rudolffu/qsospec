# Changelog

## 0.2.0

This minor release changes the default scientific model and public APIs, and
requires Python 3.10 or later. See the
[0.2 migration guide](https://qsospec.readthedocs.io/en/latest/getting_started/migration_0_2.html).

- Adopt the versioned `global_v2` default: VW01/Verner09/Park22 regional iron,
  independent UV/optical widths and soft Hγ/Balmer refinement. Add the
  `GlobalContinuumConfig.legacy_v1()` reproducibility preset.
- Clarify iron width coordinates (`fwhm_kms` + `width_mode`), polynomial
  modes (`off`/`auto`/`on`) and canonical Hγ policies.
- Expand UV/optical/NIR recipes and adaptive [O III] fitting, with Gaussian
  resolution forward modeling, line peaks and separate systemic-redshift diagnostics.
- Add native covariance propagation and matched bootstrap products with
  collision-free qualified keys; run schema 7 reads earlier schemas 5 and 6.
- Improve AGN-aware host decomposition, template/resolution reliability,
  HostSED reconstruction and explicit configuration precedence.
- Add resumable survey-scale runs, direct model-key loading, scalar resume
  planning, deferred reconciliation and timing diagnostics.
- Rename the unreleased full-spectrum preset to `recipes.extended_quasar()`;
  retain established root APIs and place new specialist utilities in their modules.
- Restructure science/API documentation and provide portable survey-input examples.

## 0.1.0

- Extract the array-based `neofit` implementation into the standalone
  `qsospec` package.
- Add a modern `src` package layout and bundled iron/Balmer resources.
- Preserve local, global, batch, Parquet archive, QA, and optional pPXF
  workflows.
- Add canonical `WorkflowResult`, `HostWorkflowResult`, and `FitWarning`
  names with deprecated `NeoFit*` aliases.
