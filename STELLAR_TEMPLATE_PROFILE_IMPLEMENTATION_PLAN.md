# Stellar-template profile implementation plan

## Baseline inspected before editing

- Repository: `/Users/yuming/tools/qsospec`
- Branch: `codex/rgs-lsf-halpha-selection`
- Commit: `709b92b51fd6c3c93185d05924ddbf9c7fdcbb38`
- Work tree: clean at inspection time.
- Native E-MILES: `spectra_emiles_9.0.npz`, SHA-256
  `6a1b1a70db4d95f67afe0adbe254f111eb2b50412b9f311714c54ffba18c6f0e`,
  keys `templates,masses,lam,ages,metals,fwhm`, template shape
  `(16675,25,6)`, 150 flattened SSPs, wavelength 1680.2--49998.29 A,
  ages 0.0631--15.8489 Gyr, metallicities -1.71--0.22, and a
  wavelength-dependent FWHM array spanning 2.51--23.57 A.
- Native XSL: `spectra_xsl_9.0.npz`, SHA-256
  `b8bee334ee1705a0d4e7ff0f161990c248d68ac1b8c8eb138b0d8c05cf37d853`,
  the same six key names but template shape `(58642,26,8)`, 208 flattened
  SSPs, wavelength 3500.0--24749.77 A, ages 0.0501--15.8489 Gyr,
  metallicities -2.2--0.2, and a wavelength-dependent FWHM array spanning
  0.357--3.111 A.
- The two libraries do not have identical SSP grids or flattened ordering.
  Neither file carries explicit IMF/isochrone labels. Their similarly named
  fields are therefore inspected and preserved per product rather than assumed
  equivalent. Both cover 5100 A and the XSL product covers the existing 1.0,
  1.6, and 2.2 micron HostSED samples, but only E-MILES extends to 5 microns.

## Required Phase-0 decisions

1. **Current hard rejection.**
   `src/qsospec/workflows/host/ppxf_host.py::_resample_stellar_templates`
   calls `additional_template_sigma()` and then applies
   `in_range &= ~invalid_resolution`. In
   `src/qsospec/resolution.py`, the compatibility helper includes
   `template > data` in that invalid mask.

2. **Callers depending on old semantics.**
   The host path and `tests/test_resolution_contract.py` are the direct users.
   `additional_template_sigma()` is exported from `qsospec.__init__`, so an
   external caller may also depend on its two-array return contract and NaNs.

3. **Compatibility choice.**
   Keep `additional_template_sigma()` unchanged. Add
   `match_template_resolution_toward_data()` with explicit class masks and use
   only the new result in host fitting and cache construction.

4. **Current fit/HostSED coupling.**
   `PPXFTemplateLibrary.flux/wave` feed both pPXF interpolation and
   `predict_host_sed()`. Compact reconstruction also verifies and reloads that
   single file/matrix. This is safe only for native products.

5. **Fit/source separation.**
   Extend `PPXFTemplateLibrary` with `fit_flux/fit_wave` and
   `source_flux/source_wave`, while keeping `flux/wave` as fit-array aliases.
   Native profiles share arrays. `xsl_preconvolved` loads the derived matrix as
   fit arrays and native XSL as source arrays. Stellar weights and the fit-time
   normalization scales are applied to source arrays for HostSED reconstruction.

6. **Existing reconstruction identity.**
   The current compact state retains the native file name/hash, wavelength and
   matrix hashes, flattening convention, original shape, scales, weights, and
   preprocessing normalization. It does not distinguish fit and source hashes.
   Version 2 of the compact state will add both identities while retaining a
   reader for version 1.

7. **pPXF grid placement.**
   `prepare_spectrum_for_host_decomp()` selects native valid pixels and fit
   range, then `_log_resample()` builds an object-specific logarithmic rest grid
   with the same number of selected native pixels. Stellar templates are
   interpolated onto `PreprocessedSpectrum.wave_log` immediately before pPXF.

8. **Exact preconvolved target.**
   A derived XSL matrix is tied to the native-XSL hash, object key, redshift,
   fit range, exact `wave_log` hash, rest-frame target sigma-lambda hash,
   valid-resolution-mask hash, source template order/hash, algorithm version,
   and normalization version. It contains the unnormalised source templates
   sampled and one-sided-convolved on that exact pPXF grid.

9. **Cache identity.**
   Canonical JSON of all fields in item 8 is SHA-256 hashed. Array identities
   include dtype, shape, and C-order bytes using the existing `array_sha256()`.
   Atomic files are written beside the requested output and promoted with
   `os.replace()` only after read-back validation.

10. **Storage compatibility.**
    Existing schema-v5 model metadata, workflow metadata, host component maps,
    and long-form host metrics can carry profile, resolution, cache, and split
    reliability diagnostics without a run-store schema bump. Compact HostSED
    state is independently versioned and can move to v2.

11. **Bounded validation availability.**
    Read-only discovery resolved the immutable 96-row AIMS-z local-coadd
    qsospec Parquet product under the external MLSpecZ data root. It contains
    complete object-specific sigma-lambda vectors and fixed pilot ordering.
    Rows 0--15 define the smoke sample; the same rows plus 16--95 define the
    bounded pilot. Validation outputs and XSL caches will be written under a
    new external temporary root without editing MLSpecZ or its products.

12. **Legacy configuration resolution.**
    Existing downstream configs set only `template_file` to
    `spectra_emiles_9.0.npz`; current APIs also pass an explicit default family.
    A single resolver in `host/templates.py` will infer `emiles_native` for
    historical defaults, `xsl_native` for the canonical XSL filename when
    family is absent/default-compatible, and `custom_native` otherwise.
    Explicit profile/family/file conflicts will fail. Workflow-level explicit
    arguments continue to override dataclass defaults using the existing rules.

## File-level implementation

- `src/qsospec/resolution.py`: add the one-sided match result and diagnostics;
  preserve the public compatibility helper.
- `src/qsospec/workflows/host/config.py`: add profile/cache fields, strict
  native-data preservation, and validation.
- `src/qsospec/workflows/host/templates.py`: add the central profile resolver,
  split fit/source library semantics, metadata/order validation, and native
  loading cache.
- `src/qsospec/workflows/host/preconvolved_templates.py`: implement exact XSL
  cache key, builder, atomic writer, loader, and provenance validation.
- `src/qsospec/workflows/host/ppxf_host.py`: replace the coarser-template veto,
  capture match diagnostics and split reliability, consume fit arrays, and use
  source arrays for HostSED state/reconstruction.
- `src/qsospec/workflows/host/plots.py`: expose profile, one-sided matching,
  cache, and reliability diagnostics without depicting coarser pixels as masks.
- `src/qsospec/workflows/host_workflow.py`, `workflows/batch.py`, and exports:
  resolve/load profiles consistently while preserving native fitting arrays.
- `src/qsospec/io/run_store.py` and `io/products.py`: round-trip diagnostics in
  existing metadata/metric structures; no schema bump unless tests prove one is
  unavoidable.
- `scripts/preconvolve_xsl_for_host_fit.py`: optional explicit CLI writing only
  to a caller-supplied external path/cache root.
- Tests: add profile resolution, one-sided matching, fit/source HostSED,
  preconvolution provenance/parity, native-array preservation, pPXF integration,
  storage compatibility, and QA diagnostics; retain existing host tests.
- Docs/reports: add the profile guide and three requested implementation/parity
  reports. E-MILES remains the documented and reported default.
