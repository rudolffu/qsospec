# Stellar-template profile implementation report

## Outcome

The pPXF host workflow now has four resolved template profiles:

- `emiles_native` remains the public default and continues to use
  `spectra_emiles_9.0.npz`.
- `xsl_native` is an explicit, optional native-XSL profile.
- `xsl_preconvolved` is an optional exact object/grid/LSF-specific fitting
  product backed by a native XSL source library.
- `custom_native` preserves legacy custom-NPZ use.

The science spectrum is never convolved to the stellar-template resolution.
Native products are convolved only where the template is sharper than the
data. A coarser template is retained, measured, and reported as a stellar
kinematics/population caveat rather than being used as a pixel or complete-fit
veto. The existing `additional_template_sigma()` helper keeps its historical
invalid-mask behavior for compatibility; the host path uses the new
`match_template_resolution_toward_data()` result.

## Fit/source separation

`PPXFTemplateLibrary` now distinguishes fit-time arrays from native source
arrays. Native profiles share the same immutable arrays. An exact XSL cache
contains only its convolved fit matrix, pPXF grid, source indices, and exact
provenance; loading it also loads and verifies the native XSL library.

HostSED reconstruction always applies the fitted stellar weights and template
normalizations to `source_flux` on `source_wave`. Host reconstruction state v2
stores both fit-product and source-library hashes while the reader retains v1
support. The final qsospec fit continues to receive the native wavelength,
flux, uncertainty/mask, and object-specific LSF after subtraction of only the
stellar model.

## Exact XSL product

The reusable builder and CLI are implemented in:

- `src/qsospec/workflows/host/preconvolved_templates.py`
- `scripts/preconvolve_xsl_for_host_fit.py`

The cache key covers the native XSL file/matrix/wavelength identities, explicit
template ordering metadata, object key, redshift, fit interval, exact pPXF log
grid, target rest-frame sigma-lambda vector, valid-resolution mask, convolution
algorithm, and normalization convention. Products are written atomically and
read back before promotion. A cache with another object, redshift, grid, fit
range, LSF, source library, or template ordering is rejected. Partially invalid
LSF samples are represented by the hashed validity mask and remain unavailable
pixels; they do not prevent exact products for otherwise complete
object-specific LSFs.

## Inspected local template products

| Property | E-MILES | XSL |
|---|---:|---:|
| SHA-256 | `6a1b1a70db4d95f67afe0adbe254f111eb2b50412b9f311714c54ffba18c6f0e` | `b8bee334ee1705a0d4e7ff0f161990c248d68ac1b8c8eb138b0d8c05cf37d853` |
| Native array shape | `(16675, 25, 6)` | `(58642, 26, 8)` |
| Flattened SSP count | 150 | 208 |
| Wavelength coverage | 1680.20–49998.29 Å | 3500.00–24749.77 Å |
| Age range | 0.0631–15.8489 Gyr | 0.0501–15.8489 Gyr |
| Metallicity range | −1.71–0.22 | −2.20–0.20 |
| FWHM metadata | wavelength-dependent, 2.51–23.57 Å | wavelength-dependent, 0.357–3.111 Å |

Both NPZ files expose `templates`, `masses`, `lam`, `ages`, `metals`, and
`fwhm`, but they are distinct SPS families. Their age/metallicity dimensions
and flattened order are not interchangeable. Neither file contains explicit
IMF/isochrone labels, so their absence is recorded rather than inferred. XSL
covers the existing 1.0, 1.6, and 2.2 μm HostSED samples; E-MILES extends
farther into the infrared.

## Diagnostics and compatibility

Run metadata and scalar products now include profile/product/source identities,
one-sided convolution and template-coarser fractions, mismatch distributions in
Å and km/s, cache validation, timings, and the split host-continuum, host-fraction,
absorption-subtraction, stellar-kinematics, stellar-population, and HostSED
reliability fields. These fit the existing schema-v5 metadata contract, so the
run-store schema was not changed.

The Aydar-inspired `agn_pseudocontinuum_masked` basis, width selection, masks,
stellar-only subtraction, single-object/batch behavior, and final quasar model
are unchanged. Host QA reports the profile and resolution semantics without
shading coarser-template pixels as excluded.

## Validation

- Complete test suite: 315 passed; one existing capfit numerical warning.
- Ruff critical checks (`E9,F`) on all changed Python files: passed.
- Sphinx HTML and doctest builds with warnings as errors: passed (offline
  intersphinx mode); 9 doctests passed.
- Wheel and sdist built in the Conda base environment with `--no-isolation`;
  both passed `twine check`.
- The immutable 96-row AIMS-z local-coadd pilot was run only after a successful
  16-row three-profile stop-gate check. All 96 final fits completed. All 51
  objects below the host redshift gate completed pPXF for E-MILES, native XSL,
  and exact preconvolved XSL.

The bounded runs and derived caches were written outside Git. No production
sample, external template, or downstream repository was modified.

## Downstream configuration notes

Existing downstream configurations that set only the canonical E-MILES file
continue to resolve to `emiles_native` and need no change. A downstream runner
that wants XSL must permit the new `template_profile`, `template_product_kind`,
and `source_template_file` fields and, for exact caches, provide a different
fit file per object. The current strict MLSpecZ configuration parser is
E-MILES-specific and would need that explicit extension before it can
orchestrate optional XSL profiles. No such downstream edit was made here.

E-MILES remains the adopted default.
