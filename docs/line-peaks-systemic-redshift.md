# Line peaks and optional systemic redshifts

New generic and built-in line fits record vacuum line peaks automatically.
The peak is the maximum of the selected fitted line profile, after excluding
continuum and unrelated features. It differs from a flux-weighted centroid.
Existing centroid measurements are unchanged.

```python
import qsospec

# Pass the catalog's adopted systemic redshift as the ordinary input redshift.
spectrum = qsospec.Spectrum.from_arrays(
    wave_obs, flux_obs, err=error_obs, z=dr20q_z_sys,
    wave_frame="observed", survey="sdss",
)
result = qsospec.fit_global_lines(spectrum)

fit = result.line_complexes["hbeta_oiii"]
peak = fit.metrics["hbeta_broad_peak_observed_angstrom"]
error = fit.metric_errors["hbeta_broad_peak_observed_angstrom"]

# Optional post-processing; never alters spectrum.z or refits the spectrum.
diagnostic = qsospec.estimate_systemic_redshift(result, method="ws22")
if diagnostic["status"] == "available":
    print(diagnostic["z_sys"], diagnostic["z_sys_error"])
```

The estimator is disabled unless called. It adds its returned dictionary to
`result.metadata["systemic_redshift"]`; saving this result through the existing
run-store API persists that diagnostic separately from the adopted redshift.
Calling it on a loaded result affects the in-memory result only. It does not
rewrite an existing run directory.

## Measurement definitions

Metric keys follow `<feature>_<role>_peak_rest_angstrom`,
`..._peak_observed_angstrom`, and `..._peak_velocity_kms`. `full` combines all
roles for that feature. For example, `oiii_5008_full` includes the core and wing,
whereas `hbeta_broad` excludes narrow Hβ. Selection fluxes are also saved as
`..._peak_selection_flux` for detection checks and matched uncertainty trials.

Definitions, exact component membership, reference wavelengths, and flags live
in `fit.metadata["line_peaks"]["measurements"]`. The recorded rest wavelength
uses the input redshift's frame; the observed wavelength multiplies it by
`1 + z_input`. Velocity is `c * log(peak_rest / reference_wave)`.

A gridded search includes seeds around component centers, then continuously
refines every candidate maximum. Boundary maxima, absent profiles, and competing
maxima within 1% in height are flagged. Their local covariance peak errors are
unavailable. Finite-difference derivatives are checked at two step sizes.
Non-identifiable coordinates, missing covariance, and unstable or zero
propagated errors are not replaced with zero uncertainties.

Errors use the full covariance of the relevant fitted parameters, including
ties. They condition on the chosen model, component selection, continuum, host,
and input frame. They are not a measure of model-choice systematics. Existing
bootstrap trials automatically collect peak measurements; their matched draws
can supply uncertainties and cross-line correlations without extra refits.

## WS22 calibration

The versioned `WS22_CALIBRATION` table is in `qsospec.systemic_redshift`.
It follows [Wu & Shen (2022), section 4.2](https://arxiv.org/html/2209.03987#S4.SS2)
and [Shen et al. (2016), Tables 1–3 and section 5](https://arxiv.org/html/1602.03894#S5).

| Line selection | Vacuum reference (Å) | Mean offset (km/s) | Intrinsic scatter (km/s) |
|---|---:|---:|---:|
| Broad Hβ | 4862.68 | −109 | 400 |
| Full [O III] | 5008.24 | −48 | 56 |
| Ca II K, if explicitly measured | 3934.78 | 0 | 0 |
| Unresolved [O II] | 3728.48 | +8 | 46 |
| Full Mg II | 2798.75 | −57 | 205 |
| Isolated C III] | 1908.73 | −143 | 243 |
| Full C IV | 1549.06 | −269 − 438(log L1700 − 45) | 415 |
| Si IV/O IV] blend | 1399.41 | −151 − 345(log L1700 − 45) | 477 |

Offsets are line-minus-systemic velocities. Isolated C III] uses the isolated
line's Shen et al. Table 3 calibration (−151 + 8 km/s, 243 km/s scatter), not
the different calibration for the C III]/Si III]/Al III complex. This choice
matches WS22's isolated-line selection and is explicit in the method version.

For C IV and Si IV the offset is zero below log L1700 = 44.5. Luminosity uses
the existing power law plus polynomial, physical flux scale, and the input
redshift, with the reference papers' flat cosmology (H0 = 70, Ωm = 0.3).
Its uncertainty is propagated from saved continuum covariance where recoverable.
Luminosity-dependent estimates requiring unavailable luminosity/errors are
excluded. No luminosity extrapolation beyond observed wavelength coverage is used.

Mg II and C IV redshift peaks discard Gaussian components below 5% of the line
flux. The ordinary unfiltered peaks remain available; filtered keys end in
`_full_ws22`. Selection is recomputed in existing bootstrap trials. [Ne V] and
He II do not enter the final estimate. Missing compatible profiles are skipped;
there is no additional Ca II fit or substitution of a generic stellar velocity.
Resolved doublet profiles are recorded but are not silently substituted for the
unresolved calibration selections.

Eligible lines require detection above 2σ, at least half the expected complex
pixels, and a usable peak error. Corrected line redshifts use
`1 + z_sys,line = (1 + z_line) / (1 + mean_offset/c)`.
The mean uses inverse total marginal variances. Intrinsic scatter is added to
measurement variance; shared fitted-parameter and luminosity correlations are
retained in the mean error when available. Existing complete matched bootstrap
trials take precedence for the combined statistical covariance. Unavailable
cross-fit correlations and intrinsic-scatter correlations are treated as zero
and reported as assumptions.

Clipping is one pass: reject distances from the initial weighted mean greater
than three unscaled MADs, where MAD is median absolute deviation from the median.
Skip clipping with fewer than three lines or zero MAD. No eligible lines yields
`status="unavailable"` and null `z_sys`, with per-line reasons. No fallback is
presented as a measured systemic redshift.

## Recovery and validation

```python
loaded = qsospec.load_model(store, object_id)
statuses = qsospec.recover_line_peaks(loaded)  # Explicit, no fit.
diagnostic = qsospec.estimate_systemic_redshift(loaded)
```

New runs save exact native profile definitions. For older native runs, recovery
is accepted only when reconstructed components match the archived arrays;
otherwise it reports `unsupported_profile_reconstruction`. Missing covariance
allows point measurements, not invented errors. Original arrays remain intact.

Run `examples/benchmark_line_peaks.py --input spectra.jsonl --output report.json`
on the extracted spectra used by the iron/Balmer comparison. It benchmarks
post-processing separately and guards the spectral optimizer entry points.
