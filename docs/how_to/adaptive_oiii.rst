Adaptive [O III] profiles
========================================

Hβ/[O III] fits use ``HbetaComplexConfig(oiii_profile_mode="adaptive")``.
The adopted redshift remains exactly the supplied value, including a DR20Q
``z_sys``. Relative velocities use the narrowest fitted gas component as a reference.
This gas component can have an offset from the systemic velocity.

.. code-block:: python

   import qsospec

   import qsospec.systemic_redshift
   from qsospec.resolution import SpectralResolution

   spectrum = qsospec.Spectrum.from_arrays(
       wavelength, flux, err=error, z=z_sys,
       wave_frame="observed", flux_unit="cgs", ra=ra, dec=dec,
       resolution=SpectralResolution(
           mode="resolving_power", values=resolving_power,
           wavelength=wavelength, source="object resolution product",
           is_object_specific=True,
       ),
   )
   result = qsospec.fit_global_lines(
       qsospec.prepare_spectrum(spectrum),
       hbeta_config=qsospec.HbetaComplexConfig(),
       complexes=("hbeta_oiii",),
   )
   fit = result.hbeta
   print(fit.metadata["oiii_components"])
   print(fit.metadata["profile_adequate"])
   print(fit.metrics["oiii_5008_full_w80_kms"])
   print(fit.metric_errors["oiii_5008_full_w80_kms"])

Omit ``resolution`` when unavailable. No survey-average resolution is assumed.
To reproduce the previous fitter, request
``HbetaComplexConfig(oiii_profile_mode="legacy")``. Setting
``fit_oiii_wings=False`` restricts either mode to one [O III] component.

Selection and optimization
----------------------------------------

The adaptive model uses the generic variable-projection engine. Narrow Hβ has
independent velocity and width parameters. Configured broad-Hβ components are
retained regardless of catalogue classification. Each [O III] component ties
velocity and width across the vacuum wavelengths 4960.30 and 5008.24 Å, with
5008/4960 integrated flux fixed to 2.98 by default. A residual constant and
slope are fitted jointly; every candidate uses identical pixels, masks,
continuum terms and Hβ components.

One- and two-component candidates are compared using convergence, a BIC
improvement of at least 20, and added-component flux S/N of at least 5. Missing
flux uncertainty fails the S/N check. Width contrast and centroid separation
are recorded as diagnostics. Initial core bounds are 70–1200 km/s
and ±1000 km/s; additional components span 70–6000 km/s and ±4000 km/s.
Centered and redshifted components are allowed.

Each candidate starts six times, including the previous solution and centered,
blue and red alternatives. The search expands to 24 starts if the two best
successful starts differ by more than one in chi-square, fewer than two succeed,
an [O III] kinematic bound is reached, or matched coherent residuals remain.
The seed defaults to 1729 (``oiii_random_seed``). Failed starts are recorded and
the successful solution with the lowest chi-square is retained.

A third component is tested only after the two-component search when three
adjacent valid 5008 pixels exceed 3σ with the same sign and have matching 4960
pixels exceeding 2σ with that sign, within ±1200 km/s of the fitted core.
Gaps break runs. The candidate must pass the BIC and S/N cuts and lower the
mean squared residual in both lines, measured on the same local pixels.

Adequacy is recorded separately from acceptance. Remaining matched residuals,
three-pixel same-sign 3σ structure in either line, local mean squared residual
above four, incomplete doublet coverage, boundary solutions, or an unsuccessful
candidate search set ``profile_adequate=False``. These diagnostics do not add
components beyond the specified doublet trigger. In particular, a real feature
visible only in 5008 can require a targeted custom model.

Resolution and measurement frames
----------------------------------------

``Spectrum.resolution`` and ``SpectrumData.resolution`` use observed-frame
vacuum wavelengths. ``sigma_lambda`` and ``fwhm_lambda`` values are observed
Angstrom, ``sigma_kms`` is in km/s, and resolving power is dimensionless. The
object survives preparation, host subtraction, noise trials and run-store
save/load. Scalar and wavelength-dependent Gaussian descriptions are supported;
invalid, missing, incompletely covered, or banded-matrix descriptions produce an
explicitly flagged observed-profile fit.

A linear Gaussian-LSF operator integrates flux into native pixel edges using
Gaussian CDF differences. Intrinsic emission bases and their derivatives pass
through this operator once. Masking selects rows after pixel edges are defined.
The already fitted host and continuum are not convolved. Approximate input
resolution remains labeled approximate. LSF descriptors and the fitted model
are archived for reconstruction.

With forward modeling, fitted component-width parameters describe the intrinsic
Gaussians. Primary profile measurements describe the observed photon profile
before detector pixel integration; quantities with ``_intrinsic_`` in their
names are separate. Intrinsic component widths at the lower bound, with missing
errors, or below three times their errors are flagged unresolved. Deconvolved
widths are conditional on the supplied LSF and model.

Full-profile outputs and uncertainties
----------------------------------------

``fit.metrics`` and ``fit.metric_errors`` include full 5008 and doublet fluxes,
continuous peaks, W80, W90, v02/v05/v10/v50/v90/v95/v98, and the fraction outside
±500 km/s of the fitted gas reference. Definitions use
``v = c log(lambda / lambda_reference)``. Examples:

- ``oiii_5008_full_flux_input`` and ``oiii_5008_full_flux_cgs``;
- ``oiii_5008_full_peak_rest_angstrom`` and ``..._observed_angstrom``;
- ``oiii_5008_full_w80_kms`` (observed-profile primary alias);
- ``oiii_5008_full_observed_input_frame_v02_kms``;
- ``oiii_5008_full_observed_core_relative_fraction_beyond_500_kms``;
- ``oiii_5008_full_intrinsic_input_frame_w80_kms`` when LSF modeling is supported.

Both doublet members are aligned in velocity before computing a doublet
fraction. The intrinsic fixed-ratio model is exactly equivalent to using 5008
alone. The observed equivalence also requires the same LSF in velocity units;
a wavelength-dependent LSF can break it. The observed doublet fraction then
uses the ratio-weighted fractions of both lines and records this distinction.

The gas reference is flagged unreliable for flux S/N below 5, flux fraction
below 5%, velocity error above 100 km/s or unavailable, or an uncertainty that
permits exchange of the two narrowest width labels. Core-relative measurements
are withheld in that case; input-frame quantiles remain available. The flags
describe the reliability of the core-relative measurements.

Uncertainties propagate the full fitted covariance, including uncertainty in
the moving core reference. Unidentified coordinates produce unavailable errors.
They condition on the fitted profile and fixed host/global continuum. Existing
requested bootstrap trials supply matched errors through the usual workflow;
these measurements never launch additional bootstrap fits. Measurement errors
do not replace intrinsic line-velocity scatter in a redshift calibration.

Saved components and diagnostics
----------------------------------------

``result.hbeta``, existing centroid measurements, broad-Hβ summary fields, and
the existing ``OIII5007_core``/``OIII5007_wing`` component keys remain available.
A third component uses ``OIII5007_wing2`` (and corresponding 4959 key).
``narrow.*`` describes the [O III] gas reference; adaptive narrow Hβ uses
``Hb_narrow.velocity_kms`` and ``Hb_narrow.fwhm_kms``. Iterate the explicit
``metadata["oiii_components"]`` list instead of assuming one wing.

Labels are sorted by width after fitting, with parameters, covariance,
component arrays and reconstruction definitions permuted together. Bounds
remain attached to the originally fitted component in the saved recipe, even
when its display label changes. The labels describe component width.
``metadata["candidate_selection"]`` records every candidate's definition,
parameters, covariance and status, start history, BIC, S/N, residuals and
acceptance reasons. The selected model and quality flags are stored separately.

Saved bundles preserve their recorded models. Reading a run never reselects
components or changes its adopted redshift. Explicit peak recovery uses saved
model definitions and validates reconstruction against archived arrays.
The optional ``qsospec.systemic_redshift.estimate_systemic_redshift(result, method="ws22")``
is a separate diagnostic and does not trigger a refit or adopt its result.

Validation artifacts
----------------------------------------

``validation/oiii_adaptive_v1/`` contains reproducible synthetic checks,
fixed-continuum comparisons for all 11 QSOFEED objects plus J1242 and J1743,
full-workflow comparisons, and a deterministic 100-object control manifest.
See its ``REPORT.md`` for measured limitations,
runtimes and the final test results.
