Preprocessing order
===================

File and batch workflows process each object in this order:

1. Read and normalize the input arrays and metadata.
2. Query E(B-V) and apply observed-frame Galactic dereddening.
3. Apply the object-level host-decomposition gate.
4. Optionally fit and subtract the pPXF stellar host.
5. Fit the global continuum.
6. Select and fit emission complexes from rest-frame coverage.
7. Estimate covariance and optional Monte Carlo uncertainty.
8. Archive models, masks, metadata, warnings, and QA inputs.

Galactic extinction
-------------------

The default correction is Planck GNILC + F99 with :math:`R_V=3.1`. Flux and
one-sigma error receive the same multiplicative factor; inverse variance is
divided by the factor squared. The applied map, E(B-V), law, correction range,
coordinates, and data path are stored in result metadata.

Correction is applied exactly once. Reapplying a different correction to
already-corrected ``SpectrumData`` is rejected because the raw arrays are no
longer available.

Continuum clipping
------------------

Continuum fitting uses configured rest-frame anchor windows and masks strong
line regions. One blue-side iteration rejects pixels more than three
uncertainties below the initial continuum for wavelengths below 3500 Å,
reducing absorption-line bias.

Balmer emission is disabled when the longest valid rest wavelength is
``<= 3600`` Å. Otherwise the continuous Balmer pseudo-continuum additionally
requires enough red-side pixels above the 3646 Å edge.

Array APIs
----------

``fit_local``, ``fit_global_continuum``, and ``fit_global_lines`` do not query
external maps. Their :class:`qsospec.Spectrum` input is assumed to be ready for
fitting. Use :func:`qsospec.correct_spectrum` when explicit array
dereddening is needed.
