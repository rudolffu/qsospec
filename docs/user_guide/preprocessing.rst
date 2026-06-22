Preprocessing order
===================

File and batch workflows process each object in this order:

1. Read and normalize the input arrays and metadata.
2. Query E(B-V) and apply observed-frame Galactic dereddening.
3. Convert wavelength and flux density to the rest frame:
   :math:`\lambda_{\rm rest}=\lambda_{\rm obs}/(1+z)` and
   :math:`F_{\lambda,\rm rest}=(1+z)F_{\lambda,\rm obs}`.
4. Apply the object-level host-decomposition gate.
5. Optionally fit and subtract the pPXF stellar host.
6. Fit the global continuum.
7. Select and fit emission complexes from rest-frame coverage.
8. Estimate covariance and optional Monte Carlo uncertainty.
9. Archive models, masks, metadata, warnings, and QA inputs.

Galactic extinction
-------------------

The default correction is Planck GNILC + F99 with :math:`R_V=3.1`. Flux and
one-sigma error receive the same multiplicative factor; inverse variance is
divided by the factor squared. The applied map, E(B-V), law, correction range,
coordinates, and data path are stored in result metadata.

Correction is applied exactly once. Reapplying a different correction to
already-corrected ``SpectrumData`` is rejected because the raw arrays are no
longer available.

The subsequent rest-frame conversion multiplies flux and one-sigma
uncertainty by :math:`1+z` and divides inverse variance by
:math:`(1+z)^2`. It is also idempotent and still occurs when Galactic
correction is disabled or the input was declared already corrected.

For arrays, :meth:`qsospec.Spectrum.from_arrays` defaults to
``galactic_extinction_corrected=False``. Supply RA/Dec and call
:func:`qsospec.prepare_spectrum`, or pass the spectrum to the high-level
:func:`qsospec.fit_object_to_store` workflow. Set
``galactic_extinction_corrected=True`` only when the supplied flux and
uncertainty are already dereddened.

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
external maps and require rest-frame-normalized :math:`F_\lambda`. Prepare
ordinary observed-frame arrays first with :func:`qsospec.prepare_spectrum`.
Rest-frame composite or model arrays may instead be constructed with
``wave_frame="rest"``. :func:`qsospec.correct_spectrum` remains available as
a compatibility helper.
