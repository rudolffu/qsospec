Signed local-line diagnostics
=============================

``measure_signed_line_amplitude`` measures a fixed-centre, fixed-width line
while fitting a constant or linear local baseline.  The line amplitude is
signed: negative noise excursions are retained rather than clipped to zero.
This makes the result suitable for companion-line detection tests without the
positive noise bias of a non-negative physical emission-line fit.

.. code-block:: python

   result = qsospec.measure_signed_line_amplitude(
       spectrum,
       archived_continuum,
       rest_wavelength=4960.30,
       fwhm_kms=900.0,
       fit_window=(4935.0, 4985.0),
   )
   print(result.flux, result.flux_error, result.snr)

The coefficient is an integrated line flux in the input flux scaling.  Its
uncertainty is the inverse weighted-normal-matrix uncertainty for the supplied
per-pixel errors.  ``fit_local_line_pattern`` extends the same linear model to
several independent or fixed-ratio components.  A small external width or
velocity grid can use ``measure_signed_line_grid``; the primitive itself does
not hide nonlinear optimisation.

Ordinary coverage limitations return structured statuses such as
``not_covered``, ``partial_coverage``, or ``insufficient_valid_pixels``.
Masked pixels, non-finite values, non-positive errors, and an explicit
``excluded_mask`` are omitted.

These functions are calibration-neutral.  Survey-specific thresholds,
redshift interpretation, and catalogue decisions belong in downstream code.

Selective redshift refits
-------------------------

The Euclid broad/narrow measurement command accepts
``--redshift-override-table`` for an explicitly reviewed subset.  Only rows
with ``accepted_vi`` or ``accepted_manual`` status are fitted.  The spectrum
and archived continuum remain pixel-aligned and are reframed at ``z_revised``;
the local residual continuum and all covered complexes are then remeasured.  A
separate explicit ``--measurement-directory`` is required, so the original
archive and measurement ledger cannot be overwritten accidentally.
