Fit arrays in memory
====================

Prerequisites
-------------

You need aligned wavelength, flux, and uncertainty arrays plus a finite
redshift. Arrays are treated as uncorrected by default.

.. code-block:: python

   import qsospec

   spectrum = qsospec.Spectrum.from_arrays(
       wave_obs,
       flux,
       err=error,
       z=redshift,
       wave_frame="observed",
       flux_unit="cgs",
       flux_scale=1e-17,
       ra=ra,
       dec=dec,
   )
   prepared = qsospec.prepare_spectrum(spectrum)
   result = qsospec.fit_global_lines(prepared)

If the arrays are already dereddened, construct the spectrum with
``galactic_extinction_corrected=True`` and fit it directly.

Expected outputs
----------------

``result.continuum`` contains the continuum model; ``result.line_complexes``
contains successful and failed selected complexes; ``result.complex_statuses``
summarizes coverage and fit outcomes.

Common failures
---------------

- Shape mismatch: verify all arrays are one-dimensional and aligned.
- No cgs metrics: supply the survey/unit preset or explicit metadata.
- Unexpected missing complex: inspect ``complex_statuses`` and rest coverage.
- Missing dust coordinates: provide RA/Dec, use ``ebv_override``, or declare
  already-corrected arrays explicitly.

Next: :doc:`../user_guide/global_fitting` and
:doc:`fit_j001554`.
