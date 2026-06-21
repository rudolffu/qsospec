Fit arrays in memory
====================

Prerequisites
-------------

You need aligned wavelength, flux, and uncertainty arrays plus a finite
redshift. Arrays should already be corrected for Galactic extinction.

.. code-block:: python

   import qsospec

   spectrum = qsospec.Spectrum.from_arrays(
       wave_obs,
       flux,
       err=error,
       z=redshift,
       wave_frame="observed",
       survey="desi",
   )
   result = qsospec.fit_global_lines(spectrum)

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

Next: :doc:`../user_guide/global_fitting` and
:doc:`../user_guide/results`.
