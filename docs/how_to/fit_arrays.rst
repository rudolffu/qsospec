Fit arrays in memory
====================

Prerequisites
-------------

You need aligned wavelength, flux, and uncertainty arrays plus a finite
redshift. The excerpt below uses your own ``wave_obs``, ``flux``, ``error``,
``redshift``, ``ra``, and ``dec`` variables and assumes vacuum wavelengths
in Angstrom. Arrays are treated as uncorrected by default. For a supplied
input that can be run directly, use :doc:`../getting_started/quickstart`.

Here ``flux_scale=1e-17`` assumes flux density and errors in the usual SDSS
numerical scale. Use ``flux_scale=1.0`` for physical cgs arrays, or
``flux_unit="relative"`` with no scale for relative flux. See
:doc:`../getting_started/spectral_basics` for the conversions.

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

If observed-frame arrays are already dereddened, construct the spectrum with
``galactic_extinction_corrected=True`` and call
``prepare_spectrum`` to perform the rest-frame flux conversion. Only arrays
that already contain rest wavelength and rest-frame :math:`F_\lambda` should
use ``wave_frame="rest"`` and enter a low-level fitter directly.

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
- Missing dust coordinates: provide RA/Dec or a target-specific ``ebv_override``.
  Use the already-corrected declaration only for dereddened data.

Next: :doc:`../user_guide/global_fitting` and
:doc:`fit_j001554`.
