Fit your own arrays
===================

Declare the wavelength frame and flux scale, prepare the spectrum, then fit
its continuum and covered line complexes.

Before you begin
----------------

Supply one-dimensional aligned arrays ``wave_obs``, ``flux`` and ``error``,
a finite ``redshift``, and object coordinates ``ra`` and ``dec`` in degrees.
Use vacuum wavelengths in Angstrom and one-sigma flux-density errors.
For inverse variance, replace ``err=error`` with ``ivar=inverse_variance``.
These are your own data; for a supplied runnable input, use the
:doc:`SDSS quickstart <../getting_started/quickstart>`.

Prepare and fit
---------------

This example assumes numerical flux and errors in the usual SDSS scale,
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
Configure :doc:`../getting_started/dustmaps` for the default Galactic
correction.

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
   print(result.continuum_success)
   print(result.complex_statuses)

Preparation applies Galactic correction at observed wavelengths, then
transforms wavelength, flux density and errors to the rest-frame convention.
The low-level fitter uses this prepared spectrum and does not query dust maps.

Match the declarations to your data
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Your input
     - Declaration
   * - Physical cgs flux density and errors
     - ``flux_unit="cgs", flux_scale=1.0``
   * - Numerical SDSS-scale flux density and errors
     - ``flux_unit="cgs", flux_scale=1e-17``
   * - Relative flux density and errors
     - ``flux_unit="relative"``; omit ``flux_scale``
   * - Already dereddened flux and errors
     - ``galactic_extinction_corrected=True``; prepare to convert frames
   * - Wavelength, flux density and errors already in the rest convention
     - ``wave_frame="rest"``; declare foreground correction separately
   * - A boolean good-pixel mask
     - ``mask=good_pixel_mask`` with ``True`` for usable pixels

``wave_frame="rest"`` declares both wavelength and flux-density frame.
Dividing wavelength alone by :math:`1+z` is insufficient. Jy inputs require
conversion from :math:`F_\nu` to :math:`F_\lambda` before construction.
See :doc:`../getting_started/spectral_basics` for these conventions.

Save while fitting
------------------

If you need to reopen the analysis later, use this **instead of** the
``prepare_spectrum``/``fit_global_lines`` calls above:

.. code-block:: python

   result = qsospec.fit_object_to_store(
       spectrum,
       "runs/my-quasar",
       object_id="my-quasar",
       write_qa=False,
   )

The saved workflow prepares the arrays automatically. Use
:doc:`inspect_run` to read the bundle and :doc:`render_qa` to plot it later.

Check the result
----------------

``result.continuum`` contains the continuum solution. ``line_complexes``
contains attempted line fits, and ``complex_statuses`` records coverage and
fit outcomes. Inspect the :doc:`QA figure <../user_guide/qa_plots>` before
extracting a :doc:`measurement <../user_guide/results>`.

For missing physical fluxes, absent complexes or dust errors, follow
:doc:`../getting_started/troubleshooting`.
