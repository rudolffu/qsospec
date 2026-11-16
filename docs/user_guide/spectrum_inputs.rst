Spectrum inputs
===============

In-memory arrays
----------------

For an end-to-end example, follow :doc:`../how_to/fit_arrays`. The variables
below represent your own data and a boolean good-pixel mask.

Create :class:`qsospec.Spectrum` from observed- or rest-frame wavelengths:

.. code-block:: python

   import qsospec

   spectrum = qsospec.Spectrum.from_arrays(
       wave,
       flux,
       err=error,             # or ivar=inverse_variance
       z=redshift,
       wave_frame="observed",
       mask=good_pixel_mask,
       flux_unit="cgs",      # or "relative"
       flux_scale=1.0,       # omit for relative spectra
       ra=ra,
       dec=dec,
   )

Internally, wavelengths are stored in the observed frame and
``spectrum.wave_rest`` is derived from redshift. The boolean ``mask`` uses
``True`` for valid pixels. In-memory arrays are assumed not to be corrected
for Galactic extinction unless ``galactic_extinction_corrected=True`` is
supplied. ``wave_frame`` declares both the wavelength frame and the matching
flux-density frame: observed inputs contain
:math:`F_{\lambda,\rm obs}`, while ``"rest"`` inputs contain
:math:`F_{\lambda,\rm rest}`. See :doc:`preprocessing`.

Files and tables
----------------

:func:`qsospec.read_spectrum` normalizes supported FITS and table inputs.
Readers recognize common wavelength, flux, error/inverse-variance, mask,
redshift, object-ID, RA, and Dec aliases.

.. code-block:: python

   data = qsospec.read_spectrum(
       "spectra.parquet",
       row_index=19,
       reader="auto",
   )

Supported paths include DESI/SPARCL-like vector-row Parquet, SDSS/LAMOST/IRAF
FITS. Inspect FITS reader selection with :func:`qsospec.detect_fits_reader`;
explicit FITS readers are ``"sdss"``, ``"lamost"`` and ``"iraf"``.

Batch input manifests are separate scalar tables listing spectrum paths and
optional row indices, redshifts, object IDs and reader choices. Use
:func:`qsospec.read_input_manifest` to inspect them, or pass a manifest to
``fit_batch``. A manifest is not itself a vector-spectrum table. The
:doc:`../how_to/sdss_parquet_inputs` guide explains normalized survey inputs.

DESI/SPARCL, SDSS, and LAMOST readers infer cgs :math:`f_\lambda` with the
usual :math:`10^{-17}` scale. IRAF inputs default to relative
:math:`f_\lambda` because their calibration is not standardized. Override a
reader explicitly when needed:

.. code-block:: python

   data = qsospec.read_spectrum(
       "spectrum.fits",
       flux_unit="cgs",
       flux_scale=1.0,
   )

Units and metadata
------------------

Every array spectrum must declare physical ``"cgs"`` or arbitrary
``"relative"`` :math:`f_\lambda`. For cgs arrays, ``flux_scale`` converts the
supplied values to
:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}` and defaults to one.
Relative spectra cannot set a scale. A DESI or SDSS survey preset counts as
unit confirmation and supplies
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
Physical cgs measurements are omitted for relative spectra.

Common checks
-------------

- Wavelength, flux, and uncertainty arrays must be one-dimensional and aligned.
- A finite redshift is required.
- ``wave_frame="rest"`` must only be used when wavelength, flux, and
  uncertainty have all already been transformed to the rest frame.
- Invalid uncertainties and masked pixels are excluded.
- File-based Galactic correction requires RA/Dec unless ``ebv_override`` is
  supplied.
