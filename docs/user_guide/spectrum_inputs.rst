Spectrum inputs
===============

In-memory arrays
----------------

Create :class:`qsospec.Spectrum` from observed- or rest-frame wavelengths:

.. code-block:: python

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
supplied. See :doc:`preprocessing`.

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
FITS, and input manifests. Inspect reader selection with
:func:`qsospec.detect_fits_reader`.

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
:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}` and defaults to one.
Relative spectra cannot set a scale. A DESI or SDSS survey preset counts as
unit confirmation and supplies
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
Physical cgs measurements are omitted for relative spectra.

Common checks
-------------

- Wavelength, flux, and uncertainty arrays must be one-dimensional and aligned.
- A finite redshift is required.
- Invalid uncertainties and masked pixels are excluded.
- File-based Galactic correction requires RA/Dec unless ``ebv_override`` is
  supplied.
