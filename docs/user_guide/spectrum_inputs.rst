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
       survey="desi",
   )

Internally, wavelengths are stored in the observed frame and
``spectrum.wave_rest`` is derived from redshift. The boolean ``mask`` uses
``True`` for valid pixels.

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

Units and metadata
------------------

``SpectrumMetadata`` records wavelength and flux-density units, survey,
source, and an optional scale to cgs. DESI and SDSS presets use
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
If the scale is unknown, fitting still works but physical cgs quantities may
be omitted.

Common checks
-------------

- Wavelength, flux, and uncertainty arrays must be one-dimensional and aligned.
- A finite redshift is required.
- Invalid uncertainties and masked pixels are excluded.
- File-based Galactic correction requires RA/Dec unless ``ebv_override`` is
  supplied.
