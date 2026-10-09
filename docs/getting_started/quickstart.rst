Quick start
===========

Fit a real SDSS DR17 quasar spectrum with the standard continuum and
emission-line models. This example uses
`spec-1198-52669-0040.fits
<https://sas.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/1198/spec-1198-52669-0040.fits>`_,
classified by the SDSS pipeline as a quasar at :math:`z=0.5764288`, with
``ZWARNING=0``.

Prerequisites
-------------

Install QSOSpec as described in :doc:`installation`, then configure the
Planck GNILC dust map following :doc:`dustmaps`. Only the default Planck map
is needed here. No pPXF host templates are required because host
decomposition is disabled.

Download and inspect the spectrum
---------------------------------

Astropy downloads the file once and reuses its local cache on subsequent
calls. Downloading requires internet access; fitting a local copy does not.

.. code-block:: python

   from astropy.utils.data import download_file
   import qsospec

   url = (
       "https://sas.sdss.org/sas/dr17/sdss/spectro/redux/26/"
       "spectra/lite/1198/spec-1198-52669-0040.fits"
   )
   path = download_file(url, cache=True)
   data = qsospec.read_spectrum(path, reader="sdss")
   print(data.redshift)        # 0.5764288306236267
   print(data.ra, data.dec)    # 135.92077, 39.286062 (degrees)
   print(data.metadata["flux_unit"], data.metadata["flux_scale"])
   # cgs 1e-17

To use a file already on disk, replace the download with:

.. code-block:: python

   path = "/path/to/spec-1198-52669-0040.fits"
   data = qsospec.read_spectrum(path, reader="sdss")

The SDSS reader converts ``loglam`` to vacuum observed wavelength in
Angstrom via ``10**loglam``. Flux density is supplied in units of
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`;
``flux_scale=1e-17`` records that scale without multiplying the input array.
``ivar`` is inverse variance in the matching supplied flux units. Non-positive
or non-finite inverse variance is excluded from fitting. Pixels with a
nonzero ``and_mask`` or ``or_mask`` are also excluded; the reader's boolean
``data.mask`` is ``True`` for unflagged pixels.

The object coordinates come from ``PLUG_RA`` and ``PLUG_DEC``, which take
precedence over the plate-pointing ``RA`` and ``DEC``. The redshift comes
from the FITS catalog extension. See :doc:`../user_guide/spectrum_inputs`
for other input formats, or :doc:`../how_to/fit_arrays` for your own arrays.

Fit the continuum and covered lines
-----------------------------------

.. code-block:: python

   result = qsospec.fit_global_lines_workflow(
       path,
       object_id="spec-1198-52669-0040",
       run_host_decomp=False,
   )
   print("Continuum converged:", result.continuum_success)
   for recipe_id, status in result.complex_statuses.items():
       print(recipe_id, status)

The file workflow applies Planck Galactic dereddening with the F99 law at
observed wavelengths, converts wavelength and flux density to the rest-frame
convention, and automatically fits the covered complexes. It retains the
input redshift from the FITS file.

For this file, the verified run has a converged continuum and ``fit`` status
for ``mgii``, ``oii_nev_neiii_hgamma`` and ``hbeta_oiii``. Hα and the UV
complexes are ``not_covered``; the rest wavelength range is approximately
2413–5825 Å.

Inspect preprocessing and plot the fit
---------------------------------------

.. code-block:: python

   dust = result.metadata["galactic_extinction"]
   print(dust["status"], dust["source"], dust["applied_ebv"])
   # applied planck 0.024079106748104095
   print("Warning codes:", result.warning_codes())
   print("Adopted redshift:", result.spectrum.z)

   figure = result.plot_qa()
   figure.savefig("sdss_quasar_qa.png", dpi=180, bbox_inches="tight")
   # In a notebook, result.show_qa() displays the figure directly.
   # To highlight excluded pixels with red crosses:
   # result.plot_qa(mark_excluded_pixels=True)

The QA figure shows the continuum, individual line components, total model
and residuals. The printed warning codes provide additional measurement
diagnostics; see :doc:`../reference/warnings` for their meanings.

The line widths in this example include instrumental broadening: the SDSS
reader does not use ``wdisp`` to construct an instrumental resolution model.

.. figure:: ../_static/sdss_quasar_qa.png
   :alt: Real SDSS quasar continuum and emission-line fit at redshift 0.5764288, with Mg II, blue optical and Hβ/[O III] zooms and fitted-pixel residuals
   :width: 100%

   The supplied SDSS spectrum after Planck/F99 correction, with
   :math:`E(B-V)=0.0241`, no host decomposition, and the original redshift.
   The QA uses rest wavelength and rest-frame flux density. The Mg II, blue
   optical and Hβ/[O III] panels show the covered line fits. Excluded input pixels
   appear in the ordinary spectrum trace, with the model evaluated across
   them. Residuals use fitted valid pixels only. Optional red markers can
   identify which input pixels were excluded.

The :download:`figure provenance <../_static/sdss_quasar_provenance.json>`
records the source URL, input SHA-256, package version, preprocessing,
complex statuses and warning codes.

Run the companion script
------------------------

Download :download:`fit_sdss_quasar.py <../../examples/fit_sdss_quasar.py>`
and run it with your configured dust map:

.. code-block:: bash

   python fit_sdss_quasar.py --output-dir validation/sdss_quickstart
   python fit_sdss_quasar.py --input /path/to/spec-1198-52669-0040.fits \
       --output-dir validation/sdss_quickstart_local

The script writes a QA PNG and ``provenance.json``. Its download cache lives
under the output directory; it does not create a run bundle. From a source
checkout, use ``python examples/fit_sdss_quasar.py`` (with ``PYTHONPATH=src``
if the package is not installed).

See :doc:`../user_guide/results` and :doc:`../user_guide/qa_plots` for
interpretation. To archive results and reload models without refitting,
continue to :doc:`../reference/run_bundles`. A separate real-spectrum example
with optional host decomposition is :doc:`../how_to/fit_j001554`.
