Configure Galactic dust maps
============================

File-based and batch workflows apply Galactic dereddening before all other
processing. The default is the Planck Collaboration (2016) GNILC map with the
Fitzpatrick (1999) law and :math:`R_V=3.1`.

Configure the external ``dustmaps`` directory after installation:

.. code-block:: python

   from dustmaps.config import config

   config["data_dir"] = "/path/to/dustmaps"

   from dustmaps import planck, sfd
   planck.fetch(which="GNILC")
   sfd.fetch()

The configured directory should contain ``planck/`` and ``sfd/``. If
correction is enabled, missing map files or missing RA/Dec fail fast rather
than silently fitting uncorrected data.

Common alternatives
-------------------

Use SFD with the Schlafly & Finkbeiner (2011) recalibration:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(map_name="sfd")

Supply a known E(B-V) without querying a map:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(ebv_override=0.035)

Use Wang & Chen (2019) for spectra whose observed wavelength grid extends
beyond the F99 implementation range:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(law="wang2019")

Disable the step explicitly:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(enabled=False)

Pass the object as ``galactic_extinction_config=extinction`` to file, host, or
batch workflows. Array spectra are uncorrected by default. High-level
``fit_object_to_store`` prepares them automatically; before low-level fitting,
call:

.. code-block:: python

   prepared = qsospec.prepare_spectrum(
       spectrum,
       galactic_extinction_config=extinction,
   )

Use ``galactic_extinction_corrected=True`` when constructing arrays that were
already dereddened.

Wavelength-domain behavior
--------------------------

The F99 implementation used by ``dust-extinction`` is valid only over its
supported wavelength range. By default, if an enabled correction encounters an
observed wavelength grid outside that range, qsospec records
``status="skipped_wavelength_out_of_range"`` in the Galactic-extinction
provenance, leaves the flux and uncertainty un-dereddened, and still performs
the rest-frame wavelength and :math:`F_\lambda` conversion. This keeps long
wavelength spectra fit-able while making the skipped correction explicit.

For stricter behavior, request a hard failure:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(
       wavelength_out_of_range="raise"
   )

See :doc:`../user_guide/preprocessing` for the order of operations and
:class:`qsospec.GalacticExtinctionConfig` for every field. The complete
single-object example is :doc:`../how_to/fit_j001554`.
