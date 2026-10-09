Fit one file and archive it
===========================

Prerequisites
-------------

Configure dust maps as described in :doc:`../getting_started/dustmaps`, and
ensure the input contains RA, Dec, and redshift. The example below uses the
file's redshift and the default Planck/F99 correction.

.. code-block:: python

   result = qsospec.fit_object_to_store(
       "spectrum.fits",
       "runs/my_object",
   )

Expected outputs
----------------

The run directory contains a concise manifest, canonical Parquet datasets,
and a main QA figure. Primary paths are available in
``result.output_files``.

Common failures
---------------

- Missing RA/Dec or map files: configure ``dustmaps`` and supply coordinates,
  or use a known target-specific ``ebv_override``.
- Existing run has a different configuration: choose a new run directory.
- Reader detection fails: pass ``reader=...`` or inspect the file format.

Next: :doc:`inspect_run` and :doc:`render_qa`.
