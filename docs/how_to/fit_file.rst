Fit one file and archive it
===========================

Prerequisites
-------------

Configure dust maps as described in :doc:`../getting_started/dustmaps`, and
ensure the input contains RA, Dec, and redshift. For a controlled test, use an
explicit E(B-V).

.. code-block:: python

   result = qsospec.fit_object_to_store(
       "spectrum.fits",
       "runs/my_object",
       redshift=1.2,
       galactic_extinction_config=qsospec.GalacticExtinctionConfig(
           ebv_override=0.0,
       ),
   )

Expected outputs
----------------

The run directory contains an immutable manifest, Parquet model and
measurement tables, compact products, and a main QA figure. Paths are also
available in ``result.output_files``.

Common failures
---------------

- Missing RA/Dec or map files: configure ``dustmaps`` or use ``ebv_override``.
- Existing run has a different configuration: choose a new run directory.
- Reader detection fails: pass ``reader=...`` or inspect the file format.

Next: :doc:`inspect_run` and :doc:`render_qa`.
