Fit a Parquet sample
====================

Fit a small subset first, inspect its measurements and QA, then apply the
same configuration to the full sample.

Before you begin
----------------

Each row needs vector wavelength, flux and uncertainty columns, plus the
object's redshift and coordinates. The input is a spectrum table, not an
ordinary scalar catalog. Follow :doc:`sdss_parquet_inputs` for normalized
SDSS/DESI inputs and :doc:`../getting_started/dustmaps` for foreground setup.
The path below represents your own table.

Test a subset
-------------

.. code-block:: python

   import qsospec

   batch = qsospec.fit_batch(
       "spectra.parquet",
       "runs/pilot",
       row_indices=range(20),
       n_workers=1,
   )
   print(batch.n_completed, batch.n_failed, batch.n_skipped)

Start in serial mode so input or configuration errors are easy to inspect.
Use :doc:`inspect_run` to read failures and measurements, then :doc:`render_qa`
to inspect representative objects. Batch fitting defers plotting by default.

Scale up and resume
-------------------

Once you have assessed the configuration, use a separate directory for the
full sample and follow :doc:`../user_guide/batch_fitting` for a script with
parallel workers. Calling ``fit_batch`` again with the same input identities
and configuration resumes completed work. Changing the scientific model
requires a new run directory.

New runs use schema 8 with indexed shards and compact models. Finalization
packs pending tables and validates the index. The
:doc:`run-bundle reference <../reference/run_bundles>` describes storage and
compatibility.
