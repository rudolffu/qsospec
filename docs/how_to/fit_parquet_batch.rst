Fit a Parquet sample
====================

Prerequisites
-------------

Each row should contain vector wavelength and flux columns plus uncertainty,
redshift, RA, Dec, and an object identifier.

.. code-block:: python

   batch = qsospec.fit_batch(
       "spectra.parquet",
       "runs/sample",
       row_indices=range(20),
       n_workers="auto",
   )

   print(batch.n_completed, batch.n_failed)

Expected outputs
----------------

The run bundle contains canonical resumable Parquet datasets. Finalization
validates them without producing duplicate compact tables. Repeating the
command with the same configuration resumes the run.

Common failures
---------------

- Dust-map preflight fails: configure the external data directory.
- Individual missing coordinates: inspect the ``failures`` table.
- Resume rejects configuration: start a new run directory.

Next: :doc:`../user_guide/batch_fitting` and
:doc:`../reference/run_bundles`.
