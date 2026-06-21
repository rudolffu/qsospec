Batch fitting
=============

:func:`qsospec.fit_batch` applies the same scientific configuration to
Parquet or FITS samples and writes a resumable run bundle.

.. code-block:: python

   batch = qsospec.fit_batch(
       ["spectra-000.parquet", "spectra-001.parquet"],
       "runs/sample",
       n_workers="auto",
   )

Execution
---------

- Parquet inputs are scanned with projected columns and bounded record batches.
- FITS inputs can be files, directories, globs, or manifests.
- ``n_workers="auto"`` uses at most eight processes and reserves one CPU.
- Each worker limits BLAS/OpenMP to one thread.
- Missing coordinates become per-object failures; globally missing dust maps
  abort before the run begins.

Resume and partitioning
-----------------------

Reusing a run directory with the identical configuration skips completed
objects. Configuration changes are rejected because the manifest is immutable.

For independent cluster jobs:

.. code-block:: python

   qsospec.fit_batch(
       inputs,
       run_directory,
       num_shards=16,
       shard_index=job_index,
       finalize=False,
   )

Finalize after all shards complete:

.. code-block:: python

   qsospec.finalize_run(run_directory)

Batch fitting does not render all QA by default. Select objects afterward with
:func:`qsospec.render_qa`. See :doc:`../reference/run_bundles`.
