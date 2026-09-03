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
- Worker processes inherit numerical-library thread settings unchanged.
- Missing coordinates become per-object failures; globally missing dust maps
  abort before the run begins.
- Workers retain a process-local run-store handle.  Parent promotion is
  decoupled from full manifest reconciliation; lightweight counters default to
  128-object intervals.
- The returned ``BatchResult.timings`` separates input loading, numerical fit,
  staging/serialization, promotion, lightweight manifest updates, and full
  reconciliation/finalization.

Resume and partitioning
-----------------------

Reusing a run directory with the identical configuration skips completed
objects. Configuration changes are rejected because the manifest is immutable.

With the default ``resume_planning="auto"``, Parquet inputs are scanned first
using only scalar identity columns.  Completed objects are classified from
their deterministic run-store shard paths.  A fully completed batch returns
before spectrum vectors, workers, or templates are loaded; a partial batch
reads only the unfinished or retried rows.  ``retry_failures=True`` retains the
historical default of retrying failures, while ``False`` treats them as
terminal skips.

``resume_planning="lightweight"`` requires this optimized behavior and raises
when row identity cannot be preserved. ``"legacy"`` retains the vector-first
path for regression checks. In particular, arbitrary Arrow filters fall back
to legacy mode under ``"auto"`` because their physical row offsets are
ambiguous. :func:`qsospec.plan_batch_resume` exposes the scalar plan without
fitting. ``BatchResult.timings`` reports identity/reconciliation time and the
numbers of vector rows loaded and avoided.

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

``compact_models=True`` is rejected in schema v5 because it was a misleading
no-op.  Per-object permanent shards remain the crash-safe authoritative layout.
