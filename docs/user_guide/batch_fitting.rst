Fit and resume a sample
=======================

``fit_batch`` applies one scientific configuration to a Parquet spectrum
table or a collection of FITS spectra. Results use the same run-bundle format
as a saved single-object fit.

Start with representative objects
---------------------------------

Follow :doc:`../how_to/fit_parquet_batch` to fit a small subset serially.
Inspect coverage, residuals, warnings and the measurements you plan to use
before processing the full sample. Configure the
:doc:`dust map <../getting_started/dustmaps>` before starting.

For FITS, inputs can be files, directories, globs or manifests. Parquet inputs
contain one spectrum per row, with vector columns; see
:doc:`spectrum_inputs` and :doc:`../how_to/sdss_parquet_inputs`.

Run in parallel
---------------

Save the following as ``fit_sample.py`` and replace the input path with your
spectrum table. Keep the main guard when using multiple processes:

.. code-block:: python

   import qsospec

   def main():
       batch = qsospec.fit_batch(
           "spectra.parquet",
           "runs/sample",
           n_workers="auto",
       )
       print("Completed:", batch.n_completed)
       print("Failed:", batch.n_failed)
       print("Skipped:", batch.n_skipped)
       print(batch.timings)

   if __name__ == "__main__":
       main()

.. code-block:: bash

   python fit_sample.py

``n_workers="auto"`` uses at most eight processes and reserves one CPU.
Use ``n_workers=1`` for serial execution. Workers inherit your numerical-library
thread settings. On a cluster, keep processes × numerical threads within the
CPUs allocated to the job.

Inspect failures and render QA
------------------------------

.. code-block:: python

   run = qsospec.open_run("runs/sample")
   failures = run.read_table("failures").to_pandas()
   print(failures)
   qsospec.render_qa(run, sample=20, random_seed=12345)

A globally missing dust map stops the run at preflight. Object-specific
errors are recorded in ``failures`` so other objects can continue. Completed
objects can still contain line-specific warnings or uncovered complexes.
Batch fitting defers QA plotting by default. Use :doc:`qa_plots` to assess
selected objects and :doc:`../how_to/inspect_run` to export measurements.

Resume an interrupted run
-------------------------

Rerun the same call with the same scientific configuration and stable input
identities. Completed objects are skipped; failed objects are retried by
default. Set ``retry_failures=False`` to leave failures as terminal skips.
A changed configuration or storage policy requires a new run directory.
Input identity can depend on the source path and row index, so preserve these
when resuming a run.

For Parquet inputs, default ``resume_planning="auto"`` checks scalar identity
columns before loading spectrum vectors. A completed sample returns without
starting fit workers. A partial sample loads unfinished or retried rows.
Unsupported filters fall back to the established vector-first path.

To inspect the scalar resume plan without fitting:

.. code-block:: python

   plan = qsospec.plan_batch_resume(
       "spectra.parquet", "runs/sample", retry_failures=True,
   )
   print(plan.completed_count, plan.unfinished_count, plan.retry_failed_count)

Pass ``retry_failures`` explicitly when comparing this plan to a batch call;
the planner defaults to ``False`` and the fitter to ``True``.
``resume_planning="lightweight"`` requires scalar planning and raises when
unsupported; ``"legacy"`` selects vector-first loading. See
:doc:`../reference/fit_performance` for diagnostics.

Partition independent jobs
--------------------------

For cluster jobs, pass the same inputs, run directory, scientific configuration
and number of partitions to every job. Each job supplies its own zero-based
``job_index`` from 0 through 15 in this example:

.. code-block:: python

   qsospec.fit_batch(
       inputs,
       run_directory,
       num_shards=16,
       shard_index=job_index,
       finalize=False,
   )

After all jobs have completed, finalize once:

.. code-block:: python

   qsospec.finalize_run(run_directory)

Job partitions are distinct from the storage shards inside a run bundle.
New schema-8 runs use indexed multi-object storage shards and compact
parameter models. See :doc:`../reference/run_bundles` for copying, conversion
and archive compatibility.
