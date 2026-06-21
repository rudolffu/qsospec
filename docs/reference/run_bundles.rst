Run-bundle reference
====================

|project_name| uses one Parquet-backed run format for a single spectrum and for a
large sample. Scalar science fields remain long-form and provisional; model
components are nested records, so adding a future line recipe does not require
adding a new Parquet column.

.. code-block:: text

   run_directory/
     manifest.json
     inputs/
     objects/
     measurements/
     warnings/
     models/
     failures/
     derived/
     compact/
     qa/
     staging/

The table directories contain canonical, collision-free object shards.
``compact/`` contains convenient scalar tables produced by finalization.
Single-object runs also compact the model table. JSON is used only for
run-level provenance and shard state.

Single object
-------------

.. code-block:: python

   import qsospec

   result = qsospec.fit_object_to_store(
       "spectrum.fits",
       "runs/my_object",
       redshift=1.2,
   )

The main QA is written by default. Set ``write_qa=False`` to defer plotting or
``write_legacy_products=True`` to request the former loose CSV/JSON products.

Batch fitting
-------------

.. code-block:: python

   batch = qsospec.fit_batch(
       ["spectra-000.parquet", "spectra-001.parquet"],
       "runs/sample",
       n_workers="auto",
   )

Parquet sources are scanned once with projected columns and bounded record
batches. FITS inputs may be files, globs, directories, or CSV/Parquet manifest
tables. FITS tasks are dynamically scheduled one file at a time; Parquet
spectra use small worker microbatches.

``n_workers="auto"`` selects at most eight spawned processes and leaves one CPU
available. Each worker limits BLAS/OpenMP to one thread. ``n_workers=1`` selects
serial execution. A restricted platform without process semaphore support
falls back to serial execution.

For independent cluster jobs, use the same run directory and configuration:

.. code-block:: python

   qsospec.fit_batch(
       inputs,
       run_directory,
       num_shards=16,
       shard_index=job_index,
       finalize=False,
   )

After every job completes:

.. code-block:: python

   qsospec.finalize_run(run_directory)

Partitioning is deterministic from the internal source-and-row object key.
Workers write checksummed private staging shards; only the coordinator promotes
validated shards.

Resume and inspection
---------------------

Reusing a run directory with the same configuration skips completed objects
and retries failures by default. A changed scientific configuration is
rejected; use a new run directory or run ID.

.. code-block:: python

   run = qsospec.open_run("runs/sample")
   model = qsospec.load_model(run, "scientific-object-id")

Object IDs need not be unique. Use the internal ``object_key`` when an ID is
ambiguous.

Catalogs, derived quantities, and QA
------------------------------------

Wide science catalogs are views over the authoritative long-form
``measurements`` table. Inspect available quantities before defining a catalog:

.. code-block:: python

   measurements = run.read_table("measurements").to_pandas()
   print(
       measurements[["section", "recipe_id", "quantity"]]
       .drop_duplicates()
   )

Derived quantities are a separate calibration stage. A calculator receives an
object record and all of its long-form measurements, and returns one or more
records containing a quantity, value, errors, unit, and optional metadata.
This permits changing cosmology, bolometric corrections, or black-hole-mass
calibrations without refitting spectra.

.. code-block:: python

   qsospec.compute_derived_quantities(run, calculators)
   qsospec.render_qa(
       run,
       warning_codes=["optional_line_fit_failed"],
       sample=20,
   )

Batch fitting does not create QA figures by default. ``render_qa(...)`` can
select object IDs, warning codes, failures, deterministic random samples, or a
query against the object table. Main QA figures distinguish final fitted
pixels, pPXF emission masks, and configured not-modelled windows. Run bundles
written with schema version 2 or later preserve the pPXF masks exactly;
schema-version-1 bundles infer them from their archived host configuration
when possible. Schema version 3 additionally preserves per-complex metadata
and excluded-pixel masks, including Lyα absorption rejection.

Model rows store the corrected arrays actually fitted plus Galactic-extinction
provenance. Raw uncorrected flux arrays are not duplicated.
