Run-bundle reference
====================

|project_name| uses one Parquet-backed run format for a single spectrum and for a
large sample. Scalar science fields remain long-form and provisional; model
components are nested records, so adding a future line recipe does not require
adding a new Parquet column.

.. code-block:: text

   run_directory/
     manifest.json
     data/
       inputs/
       objects/
       measurements/
       warnings/
       models/
       failures/
       derived/
     qa/
     .staging/

The datasets contain canonical, collision-free object shards. Finalization
validates them without creating duplicate compact copies and removes empty
staging state. JSON is used only for concise run-level provenance.

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
available. Worker processes inherit numerical-library thread settings.
``n_workers=1`` selects serial execution. A restricted platform without process semaphore support
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
validated shards.  Promotion updates in-memory authoritative key sets and writes
only lightweight manifest counters at ``manifest_update_interval`` (128 objects
by default).  Exact shard counts are reconciled at startup and finalization,
not after every promoted object.  A stale counter in an interrupted manifest is
therefore recoverable from the permanent shards.

Resume and inspection
---------------------

Reusing a run directory with the same configuration skips completed objects
and retries failures by default. A changed scientific configuration is
rejected; use a new run directory or run ID.

Schema-v5 object and failure filenames are deterministic hashes of
``object_key``. Fast resume therefore combines a scalar-only Parquet identity
scan with direct existence checks for those authoritative shards. Completed
rows never reach the spectrum-vector decoder. Use
``qsospec.plan_batch_resume(...)`` to inspect expected, completed,
failed-terminal, retry-failed, and unfinished counts without fitting.

``fit_batch`` accepts ``resume_planning="auto"`` (default), ``"lightweight"``,
or ``"legacy"``. Forced lightweight mode rejects unsupported filtered scans;
automatic mode safely falls back. Telemetry records identity and manifest
planning time, worker startup, spectral loading, fitting, serialization, and
the vector rows loaded/avoided.

.. code-block:: python

   run = qsospec.open_run("runs/sample")
   model = qsospec.load_model(run, "scientific-object-id")

For scalable downstream work, build the ID index once and load by immutable
``object_key``.  This opens a constant number of hashed object shards rather
than scanning full datasets:

.. code-block:: python

   object_key = run.build_object_index()["scientific-object-id"]
   model = qsospec.load_model_by_key(run, object_key)

Object IDs need not be unique. Use the internal ``object_key`` when an ID is
ambiguous.

Catalogs, derived quantities, and QA
------------------------------------

Wide science catalogs are views over the authoritative long-form
``measurements`` table. Inspect available quantities before defining a catalog:

.. code-block:: python

   measurements = run.read_measurements().to_pandas()
   print(
       measurements[["section", "recipe_id", "quantity"]]
       .drop_duplicates()
   )

``read_measurements()`` returns the canonical measurement vocabulary. For a
forensic view of strings exactly as stored in a historical shard, use
``run.read_table("measurements")`` or
``run.read_measurements(canonical=False)``. Canonicalization never rewrites the
Parquet files.

.. _host-fraction-vocabulary:

Host-fraction vocabulary
------------------------

Run manifests record ``measurement_vocabulary_version = 2``. Wavelength-local
host measurements use source-explicit names:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Quantity
     - Meaning
   * - ``fHost_<wave>``
     - pPXF stellar-host flux density used in the final qsospec-refined sample.
   * - ``fAGN_<wave>``
     - Final qsospec AGN-continuum flux density.
   * - ``fracHost_<wave>``
     - Final host fraction using the pPXF host and final qsospec AGN continuum.
   * - ``fHost_pPXF_<wave>``
     - Direct pPXF stellar-host model sample on the fitted grid.
   * - ``fAGN_pPXF_<wave>``
     - Direct pPXF AGN nuisance-continuum sample.
   * - ``fTotal_pPXF_<wave>``
     - Direct total pPXF model sample.
   * - ``fracHost_pPXF_<wave>``
     - Direct pPXF host/total fraction at the same wavelength.
   * - ``ppxf_agn_fraction_flux_global``
     - pPXF AGN fraction integrated over fitted spectral support.

Thus ``fAGN_5100`` is a flux density, not an AGN fraction.
``fracHost_5100`` and ``fracHost_pPXF_5100`` share the same pPXF stellar-host
solution but use different AGN/total continuum definitions. When both are
finite, ``host_metric`` also contains
``deltaFracHost_final_pPXF_<wave> = fracHost_<wave> -
fracHost_pPXF_<wave>``.

Each host-sample row records its definition identifier, component sources,
rest wavelength, direct-coverage requirement, host strategy, method, and unit
in measurement metadata. Historical ``host_sample`` names such as
``fHostFit_5100`` and ``fracHost_5100`` are mapped using their section context;
the final ``continuum_sample/fracHost_5100`` name is not changed.
An existing schema-v5 run without vocabulary version 2 remains readable, but
must not be resumed; start a new run directory to avoid mixing raw v1 and v2
names in one immutable bundle.

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
pixels, pPXF emission masks, and configured not-modelled windows. Schema
version 5 stores exact pPXF masks, per-complex excluded-pixel masks and
metadata, rest wavelength, and the rest-frame-normalized arrays used by the
fit. Older development schemas are rejected and their runs should be
recreated.

Model rows store the corrected, rest-frame-normalized arrays actually fitted
plus Galactic-extinction and frame-conversion provenance. Raw uncorrected
flux arrays are not duplicated.

For AGN-aware pPXF host fits, the nested model components also retain aligned
stellar, power-law, Fe II, Balmer-continuum, high-order Balmer, aggregate AGN,
pPXF best-fit, physical-total, closure-residual, and host-subtracted arrays.
The long-form ``host_metric`` measurements include the selected width, global
AGN fraction, closure diagnostics, and stage timings. Strategy, fallback,
coverage, template provenance, weights, and reliability remain in workflow
metadata.

Notebook display
----------------

.. code-block:: python

   figure = model.plot_qa()
   model.show_qa()
   run.plot_qa("scientific-object-id")

These methods return open Matplotlib figures and do not create additional
files. ``model.qa_path`` points to the primary saved QA image when available.
