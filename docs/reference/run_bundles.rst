Run-bundle reference
====================

|project_name| uses one Parquet-backed run format for a single spectrum and for a
large sample. Science measurements are stored in long-form tables; model
components are nested records, so adding a future line recipe does not require
adding a new Parquet column.

.. code-block:: text

   run_directory/
     manifest.json
     object-index.sqlite
     assets/
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

Schema-8 runs use indexed multi-object shards, compact parameter recipes,
and content-addressed template assets. Readers return populated model arrays.
Schemas 5--7 are also readable. Finalization packs pending records, validates
the index and collects obsolete generations. Existing runs are never converted
automatically.

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
``write_legacy_products=True`` to request per-object CSV/JSON products.

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
validated shards.  Promotion updates in-memory object-key sets and writes
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
scan with direct existence checks for the saved shards. Completed
rows are skipped before loading spectrum arrays. Use
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

Wide science catalogs are views over the long-form
``measurements`` table. Inspect available quantities before defining a catalog:

.. code-block:: python

   measurements = run.read_measurements().to_pandas()
   print(
       measurements[["section", "recipe_id", "quantity"]]
       .drop_duplicates()
   )

``read_measurements()`` returns the canonical measurement vocabulary. For a
view of names exactly as stored in an archived shard, use
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
in measurement metadata. Archived ``host_sample`` names such as
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

Compact models and conversion
-----------------------------

New runs default to ``model_storage="parameters"``. Gaussian and other supported
line profiles, local continua, power laws, polynomials, iron and Balmer models
are reconstructed from saved fitted coordinates and a versioned evaluation
recipe. pPXF models retain fitted weights, kinematics, polynomial coefficients,
preprocessing and template-transform settings. Reloading does not invoke an
optimizer or perform a new spectral fit.

Required template arrays are saved once per content hash under ``assets/``.
Runs can be moved without the original template installation. Missing or
corrupted assets and unsupported evaluator versions
raise errors during reload. Unsupported custom models and unverifiable older models
retain arrays with a reason in ``model_storage_components`` metadata.

Before omitting an array, the writer checks reconstruction with ``rtol=1e-10``
and ``atol=1e-12`` in stored flux units, including identical NaN/Inf locations.
Observed inputs, errors, wavelengths, parameters and covariance retain float64
precision. Metadata and measurements remain available without reconstruction.

Use ``model_storage="arrays"`` in :func:`qsospec.fit_object_to_store`,
:func:`qsospec.fit_batch`, or :meth:`qsospec.RunStore.create` for explicit array
storage. The storage policy cannot be changed when resuming a schema-8 run.
Lossless Zstandard level 3 and byte-stream-split encoding are used for floating
array columns in both modes.

The index commits all tables of a promoted object together. Workers write
private durable staging directories; coordinators serialize publication. Shards
hold up to 128 objects or 64 MiB of uncompressed table data by default (an
indivisible larger object occupies its own shard). Set ``shard_objects`` and
``shard_bytes`` on ``RunStore.create`` to override these limits. Readers hold a
snapshot while loading a model. Replacements create immutable generations;
collection waits for snapshot readers before removing superseded files.
``store.compact()`` repacks all tables. Measurement-only reads load
tables without opening template assets.

Copy an existing run into a new destination without refitting:

.. code-block:: python

   compact = qsospec.convert_run("runs/original", "runs/compact")
   result = qsospec.load_model(compact, "object-key")

Or use the CLI:

.. code-block:: console

   python -m qsospec.io.convert_run runs/original runs/compact

The destination must not exist. Conversion preserves scientific object keys,
input redshifts, measurements, covariance and configuration identity. Existing
parameter recipes are retained; older line definitions are recovered only after
validation against archived arrays. Missing saved continuum/host evaluation
state requires an array fallback. Failed conversions retain a marked
incomplete destination for inspection and do not edit the source.

Balmer cache
------------

Fitting uses an exact workflow-local LRU cache for Balmer-series bases and
FWHM/velocity derivatives, shared across prefit, final fitting and uncertainty
trials in the same workflow. Default limits are 128 entries and 16 MiB of cached
array data. Keys include template/grid content hashes and exact floating-point
arguments; no rounding or interpolation is introduced. Cached arrays have
immutable backing storage. Each independent spectrum starts a fresh cache.

For controlled comparisons or custom cache limits:

.. code-block:: python

   from qsospec.templates.balmer_cache import balmer_cache

   with balmer_cache(enabled=False):
       result = qsospec.fit_global_lines(spectrum, config)

   with balmer_cache(max_entries=128, max_bytes=16 * 1024**2):
       result = qsospec.fit_global_lines(spectrum, config)
