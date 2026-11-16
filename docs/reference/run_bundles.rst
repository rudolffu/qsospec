Saved run bundles
=================

A run bundle keeps fitted measurements, uncertainties, models, masks and
provenance together. Single-object and batch workflows use the same format.
Loading a model reconstructs the saved fit without running an optimizer.

Choose an operation
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Interface and guide
   * - Fit and save one object
     - ``fit_object_to_store``; :doc:`../how_to/fit_file`
   * - Fit or resume a sample
     - ``fit_batch``; :doc:`../user_guide/batch_fitting`
   * - Reopen a saved analysis
     - ``open_run`` and ``load_model``; :doc:`../how_to/inspect_run`
   * - Export selected measurements
     - ``build_science_catalog``; :doc:`../how_to/inspect_run`
   * - Plot archived objects
     - ``render_qa``; :doc:`../how_to/render_qa`
   * - Copy into the current storage format
     - ``convert_run``; see below

Directory contents
------------------

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

Copy the **whole directory** when moving a run. ``manifest.json`` records the
configuration and format. ``assets/`` contains the template arrays used to
reconstruct compact models. The index locates objects in Parquet shards;
``.staging/`` holds pending worker products.

Science measurements use long-form rows with ``section``, ``recipe_id``,
``quantity``, ``value`` and ``error`` fields. See
:doc:`measurement_dictionary` for selectors, units and uncertainty definitions.
The wide catalog is a view over these rows; it does not refit spectra.

Current format and compatibility
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Format
     - Current behavior
   * - Schema 8 (new runs)
     - Indexed multi-object shards, compact parameter models and saved
       template assets; ``model_storage="parameters"`` is the default.
   * - Schemas 5, 6 and 7
     - Readable without refitting. Existing bundles are not converted
       automatically; use a new destination for conversion.
   * - Earlier development schemas
     - Rejected by the current reader. Recreate the analysis with the
       appropriate inputs and software.

Readability and resume eligibility are separate. Resume preserves the run's
scientific configuration and storage policy; a changed model needs a new
directory. A schema-5 run without measurement-vocabulary version 2 is
readable but must not be resumed with the current vocabulary. The
:doc:`../getting_started/migration_0_2` guide also explains model identifiers
and qualified uncertainty keys.

Load by scientific ID or immutable key
--------------------------------------

The following uses the analysis written by :doc:`../how_to/fit_file`:

.. code-block:: python

   import qsospec

   run = qsospec.open_run("runs/sdss-quasar")
   model = qsospec.load_model(run, "spec-1198-52669-0040")

Scientific object IDs need not be unique. To make repeated downstream reads,
resolve the ID once and load by its internal ``object_key``:

.. code-block:: python

   object_key = run.build_object_index()["spec-1198-52669-0040"]
   model = qsospec.load_model_by_key(run, object_key)

If an ID is ambiguous, select the desired ``object_key`` from the saved object
table instead. The index supports direct object lookup without a full model
dataset scan.

Read measurements without loading models
----------------------------------------

.. code-block:: python

   measurements = run.read_measurements().to_pandas()
   print(measurements[["section", "recipe_id", "quantity"]].drop_duplicates())

``read_measurements()`` returns canonical names. For names exactly as stored
in an archived shard, use ``run.read_table("measurements")`` or
``run.read_measurements(canonical=False)``. Canonicalization does not rewrite
Parquet files. Measurement-only reads do not load reconstruction assets.

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

Compact models and explicit arrays
----------------------------------

New runs store supported line profiles, local continua, power laws,
polynomials, iron and Balmer components as fitted parameters plus versioned
evaluation recipes. pPXF models retain weights, kinematics, polynomial
coefficients, preprocessing and template-transform settings. Readers return
populated model arrays in either storage mode.

Template arrays are saved once per content hash in ``assets/``. A copied run
does not need the original template installation. Missing or corrupted assets,
or unsupported evaluator versions, raise errors when loading a model.
Unsupported custom models and unverifiable older components retain arrays;
``model_storage_components`` metadata records the fallback reason.

Before omitting an array, the writer validates reconstruction using
``rtol=1e-10`` and ``atol=1e-12`` in stored flux units, with identical NaN/Inf
locations. Input flux, errors, wavelengths, parameters and covariance retain
float64 precision. Floating array columns use lossless Zstandard compression.
The stored input is the corrected, rest-frame-normalized spectrum actually
fitted; raw uncorrected arrays are not duplicated.

For explicit array storage, set ``model_storage="arrays"`` in
``fit_object_to_store``, ``fit_batch`` or ``RunStore.create`` when creating the
run. This policy cannot be changed when resuming a schema-8 bundle.

Convert a copy without refitting
--------------------------------

Choose a destination that does not already exist:

.. code-block:: python

   compact = qsospec.convert_run("runs/sdss-quasar", "runs/sdss-quasar-compact")
   model = qsospec.load_model(compact, "spec-1198-52669-0040")

The equivalent command-line interface is:

.. code-block:: console

   python -m qsospec.io.convert_run runs/sdss-quasar runs/sdss-quasar-compact

These are alternative calls; run either one, since the first creates the
destination. Conversion preserves object keys, redshifts, measurements,
covariance and scientific configuration identity. Existing parameter recipes
are retained. Older definitions are compacted only after validation against
saved arrays; missing continuum/host state uses an array fallback.
A failed conversion leaves a marked incomplete destination for inspection
and does not edit the source.

Finalization and storage shards
-------------------------------

A normal batch finalizes when it completes. For independent jobs writing
partitions with ``finalize=False``, call ``finalize_run`` after all jobs finish,
as shown in :doc:`../user_guide/batch_fitting`. Finalization packs pending tables,
validates the index and collects obsolete generations.

Workers write checksummed private staging shards; coordinators publish them
under serialized access. The index commits all tables of an object together.
Readers hold a snapshot during model loading. Default storage shards hold up
to 128 objects or 64 MiB of uncompressed table data; a larger indivisible object
occupies its own shard. ``RunStore.create`` accepts ``shard_objects`` and
``shard_bytes`` for custom limits. ``store.compact()`` repacks tables.

Manifest counters are lightweight progress records; startup and finalization
reconcile them with saved state. An interrupted counter update does not by
itself invalidate permanent shards. See :doc:`fit_performance` for scalar
resume planning, timings and cache controls.

Host models and derived quantities
----------------------------------

AGN-aware pPXF archives retain the stellar, power-law, Fe II, Balmer,
aggregate AGN, pPXF best-fit, physical-total, closure-residual and
host-subtracted models. Host strategy, fallback, coverage, template provenance,
weights and reliability are saved with the workflow. ``host_metric`` rows
include selected width, global AGN fraction, closure diagnostics and timings.

Derived quantities are a separate calibration stage. Pass a mapping of
calibration IDs to calculators into ``compute_derived_quantities``. Each
calculator receives ``{"object": object_record, "measurements": rows}`` and
returns a quantity record or a list of records, including units and errors.
This supports changing cosmology, bolometric corrections or black-hole-mass
calibrations without refitting spectra. See :doc:`api/io` for the interface.
