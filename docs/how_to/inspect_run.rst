Inspect a run and export measurements
=====================================

Open a saved analysis, find the quantities available for each object, and
make a wide catalog. This example uses the run written by :doc:`fit_file`.
For a sample run, replace the directory and object selection.

Read the saved tables
---------------------

.. code-block:: python

   import qsospec

   run = qsospec.open_run("runs/sdss-quasar")
   objects = run.read_table("objects").to_pandas()
   measurements = run.read_measurements().to_pandas()
   warnings = run.read_table("warnings").to_pandas()
   failures = run.read_table("failures").to_pandas()
   print(objects[["object_id", "object_key"]])

``read_measurements()`` returns the current canonical measurement vocabulary.
Use ``run.read_table("measurements")`` for names exactly as stored in an
older archive. Reading canonical names does not rewrite saved data.

Load a model without refitting
------------------------------

.. code-block:: python

   model = qsospec.load_model(run, "spec-1198-52669-0040")
   print(model.complex_statuses)
   print(model.warning_codes())

Scientific object IDs need not be unique. For repeated or ambiguous IDs,
load using the saved collision-free key:

.. code-block:: python

   object_key = objects.iloc[0]["object_key"]
   model = qsospec.load_model_by_key(run, object_key)

Choose catalog quantities explicitly
------------------------------------

Inspect the available section, recipe and quantity names:

.. code-block:: python

   available = measurements[["section", "recipe_id", "quantity"]]
   print(available.drop_duplicates().sort_values(list(available.columns)))

The :doc:`measurement dictionary <../reference/measurement_dictionary>`
connects these names to physical definitions. For broad Hβ:

.. code-block:: python

   catalog = qsospec.build_science_catalog(
       run,
       specification={
           "hbeta_fwhm_kms": {
               "section": "complex_metric",
               "recipe_id": "hbeta_oiii",
               "quantity": "Hb_broad_fwhm_kms",
           },
           "hbeta_flux_cgs": {
               "section": "complex_metric",
               "recipe_id": "hbeta_oiii",
               "quantity": "Hb_broad_flux_cgs",
           },
       },
       output_path="hbeta_measurements.parquet",
   )
   print(catalog[["object_id", "hbeta_fwhm_kms", "hbeta_fwhm_kms_err"]])

Each selector must identify at most one measurement per object. The catalog
includes a matching ``_err`` column unless ``include_error=False`` is set.
Missing values remain missing; assess statuses and warnings before selecting
a science sample. A catalog without a specification contains only the object
table, so choose the measurement columns you want explicitly.

See :doc:`../reference/run_bundles` for derived quantities and
:doc:`render_qa` for plotting from an archive.
