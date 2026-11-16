Fit one file and save the analysis
==================================

Use ``fit_object_to_store`` when you want both an in-memory result and a
saved analysis that can be reopened without fitting again.

Before you begin
----------------

Configure :doc:`../getting_started/dustmaps`. Your file needs a redshift and
coordinates, or explicit overrides. The following example uses ``path``
from the :doc:`SDSS quickstart <../getting_started/quickstart>`.

Fit and write a run bundle
--------------------------

.. code-block:: python

   import qsospec

   result = qsospec.fit_object_to_store(
       path,
       "runs/sdss-quasar",
       object_id="spec-1198-52669-0040",
       run_host_decomp=False,
       write_qa=True,
   )
   print(result.continuum_success)
   print(result.complex_statuses)
   print(result.output_files)

The directory contains ``manifest.json``, an object index, Parquet datasets,
reconstruction assets and the main QA figure. ``write_qa=False`` defers saved
plotting; ``result.show_qa()`` remains available in a notebook.

For your own file, replace ``path`` and choose an object ID and output
directory. Pass ``redshift=...`` to override the file's redshift.
``reader="auto"`` selects a supported reader; use ``read_spectrum`` to
inspect the input before fitting, as in :doc:`../user_guide/spectrum_inputs`.

Reopen the saved fit
--------------------

.. code-block:: python

   run = qsospec.open_run("runs/sdss-quasar")
   loaded = qsospec.load_model(run, "spec-1198-52669-0040")
   print(loaded.complex_statuses)

New runs use compact parameter models. Reloading reconstructs model arrays
from fitted coordinates and saved assets; it does not run the optimizer.
See :doc:`../reference/run_bundles` for portability and explicit array storage.

If fitting stops
----------------

- Missing map or coordinates: follow :doc:`../getting_started/troubleshooting`.
- Different configuration in an existing directory: choose a new directory.
- Reader detection fails: inspect the format or specify ``reader`` explicitly.

Continue to :doc:`inspect_run` for measurements and catalog export, or
:doc:`render_qa` to save more figures from the bundle.
