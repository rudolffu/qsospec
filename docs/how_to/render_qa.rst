Render QA without refitting
===========================

Load saved models to inspect a single object or a selected part of a sample.
This example uses the bundle from :doc:`fit_file`.

Show one fit in a notebook
--------------------------

.. code-block:: python

   import qsospec

   run = qsospec.open_run("runs/sdss-quasar")
   model = qsospec.load_model(run, "spec-1198-52669-0040")
   figure = model.plot_qa()
   # In a notebook, model.show_qa() displays the figure directly.

``plot_qa`` returns an open Matplotlib figure. Save it with
``figure.savefig(...)`` or use ``render_qa`` to write the standard products.

Save selected figures
---------------------

.. code-block:: python

   outputs = qsospec.render_qa(
       run,
       object_ids=["spec-1198-52669-0040"],
       plot_config=qsospec.GlobalQAPlotConfig(
           show_smoothed_data=True,
           show_residual_panel=True,
       ),
   )
   print(outputs)

Files go under the run's ``qa/`` directory unless you pass ``output_dir``.
The returned mapping contains saved paths for each selected object.

Review a sample
---------------

For your own ``runs/sample`` bundle, select objects by warning code and
limit the number of figures:

.. code-block:: python

   outputs = qsospec.render_qa(
       "runs/sample",
       warning_codes=["optional_line_fit_failed", "lya_low_flux_snr"],
       sample=20,
       random_seed=12345,
   )

Selections can also use ``query`` against the object table. Set
``include_failed=True`` to include failure-summary figures. Models are
reconstructed from the archive without refitting spectra. Read
:doc:`../user_guide/qa_plots` for the meaning of the displayed curves and masks.
