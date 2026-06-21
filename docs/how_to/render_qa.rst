Render QA without refitting
===========================

Render selected archived objects:

.. code-block:: python

   outputs = qsospec.render_qa(
       "runs/sample",
       object_ids=["target-1", "target-2"],
       plot_config=qsospec.GlobalQAPlotConfig(
           show_smoothed_data=True,
           show_residual_panel=True,
       ),
   )

Select by warnings or object-table query:

.. code-block:: python

   qsospec.render_qa(
       "runs/sample",
       warning_codes=["optional_line_fit_failed", "lya_low_flux_snr"],
       sample=20,
   )

Set ``include_failed=True`` to create compact failure summary figures. Existing
model arrays are loaded from the run store; spectra are not refitted.

Next: :doc:`../user_guide/qa_plots`.
