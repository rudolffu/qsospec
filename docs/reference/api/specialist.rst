Specialist module API
=====================

Use module-qualified imports for specialist analysis and reconstruction.
Core fitting and workflow interfaces remain on :mod:`qsospec`.

Recipes
-------

.. autofunction:: qsospec.recipes.extended_quasar

Uncertainties and peaks
-----------------------

.. autofunction:: qsospec.uncertainties.measure_selected_profile
.. autofunction:: qsospec.uncertainties.recover_uncertainties
.. autofunction:: qsospec.uncertainties.measurement_key
.. autofunction:: qsospec.line_peaks.recover_line_peaks
.. autofunction:: qsospec.systemic_redshift.estimate_systemic_redshift

Resolution and luminosity
-------------------------

.. autoclass:: qsospec.resolution.SpectralResolution
   :members:
.. autofunction:: qsospec.luminosity.monochromatic_luminosity
.. autofunction:: qsospec.observed_model.reconstruct_observed_model

Classification and selected measurements
----------------------------------------

.. autoclass:: qsospec.broad_narrow_measurements.BroadNarrowMeasurementConfig
   :members:
.. autofunction:: qsospec.broad_narrow_measurements.broad_narrow_recipe
.. autofunction:: qsospec.broad_narrow_measurements.measure_broad_narrow_complex
.. autofunction:: qsospec.broad_narrow_measurements.measure_broad_narrow_complexes
.. autofunction:: qsospec.halpha_classification.fit_halpha_model_grid
.. autofunction:: qsospec.hei_pgamma_classification.fit_hei_pgamma_model_pair
.. autofunction:: qsospec.signed_lines.measure_signed_line_amplitude
.. autofunction:: qsospec.narrow_line_calibration.selection_mask

Host reconstruction and vocabulary
----------------------------------

.. autoclass:: qsospec.workflows.host.ResolvedHostTemplateProfile
   :members:
.. autofunction:: qsospec.workflows.host.reconstruct_host_sed_from_state
.. autofunction:: qsospec.workflows.host.config.resolve_host_runtime_config
.. autofunction:: qsospec.io.run_store.reconstruct_host_sed_from_run
.. autofunction:: qsospec.measurement_vocabulary.final_host_sample_name
.. autofunction:: qsospec.measurement_vocabulary.ppxf_host_sample_name
.. autofunction:: qsospec.measurement_vocabulary.canonicalize_legacy_measurement_name
