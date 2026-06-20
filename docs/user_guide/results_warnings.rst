Results and Warnings
====================

All fitting functions return structured result objects.  Warnings record
non-fatal issues that may affect measurement reliability.


FitResult
---------

.. autoclass:: qsospec.FitResult
   :noindex:
   :exclude-members: to_dict, to_table, warning_codes, summary, failed

   Result of one local line-complex optimisation.  Returned by
   :func:`~qsospec.fit_line_complex` and as individual window results in a
   :class:`LocalFitResult`.

   **Key attributes:** ``success``, ``status``, ``message``, ``theta``,
   ``param_names``, ``param_values``, ``chi2``, ``dof``, ``reduced_chi2``,
   ``model``, ``residual``, ``wave_rest_fit``, ``flux_fit``, ``err_fit``,
   ``component_models``, ``warnings``, ``metadata``.

   **Methods:** ``to_dict()``, ``to_table()``, ``warning_codes()``,
   ``summary()``, ``failed(message)`` (classmethod).  See the :doc:`API
   reference <../api>` for full documentation.


LocalFitResult
--------------

.. autoclass:: qsospec.LocalFitResult
   :noindex:

   Result of fitting multiple independent local windows.  Returned by
   :func:`~qsospec.fit_local`.

   **Key attributes:** ``success`` (overall), ``window_results`` (dict of
   :class:`FitResult` keyed by window label), ``warnings``, ``metadata``.

   Methods: ``warning_codes()``, ``to_table()``, ``summary()``.


GlobalContinuumResult
---------------------

.. autoclass:: qsospec.GlobalContinuumResult
   :noindex:

   Result of the global AGN continuum fit.  Returned by
   :func:`~qsospec.fit_global_continuum`.

   **Key attributes:** ``success``, ``status``, ``message``, ``param_values``,
   ``param_errors``, ``covariance`` (ndarray or None), ``chi2``, ``dof``,
   ``reduced_chi2``, ``wave_rest``, ``model``, ``component_models`` (power_law,
   uv_iron, optical_iron, balmer_pseudocontinuum, etc.), ``fit_mask``,
   ``clip_mask``, ``warnings``, ``metadata``, ``optimizer_result``.


EmissionComplexResult
---------------------

.. autoclass:: qsospec.EmissionComplexResult
   :noindex:

   Result of one continuum-subtracted emission-line complex fit.  Returned by
   :func:`~qsospec.fit_hbeta_complex`, :func:`~qsospec.fit_mgii_complex`,
   :func:`~qsospec.fit_halpha_complex`, and
   :func:`~qsospec.fit_lya_nv_complex`.  ``HbetaComplexResult`` is an alias.

   **Key attributes:** ``success``, ``status``, ``message``, ``selected_model``
   (e.g. ``"core"``, ``"with_wings"`` for Hβ), ``param_values``,
   ``param_errors``, ``covariance``, ``metrics`` (derived measurements: fluxes,
   FWHM, EW, moments), ``metric_errors``, ``chi2``, ``dof``, ``reduced_chi2``,
   ``bic``, ``wave_rest``, ``flux_continuum_subtracted``, ``err``, ``model``,
   ``component_models``, ``fit_mask``, ``excluded_mask``, ``warnings``,
   ``metadata``.


WorkflowResult
--------------

.. autoclass:: qsospec.WorkflowResult
   :noindex:

   Result of the complete multi-complex global workflow.  Returned by
   :func:`~qsospec.fit_global_lines` and related functions.

   **Key attributes:** ``spectrum``, ``continuum_initial``, ``continuum``,
   ``hbeta``, ``hbeta_initial``, ``mgii``, ``halpha``, ``line_complexes``
   (dict of results keyed by recipe ID), ``complex_statuses`` (dict of status
   strings), ``host_decomp_enabled``, ``total_spectrum``, ``host_fit``,
   ``host_sed``, ``host_model_on_quasar_grid``, ``host_fit_mask``,
   ``host_emission_mask``, ``host_warnings``, ``monte_carlo`` (list of
   :class:`WorkflowResult` or None), ``warnings``, ``metadata``,
   ``output_files``.

   **Properties:** ``continuum_success``, ``legacy_hbeta_success`` (deprecated).

   Methods: ``warning_codes()``, ``summary()``.


Writing output products
-----------------------

Call :func:`qsospec.write_global_line_products` to write standard output files:

.. code-block:: python

    files = qsospec.write_global_line_products(
        result,
        output_dir,
        qa_plot_config=qsospec.GlobalQAPlotConfig(
            show_host_context_in_overview=True,
        ),
    )

This writes ``*_summary.json``, ``*_measurements.csv``, ``*_full_grid.csv``,
``main_qa_<name>.png``, and optional diagnostic plots.

The Hβ compatibility wrapper :func:`qsospec.write_global_hbeta_products`
retains the legacy output paths.


FitWarning
----------

.. autoclass:: qsospec.FitWarning
   :noindex:

   An immutable warning or status record.

   **Attributes:** ``code`` (machine-readable code), ``message``
   (human-readable), ``severity`` (``"warning"``, ``"error"``, or ``"info"``),
   ``context`` (dict).

   **Method:** ``to_dict()`` — serialise the warning for storage.

   **Common warning codes:**

   ``optional_line_fit_failed``
     A non-essential complex failed.

   ``lya_fit_unreliable``
     Lyα fit has insufficient red-side coverage.

   ``lya_not_covered``
     Lyα/N V window lacks enough valid pixels.

   ``hbeta_not_covered``
     Hβ window is outside the spectrum coverage.

   ``continuum_fallback``
     Continuum fit fell back to the legacy joint solver.

   ``covariance_computation_failed``
     Hessian inversion failed.

   ``no_cgs_scale``
     Physical unit scale unknown; cgs measurements omitted.

   ``iron_no_overlap``
     Iron template has insufficient overlap with the fit window.

   ``balmer_high_n_extension_model_dependent``
     A high-n Balmer extension was used.
