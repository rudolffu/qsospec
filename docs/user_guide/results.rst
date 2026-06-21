Results and measurements
========================

Global workflow
---------------

:class:`qsospec.WorkflowResult` contains:

- ``spectrum``: the spectrum fitted after preprocessing/host subtraction.
- ``total_spectrum``: the dereddened total spectrum when available.
- ``continuum`` and ``continuum_initial``.
- ``line_complexes`` keyed by recipe ID.
- ``complex_statuses`` for fitted, partial, failed, and uncovered recipes.
- host arrays and masks when requested.
- warnings, Monte Carlo summaries, metadata, and output paths.

Check ``continuum_success`` and per-complex ``success`` separately. The
workflow deliberately does not collapse heterogeneous complex outcomes into
one scientific verdict.

Complex measurements
--------------------

Each :class:`qsospec.EmissionComplexResult` exposes fitted parameters,
covariance, component models, masks, and a ``metrics`` dictionary. Metric
names are feature-specific, for example:

.. code-block:: python

   hbeta = result.line_complexes.get("hbeta_oiii")
   if hbeta is not None and hbeta.success:
       print(hbeta.metrics["Hb_broad_flux_input"])
       print(hbeta.metrics["Hb_broad_fwhm_kms"])

   lya = result.line_complexes.get("lya_nv")
   if lya is not None:
       print(lya.metadata["lya_coverage_status"])
       print(lya.metadata["lya_fit_reliable"])

Warnings
--------

``result.warning_codes()`` combines workflow, continuum, and complex warning
codes. Warnings are structured records with severity, message, and context.
See :doc:`../reference/warnings` for recommended actions.

Archived results
----------------

Use :func:`qsospec.open_run`, :func:`qsospec.load_model`, and
:func:`qsospec.build_science_catalog` to inspect run bundles without
refitting. See :doc:`../reference/run_bundles`.
