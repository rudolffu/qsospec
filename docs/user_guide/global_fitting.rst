Global fitting
==============

The global workflow fits a shared AGN continuum followed by all requested
emission complexes that pass their coverage policies.

.. code-block:: python

   result = qsospec.fit_global_lines(
       spectrum,
       global_config=qsospec.GlobalContinuumConfig(),
       uncertainty_config=qsospec.UncertaintyConfig(covariance=True),
   )

Default model
-------------

- Single pivoted power law (broken or automatic selection is opt-in).
- Independently broadened VW01 UV and Park22 optical Fe II templates, with
  a conditional regional Verner09 bridge sharing an empirical-template kernel.
- An optional baseline-anchored quadratic correction, disabled by default.
- Continuous Balmer bound-free plus high-order series component, with
  soft Hγ ratio refinement when its covered wings support the joint fit.
- Auto-enabled covered line recipes, including Lyα/N V, C IV, C III], Mg II,
  optical complexes, and Paschen/NIR complexes.

Recipe selection
----------------

``complexes=None`` selects all covered auto-enabled recipes. An empty list
performs a continuum-only fit. An explicit sequence limits fitting:

.. code-block:: python

   result = qsospec.fit_global_lines(
       spectrum,
       complexes=["lya_nv", "civ", "ciii", "mgii"],
   )

``result.complex_statuses`` lists the outcome for each requested complex.
Scientific measurements are stored in each successful
``result.line_complexes[recipe_id].metrics`` mapping.

Continuum configuration
-----------------------

Use :class:`qsospec.GlobalContinuumConfig` for continuum windows, components,
clipping, and optimizer settings. When Lyα has usable coverage and no explicit
global configuration is supplied, the workflow uses red-side Lyα-safe
continuum windows automatically.

See :doc:`../science/continuum_model`, :doc:`../reference/recipes`, and
:doc:`../reference/configuration`.

Alternative iron and polynomial choices
---------------------------------------

The default split iron model uses VW01 in the UV plus Park22 in the
optical. To use the single Verner et al. (2009) theoretical template across
its approximately 2000–10000 Å support:

.. code-block:: python

   config = qsospec.GlobalContinuumConfig.with_single_iron("verner09")

Polynomial correction is disabled by default, including for SDSS FITS,
SDSS Parquet, and arrays declared with ``survey="sdss"``. To enable polynomial
selection by BIC for SDSS spectra:

.. code-block:: python

   from dataclasses import replace

   config = replace(
       config,
       polynomial=qsospec.PolynomialContinuumConfig(mode="auto"),
   )

This assesses a small quadratic after the polynomial-free baseline and retains
it only if it passes improvement and safety checks. Its baseline slopes are
preserved exactly. Use ``mode="on"`` to attempt correction for any survey
without requiring this improvement score (numerical checks still apply), or
``mode="off"`` to keep the baseline only.
The default correction and normalization-change limits are both 10%.

Fitting performance
-------------------

Evaluation reuse and conservative warm starts are enabled by default through
:class:`qsospec.FitPerformanceConfig`. They reduce repeated model work and
provide compatible starting points for repeated multistart line fits.
They do not change model families, component selection thresholds, or
uncertainty methods.

For a reproducible comparison with cold starts:

.. code-block:: python

   config = qsospec.GlobalContinuumConfig(
       performance=qsospec.FitPerformanceConfig(warm_starts=False),
   )
   result = qsospec.fit_global_lines(spectrum, global_config=config)

Use ``cache_evaluations=False`` as well to disable evaluation reuse.
``result.metadata["fit_performance"]`` records the settings and evaluation
counts. See :doc:`../reference/fit_performance` for cache limits, compatible
starts, retries, and solver diagnostics.
