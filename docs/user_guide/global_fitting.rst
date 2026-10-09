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

- Pivoted power law.
- Independently broadened UV and optical Fe II templates when covered.
- An optional baseline-anchored quadratic correction, disabled by default.
- Continuous KD13-style Balmer bound-free plus high-order series component.
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

Inspect ``result.complex_statuses`` before assuming a requested complex was
fitted. Scientific measurements are stored in each successful
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

The default split iron model remains VW01 in the UV plus Park22 in the
optical. To use the single Verner et al. (2009) theoretical template across
its approximately 2000–10000 Å support:

.. code-block:: python

   config = qsospec.GlobalContinuumConfig.with_single_iron("verner09")

Polynomial correction is disabled by default, including for SDSS FITS,
SDSS Parquet, and arrays declared with ``survey="sdss"``. No candidate is
assessed unless requested. To opt into BIC-gated SDSS assessment:

.. code-block:: python

   from dataclasses import replace

   config = replace(
       config,
       polynomial=qsospec.PolynomialContinuumConfig(mode="auto"),
   )

This assesses a small quadratic after the polynomial-free baseline and retains
it only if it passes improvement and safety checks. Its baseline slopes are
preserved exactly. Use ``mode="on"`` to attempt correction for any survey
without the improvement-score gate (safety checks still apply), or
``mode="off"`` to keep the baseline only.
The default correction and normalization-change limits are both 10%.

Fitting performance
-------------------

:class:`qsospec.FitPerformanceConfig` controls exact solver evaluation reuse and
conservative warm starts. Evaluation reuse is enabled by default. Its one-entry
cache is bounded to 16 MiB per solver, including retained wavelength grids and
derivatives; oversized evaluations are not retained. This applies to the
continuum, generic line recipes, and Hβ, Mg II, Hα, and Lyα adapters. It preserves
the variable-projection Jacobian arithmetic and fitting rules.

Conservative warm starts are enabled by default after the six-source validation.
They seed only the first start of repeated compatible
multistart line fits, leaving the other alternatives unchanged. Single-start
fits and continuum searches retain their original initialization. Failed or
inadequate warm searches are discarded and retried with the original cold
search. Raw-spectrum host prefits, host-subtracted fits, individual spectra and
uncertainty trials have independent sessions.

.. code-block:: python

   performance = qsospec.FitPerformanceConfig(warm_starts=False)
   # Cache-enabled, cold-start control:
   config = qsospec.GlobalContinuumConfig(performance=performance)
   result = qsospec.fit_global_lines(spectrum, global_config=config)

   # Explicit cold/cache-disabled control:
   control = qsospec.FitPerformanceConfig(
       cache_evaluations=False, warm_starts=False,
   )
   result = qsospec.fit_global_lines(spectrum, performance=control)

Standalone fitters also accept the keyword-only ``performance=`` argument;
an explicit argument overrides the configured setting. Native local joint
fitters accept the controls and record optimizer statistics but do not use the
separable-evaluation cache or add warm searches. The coupled Hγ refinement
retains its joint optimizer; its compiled contexts receive shared bookkeeping
improvements.

``result.metadata["fit_performance"]`` records settings, evaluator/cache counts,
optimizer calls, linear solves, and warm/cold-retry counts. The peak retained
cache size is the largest single solver cache observed in that session, rather
than a measurement of process RSS. Candidate ``multistart`` records preserve
warm-attempt starts, provenance, retry reasons and cold retry starts.

Cached cold fits must agree exactly with uncached cold fits. Warm starts can
change optimizer paths, so performance and scientific agreement require matched
validation. They never reuse a fit in place of optimization, alter the adopted
redshift, or add uncertainty refits. Saved bundles retain these diagnostics;
older bundles without them remain readable.
