Fitting performance reference
=============================


:class:`qsospec.FitPerformanceConfig` controls exact solver evaluation reuse and
conservative warm starts. Evaluation reuse is enabled by default. Its one-entry
cache is bounded to 16 MiB per solver, including retained wavelength grids and
derivatives; oversized evaluations are not retained. This applies to the
continuum, generic line recipes, and Hβ, Mg II, Hα, and Lyα adapters. It preserves
the variable-projection Jacobian arithmetic and fitting rules.

Warm starts are enabled by default.
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

Caching preserves the results of cold fits exactly. Warm starts provide an
initial solution for a new optimization. Saved bundles include performance
diagnostics, and older bundles without these fields are also readable.
