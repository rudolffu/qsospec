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
       polynomial=qsospec.PolynomialContinuumConfig(enabled=None),
   )

This assesses a small quadratic after the polynomial-free baseline and retains
it only if it passes improvement and safety checks. Its baseline slopes are
preserved exactly. Use ``enabled=True`` to attempt correction for any survey
without the improvement-score gate (safety checks still apply), or
``enabled=False`` to keep the baseline only.
The default correction and normalization-change limits are both 10%.
