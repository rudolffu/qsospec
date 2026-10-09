Coverage and reliability
========================

Coverage asks whether the usable spectrum supports the requested model.
The answer depends on the recipe. It is separate from optimizer success,
feature detection, and the suitability of a particular measurement.

Whole-window recipes
--------------------

Ordinary recipes first compare their declared window with the span from the
first to the last valid rest-frame pixel. The geometric overlap must meet
``min_coverage_fraction``. The fit also needs ``min_valid_pixels`` after
input and recipe masks, and required laboratory line centers must lie inside
``edge_margin_kms`` on both sides of the valid wavelength span.

``ComplexRecipe`` defaults to 80% overlap, 30 valid pixels, and a 1000 km/s
edge margin; particular recipes and adapters can override these values.
The fraction is geometric overlap, not the fraction of unmasked pixels.
For example, the Hα adapter uses a 60% overlap threshold. Consult
:doc:`../reference/recipes` or ``qsospec.recipes.describe(recipe_id)`` for
the configuration used by a fit.

For ``coverage_mode="full"``, all required lines must pass the center rule.
For ``"component_adaptive"`` without local support, the total window must
still pass its overlap threshold, after which only covered component groups
are fitted. A component tied by a fixed flux ratio is retained only when
its amplitude parent is also active.

Compact local-support recipes
-----------------------------

Recipes with ``local_support=True`` assess their fitting windows separately.
Within each window, ``min_coverage_fraction`` is the fraction of available
native pixels that remain valid after masks, and ``min_valid_pixels`` sets
the minimum valid count. A distant missing window cannot reject a locally
supported line. This is useful for [S III] when one doublet member is outside
the observed wavelength range.

An active line must have valid support spanning its laboratory center plus
``edge_margin_kms`` on both sides. Its core, within ``core_margin_kms``,
needs at least ``min_core_pixels`` and half the expected core pixel count
estimated from the local grid spacing. The default core margin is 500 km/s
and the minimum core count is three. Masked gaps contribute no core pixels.
The upper configured component width also sets a check on support for both
half-width regions. The recorded component states are:

``observed``
   Center, core, and configured half-width support pass.

``truncated``
   Center and core pass, but the configured half-width support is incomplete.
   The component can be fitted; inspect its measured profile support.

``masked_core``
   The core has too few usable pixels; the component is excluded.

``not_observed``
   The local window or center does not pass the support rules.

``not_selected``
   The component is inactive, for example because it was disabled in the recipe.

These states are stored in ``fit.metadata["component_coverage_status"]``; local
window fractions and counts are in ``fit.metadata["window_coverage"]``.
See :doc:`nir_line_coverage` for profile-support and blend diagnostics.

Lyα/N V
-------

Lyα uses :class:`qsospec.LyaNVComplexConfig` and a separate classifier:

``full``
   Coverage reaches the configured blue and red limits, and passes geometric
   overlap, valid-pixel fraction, and valid-count cuts.

``red_side_only``
   The center is safely inside the wavelength span, the spectrum reaches
   the red limit, and the red side passes its valid-pixel fraction and
   count cuts. Blue-side coverage is insufficient for ``full``.

``edge_truncated``
   Useful overlap and enough valid pixels exist, but neither fit-eligible
   state passes.

``not_covered``
   The useful-overlap or valid-count requirement fails.

Full and red-side-only cases are fitted. Edge-truncated and not-covered cases
are skipped. After a preliminary fit, contiguous residual runs below
:math:`-3\sigma` and no wider than 2000 km/s are masked and the complex is
refitted once.

``lya_fit_reliable`` requires a successful full fit, broad Lyα flux S/N of
at least three, no more than 20% absorption-masked pixels, available
covariance, and no active Lyα kinematic bound. Red-side-only measurements
are retained with limited coverage and ``lya_fit_reliable=False``.
These are Lyα-specific requirements, not a universal line-quality rule.

Fit outcomes and QA zooms
-------------------------

``result.complex_statuses`` reports workflow outcomes such as ``fit``,
``failed``, and ``not_covered``. A fitted complex can still have partial
coverage in ``fit.metadata["coverage_status"]`` or inactive components.
A failure means an attempted model did not fit successfully; it does not
mean the wavelength region was absent. An unavailable metric or error
requires its own diagnostic. None of these states alone means nondetection
or supplies a calibrated upper limit.

QA display has a separate coverage policy. For example, a partial
optical-blue fit is archived, while its QA zoom is displayed only when
[Ne V] 3427, both [O II] lines, [Ne III] 3870, and Hγ are covered.
A missing zoom therefore need not mean a missing fit. See
:doc:`../user_guide/qa_plots` for practical assessment.

Host decomposition
------------------

Host decomposition runs when requested and the input redshift is finite
and below 1.2. Wavelength coverage and resolution then determine which
host quantities are constrained. Fractions outside the observed rest-frame
range are not reported as constrained measurements. See
:doc:`../user_guide/host_decomposition`.
