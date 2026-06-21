Reading QA plots
================

The main QA figure is designed to show what the workflow actually attempted,
not merely a continuous model curve.

Overview semantics
------------------

- Thin grey: observed spectrum.
- Darker grey: observed spectrum smoothed for display.
- Solid near-black: total model on final fitted pixels only.
- Faint dashed continuum: extrapolated continuum outside fitted regions.
- Grey shading: pixels masked during the earlier pPXF host fit.
- Hatched blue-grey: selected but failed, truncated, or explicitly unmodelled
  regions.
- Residual strip: :math:`(\mathrm{data}-\mathrm{model})/\sigma` on fitted
  pixels only, with references at zero and :math:`\pm3`.

A successful emission-line fit takes display precedence over a pPXF emission
mask. Absorption pixels rejected by the Lyα refit remain marked separately.

Zoom panels
-----------

Zooms show only locally relevant broad, narrow, and wing components. The
optical-blue adaptive complex receives a zoom only when all display lines are
covered. Red-side-only Lyα panels are labeled limited and
continuum-extrapolated.

Scaling
-------

When Lyα is covered but not fitted, the overview upper limit uses the 99.8th
percentile of the unsmoothed displayed data rather than an incomplete model.
Clipped peaks are marked with upward indicators.

Configuration
-------------

Use :class:`qsospec.GlobalQAPlotConfig` to select raw-plus-smoothed,
smoothed-only, or raw-only display; residuals; fitted-region shading; output
format; and zoom count.
