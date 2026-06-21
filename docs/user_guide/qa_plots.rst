Reading QA plots
================

The main QA figure combines a continuous model view with explicit shading and
residual masks that show which pixels constrained the fit.

Overview semantics
------------------

- Thin grey: observed spectrum.
- Darker grey: observed spectrum smoothed for display when the input has more
  than 4,000 wavelength pixels.
- Solid near-black: total model across all valid displayed pixels.
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

Physical spectra are displayed in
:math:`10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,
\mathring{A}^{-1}` regardless of their input ``flux_scale``. This is a
plot-only transformation; fitted values, measurements, and archived arrays
retain their native input scaling. Relative spectra retain relative
:math:`F_\lambda` units.

When Lyα is covered but not fitted, the overview upper limit uses the 99.8th
percentile of the unsmoothed displayed data rather than an incomplete model.
Clipped peaks are marked with upward indicators.

Configuration
-------------

Use :class:`qsospec.GlobalQAPlotConfig` to select raw-plus-smoothed,
smoothed-only, or raw-only display; residuals; fitted-region shading; output
format; and zoom count.

In notebooks, call ``result.plot_qa()`` for an open Matplotlib figure or
``result.show_qa()`` to display it immediately. Opened run bundles provide the
same methods with an object identifier.
