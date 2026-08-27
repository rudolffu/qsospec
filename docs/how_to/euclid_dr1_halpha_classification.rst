Compare RGS H-alpha narrow and broad models
=============================================

.. note::

   This command remains the H-alpha-only compatibility entry point.  New DR1
   work should use :doc:`euclid_dr1_narrow_line_classification`, which combines
   H-alpha with the dedicated He I 10833 + Pa-gamma N0/B1 comparison and writes
   the complete gold-sample evidence ledger.

The DR1 gold workflow fits observed line widths.  For the opt-in physical-type
analysis, ``qsospec`` converts intrinsic width bounds to observed bounds with a
Gaussian, constant-resolving-power line-spread function.  The initial Euclid
RGS point-source approximation is ``R=480``, or 624.6 km/s instrumental FWHM.
Consequently, the intrinsic 1200 km/s narrow/broad boundary corresponds to an
observed fitted FWHM of 1352.8 km/s.

Run the model comparison after the gold run has been finalized::

   export MLSPECZ_DATA_ROOT=/path/to/mlspecz_data
   /Users/yuming/miniforge3/bin/python \
     scripts/classify_euclid_dr1_halpha.py

The command is resumable and writes chunk products plus final tables under::

   $MLSPECZ_DATA_ROOT/outputs/qsospec/dr1_identified_gold_rgs_v1/
     classification/halpha_narrow_r480_v1/

Each covered spectrum is fit with the same continuum and H-alpha window using
two alternatives: tied narrow lines only (``N0``), and one flexible broad
H-alpha component plus the tied narrow lines (``B1``).  The output records BIC values,
observed and quadrature-deconvolved narrow widths, a two-sigma intrinsic-width
upper bound, and the same-line fraction

.. math::

   f_{\mathrm{broad},\mathrm{H}\alpha} =
   \frac{F_{\mathrm{broad},\mathrm{H}\alpha}}
        {F_{\mathrm{broad},\mathrm{H}\alpha}+
         F_{\mathrm{narrow},\mathrm{H}\alpha}}.

The fraction deliberately excludes [N II] and [S II], which would dilute the
denominator in a whole-complex fraction.

Interpretation boundary
-----------------------

This first pass is a measurement and model-comparison workflow, not a frozen
type-2 classifier.  It does not assign ``physical_class`` and it does not write
a narrow-line candidate catalogue.  In particular, it does not adopt a broad
fraction threshold or treat a BIC anchor as a purity criterion.  VI classes
are copied only for post-hoc auditing; they never enter fitting or selection.

Before publication selection, calibrate completeness and contamination with
injection/recovery spanning redshift, S/N, source extent, intrinsic narrow
width, and broad fraction.  Compare the result with independent
higher-resolution decompositions.  Classification deliberately retains a
fixed conservative ``R=480`` LSF for resolved sources rather than deriving an
object-specific width from morphology; source extension is evaluated as a
calibration stratum.

The default uses eight bounded worker processes and 32 input rows per resumable
chunk.  Use ``--workers 1`` for a deterministic serial diagnostic.  For a short
smoke run, use ``--max-chunks 1``.  Omit it to finalize all current
H-alpha-covered objects.  Re-running after the production fit has gained more
objects updates only changed input-row chunks.  ``--force`` is required to
recompute already aligned parts deliberately.

The more complex ``B2`` and ``B3`` alternatives remain available for selected
objects with asymmetric residuals through ``--broad-component-counts 1 2 3``.
They are not part of the full-sample default.
