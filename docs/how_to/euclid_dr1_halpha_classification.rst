Compare RGS H-alpha narrow and broad models
=============================================

.. note::

   This command fits H-alpha only. For joint DR1 line evidence, use
   :doc:`euclid_dr1_narrow_line_classification`, which combines
   H-alpha with the dedicated He I 10833 + Pa-gamma N0/B1 comparison and writes
   the gold-sample line-evidence table.

The DR1 gold workflow fits observed line widths.  For the opt-in physical-type
analysis, ``qsospec`` converts intrinsic width bounds to observed bounds with a
Gaussian, constant-resolving-power line-spread function.  The initial Euclid
RGS point-source approximation is ``R=480``, or 624.6 km/s instrumental FWHM.
Consequently, the intrinsic 1200 km/s narrow/broad boundary corresponds to an
observed fitted FWHM of 1352.8 km/s.

Run the model comparison after the gold run has been finalized::

   export MLSPECZ_DATA_ROOT=/path/to/mlspecz_data
   python \
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

The fraction uses H-alpha flux only; [N II] and [S II] are excluded.

Interpretation
--------------

The output contains line measurements and model comparisons. It leaves
``physical_class`` unassigned and reports broad fractions and BIC values
without a physical-type threshold. VI classes are included for comparison
after fitting and selection.

Before publication selection, calibrate completeness and contamination with
injection/recovery spanning redshift, S/N, source extent, intrinsic narrow
width, and broad fraction.  Compare the result with independent
higher-resolution decompositions.  The fit uses the same ``R=480`` LSF for
resolved sources. Calibration
results are grouped by source extent to assess morphological broadening.

The default uses eight worker processes and 32 input rows per resumable
chunk.  Use ``--workers 1`` for a deterministic serial diagnostic.  For a short
smoke run, use ``--max-chunks 1``.  Omit it to finalize all current
H-alpha-covered objects.  Re-running after the production fit has gained more
objects updates only changed input-row chunks.  ``--force`` is required to
recompute existing compatible parts.

The more complex ``B2`` and ``B3`` alternatives remain available for selected
objects with asymmetric residuals through ``--broad-component-counts 1 2 3``.
They are not part of the full-sample default.
