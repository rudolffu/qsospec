Classify provisional RGS narrow-line evidence
================================================

The DR1 gold workflow has a unified, opt-in model comparison for H-alpha and
He I 10833 + Pa-gamma.  It reuses archived spectra and continua; the original
production fit does not need to be rerun.  Both complexes use only two local
models: narrow-only (``N0``) and narrow plus one broad component (``B1``).

The fitted widths are observed widths.  Classification uses one fixed
conservative Gaussian line-spread function at ``R=480`` for every object,
giving an instrumental FWHM of 624.6 km/s.  The intrinsic 1200 km/s
narrow/broad boundary is therefore 1352.8 km/s in fitted observed width.  It
does not derive an effective resolution or a new FWHM from object morphology.
For an extended source the point-source instrumental width under-corrects
morphological broadening, so the inferred intrinsic width is larger and the
narrow-width gate is conservative.

Run the real 64-object smoke test first::

   export MLSPECZ_DATA_ROOT=/path/to/mlspecz_data
   /Users/yuming/miniforge3/bin/python \
     scripts/classify_euclid_dr1_narrow_lines.py \
     --mode smoke --workers 2

Then resume or start the current archived production sample::

   /Users/yuming/miniforge3/bin/python \
     scripts/classify_euclid_dr1_narrow_lines.py \
     --mode production --workers 2

``--mode`` is mandatory.  During method development, run only smoke mode; do
not launch the several-thousand-object production target merely to test code.
The production command above is retained for the later frozen analysis.

The command uses 32-object resumable parts and exact gold-membership order.  It
writes under::

   $MLSPECZ_DATA_ROOT/outputs/qsospec/dr1_identified_gold_rgs_v1/
     classification/narrow_line_r480_v2/

``line_model_comparison.parquet`` contains one row for each fitted or failed
object/complex pair. ``narrow_line_evidence.parquet`` always contains all
8,530 gold objects in input order; objects without an archived completed fit
are ``not_available``. ``provisional_narrow_candidates.parquet`` is an
analysis product, not a physical-class catalogue.

Model contracts
---------------

H-alpha N0 contains the tied narrow H-alpha, [N II], and [S II] components;
B1 adds one broad H-alpha component.  The dedicated He I/Pa-gamma recipe fits
rest 10550--11150 Angstrom.  In N0, narrow He I and Pa-gamma share velocity and
width but have independent non-negative fluxes.  B1 adds broad He I and
Pa-gamma with shared broad kinematics and independent non-negative fluxes.
Pa-gamma S/N is diagnostic and is never required: a secure narrow He I fit can
provide the He I-branch evidence by itself.  The existing broad-only
``paschen_nir`` production recipe is unchanged.

The locked provisional complex rule requires:

* both N0 and B1 to succeed;
* narrow-line S/N at least 5 (narrow He I for the He I branch);
* a two-sigma intrinsic narrow-width upper bound below 1200 km/s;
* no relevant active-bound warning in the narrow kinematics.
* the N0 standardized residual has absolute 95th percentile at most 3 and
  absolute maximum at most 8;
* no secure broad evidence.

Broad evidence requires ``BIC(N0)-BIC(B1) >= 10``, at least one fitted broad
line with S/N at least 5, and an improvement of at least 0.25 in the absolute
95th-percentile standardized residual.  Preference for N0 in BIC is diagnostic
only.  At object level, broad evidence has precedence; in its absence, either
complex may supply provisional narrow evidence.  The possible statuses are
``provisional_narrow``, ``broad_evidence``, ``poor_fit``, ``indeterminate``,
``fit_failed``, ``not_covered``, and ``not_available``.  ``physical_class``
remains null.

Calibration boundary
--------------------

Use the real-pattern injection command for a smoke test and then a larger
calibration::

   /Users/yuming/miniforge3/bin/python \
     scripts/calibrate_euclid_dr1_hei_pgamma.py --mode smoke

   /Users/yuming/miniforge3/bin/python \
     scripts/calibrate_euclid_dr1_hei_pgamma.py \
     --mode production --n-injections 2000

It spans He I S/N, intrinsic narrow width, broad width and fraction,
Pa-gamma/He I ratio, redshift, and true effective resolving power using real
RGS wavelength/error/mask patterns.  The varied true resolution tests the
fixed-R=480 classifier under source extension; it is never supplied as a
per-object fit input.  Threshold sweeps and stratified completeness, purity,
and broad-source false-narrow rates are diagnostic.  Injection purity
must reach at least 95 percent, but injection results alone never promote the
label.  Independent higher-resolution decompositions must quantify broad-line
contributions.  VI and PCF are post-selection audit references only; no VI or
PCF field enters a fit or selection expression.

After finalization, render only the selected QA rows (failures, decision
boundaries, high residuals, broad-evidence cases, and deterministic provisional
controls)::

   /Users/yuming/miniforge3/bin/python \
   scripts/render_euclid_dr1_narrow_line_qa.py --workers 2

The renderer writes PNGs plus a row-level render-status table.  It does not
create a QA plot for every fitted object.
