Euclid DR1 broad+narrow measurements
====================================

The measurement-first workflow reuses the immutable spectrum, global
continuum, uncertainty, mask, and adopted redshift archived by the 8,530-row
identified-gold run.  It does not rerun Galactic dereddening, the global
continuum, Fe II, or host decomposition, and it does not assign a physical
type.  VI, PCF, morphology, and photometry are copied only as audit metadata
after the spectral fit.

Scientific model
----------------

Each covered permitted-line complex uses one narrow and one broad Gaussian
family, non-negative integrated fluxes, covariance errors, the fixed
point-source ``R=480`` Gaussian LSF, and a residual straight line on top of the
archived global continuum.  The fitted widths are observed widths.  Approximate
intrinsic widths are secondary quadrature-deconvolved columns.

The four version-1 recipes are:

* H-alpha plus narrow [N II] and optional [S II].  The narrow kinematics are
  shared and the [N II] ratio is fixed.  The broad fraction is
  ``Ha_broad / (Ha_broad + Ha_narrow)``; [N II] and [S II] never enter it.
* H-beta plus the tied [O III] core pair.  No [O III] wing is fitted.  The
  fraction uses H-beta alone.
* the existing effective Mg II blend, with exactly one narrow and one broad
  component rather than the production two-broad parameterization;
* He I 10833 plus Pa-gamma with tied kinematics within each family and
  independent line fluxes.  Pa-gamma can be zero or absent without invalidating
  He I.  He I, Pa-gamma, and joint broad fractions are reported.

Every successful row uses common columns for narrow/broad/total flux and
uncertainty, flux S/N, broad fraction and covariance-propagated uncertainty,
velocities, observed component widths, approximate intrinsic widths,
equivalent width, fit statistics, bound warnings, optimizer telemetry, and
standardized residual statistics.  A fixed 2 km/s velocity grid from -30,000
to +30,000 km/s evaluates the summed narrow+broad primary-line profile.  Its
outer half-maximum span is
``total_profile_fwhm_observed_kms``; its flux-weighted second moment is
``total_profile_sigma_kms``.  A separate flag marks multi-peaked cases where a
single FWHM interval is ambiguous.

Run and resume
--------------

A bounded smoke run is mandatory during development and is capped at 100
objects.  It samples redshift coverage for every complex that is already
present in the source archive.  An active source run requires an explicit
override; part fingerprints include current object availability, so a
``not_available`` row is automatically reconsidered after that object appears.

Set the shared gold-sample root before invoking either mode:

.. code-block:: bash

   export GOLD_ROOT="$MLSPECZ_DATA_ROOT/outputs/qsospec/dr1_identified_gold_rgs_v1"

.. code-block:: bash

   python scripts/measure_euclid_dr1_broad_narrow_lines.py \
     --mode smoke \
     --output-root "$GOLD_ROOT" \
     --input "$GOLD_ROOT/input/spectra.parquet" \
     --run-directory "$GOLD_ROOT/runs/production_no_balmer_no_host_v1" \
     --workers 4 \
     --allow-active-source-run

After the source run is finalized:

.. code-block:: bash

   python scripts/measure_euclid_dr1_broad_narrow_lines.py \
     --mode production \
     --output-root "$GOLD_ROOT" \
     --input "$GOLD_ROOT/input/spectra.parquet" \
     --run-directory "$GOLD_ROOT/runs/production_no_balmer_no_host_v1" \
     --workers 24 \
     --chunk-size 128

Repeating the same command validates and skips compatible atomic part files.
Use ``--selection`` for an explicit Parquet/CSV object list, ``--max-chunks``
for bounded server tests, ``--finalize-only`` to rebuild catalogue views, and
``--force`` only for an intentional rebuild.  A changed model or source
configuration is rejected instead of mixing rows.

Products
--------

The default directory is
``$GOLD_ROOT/measurements/broad_narrow_r480_v1``.  It contains:

* atomic ``parts/part-*.parquet`` files and timing sidecars;
* authoritative ``broad_narrow_line_measurements.parquet`` with one explicit
  status per selected object/complex;
* ``broad_narrow_measurement_ledger.parquet`` in exact membership/complex
  order, including post-fit audit columns;
* convenient ``broad_narrow_measurements_wide.parquet``;
* ``qa_selection.parquet`` without automatic plot rendering;
* configuration, manifest, summary, and column dictionary JSON files.

``complete``, ``not_covered``, ``continuum_unavailable``, ``fit_failed``, and
``not_available`` are explicit statuses.  Failed and uncovered fluxes are
missing, never zero.  No product contains a physical-class count.

A finalized source run may contain terminal upstream fit failures as well as
successful archives.  Such objects remain in the measurement ledger with
``not_available`` status and their source exception recorded.  An object that
is neither archived nor present in the source failure table remains a hard
provenance error.

Exploration and selected QA
---------------------------

.. code-block:: bash

   python scripts/explore_euclid_dr1_broad_narrow_population.py \
     --measurements "$GOLD_ROOT/measurements/broad_narrow_r480_v1/broad_narrow_line_measurements.parquet"

   python scripts/render_euclid_dr1_broad_narrow_qa.py \
     --selection "$GOLD_ROOT/measurements/broad_narrow_r480_v1/qa_selection.parquet" \
     --run-directory "$GOLD_ROOT/runs/production_no_balmer_no_host_v1" \
     --output-directory "$GOLD_ROOT/measurements/broad_narrow_r480_v1/qa"

The exploration helper writes PDF and 300-dpi PNG population diagnostics and a
threshold grid.  User-supplied VI labels are permitted only as post-hoc counts.
The QA renderer is capped at 100 explicitly selected rows and refits only those
local complexes for visualization.

Run-store benchmark
-------------------

.. code-block:: bash

   python benchmarks/benchmark_qsospec_run_store.py \
     --run-directory "$GOLD_ROOT/runs/production_no_balmer_no_host_v1" \
     --sample-size 32 \
     --worker-counts 8 16 24 32 \
     --include-full-scan-comparison \
     --output run_store_benchmark.json

The benchmark is read-only.  It reports shard counts, completed-key and index
scan costs, direct per-object load percentiles, and an optional legacy
dataset-filter comparison.  It contains no hardware-dependent pass/fail rule.
