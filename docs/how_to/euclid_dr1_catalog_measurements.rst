Euclid DR1 catalogue measurements
=================================

The Euclid bridge prepares matched DESI or SDSS array spectra for the existing
resumable :func:`qsospec.fit_batch` workflow. It selects current identified,
QSO-prefixed, finite-positive-redshift rows and attaches primary/supplementary
tier and Euclid redshift provenance from the MLSpecZ science ledger.

.. code-block:: bash

   export MLSPECZ_DATA_ROOT=/path/to/mlspecz_data
   /Users/yuming/miniforge3/bin/python scripts/build_euclid_dr1_qsospec_input.py \
     --identified "$MLSPECZ_DATA_ROOT/dr1_public_catalog_v1/catalog_identified.parquet" \
     --ledger "$MLSPECZ_DATA_ROOT/dr1_public_catalog_v1/analysis/dr1_science_ledger.parquet" \
     --external-spectra /path/to/matched_desi_spectra.parquet \
     --external-survey DESI \
     --output /path/to/euclid_dr1_desi_qsospec_input.parquet

The input spectra must contain exact integer ``object_id``, wavelength and
flux arrays, and ``ivar``, ``variance``, or ``var``. Declare
``--galactic-extinction-corrected`` only when it is true for the supplied
arrays. Do not report a line or continuum measurement outside actual spectral
coverage; the normal recipe coverage gates remain authoritative.

Fit the result with :func:`qsospec.fit_batch`. Reusing the same run directory
and scientific configuration resumes completed objects. Build scalar science
tables with :func:`qsospec.build_science_catalog`; preserve continuum success,
per-complex success, warnings, coverage status, and host-decomposition status
as separate fields.

Identified-gold RGS workflow
----------------------------

The initial Euclid RGS line-measurement run has a separate, snapshot-checked
workflow.  It keeps all 8,530 strict-gold rows selected with production QC:
QSO-prefixed final class, finite positive ``z_final``, ``med_snr > 3``,
``n_invalid <= 15``, ``p_uncertain < 0.1``, and ``flag_uncertain == 0``.
The production ``n_invalid`` is authoritative and is not recomputed from raw
flux.  The 12 newly classified galaxies and four ``LIKELY_Q`` rows remain in
the input as explicit VI-class audit flags.

Large products are written outside Git under
``$MLSPECZ_DATA_ROOT/outputs/qsospec/dr1_identified_gold_rgs_v1``. Build the
reader-compatible input first:

.. code-block:: bash

   export MLSPECZ_DATA_ROOT=/Users/yuming/astro/ml_projects/dr1agn/mlspecz_data
   /Users/yuming/miniforge3/bin/python \
     scripts/build_euclid_dr1_gold_rgs_input.py

The builder consumes the materialized composite stack, its membership table,
and the latest SpecBox VI CSV. It preserves all 8,530 unique signed IDs on the
common 500-bin 12047--18734 Angstrom grid. It writes ``mask=0`` for usable
pixels, ``mask=1`` for invalid pixels, and zeros invalid inverse variances.
Redshift precedence is 688 latest SpecBox VI, 98 earlier catalogue VI, then
7,744 hybrid redshifts. The manifest records input/output hashes, schemas,
counts, mask conversion, common-grid validation, and repository commits.

The fixed first-pass fit disables both the Balmer pseudo-continuum and host
decomposition. Balmer emission lines remain fitted, as do the default UV and
optical Fe II templates and every automatically covered line complex. It uses
Planck GNILC plus F99 Galactic dereddening with ``R_V=3.1``, covariance errors,
zero Monte Carlo trials, eight local workers, bounded Parquet batches, and no
automatic failure retry. The fit writes model archives but no QA images or
legacy products.

.. code-block:: bash

   /Users/yuming/miniforge3/bin/python \
     scripts/run_euclid_dr1_gold_rgs.py --mode smoke
   /Users/yuming/miniforge3/bin/python \
     scripts/run_euclid_dr1_gold_rgs.py --mode production

The production command requires at least 12 GiB free space and a passing
64-object smoke validation. Repeating either command resumes validated object
products in the same run directory; recorded failures are not retried. The
smoke sample contains all 16 newly reviewed non-QSO/likely rows plus fixed
redshift-spanning samples from catalogue VI, new confirmed-QSO VI, and hybrid
redshift groups.

Generate the reconciliation and science tables before rendering QA:

.. code-block:: bash

   /Users/yuming/miniforge3/bin/python \
     scripts/report_euclid_dr1_gold_rgs.py
   /Users/yuming/miniforge3/bin/python \
     scripts/render_euclid_dr1_gold_rgs_qa.py

The report defines hard bad fits as batch exceptions, failed/non-finite
continuum fits, or covered complexes with status ``failed``. Parameter-bound,
covariance, optimizer-fallback, and reliability warnings remain a distinct
caution tier. Deferred QA contains every hard bad fit, at most 25 examples per
warning code, the 25 highest-chi-square otherwise completed fits, and 20
deterministic redshift/source-stratified good controls. The priority order
``hard_bad > warning > chi2_tail > good_control`` prevents duplicate plots.
This initial workflow does not perform physical type-1/type-2
reclassification, luminosities, Balmer-continuum fitting, or host
decomposition.

For the measurement-first catalogue that reuses this archive and fits one
narrow plus one broad family in H-alpha, H-beta, Mg II, and He I/Pa-gamma, see
:doc:`euclid_dr1_broad_narrow_measurements`.  That catalogue reports continuous
flux fractions and profile widths and does not assign a type-1/type-2 class.

Luminosities
------------

:func:`qsospec.monochromatic_luminosity` converts observed-frame physical
``f_lambda`` to ``L_lambda``, ``lambda L_lambda``, and ``L_nu`` with an explicit
Astropy cosmology. It propagates flux and optional host-fraction uncertainty.
:func:`qsospec.bolometric_luminosity` requires a named correction prescription;
there is intentionally no universal default correction.

Host subtraction remains object-level until the external-to-Euclid aperture,
synthetic-photometry scaling, and variability transfer is validated. A pPXF
host fraction from an external fibre spectrum must not be copied directly into
Euclid slitless flux space.
