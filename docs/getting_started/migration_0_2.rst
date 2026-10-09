Migrating to 0.2.0
==================

Version 0.2.0 requires Python 3.10 or later. It introduces a new default
science model and clarifies the APIs developed for this release. Existing
public symbols from 0.1 remain available at package root.

Default continuum model
-----------------------

``GlobalContinuumConfig()`` identifies ``global_v2`` independently of the
package version. It retains VW01 UV and Park22 optical Fe II, independent
widths, the regional Verner09 bridge when eligible, soft Hγ refinement with
0.30 dex model-ratio tolerance, and polynomial correction off. The main
:doc:`../science/continuum_model` chapter explains activation, alternatives,
and width conventions. Workflow, continuum, and
run provenance record the model identifier alongside the effective config.

For reproduction of the previous default model, use:

.. code-block:: python

   import qsospec

   config = qsospec.GlobalContinuumConfig.legacy_v1()

This ``global_v1`` preset disables the regional bridge and uses
automatic hard Hγ synchronization. It reproduces prior continuum defaults
within the current implementation; it does not restore every old line fitter.

Configuration coordinates and modes
-----------------------------------

Iron has one width coordinate: ``fwhm_kms`` and ``fwhm_bounds`` both use
``width_mode="legacy"``, ``"kernel"``, or ``"target"``. Legacy empirical widths
are additional kernels; legacy Verner09 widths are target Gaussian-equivalent
widths. Target mode requires a justified native width and prohibits sharpening.
Unknown empirical effective widths remain unavailable. The unreleased
``kernel_fwhm_kms`` and ``target_fwhm_kms`` config fields have been removed;
these quantities remain in measurement metadata.

.. code-block:: python

   iron = qsospec.IronTemplateConfig.verner09(
       fwhm_kms=4000., width_mode="target", fwhm_bounds=(900., 10000.)
   )
   polynomial = qsospec.PolynomialContinuumConfig(mode="auto")

Polynomial modes are ``off`` (default), ``auto`` (existing SDSS eligibility and
BIC selection), and ``on`` (skip BIC selection while keeping numerical and fractional checks).
The unreleased tri-state ``enabled`` field has been removed. Hγ modes are
``off``, ``soft``, ``hard``, and ``require``; aliases ``none``/``never`` normalize
to ``off`` and ``auto``/``hard_legacy`` to ``hard``. Hβ policies are unchanged.

Host convenience arguments now default to ``None``. An explicit
``template_root``, ``template_file``, or ``host_fit_range`` overrides the
corresponding ``HostDecompConfig`` value, which overrides the package default.
Equivalent resolved configurations produce identical hash inputs.

Recipes and specialist modules
------------------------------

``qsospec.recipes.extended_quasar()`` replaces the unreleased ``nir_complete()``
name and identifies ``extended_quasar_v1``. It preserves the same full-spectrum
inventory, including standard UV/optical recipes and compact NIR regions.
``complexes=None`` keeps the normal auto-enabled inventory, including the
``paschen_nir`` umbrella. Overlapping umbrella/compact regions
remain rejected.

Specialist functions introduced in this release use their modules:

.. code-block:: python

   from qsospec.uncertainties import measure_selected_profile
   from qsospec.line_peaks import recover_line_peaks
   from qsospec.systemic_redshift import estimate_systemic_redshift
   from qsospec.resolution import SpectralResolution
   from qsospec.halpha_classification import fit_halpha_model_grid

Core fitting, configuration, host configuration, batch/resume, and model-loading
interfaces remain at root. See :doc:`../reference/api/specialist`.

Matched uncertainty products
----------------------------

Matched draws and summaries use ``measurement_key_schema="qualified_v1"``:

.. code-block:: text

   continuum_sample:fAGN_5100
   continuum_param:power_law.slope
   derived:ws22_log_l1700
   line:hbeta_oiii:Hb_broad_fwhm_kms
   line:mgii:MgII_broad_flux_input

Native ``fit.metrics``, ``fit.metric_errors`` and continuum sample names are
unchanged. Run schema 8 persists the qualified draw namespace and compact model recipes. Readers retain
schemas 5, 6 and 7, convert older qualified entries explicitly, and discard
ambiguous unqualified line aliases. Older point estimates and recorded
models are preserved; loading never refits them.
