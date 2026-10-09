Understand and estimate uncertainties
=========================================

A measurement error describes variation under a specified fit and noise
model. Start with the statistical errors returned by the fit, then assess
continuum, template and profile choices that matter for your measurement.
The :doc:`../reference/measurement_dictionary` lists the exact value and error
keys; :doc:`../user_guide/results` shows how to read them for the SDSS example.

Default calculation
-----------------------

The workflow defaults are:

.. code-block:: python

   import qsospec

   uncertainty = qsospec.UncertaintyConfig(
       covariance=True,
       monte_carlo_trials=0,
       refit_host_in_mc=True,
       pixel_covariance=None,
   )

This requests local covariance errors and performs no Monte Carlo trials.
``refit_host_in_mc`` takes effect only when host decomposition runs and the
trial count is positive. The covariance calculation uses the fitted
parameters and their local correlations to propagate errors into measurements.
For a quantity :math:`q(\boldsymbol{\theta})`, this is the local approximation

.. math::

   \operatorname{Var}(q) \simeq
   \nabla q^{\mathsf T} C_{\theta}\nabla q.

The gradient follows the complete measured profile or continuum sample.
Off-diagonal terms in the fitted parameter block matter: adding component
errors in quadrature generally does not reproduce an error on their sum.

What varies and what stays fixed
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Calculation
     - Variation represented
     - Fixed assumptions
   * - Continuum covariance
     - Local fitted continuum parameters, including amplitude and shape
       correlations within their fitted block.
     - Host subtraction, selected templates and continuum model. A
       baseline-anchored polynomial block is conditional on the baseline
       slopes; unknown correlations with those slopes are not supplied.
   * - Sequential line covariance
     - Fitted line parameters and their correlations, propagated to the
       selected line sum.
     - Global continuum, host subtraction, component selection and fixed
       ratios. Independent fit blocks have no measured cross-covariance.
   * - Joint Hγ refinement
     - Joint continuum/blue-complex parameters, including their correlations
       on the union of the fitted masks.
     - Host subtraction, templates and the adopted Hγ model-ratio tolerance.
       Other separately fitted complexes are outside this joint block.
   * - AGN-workflow bootstrap
     - Pixel noise added to the final fitted model, followed by continuum
       and line refitting. Requested recipes and their selection rules run
       on each realization.
     - Input redshift, masks, templates, configuration and any host model
       already subtracted.
   * - Host-refit bootstrap
     - Pixel noise added to the observed input, followed by the selected host
       strategy and the final AGN fit. Host and AGN measurements share trial
       identifiers.
     - Input redshift, templates and scientific configuration. This uses
       observed-spectrum perturbations rather than final-model realizations.

Ordinary covariance blocks use residual noise scaling. When a width prior is
present, only the data information receives that scale. The joint Hγ block
uses absolute pixel errors and the configured prior scatter. The saved
``covariance_noise_scaling`` and covariance-block metadata identify the rule.
The default 0.30-dex Hγ ratio tolerance is a modelling constraint; it is not
the observational error on Hγ.

Errors near a bound can be asymmetric, and degenerate parameters can have
unavailable or unstable local errors. Inspect ``parameter_at_bound``,
``covariance_rank_deficient`` and the relevant model diagnostics. A missing,
``None`` or nonfinite error is unavailable; it does not mean zero error.

Request repeated fits
-------------------------

For a FITS spectrum readable by qsospec, with coordinates and redshift in
its metadata and the :doc:`../getting_started/dustmaps` installed:

.. code-block:: python

   import qsospec

   result = qsospec.fit_global_lines_workflow(
       "spectrum.fits",  # replace with your input path
       run_host_decomp=False,
       uncertainty_config=qsospec.UncertaintyConfig(
           monte_carlo_trials=20,
           random_seed=21,
       ),
   )

Twenty trials are a short check of the resampling workflow, not evidence
that interval tails are stable. Assess stability with larger independent
trial sets before using percentile intervals in an analysis. The original
best-fit point estimates remain unchanged. When trials are requested, their
standard deviations replace the available native measurement errors, and
their 16th, 50th and 84th percentiles are stored separately.

For host-inclusive trials, use the same call with ``run_host_decomp=True``,
your external stellar-template paths and your assessed ``host_config``;
keep ``refit_host_in_mc=True``. See
:doc:`../how_to/agn_aware_ppxf_host_decomposition` for prerequisites and the
host iteration structure. Setting it to ``False`` keeps the fitted host
fixed while resampling the subsequent AGN fit.

``UncertaintyConfig(pixel_covariance=matrix)`` accepts a finite, symmetric,
positive-semidefinite covariance on the original input grid in its flux
units. The matrix controls correlated noise draws. The fitting objective
still uses the spectrum's pixel errors. For a host-wrapper call, supply the
matrix in the observed input frame used by that wrapper.

Check usable trials
-----------------------

Trial counts answer different questions:

.. code-block:: python

   trials = result.monte_carlo
   print(trials["method"])
   print(trials["n_requested"], trials["n_successful"])
   print(trials["continuum_success_count"])
   print(trials["complex_success_counts"])
   print(trials["failures"])

   key = "line:hbeta_oiii:Hb_broad_fwhm_kms"
   print(trials["valid_trial_counts"].get(key, 0))
   print(trials["errors"].get(key))
   print(trials["percentiles"].get(key))

The AGN-only bootstrap rejects a realization when its continuum fit fails.
In the host-refit path, ``n_successful`` counts returned workflows without an
exception; continuum and complex convergence counts are separate. For either
path, use ``valid_trial_counts`` for the particular quantity. A failed complex
does not contribute its metrics. A percentile can be recorded with one usable
value, but a standard deviation requires at least two. Very small usable
counts provide weak interval estimates even when an error is finite.

``measurement_covariance`` uses complete matched realizations of its recorded
``covariance_measurement_names``. ``covariance_trial_ids`` gives that common
subset; it can be smaller than the usable count of any single measurement.
Names use the ``qualified_v1`` namespaces ``line:<recipe>:<metric>``,
``continuum_sample:<name>``, ``continuum_param:<name>`` and ``derived:<name>``.
Native ``metrics`` and ``metric_errors`` retain their feature-specific names.

For a fitted generic complex, ``qsospec.uncertainties.measure_selected_profile``
can propagate its local covariance or reuse supplied matched parameter draws
for an explicit component selection. The selection and definitions are saved
in complex metadata. Its optional EW error is conditional on the continuum
supplied to that measurement. Specialized adapter results are not accepted
by this function. See :doc:`measurements` for its EW and half-maximum
conventions.

Assess uncertainty beyond pixel noise
-----------------------------------------

Refitting noise realizations measures sensitivity to the chosen noise and
refitting scheme. It does not vary template families, calibration relations,
foreground maps or an alternative scientific model unless you explicitly run
those alternatives. Compare plausible continuum/host and line models on the
same data and masks when their differences could affect the measurement.
Report that model sensitivity separately from the statistical error.

Host fractions need matched host--AGN information or an explicitly supplied
joint covariance. ``qsospec.uncertainties.host_agn_covariance`` transforms a
supplied joint host/AGN block; it does not infer missing correlations. A
luminosity calibration, black-hole-mass relation or systemic-redshift
calibration has its own intrinsic scatter and calibration uncertainty.
These are separate from conditional spectral-fit errors.

Save and recover the assumptions
------------------------------------

Run bundles retain available parameter ordering, covariance blocks, trial
identifiers, usable counts, intervals and uncertainty-method metadata. Loading
them does not refit the spectrum. Missing saved covariance remains unavailable.
See :doc:`../reference/run_bundles` and the recovery notes in
:doc:`iron_balmer_uncertainties`.

For a paper, record the following alongside the reported measurements:

* software version and science-model identifier, separately from the archive
  schema version;
* input data identifiers, adopted redshift, wavelength/flux conventions and
  preprocessing, including the foreground map and extinction law;
* enabled continuum, line and host models, templates, fixed ratios and bounds;
* instrumental-resolution treatment and the measured component/profile
  definition, including the EW continuum;
* covariance or resampling method, conditioning assumptions, seed, attempted
  and usable trial counts, and any separately assessed model sensitivity;
* selection/quality rules and method/template citations relevant to the
  components used.

The :doc:`references` lists component-specific citations. Use the references
for the models used in your analysis.
