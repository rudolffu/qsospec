Results and measurements
========================

.. _read-first-measurement:

Read your first measurement
---------------------------

Continue with ``result`` from :doc:`../getting_started/quickstart`: the SDSS
quasar ``spec-1198-52669-0040`` at :math:`z=0.5764288`, fitted with the default
continuum, automatic line selection, Planck/F99 correction and no host
decomposition. The ``hbeta_oiii`` complex fitted successfully.

The broad-Hβ profile is the sum of ``Hb_broad1``, ``Hb_broad2`` and
``Hb_broad3``. Narrow Hβ, [O III] and continuum are excluded from that sum.
Its full width at half maximum (FWHM) is the width around the profile's
highest peak, rather than an average of the three component widths.

The following function comes from the tested companion script. It reads the
native metric and matching covariance error, returning ``None`` when the
error is unavailable.

.. code-block:: python

   import numpy as np

.. literalinclude:: ../../examples/fit_sdss_quasar.py
   :language: python
   :start-after: # broad-hbeta-start
   :end-before: # broad-hbeta-end

.. code-block:: python

   measurement = read_broad_hbeta(result)
   print(measurement)
   # {'status': 'available', 'fwhm_kms': 5841.08..., 'fwhm_error_kms': 241.90...}

The verified fit gives :math:`5841\pm242\,\mathrm{km\,s^{-1}}`. This is an
observed-profile width, including instrumental broadening, because the SDSS
reader does not obtain a resolution model from ``wdisp``.

The error is a one-standard-deviation covariance estimate for the chosen line
model, conditional on the fitted global continuum and input redshift. There
are no Monte Carlo trials in this example. Continuum-model sensitivity and
instrumental or calibration uncertainty require separate checks. If an error
is unavailable, report that fact rather than interpreting it as zero.
See :doc:`../science/uncertainties` for the available uncertainty methods.

Two individual broad-Hβ coordinates reach their configured upper bounds:
``Hb_broad1.fwhm_kms`` and ``Hb_broad3.velocity_kms``. Their component
decomposition may change with the chosen model. Inspect the summed profile
and residuals in the Hβ zoom when using the aggregate width; a model-sensitivity
comparison answers a different question from its covariance error.
The [O III] coverage and resolution diagnostics concern the neighboring
doublet and its kinematic interpretation; they do not change which Hβ
components enter this measurement.

The same fit also gives the broad-Hβ integrated flux and rest-frame equivalent
width (EW):

.. code-block:: python

   hbeta = result.line_complexes["hbeta_oiii"]
   flux = hbeta.metrics["Hb_broad_flux_cgs"]
   flux_error = hbeta.metric_errors.get("Hb_broad_flux_cgs", float("nan"))
   ew = hbeta.metrics["Hb_broad_ew_rest"]
   ew_error = hbeta.metric_errors.get("Hb_broad_ew_rest", float("nan"))
   print(flux, flux_error, ew, ew_error)
   # 1.05882e-14, 6.98704e-16, 93.3685, 6.15269

Here the flux is :math:`(1.059\pm0.070)\times10^{-14}\,
\mathrm{erg\,s^{-1}\,cm^{-2}}` and the EW is :math:`93.37\pm6.15\,
\mathring{A}` in the input rest frame. The input-scaled integrated flux,
``Hb_broad_flux_input=1058.82``, is multiplied by ``flux_scale=1e-17`` to
obtain cgs flux. The EW divides the summed broad-Hβ flux by the full fitted
global continuum sampled at its fitted centroid. It includes the enabled
iron and Balmer components in that continuum denominator.
See :ref:`measurement-hbeta` for exact definitions, support and archive keys.

A concise report is: “The summed broad-Hβ FWHM is
:math:`5841\pm242\,\mathrm{km\,s^{-1}}`, including instrumental broadening.
The quoted statistical error is conditional on the fitted continuum and
the adopted line model.”

.. _save-and-reload-this-analysis:

Save and reload this analysis
-----------------------------

To preserve the analysis, choose the persistence call **before fitting**.
Use this in place of ``fit_global_lines_workflow`` in the quick start:

.. code-block:: python

   import qsospec

   run_directory = "runs/sdss-1198-0040"
   object_id = "spec-1198-52669-0040"
   result = qsospec.fit_object_to_store(
       path, run_directory, object_id=object_id,
       run_host_decomp=False, write_qa=False,
   )
   measurement = read_broad_hbeta(result)

This performs the same default fit once and writes its parameters, measurements,
errors, masks and reconstruction assets into ``run_directory``. Setting
``write_qa=False`` skips saved plotting; ``result.plot_qa()`` is still available.
An in-memory ``WorkflowResult`` from an unsaved fit has no bundle to reload.

Then, in a later session, reconstruct the saved model and recover the same
measurement without fitting:

.. code-block:: python

   import qsospec
   from fit_sdss_quasar import read_broad_hbeta  # Downloaded companion script.

   run = qsospec.open_run("runs/sdss-1198-0040")
   loaded = qsospec.load_model(run, "spec-1198-52669-0040")
   recovered = read_broad_hbeta(loaded)
   print(recovered)
   print(loaded.complex_statuses["hbeta_oiii"])  # fit
   print(loaded.line_complexes["hbeta_oiii"].success)  # True

The numerical verification archived this object and recovered its width,
error, fit status and adopted redshift. For complete bundle and catalog
operations, see :doc:`../reference/run_bundles`.
Keep the downloaded ``fit_sdss_quasar.py`` beside the session when importing
its helper, or use the function defined above.

Optional exercise: choose a line window
---------------------------------------

Before running another fit, predict what happens if you set
``complexes=["hbeta_oiii"]`` in either fitting call. Which continuum components
and which line outputs should remain?

The selector requests only Hβ/[O III]; the global continuum is still fitted.
Mg II is inside this spectrum's wavelength range but is not requested, so
it has no fitted complex in that result. This differs from Hα in the full
example, which is requested by automatic selection but is outside coverage.
Selecting only Hβ also removes the broad-Hγ fit used for the default soft
Balmer refinement, so the final continuum and Hβ measurements can change.
This is useful when studying a deliberately restricted analysis, rather than
assuming the selector only changes the plotted panels.
For this object the restricted fit gives about
:math:`5840\pm245\,\mathrm{km\,s^{-1}}`, close to the complete example;
the change in refinement is small for this spectrum.

Global workflow
---------------

:class:`qsospec.WorkflowResult` contains:

- ``spectrum``: the spectrum fitted after preprocessing/host subtraction.
- ``total_spectrum``: the dereddened total spectrum when available.
- ``continuum`` and ``continuum_initial``.
- ``line_complexes`` keyed by recipe ID.
- ``complex_statuses`` for fitted, partial, failed, and uncovered recipes.
- host arrays and masks when requested.
- warnings, Monte Carlo summaries, metadata, and output paths.

``continuum_success`` reports the continuum fit status. Each line complex
has its own ``success`` flag.

Complex measurements
--------------------

Each :class:`qsospec.EmissionComplexResult` exposes fitted parameters,
covariance, component models, masks, and a ``metrics`` dictionary. Metric
names are feature-specific, for example:

.. code-block:: python

   hbeta = result.line_complexes.get("hbeta_oiii")
   if hbeta is not None and hbeta.success:
       print(hbeta.metrics["Hb_broad_flux_input"])
       print(hbeta.metrics["Hb_broad_fwhm_kms"])

   lya = result.line_complexes.get("lya_nv")
   if lya is not None:
       print(lya.metadata["lya_coverage_status"])
       print(lya.metadata["lya_fit_reliable"])

Warnings
--------

``result.warning_codes()`` combines workflow, continuum, and complex warning
codes. Warnings are structured records with severity, message, and context.
See :doc:`../reference/warnings` for recommended actions.

Archived results
----------------

Use :func:`qsospec.open_run`, :func:`qsospec.load_model`, and
:func:`qsospec.build_science_catalog` to inspect run bundles without
refitting. See :ref:`save-and-reload-this-analysis` for the worked example and
:doc:`../reference/run_bundles` for the format. The
:doc:`../reference/measurement_dictionary` maps common measurements to their
native and archived keys.

For host-decomposed spectra, keep the final ``fracHost_<wave>`` quantity
separate from the direct ``fracHost_pPXF_<wave>`` measurement. The former uses
the final qsospec AGN continuum; the latter divides the pPXF stellar component
by the direct pPXF total. Their complete definitions are listed in
:ref:`host-fraction-vocabulary`.
