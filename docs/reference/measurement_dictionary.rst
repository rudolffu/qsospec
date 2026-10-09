Measurement dictionary
======================

This page maps reported quantities to their fitted profiles, units, and
uncertainties. Start with :doc:`../user_guide/results` for the worked SDSS
measurement, or :doc:`../science/measurements` for the mathematical definitions.

Access and archive conventions
------------------------------

For a native line measurement, select its complex first:

.. code-block:: python

   fit = result.line_complexes["hbeta_oiii"]
   value = fit.metrics["Hb_broad_fwhm_kms"]
   error = fit.metric_errors.get("Hb_broad_fwhm_kms", float("nan"))

The same native key identifies its value and error. Fitted component parameters
are in ``fit.param_values`` and ``fit.param_errors``; they are not necessarily
summed-profile measurements. ``fit.metadata["measurement_uncertainty_method"]``
identifies a resampling method when one replaced the covariance errors.
The default is local fitted-parameter covariance, conditional on the selected
model and fitted global continuum/host. Joint refinement can include additional
correlations; see :doc:`../science/uncertainties`.

In a run bundle, these native measurements have
``section="complex_metric"``, the fitted ``recipe_id``, and
``quantity=<native key>``. ``value`` and ``error`` are separate columns.
Fitted parameters instead use ``complex_parameter`` or
``continuum_parameter``. A matched-draw key is qualified, for example
``line:hbeta_oiii:Hb_broad_fwhm_kms``; this is not the key in ``fit.metrics``.
Matched errors, percentile intervals, and usable counts are respectively in
``result.monte_carlo["errors"]``, ``["percentiles"]``, and
``["valid_trial_counts"]`` under that qualified key.

Unavailable in-memory measurements or errors can be absent or NaN. The
archive omits nonfinite values and stores an unavailable error as null.
Neither supplies zero flux, zero uncertainty, a nondetection, or an upper
limit. Some native line rows have no explicit ``unit`` field; use the
quantity definition below and the saved input scaling rather than assuming
that a blank field means dimensionless. Always inspect ``fit.success``,
``result.complex_statuses``, coverage metadata, and relevant warnings.

.. _measurement-hbeta:

Broad Hβ
--------

``result.line_complexes["hbeta_oiii"]`` (also ``result.hbeta``) contains these
fields. The default adaptive model sums all retained broad Hβ components,
normally ``Hb_broad1``, ``Hb_broad2``, and ``Hb_broad3``. It excludes narrow
Hβ, [O III], He II, local continuum, and unrelated lines. The aliases below
match native ``hbeta_broad_*`` generic metrics in the adaptive fit; an explicit
legacy Hβ fit uses its dedicated measurement grid.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Native key in ``metrics`` and ``metric_errors``
     - Definition
     - Unit
   * - ``Hb_broad_flux_input``
     - Adaptive: sum of broad integrated amplitudes over the full model
       domain. Legacy: numerical broad-profile integral on 4500–5220 Å.
     - Input flux-density scale × Å
   * - ``Hb_broad_flux_cgs``
     - ``Hb_broad_flux_input`` multiplied by
       ``result.spectrum.flux_density_scale_to_cgs``
     - :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`
   * - ``Hb_broad_fwhm_kms``
     - Nearest half-maximum crossings enclosing the summed broad peak,
       divided by the 4862.68 Å vacuum reference and multiplied by :math:`c`
     - km/s
   * - ``Hb_broad_sigma_kms``
     - Second central moment of the summed broad profile, divided by the
       same reference wavelength and multiplied by :math:`c`
     - km/s
   * - ``Hb_broad_centroid``
     - Flux-weighted mean rest wavelength of the summed broad profile
     - Rest-frame vacuum Å
   * - ``Hb_broad_velocity_kms``
     - :math:`c\ln(\bar\lambda/(4862.68\,\mathring{A}))`
     - km/s in the adopted input rest frame
   * - ``Hb_broad_ew_rest``
     - Integrated broad flux / full global continuum at the broad centroid
       (adaptive) or 4862.68 Å (legacy)
     - Rest-frame Å

The adaptive profile grid has 2401 samples, centered on the laboratory
reference, with half-span
:math:`\max(50\,\mathring{A},5\,\mathrm{FWHM}_{\max}\lambda_\mathrm{ref}/c)`.
Native flux is the full-domain amplitude sum; centroid, dispersion, and FWHM
use this finite grid. This distinction matters for broad or truncated wings.
FWHM is unavailable when both crossings cannot be found; moments require
positive profile flux. A cgs flux also requires a known physical scale.

The primary widths include instrumental broadening. Adaptive forward modelling
uses a supported supplied line-spread function (LSF) while returning observed
profile metrics. Inspect ``fit.metadata["line_lsf"]`` for
``status="forward_modeled"`` or the observed-profile fallback reason.
The SDSS tutorial supplies no LSF, so its reported Hβ widths include
instrumental broadening. A component's ``*.fwhm_kms`` can be an intrinsic
parameter in a forward model; it is not interchangeable with the summed
observed ``Hb_broad_fwhm_kms``.

For every key above, archive selection is ``complex_metric/hbeta_oiii/<key>``
and the matched uncertainty key is ``line:hbeta_oiii:<key>``. Errors propagate
the full fitted line covariance through that quantity; the sampled EW
continuum is fixed in the sequential Hβ calculation. Input-redshift uncertainty
and changes in continuum model are not included in that covariance.

Other native broad and generic profiles
---------------------------------------

Dedicated Mg II and Hα adapters use ``result.line_complexes["mgii"]`` and
``["halpha_nii_sii"]``. Their broad keys begin ``MgII_broad_`` and
``Ha_broad_`` respectively, with the same suffixes as the Hβ table.
They integrate profiles on 2550–3050 Å and 6200–6900 Å, respectively,
using 7201 samples. Their reference wavelengths are 2798.75 Å and
6564.61 Å. Native EW samples the full global continuum at that reference.
Broad components are summed; narrow and neighboring lines are excluded.
Errors use the same native keys; archive rows and matched keys use their
own recipe IDs. These dedicated-adapter widths describe fitted observed
profiles; do not apply the adaptive Hβ LSF metadata convention to them.

Generic recipes group components by canonical feature and role, for example
``hgamma_broad_flux_input`` or ``civ_blend_broad_fwhm_kms``. The exact
prefix follows the registry feature ID; inspect ``fit.metrics`` and
``fit.metadata["active_components"]`` rather than constructing an alias.
``*_flux_input`` is the full-domain amplitude sum; ``*_flux_window_input``
and ``*_flux_window_fraction`` record the selected-grid window contribution.
The window integral applies the trapezoidal rule to the selected samples;
for disjoint windows, it includes intervals connecting their selected edges. Generic
centroid, sigma, and FWHM use the finite support specified for adaptive Hβ
above, centered on their own reference wavelength. Generic moment integrals
are divided by full-domain summed amplitudes, not by the sampled integral;
this distinction is relevant to Lorentzian tails. Supported LSF treatment,
if active for that fitting path, is recorded in ``line_lsf``.

.. _measurement-ew:

Equivalent width and selected profiles
--------------------------------------

Native ``*_ew_rest`` values are integrated flux divided by sampled full global
continuum, in rest-frame Å. The denominator includes enabled power law,
iron, Balmer, and polynomial components; it excludes the stellar host and
local line-fit continuum. The generic and adaptive Hβ sample is the fitted
centroid; dedicated adapters use the laboratory reference. See
:ref:`equivalent-width-conventions`.

For an explicit generic-profile selection:

.. code-block:: python

   from qsospec.uncertainties import measure_selected_profile

   selected = measure_selected_profile(
       fit,
       ("Hb_broad1", "Hb_broad2", "Hb_broad3"),
       continuum=result.continuum.model,
   )
   ew = selected["values"]["ew_angstrom"]
   ew_error = selected["errors"].get("ew_angstrom", float("nan"))

This example assumes those three IDs exist in the selected generic Hβ fit.
The helper integrates profile/continuum over a uniform grid spanning the saved
spectrum range, with at least 4096 samples. It requires positive continuum
throughout that grid. Its ``flux`` has the input integrated-flux scale;
``centroid_angstrom`` is rest vacuum Å; ``fwhm_kms`` and ``sigma_kms`` use
the measured centroid for velocity conversion. It reconstructs the selected
profile with the saved LSF when supported.

The returned values/errors/covariance and fixed selection are also retained in
``fit.metadata["selected_profile_measurements"]["Hb_broad1|Hb_broad2|Hb_broad3"]``.
They are persisted as complex metadata, not added as uniform native measurement
rows. Supplied matched parameter draws use, for example,
``line:hbeta_oiii:selected:Hb_broad1|Hb_broad2|Hb_broad3:ew_angstrom``.
Draw summaries are in ``selected["bootstrap"]``; its qualified keys describe
the fixed selection. The covariance error conditions on the supplied
continuum; it does not add
continuum uncertainty simply because a continuum array was supplied.

The specialist broad/narrow record's ``total_equivalent_width_rest`` instead
sums narrow+broad primary-line flux and divides by the global plus fitted
local continuum at the reference. Its error is
``total_equivalent_width_rest_error`` in that record. It includes local
continuum covariance while conditioning on the fixed global continuum.
Its ``total_profile_fwhm_observed_kms`` uses outermost half-maximum crossings,
with ``total_profile_fwhm_ambiguous`` marking disconnected regions. These
specialist records are separate products, not native ``Hb_broad_*`` aliases.

.. _measurement-continuum:

Continuum flux densities
------------------------

``result.metadata["continuum_samples"]`` stores covered samples at rest
1350, 3000, and 5100 Å. Substitute one of these wavelengths for ``<wave>``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Component selection
   * - ``fAGN_<wave>``
     - Full final global continuum: power law, enabled iron and Balmer,
       and any accepted polynomial. It is a flux density, not an AGN fraction.
   * - ``f_powerlaw_<wave>``
     - Power-law component only; polynomial, iron, and Balmer are excluded.
   * - ``fHost_<wave>``
     - Stellar host aligned to the quasar grid, when host subtraction is enabled.
   * - ``fracHost_<wave>``
     - Stellar host / (stellar host + final global continuum)

Flux densities use the prepared rest-frame input scale; multiply by
``result.spectrum.flux_density_scale_to_cgs`` for physical rest-frame
:math:`F_\lambda`. This does not convert flux density to luminosity.
A sample is omitted outside the valid-pixel wavelength span; local masks
can still limit its constraint. Host samples also need their host support.

Errors are in ``result.metadata["continuum_sample_errors"]`` under the same
key. Full-continuum errors include correlations among the fitted continuum
parameters; power-law errors select its component. Availability depends on
local fitted support and covariance. A host fraction does not automatically
have a covariance error. Matched refitting can supply sample errors and
intervals. Archive selection is ``section="continuum_sample"``,
``recipe_id=None``, ``quantity=<key>``; matched keys are
``continuum_sample:<key>``. Check
``result.metadata.get("continuum_sample_uncertainty_method")`` and
:doc:`../science/uncertainties` for what was varied.

.. _measurement-host:

Direct pPXF host quantities
---------------------------

``result.metadata["host_fit_samples"]`` stores direct pPXF samples:
``fHost_pPXF_<wave>``, ``fAGN_pPXF_<wave>``, ``fTotal_pPXF_<wave>``, and
``fracHost_pPXF_<wave>``. Flux densities are interpolated from the fitted
pPXF stellar, AGN, and total models on their rest grid without extrapolation.
The fraction is stellar host / pPXF total; that total can include polynomial
terms according to the host strategy. It differs from the final
``fracHost_<wave>`` denominator. The archive section is ``host_sample``
and its quantity is the same name; native errors are unavailable.
See :ref:`host-fraction-vocabulary` for component-source metadata.

``result.metadata["ppxf_agn_fraction_flux_global"]`` is the integral of the
pPXF AGN model divided by the integral of AGN+stellar models on fitted support.
It is dimensionless, is not a 5100 Å fraction, and is archived under
``host_metric`` with the same quantity. Its native error is unavailable.
Support, definition, resolution, and host quality are recorded in the host
metadata. When both local fractions are finite, the archive also reports
``host_metric/deltaFracHost_final_pPXF_<wave>`` as their signed difference.

For an in-memory host fit, ``result.host_fit.stellar_velocity`` and
``result.host_fit.stellar_sigma`` are direct pPXF stellar kinematics in km/s;
``result.host_fit.ppxf_result.error`` contains pPXF's formal solution errors
in its parameter ordering when supplied. These are not line-centroid or
systemic-redshift measurements. They are not standardized native measurement
rows, and ``load_model`` does not restore a live pPXF optimizer object.
Inspect host kinematic-resolution status before reporting them, and retain
the host result/provenance needed for that analysis. The final and direct
fraction definitions above remain accessible after model reload.

.. _measurement-peaks:

Peaks, centroids, and systemic-redshift diagnostics
---------------------------------------------------

For each feature/role or full-line selection, native peak fields are
``<selection>_peak_rest_angstrom``, ``<selection>_peak_observed_angstrom``,
and ``<selection>_peak_velocity_kms``, with errors under identical keys in
``fit.metric_errors``. For example, broad Hβ uses
``hbeta_broad_peak_rest_angstrom`` and full Hβ uses
``hbeta_full_peak_rest_angstrom``. The full selection includes narrow and
broad Hβ but excludes [O III] and continuum. Observed peak wavelength is
rest peak × (1+input redshift); velocity uses
:math:`c\ln(\lambda_\mathrm{peak}/\lambda_\mathrm{ref})`.

``fit.metadata["line_peaks"]["measurements"][<selection>]`` records
``component_ids``, definitions, bounds, reference wavelength, peak status,
uncertainty status, and method. The maximum is continuously refined;
boundary or competing peaks and non-identifiable errors are flagged.
``<selection>_peak_selection_flux`` is the selected integrated flux in the
input scale. These fields use ``complex_metric`` archive rows and
``line:<recipe_id>:<native key>`` matched uncertainty keys. Covariance errors
condition on the component selection, continuum/host, and input redshift;
existing matched draws can provide resampling errors. Ordinary and
``*_ws22`` filtered selections are separate.

Centroids are profile means, as defined above, and remain in their native
``*_centroid``/``*_velocity_kms`` fields. An [O III] fitted core is a gas
reference, not automatically the systemic velocity. See
:doc:`../science/line_peaks_systemic_redshift` and
:doc:`../how_to/adaptive_oiii` for peak and gas-reference restrictions.

The optional ``estimate_systemic_redshift(result, method="ws22")`` returns
``z_sys``, ``z_sys_error``, ``offset_kms``, ``status``, per-line decisions,
calibration version, and uncertainty assumptions. It also stores the complete
diagnostic at ``result.metadata["systemic_redshift"]``. That workflow metadata
is persisted with the model; it is not an adopted-redshift replacement or a
native measurement-table alias. Its uncertainty combines available line and
luminosity covariance with intrinsic calibration scatter, recording the
independence approximation where cross-fit correlations are unavailable.
With no eligible lines, ``status="unavailable"`` and the estimate/error
are ``None`` with per-line reasons. ``result.spectrum.z`` is unchanged.
