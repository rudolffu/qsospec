Continuum and preprocessing model
=================================

Galactic foreground
-------------------

File workflows query a two-dimensional foreground map and evaluate the
Fitzpatrick (1999) Milky Way law in the observed frame:

.. math::

   f_{\lambda,\mathrm{corrected}}
   = f_{\lambda,\mathrm{observed}}\,10^{0.4 A_\lambda}.

Planck GNILC is the default. SFD values are multiplied by 0.86 following
Schlafly & Finkbeiner (2011).

Default continuum
-----------------

``GlobalContinuumConfig()`` selects the ``global_v2`` science model: a single
power law, empirical UV and optical iron templates, a conditional regional
iron bridge, and the Balmer pseudo-continuum with soft Hγ refinement. The
additive polynomial is disabled. Components need sufficient fitted-pixel
support to be activated; a model component need not be present in every
spectrum.

Power law and Fe II
-------------------

The AGN continuum includes a pivoted power law. The default
``mode="single"`` uses one slope. ``mode="double"`` uses a continuous broken law with independent slopes
on either side of a configurable 4661 Å break. ``mode="auto"`` compares both
models on a shared accepted-pixel mask and selects the broken law only for a
default :math:`\Delta\mathrm{BIC}\ge10`.

.. math::

   f_\lambda = N\left(\frac{\lambda}{\lambda_\mathrm{pivot}}\right)^\alpha,

plus independently broadened UV and optical Fe II templates when the
spectrum and template overlap sufficiently. The default combines
VW01 in the UV and Park22 in the optical. Their additional broadening
kernels are independent by default. ``iron_width_coupling="soft"`` adds an
optional tolerance on their log kernel-width ratio. Use it when a physically
motivated relation is needed, and inspect its effect on the continuum and
line measurements.

With compatible VW01/Park22 settings and enough continuum pixels, a regional
Verner09 template fills the handoff between the empirical templates. The
default handoff intervals are 3300–3450 Å and 4100–4250 Å; template support
and broadening guards can adjust the effective intervals. Smooth weights
join the three regions. The bridge adds ``middle_iron.amp`` and shares the
optical template's additional broadening kernel when optical iron is active,
otherwise the UV kernel. With neither empirical template active, it uses the
configured fixed kernel. It does not add a third free iron width.

``result.continuum.metadata["regional_iron"]`` records activation, effective
intervals, normalization, and width parent. The amplitude uses a fixed
reference normalization, so it is not itself the final bridge-band flux.
The bridge is inactive for an exclusive full-range template or incompatible
empirical choices. ``RegionalIronConfig(enabled=False)`` explicitly disables
it.

As an alternative, ``GlobalContinuumConfig.with_single_iron("verner09")``
uses one Verner et al. (2009) theoretical template over approximately
2000–10000 Å. Its fitted width is the final target FWHM. qsospec records the
template's assumed native 900 km/s FWHM and applies only the additional
quadrature broadening

.. math::

   \mathrm{FWHM}_{\rm conv} =
   \sqrt{\mathrm{FWHM}_{\rm target}^2 - (900\,\mathrm{km\,s^{-1}})^2}.

Iron width conventions
----------------------

A template convolution width describes the additional smoothing kernel.
An effective or target width also includes the template's native profile;
for a known Gaussian-equivalent native width, the two add in quadrature.
Empirical template native widths are not always known, so an effective width
can be unavailable even when the fitted kernel width is well measured.
``IronTemplateConfig.width_mode`` specifies the coordinate used by both
``fwhm_kms`` and ``fwhm_bounds``. The Verner09 target convention above uses
an assumed 900 km/s native width.

Instrumental broadening is a separate operation and is not supplied by an
iron-template kernel parameter. See
:doc:`iron_balmer_uncertainties` for template support, width metadata, and
bridge normalization, and :doc:`../how_to/adaptive_oiii` for line-resolution
forward modelling.

Optional polynomial correction
------------------------------

The global continuum can include a signed additive polynomial without a
constant term:

.. math::

   P(\lambda)=\sum_{j=1}^{d} c_j
   \left(\frac{\lambda-\lambda_{\rm pivot}}{\lambda_{\rm scale}}\right)^j.

Correction is disabled by default for every survey, including SDSS. When
explicitly requested, the default degree is two, with pivot and scale inherited from the
power-law pivot (3000 Å). The polynomial models a small residual correction
to the power-law, iron, and Balmer continuum.

First, qsospec fits and clips a polynomial-free power law + iron + Balmer
baseline. Automatic single/broken power-law selection happens at this stage.
It then fixes the baseline slope(s) and fits the correction on exactly the
same accepted pixels without further clipping. Normalization may move by
at most 10%; iron and Balmer retain their configured constraints.

The signed correction is limited to 10% of the fixed baseline power law at
every valid input wavelength, including pixels outside continuum anchors.
Coefficient limits enforce this bound. Both limits are configurable.

Explicitly setting ``mode="auto"`` opts into quadratic assessment only for SDSS provenance.
It accepts a numerically safe candidate when
:math:`\chi^2_{\rm baseline}-\chi^2_{\rm candidate}-d\ln n\ge10` on the same
accepted pixels. This score compares the correction with the fitted baseline.
Otherwise the baseline is returned unchanged.
``mode="on"`` bypasses this score requirement but not safety checks;
``mode="off"`` (the default) disables the correction. Degree three is opt-in.

Insufficient coverage, a nonpositive baseline power law, incompatible bounds,
failed optimization, or a rank-deficient/ill-conditioned combined Jacobian
retains the baseline with a recorded reason. Candidate covariance is conditional
on the baseline slopes; their baseline uncertainties are retained and unknown
cross-covariances are NaN. Monte Carlo trials re-estimate the baseline and
reassess the correction. The signed QA strip exposes both positive and negative
corrections that could otherwise be hidden below a flux axis starting at zero.
The polynomial models smooth residual structure. Instrumental flux steps
require a separate model. Host pPXF polynomial settings are independent.

Balmer pseudo-continuum
-----------------------

The Balmer component is continuous at the 3646 Å edge. Above the
edge it uses the velocity-shifted, velocity-broadened high-order Balmer series
:math:`H(\lambda)`. Below the edge, the bound-free shape :math:`C(\lambda)` is
normalized by the high-order blend at the edge:

.. math::

   F(\lambda)=A
   \begin{cases}
   H(3646)\,C(\lambda)/C(3646), & \lambda \le 3646\\
   H(\lambda), & \lambda > 3646.
   \end{cases}

The default series uses :math:`n=6`–400, fixed
:math:`T_e=15000\,\mathrm{K}` and :math:`\tau_{3646}=1`, with fitted
amplitude, FWHM, and velocity. Diagnostic outputs retain separate
``balmer_bound_free`` and ``balmer_high_order_series`` arrays.

Hγ is fitted in the optical-blue emission-line complex; Hδ and higher
orders are included in the pseudo-continuum template. The default
``sync_with_hgamma="soft"`` jointly refines the continuum and optical-blue
lines on their combined fitted-pixel mask. The bundled Storey & Hummer
Case-B ratio sets the center of a log-ratio constraint:

.. math::

   F_{\mathrm{H}\gamma} =
   r_{\gamma/\beta}\,A_\mathrm{Balmer}\,10^{\delta_\gamma},
   \qquad \delta_\gamma \sim \mathcal{N}(0,0.30^2).

The default ``hgamma_ratio_scatter_dex=0.30`` is a modelling tolerance on
the ratio, not the measured Hγ error. The amplitude and Hγ flux can move
together; the relation is not fixed exactly. The joint covariance includes
these fitted correlations.

Soft refinement requires an active broad Hγ component and usable support
for both wings. Missing coverage or truncated wings leave the initial
free-amplitude continuum in place; an unsuccessful joint optimization also
keeps that fit. ``result.continuum.metadata["hgamma_joint_status"]`` records
``fit``, ``unavailable_coverage``, ``unavailable_truncated``, or ``failed``
when the refinement is attempted. Inspect line coverage and covariance
alongside that status.

``sync_with_hgamma="off"`` leaves the amplitude free. ``"hard"`` fixes it
from a successful broad Hγ measurement that passes the configured flux S/N
and bound checks; it falls back to a free amplitude if the measurement is
inadequate. ``"require"`` requests that hard relation and reports failure
to establish it. The Hβ width policy is separate: the default ``"auto"``
uses a reliable summed broad Hβ FWHM when available, otherwise the Balmer
width remains free. Neither policy changes the adopted redshift.

For the explicit ``global_v1`` continuum preset, use
``GlobalContinuumConfig.legacy_v1()``. It disables the regional iron bridge
and selects hard Hγ synchronization. This identifies the continuum model;
line-profile choices such as adaptive or legacy [O III] are separate settings.
See :doc:`../reference/configuration` for configuration choices and
:doc:`iron_balmer_uncertainties` for the template and refinement details.

Continuum masks
---------------

The initial continuum fit uses configured continuum windows. Additional mask
windows remove known line contamination. Blue-side pixels below the initial
continuum by more than three spectral uncertainties are rejected once below
3500 Å. Soft Hγ refinement then uses the union of the accepted continuum
pixels and fitted optical-blue pixels, modelling the lines jointly; each
pixel enters that refinement once.

See :doc:`../reference/configuration` for configurable behavior.
