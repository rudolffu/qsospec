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

Power law and Fe II
-------------------

The AGN continuum includes a pivoted power law. ``mode="single"`` uses one
slope. ``mode="double"`` uses a continuous broken law with independent slopes
on either side of a configurable 4661 Å break. ``mode="auto"`` compares both
models on a shared accepted-pixel mask and selects the broken law only for a
default :math:`\Delta\mathrm{BIC}\ge10`.

.. math::

   f_\lambda = N\left(\frac{\lambda}{\lambda_\mathrm{pivot}}\right)^\alpha,

plus independently broadened UV and optical Fe II templates when the
spectrum and template overlap sufficiently. The unchanged default combines
VW01 in the UV and Park22 in the optical.

As an alternative, ``GlobalContinuumConfig.with_single_iron("verner09")``
uses one Verner et al. (2009) theoretical template over approximately
2000–10000 Å. Its fitted width is the final target FWHM. qsospec records the
template's assumed native 900 km/s FWHM and applies only the additional
quadrature broadening

.. math::

   \mathrm{FWHM}_{\rm conv} =
   \sqrt{\mathrm{FWHM}_{\rm target}^2 - (900\,\mathrm{km\,s^{-1}})^2}.

Optional polynomial correction
------------------------------

The global continuum can include a signed additive polynomial without a
constant term:

.. math::

   P(\lambda)=\sum_{j=1}^{d} c_j
   \left(\frac{\lambda-\lambda_{\rm pivot}}{\lambda_{\rm scale}}\right)^j.

Correction is disabled by default for every survey, including SDSS. When
explicitly requested, the default degree is two, with pivot and scale inherited from the
power-law pivot (3000 Å). The polynomial is a small residual correction, not
an independently fitted alternative to the physical continuum.

First, qsospec fits and clips a polynomial-free power law + iron + Balmer
baseline. Automatic single/broken power-law selection happens at this stage.
It then fixes the baseline slope(s) and fits the correction on exactly the
same accepted pixels without further clipping. Normalization may move by
at most 10%; iron and Balmer retain their configured constraints.

The signed correction is limited to 10% of the fixed baseline power law at
every valid input wavelength, including pixels outside continuum anchors.
Conservative per-coefficient envelopes guarantee this bound. These limits
are configurable engineering defaults, not universal physical thresholds.

Explicitly setting ``enabled=None`` opts into quadratic assessment only for SDSS provenance.
It accepts a numerically safe candidate when
:math:`\chi^2_{\rm baseline}-\chi^2_{\rm candidate}-d\ln n\ge10` on the same
accepted pixels. This is a conservative staged selection score, not exact
Bayesian evidence. Otherwise the baseline is returned unchanged.
``enabled=True`` bypasses this score requirement but not safety checks;
``enabled=False`` (the default) disables the correction. Degree three remains opt-in.

Insufficient coverage, a nonpositive baseline power law, incompatible bounds,
failed optimization, or a rank-deficient/ill-conditioned combined Jacobian
retains the baseline with a recorded reason. Candidate covariance is conditional
on the baseline slopes; their baseline uncertainties are retained and unknown
cross-covariances are NaN. Monte Carlo trials re-estimate the baseline and
reassess the correction. The signed QA strip exposes both positive and negative
corrections that could otherwise be hidden below a flux axis starting at zero.
A global smooth polynomial must not be interpreted as an instrumental flux-step
model. Host pPXF polynomial controls remain independent.

Balmer pseudo-continuum
-----------------------

The production Balmer component is continuous at the 3646 Å edge. Above the
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

The Hγ line is fitted as part of the optical-blue emission-line complex,
while Hδ and higher Balmer orders are included in the pseudo-continuum
template. When broad Hγ is covered and reliably measured, qsospec uses the
Storey & Hummer Case-B ratios bundled with the template to set the
pseudo-continuum amplitude from Hγ. This fixes the integrated Hγ/Hδ relation
instead of letting the fitted Hγ complex and the Hδ+high-order template drift
independently. If Hγ is unavailable or unreliable, the pseudo-continuum falls
back to the usual free amplitude and records the skip reason in the result
metadata.

Continuum masks
---------------

Only configured continuum windows contribute to the fit. Additional mask
windows remove known line contamination. Blue-side pixels below the initial
continuum by more than three spectral uncertainties are rejected once below
3500 Å.

See :doc:`../reference/configuration` for configurable behavior.
