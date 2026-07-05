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
spectrum and template overlap sufficiently.

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
