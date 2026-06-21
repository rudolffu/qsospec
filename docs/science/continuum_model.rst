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

The AGN continuum includes a pivoted power law,

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

Continuum masks
---------------

Only configured continuum windows contribute to the fit. Additional mask
windows remove known line contamination. Blue-side pixels below the initial
continuum by more than three spectral uncertainties are rejected once below
3500 Å.

See :doc:`../reference/configuration` for configurable behavior.
