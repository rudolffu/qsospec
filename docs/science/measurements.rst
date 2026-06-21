Scientific measurements
=======================

Integrated flux
---------------

For a line model :math:`f_\lambda^\mathrm{line}`,

.. math::

   F = \int f_\lambda^\mathrm{line}(\lambda)\,d\lambda.

The result reports input-unit flux and cgs flux when the input metadata
provides a cgs scale.

Profile centroid and dispersion
-------------------------------

For the summed broad profile,

.. math::

   \bar{\lambda} =
   \frac{\int \lambda f(\lambda)\,d\lambda}{\int f(\lambda)\,d\lambda},

.. math::

   \sigma_\lambda =
   \sqrt{\frac{\int(\lambda-\bar{\lambda})^2f(\lambda)\,d\lambda}
   {\int f(\lambda)\,d\lambda}}.

Centroid offsets and dispersion are also reported in velocity units.

Numerical FWHM
--------------

FWHM is measured on the summed broad profile, not by adding individual
Gaussian widths. The fitter locates the blue and red half-maximum crossings
of an interpolated profile and converts their separation to km/s.

Equivalent width
----------------

Rest-frame equivalent width is

.. math::

   \mathrm{EW} =
   \int \frac{f_\lambda^\mathrm{line}(\lambda)}
   {f_\lambda^\mathrm{continuum}(\lambda)}\,d\lambda.

Host fractions
--------------

Where constrained by observed coverage,

.. math::

   f_\mathrm{host}(\lambda)=
   \frac{f_\lambda^\mathrm{host}}
   {f_\lambda^\mathrm{host}+f_\lambda^\mathrm{AGN}}.

Uncertainty
-----------

Covariance errors describe the fitted statistical model. They do not include
all continuum, host-decomposition, calibration, or model-choice systematics.
Monte Carlo summaries are stored separately. Warning
``statistical_uncertainty_excludes_continuum_host`` records this limitation
where relevant.

Units
-----

Wavelengths are in Å and velocities in km/s. DESI/SDSS flux-density presets
use :math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
When the physical scale is unknown, cgs-derived quantities are omitted.
