Scientific measurements
=======================

A measurement applies to a selected profile, not automatically to every line
in an emission complex. For example, broad Hβ sums the fitted broad Hβ
components and excludes narrow Hβ, [O III], He II, and the continuum. The
:doc:`../reference/measurement_dictionary` gives exact keys, component
selections, units, and error locations.

Integrated flux
---------------

For a line model :math:`f_\lambda^\mathrm{line}`,

.. math::

   F = \int f_\lambda^\mathrm{line}(\lambda)\,d\lambda.

Native generic ``*_flux_input`` fields sum the integrated amplitudes of the
selected feature and role. Gaussian and Lorentzian amplitudes represent
full-domain model flux. ``*_flux_window_input`` instead uses a trapezoidal
integral of the measurement-grid samples selected by the declared fitting
windows; ``*_flux_window_fraction`` compares it with the full-domain value. A model total can include extrapolated wings.

Dedicated Hβ, Mg II, and Hα adapters integrate the summed broad profile on
their measurement grids. :func:`qsospec.uncertainties.measure_selected_profile`
integrates explicitly selected components over the saved spectrum's
rest-wavelength range. These supports need not produce identical fluxes for
a truncated profile.

The result reports flux integrated from the prepared rest-frame
:math:`F_\lambda` in the input numerical scale. Multiplication by
``result.spectrum.flux_density_scale_to_cgs`` gives
:math:`\mathrm{erg\,s^{-1}\,cm^{-2}}` when that scale is known. The rest-frame
coordinate conversion preserves integrated flux; measurement conversion adds
no redshift factor. See :doc:`../user_guide/preprocessing`.

Profile centroid and dispersion
-------------------------------

For the selected summed profile,

.. math::

   \bar{\lambda} =
   \frac{\int \lambda f(\lambda)\,d\lambda}{\int f(\lambda)\,d\lambda},

.. math::

   \sigma_\lambda =
   \sqrt{\frac{\int(\lambda-\bar{\lambda})^2f(\lambda)\,d\lambda}
   {\int f(\lambda)\,d\lambda}}.

Native ``*_centroid`` and ``*_sigma_kms`` use the path's measurement grid.
Native velocity conversion divides the wavelength dispersion by the
laboratory reference wavelength and multiplies by :math:`c`.
The selected-profile helper divides by the measured centroid instead.
Generic moment integrals use their finite grid but normalize by the summed
full-domain amplitudes. Gaussian tails on that grid are usually negligible;
Lorentzian tails need separate attention. Their native moments are finite-grid
measurements, not infinite-domain Lorentzian moments.

Native broad centroid velocities use
:math:`c\ln(\bar{\lambda}/\lambda_\mathrm{ref})`, relative to the adopted
input rest frame. A centroid is a flux-weighted mean; a peak is the maximum
of the profile. Neither changes the input redshift.

Profile FWHM
------------

Full width at half maximum (FWHM) is measured on the summed selected profile.
It is not the sum or average of individual Gaussian widths.
Native generic and dedicated-adapter FWHM fields use the nearest half-maximum
crossings enclosing the global profile peak, interpolated between samples.
Their velocity conversion uses the laboratory reference wavelength.
``measure_selected_profile`` uses the same peak-enclosing convention but
normalizes the separation by its measured centroid.

The specialist :mod:`qsospec.broad_narrow_measurements` product is different:
``total_profile_fwhm_observed_kms`` includes both narrow and broad components
and uses the outermost half-maximum crossings. Disconnected half-maximum
regions set ``total_profile_fwhm_ambiguous`` and withhold its FWHM error.
Use the definition attached to the output you report.

Primary line widths describe the observed, instrument-broadened model.
Where supported forward modelling is active, separately named intrinsic
measurements and intrinsic component parameters are available. Inspect the
saved ``line_lsf`` metadata rather than inferring resolution correction from
the presence of a resolution descriptor alone.

.. _equivalent-width-conventions:

Equivalent width
----------------

The wavelength-dependent definition of rest-frame equivalent width is

.. math::

   \mathrm{EW}_\mathrm{integral} =
   \int \frac{f_\lambda^\mathrm{line}(\lambda)}
   {f_\lambda^\mathrm{continuum}(\lambda)}\,d\lambda.

Native ``*_ew_rest`` fields use a sampled-continuum convention:

.. math::

   \mathrm{EW}_\mathrm{native} =
   \frac{F_\mathrm{line}}
   {f_\lambda^\mathrm{continuum}(\lambda_\mathrm{sample})}.

Both are in rest-frame Å. The sampled approximation approaches the ratio
integral when the continuum changes little across the line.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Output
     - Selected line profile and support
     - Continuum denominator
   * - Native generic ``<feature>_<role>_ew_rest``
     - All components of that feature and role; full-domain amplitude sum
     - Full fitted global continuum at the fitted centroid
   * - Adaptive ``Hb_broad_ew_rest``
     - All broad Hβ components; alias of ``hbeta_broad_ew_rest``
     - Full fitted global continuum at the fitted broad Hβ centroid
   * - Dedicated adapters: legacy ``Hb_broad_ew_rest``,
       ``MgII_broad_ew_rest``, ``Ha_broad_ew_rest``
     - Summed broad profile on the adapter measurement grid
     - Full fitted global continuum at the laboratory reference wavelength
   * - Selected-profile ``values["ew_angstrom"]``
     - Explicit ``component_ids``; ratio integrated over the saved
       rest-wavelength range
     - Supplied ``continuum=`` array, interpolated across the integration grid
   * - Specialist broad/narrow ``total_equivalent_width_rest``
     - Primary line's narrow plus broad integrated amplitudes
     - Global continuum plus fitted residual local continuum at the reference
       wavelength

The native global denominator includes all enabled fitted components:
power law, iron templates, Balmer pseudo-continuum, and any accepted additive
polynomial. It excludes the separately subtracted stellar host and the
line complex's residual local continuum. It is not automatically the power
law alone. A nonpositive denominator produces an unavailable EW.

Selected-profile EW requires a positive supplied continuum throughout its
integration grid. Choose that array explicitly: ``result.continuum.model``
supplies the final global continuum; adding a local residual continuum is
a different stated measurement choice. See
:ref:`measurement-ew` for access paths and uncertainty keys.

Host fractions
--------------

The final ``fracHost_<wave>`` divides the fitted stellar host by that host
plus the final global AGN continuum. The direct ``fracHost_pPXF_<wave>``
divides the pPXF stellar host by the pPXF total model. These denominators
differ. See :ref:`host-fraction-vocabulary` and
:ref:`measurement-host` for definitions and coverage restrictions.

Uncertainty
-----------

``metric_errors`` contains statistical errors propagated through the selected
measurement and fitted covariance. Sequential line errors condition on the
fitted continuum and host; the joint Hγ refinement includes its fitted
continuum correlations. Matched resampling can replace native errors with
trial scatter and adds separately stored intervals.

An unavailable error is not zero. Model, template, and calibration sensitivity
requires the corresponding additional analysis. See
:doc:`uncertainties` and the
:doc:`../reference/measurement_dictionary` for the method attached to each
quantity.

Units
-----

Wavelengths and EWs are in Å; velocities and velocity widths are in km/s.
DESI/SDSS flux-density presets use
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
When the physical scale is unknown, cgs-derived quantities are unavailable.
