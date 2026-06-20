Scientific Definitions
=======================

This page documents the scientific quantities computed by qsospec and recorded
in the ``metrics`` dictionaries of :class:`~qsospec.EmissionComplexResult`
objects and the measurement tables.


Flux measurements
-----------------

**Integrated line flux**

   :math:`F = \int_{\lambda_1}^{\lambda_2} f_\lambda(\lambda) \, d\lambda`

   where :math:`f_\lambda` is the emission-line model in the continuum-subtracted
   rest-frame spectrum.  Integration bounds are recipe-defined and recorded in
   ``metric_errors["flux_bounds"]``.

**Continuum sample flux** (:math:`L_{5100}`)

   The continuum model (power-law + iron + Balmer) evaluated at 5100 Å
   rest-frame.  The fiducial luminosity :math:`\lambda L_\lambda(5100\,\text{Å})`
   is computed when the flux-density unit scale to cgs is known.


Broad-line profile moments
--------------------------

For a broad emission-line profile :math:`f(\lambda)`, defined over a recipe-
specific wavelength window:

**Profile flux** (0th moment)

   :math:`F = \int f(\lambda) \, d\lambda`

**Profile centroid** (1st moment)

   :math:`\bar{\lambda} = \frac{1}{F} \int \lambda \, f(\lambda) \, d\lambda`

   Recorded as both wavelength and velocity offset from the line's vacuum
   rest wavelength.

**Profile dispersion** (2nd moment, RMS)

   :math:`\sigma_\lambda = \sqrt{\frac{1}{F} \int (\lambda - \bar{\lambda})^2 f(\lambda) \, d\lambda}`

   Recorded in Angstroms and km/s.


Full Width at Half Maximum (FWHM)
----------------------------------

The **numerical FWHM** is computed by finding the half-maximum points of the
broad profile:

1. Find the profile peak :math:`f_\mathrm{max}` from the interpolated model.
2. Locate :math:`\lambda_\mathrm{blue}` and :math:`\lambda_\mathrm{red}` where
   :math:`f(\lambda) = f_\mathrm{max} / 2`.
3. Report :math:`\mathrm{FWHM} = \lambda_\mathrm{red} - \lambda_\mathrm{blue}`
   in km/s at the line's vacuum rest wavelength.

The FWHM error is propagated from the parameter covariance matrix through the
half-maximum intersection.

For multi-component broad profiles, the FWHM is computed on the summed broad
profile (excluding narrow components).


Equivalent Width (EW)
---------------------

**Rest-frame equivalent width**:

   :math:`\mathrm{EW} = \int \frac{f_\lambda^\mathrm{line}(\lambda)}{f_\lambda^\mathrm{cont}(\lambda)} \, d\lambda`

   where :math:`f_\lambda^\mathrm{line}` is the continuum-subtracted emission-line
   model and :math:`f_\lambda^\mathrm{cont}` is the global continuum model.


Host galaxy fractions
---------------------

When host decomposition is enabled, the host fraction at a given wavelength
is:

   :math:`f_\mathrm{host}(\lambda) = \frac{f_\lambda^\mathrm{host}}{f_\lambda^\mathrm{total}}`

   where :math:`f_\lambda^\mathrm{total} = f_\lambda^\mathrm{host} + f_\lambda^\mathrm{AGN}`.

Standard host fractions are reported at:

- 3000 Å rest-frame (UV)
- 5100 Å rest-frame (optical, near Hβ)


Iron template flux
------------------

In local fitting, the iron flux is:

   :math:`F_\mathrm{iron} = A_\mathrm{Fe} \int T(\lambda; \mathrm{FWHM}) \, d\lambda`

   where :math:`T` is the area-normalised iron template broadened to the fitted
   FWHM, and :math:`A_\mathrm{Fe}` is the fitted iron amplitude.  The integration
   is performed over the local fitted wavelength grid.


Balmer pseudo-continuum
-----------------------

The Kovačević & Popović (2013) Balmer pseudo-continuum consists of:

1. **Bound-free component** — a Planck-weighted opacity shape below the Balmer
   edge (3646 Å):

   .. math::

      f_\lambda^\mathrm{BF} \propto B_\lambda(T_e) \, (1 - e^{-\tau_\lambda})

   where :math:`\tau_\lambda = \tau_e \, (\lambda / \lambda_e)^3` and
   :math:`\tau_e` is the edge optical depth.

2. **High-order Balmer series** — area-normalised, broadened Balmer lines from
   :math:`n_\mathrm{min}` up to :math:`n_\mathrm{max}` (50 or 400, depending
   on provenance).

The total pseudo-continuum is their sum, multiplied by a fitted amplitude and
optionally shifted in velocity.


Units
-----

``SpectrumMetadata`` carries explicit unit information:

- **Wavelength**: Angstroms (``"angstrom"``)
- **Flux density**: :math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`
  (``"1e-17 erg s^-1 cm^-2 Angstrom^-1"``, the DESI/SDSS convention)

When the cgs scale is known, measurements are reported in physical units
(:math:`\mathrm{erg\,s^{-1}\,cm^{-2}}` for fluxes,
:math:`\mathrm{erg\,s^{-1}}` for luminosities, Å for wavelengths, km/s for
velocities).  When unknown, cgs quantities are omitted and a
``"no_cgs_scale"`` warning is recorded.
