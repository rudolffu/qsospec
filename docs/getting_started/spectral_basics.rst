Spectral quantities and conventions
=======================================

Use this page to connect the input arrays with the measurements in a fit.
The :doc:`quickstart` applies these conventions to a real SDSS spectrum.

Flux density and line flux
------------------------------

A spectrum gives flux density per unit wavelength, :math:`F_\lambda`.
An emission-line flux is the integral of its fitted profile after separating
the continuum:

.. math::

   F_{\rm line}=\int F_{\lambda,\rm line}\,d\lambda.

For physical cgs inputs, flux density has units
:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`, while integrated
flux has units :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.

SDSS stores numerical flux densities in units of :math:`10^{-17}` times
the physical cgs unit. An array value of 5 with ``flux_scale=1e-17`` means
:math:`5\times10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}`.
The fitted arrays retain the numerical scale; ``*_flux_cgs`` measurements
apply it. Relative-flux spectra have no physical cgs measurements.

QSOSpec fits :math:`F_\lambda`, not :math:`F_\nu`. Convert Jy or other
frequency-density inputs before constructing a spectrum. With wavelength in
Angstrom and :math:`c` in Angstrom per second,

.. math::

   F_\lambda=F_\nu\frac{c}{\lambda^2},\qquad
   1\,\mathrm{Jy}=10^{-23}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}.

Apply the same conversion to the flux-density errors. See
:doc:`../user_guide/spectrum_inputs` for declaring units and scale.

Wavelength and redshift
---------------------------

Line recipes use vacuum rest wavelengths in Angstrom. SDSS ``loglam`` stores
the logarithm of vacuum observed wavelength; its reader evaluates
``10**loglam``. For your own arrays, convert air wavelengths to vacuum
wavelengths before fitting. Array construction does not perform an
air-to-vacuum conversion. Other file formats need a check of their wavelength
convention as described in :doc:`../user_guide/spectrum_inputs`.

QSOSpec uses the supplied redshift to transform both the wavelength coordinate
and its matching flux density:

.. math::

   \lambda_{\rm rest}=\frac{\lambda_{\rm obs}}{1+z},\qquad
   F_{\lambda,\rm rest}=(1+z)F_{\lambda,\rm obs}.

For example, at :math:`z=1`, a 10-Angstrom observed interval becomes a
5-Angstrom rest interval, and flux density 3 becomes 6 in the same numerical
scaling. Their integrated fluxes are both 30. In general,
:math:`F_{\lambda,\rm rest}\,d\lambda_{\rm rest}
=F_{\lambda,\rm obs}\,d\lambda_{\rm obs}`.

One-sigma errors also gain the factor :math:`1+z`; inverse variance is divided
by :math:`(1+z)^2`. This is a coordinate transformation of flux, not a
luminosity calculation. ``wave_frame="rest"`` declares that the wavelength,
flux density, and uncertainty already follow this convention. Whether the
data were dereddened is a separate declaration. Use :func:`qsospec.prepare_spectrum`
for ordinary observed-frame inputs; see :doc:`../user_guide/preprocessing`.

Errors and pixel masks
--------------------------

``err`` is the one-sigma flux-density error. Inverse variance is
:math:`\mathrm{ivar}=1/\sigma^2`. Zero, negative, or nonfinite inverse variance
does not provide a usable fitting error.

The boolean mask passed to ``Spectrum.from_arrays`` uses ``True`` for valid
pixels. Raw survey masks often use nonzero integers for bad-pixel flags.
Convert these flags into a boolean good-pixel mask, for example
``good_pixel_mask = (survey_flags == 0)``. The SDSS file reader handles its
survey flags and returns a boolean valid-pixel mask.

QA plots show finite excluded pixels in the spectrum trace. The residuals and
fit statistics use fitted valid pixels; the plotted model can also be evaluated
at excluded pixels. See :doc:`../user_guide/qa_plots`.

Foreground correction and model components
----------------------------------------------

Galactic dereddening removes the Milky Way foreground using the object's
coordinates and a dust map. It is applied at observed wavelengths before the
rest-frame transformation. It does not correct attenuation within the quasar
or its host. The default tutorial needs the Planck GNILC map described in
:doc:`dustmaps`.

The fitted active galactic nucleus (AGN) continuum includes a power law,
broadened iron templates,
and a Balmer pseudo-continuum when covered. Iron consists of many blended lines
that can resemble a smooth continuum. The stellar host is a separate optional
component. An emission complex is a group of nearby lines fitted together;
a blend contains overlapping profiles that share spectral pixels.

The default continuum and optional corrections are explained in
:doc:`../science/continuum_model`. The individual Gaussian components describe
the profile shape. Their number and labels alone do not identify separate
physical regions or an outflow.

Widths and velocity measurements
------------------------------------

The full width at half maximum (FWHM) measures a profile's width at half its
peak height. Line dispersion is its flux-weighted standard deviation. For
one Gaussian, :math:`\mathrm{FWHM}=2\sqrt{2\ln2}\,\sigma\simeq2.355\sigma`.
For a sum of Gaussians this relation generally differs, and the full-profile
FWHM is not an average of the component FWHMs. Different products can also
use different half-maximum conventions; consult the
:doc:`../reference/measurement_dictionary` for the exact field.

Observed widths include instrumental broadening. Intrinsic widths describe
the source profile under the supplied line-spread function (LSF). An LSF is
the response of the instrument to an intrinsically narrow line. Without a
usable LSF, QSOSpec reports observed-profile measurements. The SDSS quickstart
does not obtain an LSF from ``wdisp``, so its widths include instrumental
broadening.

An iron-template convolution width describes broadening applied to a template
that already contains line structure. A target/effective template width also
accounts for the template's native width. These are distinct from an
instrumental correction; see :doc:`../science/continuum_model` and
:doc:`../how_to/stellar_template_resolution_profiles`.

Interpreting an uncertainty
-------------------------------

Statistical covariance describes parameter uncertainty near the fitted
solution under the adopted model. Bootstrap trials perturb spectra and repeat
the configured fit. Both depend on which components and stages are fitted or
held fixed. A modelling tolerance, such as the Hγ/Balmer ratio constraint,
is not an observed Hγ error bar.

Sensitivity to continuum windows, templates, or component choices needs a
comparison of those choices. Calibration uncertainty belongs to a derived
relation, such as a luminosity correction or systemic-redshift estimator.
Unavailable errors are missing values, not zero. See
:doc:`../science/uncertainties` and then :doc:`../user_guide/results` for
your first reported measurement.
