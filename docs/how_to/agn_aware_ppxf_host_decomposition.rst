Use the AGN-aware masked pPXF host mode
========================================

The default host strategy, ``masked_simple``, fits E-MILES stellar templates,
a small power-law basis, and an additive polynomial while masking emission
lines.

The optional ``agn_pseudocontinuum_masked`` strategy reduces the chance that
optical Fe II or the Balmer pseudo-continuum is assigned to the stellar host.
It performs a preliminary, nonrecursive qsospec Hα/Hβ fit, selects the nearest
published broad-line width, and fits pPXF with:

* E-MILES stellar templates;
* :math:`F_\lambda` power laws with slopes from -3.0 through 0.0 in steps of
  0.1;
* BG92 optical Fe II at the selected broad width, or the final fit's
  exclusive full-range iron template when configured; and
* the qsospec KD13/Storey-Hummer Balmer continuum and high-order series at the
  same width.

Native E-MILES is the default stellar profile. Native XSL and exact,
object-specific preconvolved XSL are optional alternatives; see
:doc:`stellar_template_resolution_profiles`. In every profile, qsospec keeps
the native science spectrum unchanged and convolves only a template that is
sharper than the data. A coarser template is retained with a diagnostic rather
than causing the pixel to be discarded.

Strong emission lines remain masked in pPXF. Only the fitted stellar SSP
component is subtracted; the power law, Fe II, Balmer emission, and spectral
lines remain for the final standard qsospec fit.

Setup
-----

Install ``qsospec[host]`` and obtain the external E-MILES NPZ bundle from
`micappe/ppxf_data <https://github.com/micappe/ppxf_data>`__. For example,
place ``spectra_emiles_9.0.npz`` under ``/path/to/ppxf_data``. These stellar
templates are not included in the qsospec wheel.

.. code-block:: python

   import qsospec

   host_config = qsospec.HostDecompConfig(
       strategy="agn_pseudocontinuum_masked"
   )

   result = qsospec.fit_global_lines_workflow(
       "spectrum.fits",
       run_host_decomp=True,
       template_root="/path/to/ppxf_data",
       template_file="spectra_emiles_9.0.npz",
       host_config=host_config,
   )

The public width grid is 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800,
3400, 4000, 4800, 5800, 7000, 8400, 10000, and 11800 km/s. Hα is preferred
over Hβ when both broad measurements pass the configured flux- and width-S/N
thresholds. With the default fallback policy, an object without a reliable
broad Balmer width uses ``masked_simple`` and records the reason.

Interpret the result
--------------------

Inspect the strategy and reliability before using the host model:

.. code-block:: python

   print(result.metadata["host_strategy_requested"])
   print(result.metadata["host_strategy_used"])
   print(result.metadata["host_strategy_fallback"])
   print(result.metadata["host_coverage_class"])
   print(result.metadata["host_fit_reliable"])
   print(result.metadata["host_fit_quality"])

``result.host_component_models`` contains aligned stellar, power-law, optical
Fe II, optional UV Fe II, Balmer-continuum, high-order Balmer, aggregate AGN,
pPXF best-fit, physical-total, closure-residual, and host-subtracted arrays
when applicable. The run-store model records and optional full-grid CSV retain
the same components.

For example, this uses Verner09 in both the host AGN basis and final global
continuum:

.. code-block:: python

   global_config = qsospec.GlobalContinuumConfig.with_single_iron(
       "verner09"
   )
   result = qsospec.fit_global_lines_workflow(
       "spectrum.fits",
       run_host_decomp=True,
       host_config=host_config,
       global_config=global_config,
   )

An explicit ``full_feii_template`` in ``HostAgnPseudoContinuumConfig`` takes
precedence over this inheritance.

``ppxf_agn_fraction_flux_global`` is the wavelength-integrated AGN
pseudo-continuum fraction over valid, non-emission-line pPXF pixels. Its
wavelength support is recorded with the fit. Values above 0.8 produce a
warning; reliability is assessed from the full set of host diagnostics. Direct
local pPXF samples use
names such as ``fAGN_pPXF_5100`` and ``fracHost_pPXF_5100``. ``fAGN_5100`` is
the final qsospec AGN-continuum flux density,
and ``fracHost_5100`` is the final fraction using that continuum with the pPXF
stellar host. See :ref:`host-fraction-vocabulary`.

Coverage and model closure
--------------------------

Coverage is classified from valid rest-frame pixels as ``full_optical``,
``optical_core``, ``blue_optical``, or ``insufficient``. The classifier also
records support for Ca H+K, both sides of the 4000-Å break, G band, the Hβ
absorption region, Mg b, and Na D. Blue-only fits are returned for inspection
but carry ``limited_wavelength_leverage``; insufficient coverage is not marked
reliable.

The physical pPXF components are summed and compared with ``ppxf_bestfit``.
With the default ``agn_pseudocontinuum_masked`` configuration, no additive or
multiplicative polynomial is fitted, so closure should be numerical. An
unexplained closure mismatch makes the host fit unreliable.

Provenance and limitations
--------------------------

This mode implements an AGN-aware pPXF host-decomposition method similar to
that used by `Aydar et al. (2026)
<https://ui.adsabs.harvard.edu/abs/2026A%26A...710A.141A>`__. Stellar and AGN
pseudo-continuum templates are fitted jointly, with strong emission lines
masked during the host fit.

The stellar templates and AGN templates receive separate pPXF
components: stellar velocity and dispersion are fitted, while the physically
prebroadened AGN templates have fixed independent kinematics. An available
object-specific instrumental LSF is applied once to the fit-time templates,
never to the input data. Intrinsic bundled AGN templates and native stellar
libraries are cached, while object-specific runtime convolution remains per
object unless an exact preconvolved XSL product is selected. HostSED
reconstruction always uses the native source SSP library.

Host decomposition runs only when requested and the input redshift is finite
and below 1.2. Actual wavelength coverage determines reliability. Monte Carlo
is off by default; a configured host-refit Monte Carlo uses the selected strategy but can be
expensive. Compare both strategies on representative spectra before choosing one
for a sample.
