Use the AGN-aware masked pPXF host mode
========================================

The default host strategy, ``masked_simple``, fits E-MILES stellar templates,
a small power-law basis, and the historical additive polynomial while masking
emission lines. It remains the production default.

The optional ``agn_pseudocontinuum_masked`` strategy reduces the chance that
optical Fe II or the Balmer pseudo-continuum is assigned to the stellar host.
It performs a preliminary, nonrecursive qsospec Hα/Hβ fit, selects the nearest
published broad-line width, and fits pPXF with:

* E-MILES stellar templates;
* :math:`F_\lambda` power laws with slopes from -3.0 through 0.0 in steps of
  0.1;
* BG92 optical Fe II at the selected broad width; and
* the qsospec KD13/Storey-Hummer Balmer continuum and high-order series at the
  same width.

Strong emission lines remain masked in pPXF. Only the fitted stellar SSP
component is subtracted; the power law, Fe II, Balmer emission, and spectral
lines remain for the final standard qsospec fit.

Setup
-----

Install ``qsospec[host]`` and obtain the external E-MILES NPZ bundle from
`micappe/ppxf_data <https://github.com/micappe/ppxf_data>`__. For example,
place ``spectra_emiles_9.0.npz`` under ``~/tools/ppxf_data``. These stellar
templates are not included in the qsospec wheel.

.. code-block:: python

   import qsospec

   host_config = qsospec.HostDecompConfig(
       strategy="agn_pseudocontinuum_masked"
   )

   result = qsospec.fit_global_lines_workflow(
       "spectrum.fits",
       run_host_decomp=True,
       template_root="~/tools/ppxf_data",
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

``ppxf_agn_fraction_flux_global`` is the wavelength-integrated AGN
pseudo-continuum fraction over valid, non-emission-line pPXF pixels. Its
wavelength support is recorded with the fit. Values above 0.8 produce a
warning, not an automatic reliability veto. In contrast, ``fAGN_5100`` is a
flux-density sample and is not a fraction.

Coverage and model closure
--------------------------

Coverage is classified from valid rest-frame pixels as ``full_optical``,
``optical_core``, ``blue_optical``, or ``insufficient``. The classifier also
records support for Ca H+K, both sides of the 4000-Å break, G band, the Hβ
absorption region, Mg b, and Na D. Blue-only fits are returned for inspection
but carry ``limited_wavelength_leverage``; insufficient coverage is not marked
reliable.

The physical pPXF components are summed and compared with ``ppxf_bestfit``.
The default new mode has no additive or multiplicative polynomial, so closure
should be numerical. An unexplained closure mismatch makes the host fit
unreliable.

Provenance and limitations
--------------------------

This mode is inspired by `Aydar et al. (2026)
<https://ui.adsabs.harvard.edu/abs/2026A%26A...710A.141A>`__, but it is not an
exact reproduction. It reuses qsospec's bundled BG92 and KD13/Storey-Hummer
templates because the paper's private Fe II/Balmer template arrays are not
distributed. Emission lines are masked rather than fitted as pPXF gas
components. The result therefore records
``host_pseudocontinuum_exact_replication=False``.

The E-MILES stellar templates and AGN templates receive separate pPXF
components: stellar velocity and dispersion are fitted, while the physically
prebroadened AGN templates have fixed independent kinematics. An available
object-specific instrumental LSF is applied once. Intrinsic bundled templates
are cached, while object-specific LSF convolution remains per object.

The historical :math:`z<1.2` request gate is retained. Actual wavelength
coverage still determines reliability. Monte Carlo is off by default; a
configured host-refit Monte Carlo uses the selected strategy but can be
expensive. Compare both strategies on a bounded validation sample before
changing a production pipeline.

