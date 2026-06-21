Configuration
=============

All configuration objects in ``qsospec`` are immutable dataclasses.  Construct
them directly, and the fitter will validate them before use.

Galactic extinction configuration
---------------------------------

.. autoclass:: qsospec.GalacticExtinctionConfig
   :noindex:

File-based, host-decomposition, and batch workflows apply foreground
dereddening before every other processing step.  The default configuration
uses ``PlanckGNILCQuery`` (``map_name="planck"``; ``"planck16"`` is an alias)
and the Fitzpatrick (1999) law with :math:`R_V=3.1`.

Use the SFD map with the Schlafly & Finkbeiner (2011) recalibration:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(map_name="sfd")

SFD E(B-V) values are multiplied by 0.86 before evaluating F99.  To bypass a
map query, supply a known value:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(ebv_override=0.035)

To disable correction explicitly:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(enabled=False)

Pass the configuration as ``galactic_extinction_config=extinction`` to
``fit_global_lines_workflow``, ``fit_with_optional_host_decomp``,
``fit_object_to_store``, or ``fit_batch``.  These workflows fail fast when
correction is enabled but coordinates or map files are unavailable.

The lower-level ``fit_local``, ``fit_global_continuum``, and
``fit_global_lines`` functions do not query maps and treat ``Spectrum`` inputs
as already corrected.  Use :func:`qsospec.correct_spectrum` for explicit
in-memory preprocessing.  See the `dustmaps Planck/SFD documentation
<https://dustmaps.readthedocs.io/en/latest/modules.html>`__ and the
`dust-extinction F99 documentation
<https://dust-extinction.readthedocs.io/en/latest/api/dust_extinction.parameter_averages.F99.html>`__
for implementation details.

Global continuum configuration
------------------------------

.. autoclass:: qsospec.GlobalContinuumConfig
   :noindex:

   The top-level configuration for the global AGN continuum fit.  Controls the
   power-law, iron templates, Balmer pseudo-continuum, continuum windows, and
   optimizer settings.

   **Parameters**

   *power_law* (:class:`PowerLawConfig`)
     The pivoted power-law continuum model.

   *uv_iron* (:class:`IronTemplateConfig`)
     UV iron template configuration (default: VW01).

   *optical_iron* (:class:`IronTemplateConfig`)
     Optical iron template configuration (default: Park22).

   *balmer_pseudocontinuum* (:class:`BalmerPseudoContinuumConfig`)
     Balmer pseudo-continuum configuration.

   *continuum_windows* (tuple of ``(start, end)``, optional)
     Rest-frame wavelength windows used for continuum anchoring.  Defaults to
     ``LEGACY_CONTINUUM_WINDOWS`` (24 regions).  Use
     :meth:`GlobalContinuumConfig.lya_safe` for Lyα-safe windows that avoid the
     forest and line peak.

   *mask_windows* (tuple of ``(start, end)``, optional)
     Additional rest-frame windows to mask during continuum fitting.

   *min_component_pixels* (int, default 10)
     Minimum pixels required to include a continuum component.

   *clip_blue_absorption* (bool, default False)
     If True, clip deep blue-side absorption troughs (BAL-like) during continuum
     fitting.

   *clip_threshold* (float, default 5.0)
     Absorption-clip threshold in sigma.

   *optimizer_method* (str, default ``"auto"``)
     Solver selection: ``"auto"``, ``"variable_projection"``, or
     ``"legacy_joint"``.

   *jacobian_method* (str, default ``"semi_analytic"``)
     Derivative method: ``"semi_analytic"`` or ``"2-point"``.

   *max_nfev* (int, default 400)
     Maximum nonlinear function evaluations.

   ``lya_safe()`` (classmethod) — return a copy with Lyα-safe continuum windows
   (anchor points redward of 3400 Å rest-frame, avoiding Lyα forest and peak
   contamination).


Power-law configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qsospec.PowerLawConfig
   :noindex:

   A pivoted :math:`f_\\lambda` power-law for the AGN continuum:
   :math:`f_\\lambda = \\mathrm{norm} \\cdot (\\lambda / \\mathrm{pivot})^{\\mathrm{slope}}`.

   *enabled* (bool, default True)
     Whether the power-law is included.

   *pivot* (float, default 3000.0)
     Pivot wavelength in rest-frame Angstroms.

   *norm* (float, default 1.0)
     Initial flux-density normalisation at the pivot wavelength.

   *norm_bounds* (tuple, default ``(0, None)``)
     Bounds on the normalisation.

   *slope* (float, default -1.5)
     Initial power-law index.

   *slope_bounds* (tuple, default ``(-5, 0.5)``)
     Bounds on the slope.


Iron template configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qsospec.IronTemplateConfig
   :noindex:

   Iron template configuration for local and global fitting.  Bundled templates
   are resolved by name; external templates require an explicit path.

   *template* (str)
     Template name.  Predefined: ``"bg92_optical"``, ``"park22_optical"``,
     ``"veron04_optical"``, ``"vw01_uv"``, or ``"external"``.

   *template_path* (str or Path, optional)
     Path to an external two-column ASCII template.

   *amp* (float, default 1.0)
     Initial iron amplitude.

   *amp_bounds* (tuple, default ``(0, None)``)
     Bounds on iron amplitude.

   *fwhm_kms* (float, default 3000.0)
     Initial iron broadening FWHM in km/s.

   *fwhm_bounds* (tuple, default ``(10, 20000)``)
     Bounds on iron FWHM.

   Class methods: :meth:`.bg92`, :meth:`.park22`, :meth:`.veron04`, :meth:`.vw01`
   return presets with reasonable default FWHM values.


Balmer pseudo-continuum configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qsospec.BalmerPseudoContinuumConfig
   :noindex:

   The Kovačević & Popović (2013) Balmer pseudo-continuum: bound-free emission
   plus a high-order broadened Balmer series.

   *enabled* (bool, default True)
     Whether the Balmer pseudo-continuum is included.

   *edge* (float, default 3646.0)
     Balmer edge wavelength in rest-frame Angstroms.

   *temperature_k* (float, default 15000.0)
     Electron temperature in Kelvin.

   *tau_edge* (float, default 1.0)
     Optical depth at the Balmer edge.

   *log10_ne* (int, default 9)
     Log10 electron density.  Options: 9 or 10.

   *n_min* (int, default 6)
     Minimum principal quantum number for Balmer-series lines.

   *provenance* (str, default ``"sh95"``)
     Balmer line-list provenance: ``"sh95"`` (pure Storey & Hummer 1995 through
     n=50), ``"k13"`` (K13-full extension through n=400), or ``"asymptotic"``.

   *amplitude* (float, default 1.0)
     Initial Balmer amplitude.

   *amplitude_bounds* (tuple, default ``(0, None)``)
     Bounds on the amplitude.

   *fit_fwhm* (bool, default True)
     Whether to fit the Balmer-series FWHM.

   *fwhm_kms* (float)
     Initial Balmer-series FWHM in km/s.

   *fwhm_bounds* (tuple)
     Bounds on the Balmer-series FWHM.

   *velocity_kms* (float, default 0.0)
     Balmer-series velocity shift in km/s.

   *velocity_bounds* (tuple)
     Bounds on velocity shift.

   *sync_with_hbeta* (bool, default False)
     If True and Hβ is successfully fitted, replace the Balmer FWHM with the
     broad Hβ FWHM in a post-continuum synchronisation step.

   *sync_min_fwhm_snr* (float, default 3.0)
     Minimum Hβ S/N required for synchronization.


Emission complex configurations
--------------------------------

Hβ/[O III] complex
^^^^^^^^^^^^^^^^^^

.. autoclass:: qsospec.HbetaComplexConfig
   :noindex:

   Constrained Hβ plus [O III] λλ4959,5007 fitting configuration.

   *fit_windows* (tuple, default ``(4200, 5600)``)
     Rest-frame fit windows as ``(blue_min, red_max)``.

   *broad_fwhm_bands* (tuple)
     Ordered broad Hβ component FWHM ranges, e.g. ``(1000, 6000), (6000, 15000), (15000, None)``.

   *broad_velocity_bounds* (tuple, default ``(-3000, 3000)``)
     Velocity bounds in km/s for broad components.

   *narrow_fwhm_bounds* (tuple, default ``(50, 1500)``)
     Narrow FWHM bounds in km/s.

   *narrow_velocity_bounds* (tuple, default ``(-500, 500)``)
     Narrow velocity bounds.

   *oiii_ratio_5007_4959* (float, default 2.98)
     Fixed [O III] λ5007 / λ4959 flux ratio.

   *fit_oiii_wings* (bool, default True)
     Whether to fit blue/red [O III] outflow wings (BIC selection).

   *heii_enabled* (bool, default False)
     Whether to include He II λ4686.

   *optimizer_method* (str, default ``"auto"``)
     Same options as global continuum.

   *jacobian_method* (str, default ``"semi_analytic"``)
     Same options as global continuum.

   *max_nfev* (int, default 400)
     Maximum nonlinear evaluations.


Mg II complex
^^^^^^^^^^^^^

.. autoclass:: qsospec.MgIIComplexConfig
   :noindex:

   Mg II λλ2796,2803 fitting configuration.

   *fit_windows* (tuple, default ``(2200, 3100)``)
     Rest-frame fit windows.

   *broad_fwhm_bands* (tuple)
     Two ordered broad Mg II components.

   *broad_velocity_bounds* (tuple, default ``(-3000, 3000)``)

   *narrow_fwhm_bounds* (tuple, default ``(50, 1500)``)

   *narrow_velocity_bounds* (tuple, default ``(-500, 500)``)

   *optimizer_method*, *jacobian_method*, *max_nfev*
     Same as above.


Hα/[N II]/[S II] complex
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qsospec.HalphaComplexConfig
   :noindex:

   Broad Hα plus tied narrow Hα, [N II] λλ6548,6585, and [S II] λλ6716,6731.

   *fit_windows* (tuple, default ``(6300, 6800)``)
     Rest-frame fit windows.

   *broad_fwhm_bands* (tuple)
     Three ordered broad Hα components.

   *broad_velocity_bounds* (tuple, default ``(-3000, 3000)``)

   *narrow_fwhm_bounds* (tuple, default ``(50, 1500)``)

   *narrow_velocity_bounds* (tuple, default ``(-500, 500)``)

   *nii_ratio_6585_6549* (float, default 2.94)
     Fixed [N II] λ6585 / λ6549 flux ratio.

   *optimizer_method*, *jacobian_method*, *max_nfev*
     Same as above.


Lyα/N V complex
^^^^^^^^^^^^^^^

.. autoclass:: qsospec.LyaNVComplexConfig
   :noindex:

   Lyα λ1216 and N V λλ1238,1242 fitting with coverage classification and
   absorption masking.

   *fit_lya* (bool, default True)
     Whether to fit Lyα.

   *fit_nv* (bool, default True)
     Whether to fit N V.

   *window* (tuple, default ``(1100, 1290)``)
     Rest-frame fit window.

   *n_gaussians_broad* (int, default 3)
     Number of broad Lyα Gaussians.

   *n_gaussians_narrow* (int, default 1)
     Number of narrow Lyα Gaussians.

   *broad_fwhm_bands* (tuple)
     Ordered broad Lyα FWHM bands.

   *broad_velocity_bounds* (tuple, default ``(-3000, 3000)``)

   *narrow_fwhm_bounds* (tuple, default ``(50, 1500)``)

   *narrow_velocity_bounds* (tuple, default ``(-500, 500)``)

   *nv_mode* (str, default ``"effective_blend"``)
     N V treatment: ``"effective_blend"`` (single profile), ``"equal_doublet"``
     (shared-kinematics doublet), or ``"none"``.

   *coverage_min_fraction* (float, default 0.5)
     Minimum rest-frame coverage fraction for full-fit eligibility.

   *absorption_mask_threshold* (float, default 3.0)
     Standardised-residual threshold for identifying narrow absorption features.

   *absorption_min_width_pixels* (int, default 3)
     Minimum contiguous pixel width for absorption masking.

   *reliability_min_red_fraction* (float, default 0.6)
     Minimum red-side valid fraction for a reliable fit.


Uncertainty configuration
-------------------------

.. autoclass:: qsospec.UncertaintyConfig
   :noindex:

   Controls covariance and Monte Carlo uncertainty estimation.

   *covariance* (bool, default True)
     Whether to compute parameter covariance matrices.

   *monte_carlo_trials* (int, default 0)
     Number of Monte Carlo realisations (0 disables).

   *random_seed* (int, optional)
     Random seed for reproducibility.

   *refit_host_in_mc* (bool, default False)
     Whether to refit the host decomposition in each Monte Carlo trial.


Local fitting configuration
-----------------------------

.. autoclass:: qsospec.LocalFitConfig
   :noindex:

   Configuration for fitting one or more independent local emission-line windows.

   *windows* (list of :class:`LineComplexConfig`)
     One or more local window configurations.

   *mode* (str, default ``"independent"``)
     Fitting mode (only ``"independent"`` is currently supported).

   *require_min_pixels* (int, default 5)
     Minimum valid pixels in a window to attempt fitting.

   *edge_buffer* (float, default 0.0)
     Rest-frame Angstrom buffer at window edges.


.. autoclass:: qsospec.LineComplexConfig
   :noindex:

   Configuration for one local emission-line complex fit.

   *center* (float)
     Rest-frame central wavelength of the complex.

   *window* (tuple)
     Rest-frame fit window as ``(blue, red)``.

   *components* (list of :class:`GaussianComponent` or :class:`LorentzianComponent`)
     Emission-line profile components.

   *local_continuum* (str, optional)
     Local continuum mode: ``"constant"``, ``"linear"``, or ``None``.

   *iron* (:class:`IronTemplateConfig`, optional)
     Optional iron template for this complex.

   *fit_windows* (tuple, optional)
     Explicit fit-window sub-regions.

   *mask_windows* (tuple, optional)
     Wavelength regions to mask within the fit window.

   *plot_window* (tuple, optional)
     Wavelength range for QA plotting.

   *jacobian* (str, default ``"dense"``)
     Jacobian type: ``"dense"`` or ``"sparse"``.

   *max_nfev* (int, default 200)
     Maximum function evaluations for this complex.


Profile component configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qsospec.GaussianComponent
   :noindex:

   Initial values and bounds for one Gaussian profile.

   *name* (str) -- Component label.
   *center* (float) -- Rest-frame centroid in Angstroms.
   *amp* (float) -- Initial amplitude.
   *sigma* (float) -- Initial Gaussian sigma in Angstroms.
   *bounds* (dict) -- Per-parameter bounds ``{param: (low, high)}``.

.. autoclass:: qsospec.LorentzianComponent
   :noindex:

   Initial values and bounds for one Lorentzian profile.

   *name* (str), *center* (float), *amp* (float)
   *gamma* (float) -- Lorentzian half-width at half-maximum in Angstroms.
   *bounds* (dict) -- Per-parameter bounds.


QA plot configuration
---------------------

.. autoclass:: qsospec.GlobalQAPlotConfig
   :noindex:

   Rendering options for the main global QA figure.

   *figure_width* (float, default 10.5)
   *figure_height* (float, default 6.2)
   *max_zoom_panels* (int, default 4)
   *show_smoothed_data* (bool, default True)
   *smooth_original_spectrum_for_display* (bool, default False)
   *smoothing_window_pixels* (int, default 7)
   *show_residual_panel* (bool, default True)
   *show_fit_regions* (bool, default True)
   *unmodelled_windows* (tuple, optional)
   *show_host_context_in_overview* (bool, default False)
   *object_name* (str, optional)
   *object_label* (str, optional)
   *show_coordinates* (bool, default True)
   *output_format* (str, default ``"png"``) -- ``"png"``, ``"pdf"``, or ``"both"``.
   *write_other_diagnostics* (bool, default False)
