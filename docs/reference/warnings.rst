Warning reference
=================

Warnings are structured :class:`qsospec.FitWarning` records with a code,
severity, message, and context. Use ``result.warning_codes()`` for filtering.

Complex fitting
---------------

``optional_line_fit_failed`` (warning)
   A selected non-essential complex did not converge. Inspect its zoom,
   coverage metadata, and optimizer message.

``line_complex_not_covered`` (info)
   The complex failed its coverage policy. No action is needed unless the
   input was expected to cover it.

``complex_partially_covered`` (info)
   A component-adaptive fit retained only covered lines. Check archived
   component metadata before using measurements.

``parameter_at_bound`` (warning)
   A fitted parameter reached a configured limit. Treat derived metrics
   cautiously and review the profile model or bounds.

``optimizer_fallback_legacy`` (warning)
   Variable projection fell back to the legacy joint solver. Compare fit
   quality and consider simplifying the model.

``covariance_rank_deficient`` (warning)
   Statistical covariance is not fully constrained. Do not rely on missing or
   unstable covariance errors.

Lyα/N V
-------

``lya_red_side_only`` (info)
   Lyα was fitted from limited red-side coverage. Measurements remain
   explicitly limited and unreliable.

``lya_low_flux_snr`` (warning)
   Broad Lyα flux S/N is below the reliability threshold. Require
   ``lya_fit_reliable`` before precision use.

``lya_absorption_dominated`` (warning)
   Too much of the Lyα window was absorption-masked. Inspect rejected pixels.

``lya_kinematics_at_bound`` (warning)
   Lyα velocity or width reached a bound. Treat the profile measurement as
   unreliable.

Continuum and templates
-----------------------

``balmer_components_disabled_short_coverage`` (info)
   Maximum valid rest wavelength is at or below 3600 Å. Balmer emission was
   intentionally disabled.

``balmer_high_n_extension_model_dependent`` (info)
   The n=51–400 extension contributes to the Balmer model. Preserve template
   provenance in scientific reporting.

``iron_template_partial_coverage`` (info/warning)
   Only part of an iron template overlaps the fit. Check continuum stability
   near template edges.

Workflow and uncertainty
------------------------

``host_decomp_skipped_redshift`` (info)
   Host fitting was requested but failed the :math:`z<1.2` gate.

``statistical_uncertainty_excludes_continuum_host`` (info)
   Reported statistical errors omit some continuum, host, calibration, or
   model-choice systematics. Do not interpret them as total uncertainty.

Additional low-level codes may appear for invalid pixels, insufficient window
pixels, missing line centers, fixed ratios, optional components, or backend
availability. Always preserve warning context in archived products.
