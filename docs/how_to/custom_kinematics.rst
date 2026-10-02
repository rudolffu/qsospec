Custom gas kinematics
=====================

The opt-in ``qsospec.custom_kinematics`` module builds generic recipes without
changing registered recipes or the standard Hbeta wing-selection policy.

``custom_kinematics(spec)`` accepts a JSON-compatible dictionary with ``id``,
``window``, and ``components``. Component fields are those of
``ComponentRecipe``: line IDs, role, nonnegative flux bounds, velocity bounds,
FWHM bands, velocity/width groups, and optional fixed-ratio parent. The child
flux is the parent flux divided by ``fixed_ratio`` (2.98 for [O III] 4960 linked
to 5008). Set ``continuum_mode`` to ``residual_linear`` to fit a local correction.
``initial_values`` maps resolved parameter names to starting values.

``fit_custom_kinematics(spectrum, continuum, spec, n_starts=24, seed=1729)``
runs deterministic multistart fitting, including a mirrored velocity start,
and retains the converged fit with the smallest chi-square. It records each
start and its outcome in result metadata. An all-failed run remains a failed
fit. Explicit candidate recipes must implement their own model selection;
implicit component selection is disallowed with multistart.

``component_measurements`` exports velocity-ordered component records, retaining
original IDs, parameter ties, and covariance errors. ``mixture_kinematics``
measures any positive-component Gaussian mixture for one exact line ID, using
wavelength-Gaussian profiles and logarithmic velocities. It excludes all other
lines, including [O III] 4960 when measuring 5008. Negative or nonfinite fluxes
invalidate the profile; zero-flux components contribute nothing.

The generic engine also accepts optional ``initial_values``, ``n_starts``, and
``random_seed`` keyword arguments. The default remains one unmodified start.
Statistical model selection, MC refits, and the choice of velocity reference
are analysis policies, not new default scientific assumptions in QSOSpec.

``select_nested_candidates`` implements explicit nested comparisons with
configurable BIC and summed-component significance thresholds. Free doublet
fluxes use their full covariance when calculating summed significance. The
``peer_widths`` option requires the tied-width model before testing a release.

``refit_custom_continuum`` warm-refits the selected continuum family on its
accepted continuum pixels. When the baseline has the native soft Hgamma joint
fit, that joint objective is reoptimized, including the continuum/Fe II and
Hgamma parameters. Hbeta synchronization and host decomposition stay disabled.
