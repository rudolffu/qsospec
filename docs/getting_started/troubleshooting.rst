Troubleshooting and support
===========================

Start with the stage that failed: installation, input preparation, fitting or
loading a saved run. For a fit that completed, inspect the status of the
specific measurement you need.

Wrong version or missing import
-------------------------------

Print the interpreter, package version and imported source path using the
:doc:`installation check <installation>`. A notebook can use a different
Python installation from your terminal. Install into the intended interpreter
and restart its kernel after an upgrade. Refresh an editable installation
after pulling version or dependency changes.

If a specialist function is missing from ``qsospec``, check its module in
:doc:`../reference/api/specialist`. New specialist tools in 0.2 use module
imports; the core workflows remain at package root.

Dust-map or coordinate error
----------------------------

The standard workflow queries the Planck GNILC map using the object's RA and
Dec in degrees. Follow :doc:`dustmaps` to configure and fetch it once. Missing
map data stop a batch at preflight; missing coordinates can fail individual
objects.

If you already know the target's E(B-V), pass a
``GalacticExtinctionConfig(ebv_override=...)``. Declare
``galactic_extinction_corrected=True`` only when the supplied flux and
uncertainties are already dereddened. A zero override represents an explicit
zero-reddening assumption.

Skipped foreground correction at long wavelengths
-------------------------------------------------

``skipped_wavelength_out_of_range`` means the observed grid extends beyond
the configured extinction law. The default policy continues fitting and
records the skipped correction. For longer wavelengths, consider
``GalacticExtinctionConfig(law="wang2019")``. Set
``wavelength_out_of_range="raise"`` if your analysis requires correction to
succeed. See :doc:`../user_guide/preprocessing`.

Array fitter asks for rest-frame flux
-------------------------------------

Construct a ``Spectrum`` with the actual input frame, then call
``prepare_spectrum`` before a low-level fitter. It transforms flux density
and errors as well as wavelengths. Setting ``wave_frame="rest"`` declares
that all these transformations have already been made. Follow
:doc:`../how_to/fit_arrays` for a complete preparation sequence.

Line absent or measurement unavailable
--------------------------------------

Check ``result.complex_statuses`` first. A line may be unrequested, outside
usable coverage, masked at its core or part of a failed complex. The minimum
and maximum wavelength alone do not establish usable coverage.

For a fitted complex, check its ``success``, ``metrics``, ``metric_errors``
and warnings. A missing or nonfinite value/error remains unavailable; it
does not represent zero flux or zero uncertainty. See
:doc:`../science/coverage_reliability` and :doc:`../reference/warnings`.

Cgs flux or luminosity unavailable
----------------------------------

Relative-flux spectra can yield profile widths and relative measurements, but
have no physical flux scale. Supply the correct ``flux_unit`` and
``flux_scale`` when your data are calibrated. For SDSS numerical flux values,
``flux_scale=1e-17`` is the usual convention. The
:doc:`spectral conventions <spectral_basics>` explain the conversion.

A fit converged but looks poor
------------------------------

Optimizer success describes completion under the chosen model. Inspect the
unsmoothed residuals, component sum, coverage and parameters at bounds.
Compare scientifically motivated continuum, profile or host alternatives
in separate runs. Use :doc:`../user_guide/qa_plots` to assess the measurement;
automatic convergence alone does not establish a line detection.

Host fitting skipped or unreliable
----------------------------------

Check ``result.host_decomp_enabled`` and
``result.metadata.get("host_decomp_skip_reason")``. Host fitting must be
requested explicitly, needs pPXF and external templates, and is restricted
to the supported redshift range. Coverage and resolution diagnostics can
limit a host quantity even when fitting was attempted. Follow
:doc:`../how_to/fit_with_host` and :doc:`../user_guide/host_decomposition`.

Resume rejects a run directory
------------------------------

Saved runs keep an immutable configuration and storage policy. Reuse the
same scientific configuration to resume; use a new directory to compare
models or change the storage mode. Archive readability does not guarantee
that an older run is eligible for resume. See
:doc:`../reference/run_bundles` for schema and measurement-vocabulary rules.

Saved model cannot be loaded
----------------------------

Use ``object_key`` if scientific object IDs are duplicated. Move the entire
run directory, including ``assets/`` and its index, when copying an analysis.
Compact models need their saved template assets and a supported evaluator.
A run from an unsupported schema must be recreated with the appropriate
inputs/software; loading does not refit or silently convert it.

Ask for help or report a problem
--------------------------------

Use `GitHub issues <https://github.com/rudolffu/qsospec/issues>`__. Include:

- The package version, Python version and imported source path.
- The workflow, input format, flux units/scale and wavelength frame.
- The relevant configuration, line ID and warning code or full traceback.
- A small reproducible example, with shareable input or synthetic data.

For unexpected measurements, include the relevant QA region and explain
which quantity you expected to recover. This makes it possible to separate
input-convention problems from model and fitting behavior.
