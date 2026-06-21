Workflows
=========

qsospec provides three fitting modes: local (independent windows), global
(multi-complex continuum + line workflow), and batch (process-parallel sample
fitting with Parquet run storage).


Local fitting
-------------

Local fitting treats each emission-line complex independently within its own
rest-frame wavelength window.  It is the fastest mode and suitable for
exploratory work, targeted line measurements, or when only a subset of the
spectrum is available.

Create a ``Spectrum`` from arrays and pass it to :func:`qsospec.fit_local`
with a :class:`~qsospec.LocalFitConfig`.

Array-based ``Spectrum`` inputs are treated as already corrected for Galactic
extinction.  Use :func:`qsospec.correct_spectrum` first when correction is
needed.

.. code-block:: python

    import qsospec

    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=redshift)

    config = qsospec.LocalFitConfig(
        windows=[
            qsospec.recipes.local_mgii(),
            qsospec.recipes.local_hbeta(profile="lorentzian"),
            qsospec.recipes.local_halpha(),
        ],
    )
    result = qsospec.fit_local(spectrum, config)

Each window produces a :class:`~qsospec.FitResult` (accessible via
``result.window_results``).  Windows that fail (e.g. insufficient coverage,
optimizer failure) record the failure independently, so one bad window does
not affect the others.

You can also fit a single complex directly:

.. code-block:: python

    config = qsospec.recipes.local_hbeta(
        profile="gaussian",
        iron_template=qsospec.IronTemplateConfig.park22(),
    )
    result = qsospec.fit_line_complex(spectrum, config)

Local models support:

- Gaussian and Lorentzian emission-line profiles
- Constant, linear, or absent local continua
- Optional Fe II templates with fitted amplitude and FWHM
- Explicit fit and mask sub-windows
- Dense or sparse analytic Jacobians

**Limitations**: local windows do not share continuum or kinematic parameters.
There is no global continuum model, and the iron/continuum in one window does
not affect another.


Global fitting
--------------

The global workflow is the primary science interface.  It fits a full AGN
continuum model (power-law, iron, Balmer pseudo-continuum) and then multiple
emission-line complexes on the continuum-subtracted spectrum.

.. code-block:: python

    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=redshift)
    result = qsospec.fit_global_lines(spectrum)

This uses sensible defaults.  Each step can be configured:

.. code-block:: python

    result = qsospec.fit_global_lines(
        spectrum,
        global_config=qsospec.GlobalContinuumConfig(
            power_law=qsospec.PowerLawConfig(slope=-1.7),
            optimizer_method="variable_projection",
        ),
        hbeta_config=qsospec.HbetaComplexConfig(
            fit_oiii_wings=True,
        ),
        mgii_config=qsospec.MgIIComplexConfig(),
        halpha_config=qsospec.HalphaComplexConfig(),
        uncertainty_config=qsospec.UncertaintyConfig(
            covariance=True,
            monte_carlo_trials=100,
            random_seed=42,
        ),
        lya_nv_config=qsospec.LyaNVComplexConfig(
            nv_mode="equal_doublet",
        ),
    )


Fitting steps
^^^^^^^^^^^^^

The global workflow executes these steps in order:

1. **Galactic dereddening** — file-based workflows query the configured dust
   map and apply F99 to observed-frame flux and uncertainty.  Direct
   ``fit_global_lines(Spectrum)`` calls treat their input as pre-corrected.

2. **Continuum fitting** — fits the power-law, UV iron, optical iron, and
   Balmer pseudo-continuum on line-free windows using bounded variable
   projection.  Returns a :class:`~qsospec.GlobalContinuumResult`.

3. **Optional Hβ synchronisation** — if
   ``BalmerPseudoContinuumConfig.sync_with_hbeta=True`` and the later Hβ fit
   succeeds, the Balmer-series FWHM is replaced with the broad Hβ FWHM.

4. **Adaptive complex selection** — from the rest-frame coverage, qsospec
   determines which emission-line recipes are executable.  By default
   (``complexes=None``), all covered, auto-enabled recipes are selected.

5. **Hβ/[O III] fitting** — constrained broad + narrow Hβ model with ordered
   components, optional [O III] wings, and BIC-based wing selection.

6. **Mg II fitting** — two ordered broad + one narrow Mg II components.

7. **Hα/[N II]/[S II] fitting** — three ordered broad Hα + tied narrow lines.

8. **Optional complexes** — Lyα/N V (with coverage classification and
   absorption masking), C IV, C III], K/NIR (Paschen/NIR), [O II]/Hγ/Ne III,
   and generic narrow-line recipes.  Each records independent success/failure.

9. **Uncertainty estimation** — covariance errors from the Hessian at the
   solution, and optionally Monte Carlo perturbation trials.


Host decomposition workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optional pPXF host-decomposition workflow reads a spectrum, optionally
subtracts the host galaxy, and then runs qsospec fitting.  This requires
``qsospec[host]``.

.. code-block:: python

    result = qsospec.fit_global_lines_workflow(
        "spectrum.fits",
        row_index=0,
        run_host_decomp=True,
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(
            map_name="planck",
        ),
    )

Or for local fitting with host subtraction:

.. code-block:: python

    result = qsospec.fit_with_optional_host_decomp(
        "spectrum.fits",
        local_config,
        fit_kind="global",
        run_host_decomp=True,
    )


Controlling complexes
^^^^^^^^^^^^^^^^^^^^^

Pass an explicit list of recipe IDs or :class:`~qsospec.ComplexRecipe` objects
to control which complexes are fitted:

.. code-block:: python

    # Fit only Hβ and Mg II
    result = qsospec.fit_global_lines(
        spectrum,
        complexes=["hb", "mgii"],
    )

    # Continuum only (no emission lines)
    result = qsospec.fit_global_lines(
        spectrum,
        complexes=[],
    )

    # Custom selection including Lyα
    result = qsospec.fit_global_lines(
        spectrum,
        complexes=["hb", "mgii", "ha", "lya_nv"],
    )

Use :func:`qsospec.recipes.list_complexes` to see all available recipes and
:func:`qsospec.recipes.describe` for details on a specific recipe.


Solver and Jacobian options
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``optimizer_method`` and ``jacobian_method`` on each configuration control
how the nonlinear optimisation is performed:

- ``"variable_projection"`` — bounded variable projection (default for global
  fitting).  Linear parameters (amplitudes, fluxes) are solved exactly at each
  nonlinear step; only slopes, widths, and velocities are optimised nonlinearly.
- ``"legacy_joint"`` — SciPy ``least_squares`` on all parameters jointly
  with finite differences.
- ``"auto"`` — tries variable projection first; falls back to ``"legacy_joint"``
  on failure.

Jacobian methods:

- ``"semi_analytic"`` — analytic derivatives for Gaussians, Lorentzians, power-law, Balmer series; differentiated kernel for iron FWHM.
- ``"2-point"`` — finite-difference derivatives.


Batch fitting
-------------

Large samples use the same Parquet-backed run format as single objects.
:func:`qsospec.fit_batch` provides process-parallel batch execution:

.. code-block:: python

    batch = qsospec.fit_batch(
        ["spectra-000.parquet", "spectra-001.parquet"],
        "runs/sample",
        n_workers="auto",
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(
            map_name="planck",
        ),
    )

Key features:

- **Resumable**: re-running with the same configuration skips completed objects.
- **Deterministic partitioning**: for cluster jobs, use ``num_shards`` and
  ``shard_index`` to split work across nodes.
- **Process parallelism**: spawned workers with one BLAS/OpenMP thread each.
- **Multiple input formats**: DESI/SPARCL-like Parquet, SDSS/LAMOST/IRAF FITS,
  CSV/Parquet manifests.

See :doc:`../run_bundles` for the run-bundle format, catalog construction,
derived quantities, and QA rendering.
