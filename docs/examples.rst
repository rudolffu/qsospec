Examples
========

Create a Spectrum
-----------------

The most common way to create a :class:`~qsospec.Spectrum` is from NumPy arrays:

.. code-block:: python

    import numpy as np
    import qsospec

    wave_obs = np.linspace(4000, 9000, 5000)   # observed-frame Angstroms
    flux = np.random.normal(1, 0.1, len(wave_obs))
    err = np.full_like(flux, 0.1)

    spectrum = qsospec.Spectrum.from_arrays(
        wave_obs,
        flux,
        err=err,
        z=0.5,
        wave_frame="observed",
    )

For SDSS or DESI spectra, the unit presets are applied automatically:

.. code-block:: python

    spectrum = qsospec.Spectrum.from_arrays(
        wave_obs, flux, err=err, z=0.5,
        survey="sdss",
    )


Local fitting
-------------

.. code-block:: python

    # Fit Hβ with optical iron
    hb_config = qsospec.recipes.local_hbeta(
        profile="gaussian",
        iron_template=qsospec.IronTemplateConfig.park22(fwhm_kms=3000),
    )
    result = qsospec.fit_line_complex(spectrum, hb_config)

    if result.success:
        print(result.summary())
        table = result.to_table()
        print(table[["component", "flux", "flux_err", "fwhm_kms"]])

    # Fit multiple windows independently
    config = qsospec.LocalFitConfig(
        windows=[
            qsospec.recipes.local_mgii(iron_template=qsospec.IronTemplateConfig.vw01()),
            qsospec.recipes.local_hbeta(profile="lorentzian"),
            qsospec.recipes.local_halpha(),
        ],
    )
    local_result = qsospec.fit_local(spectrum, config)

    for name, r in local_result.window_results.items():
        status = "OK" if r.success else f"FAILED: {r.message}"
        print(f"{name}: {status}")


Global fitting (basic)
-----------------------

.. code-block:: python

    result = qsospec.fit_global_lines(spectrum)

    print(f"Continuum: {'OK' if result.continuum_success else 'FAILED'}")
    for rid, status in result.complex_statuses.items():
        print(f"  {rid}: {status}")

    # Extract Hβ measurements
    if result.hbeta is not None and result.hbeta.success:
        for key in ("fwhm_broad_kms", "fwhm_broad_err_kms", "ew_rest_broad",
                     "log_l_hbeta_broad", "log_l_hbeta_broad_err"):
            if key in result.hbeta.metrics:
                print(f"  {key} = {result.hbeta.metrics[key]:.3f}")


Global fitting (customised)
----------------------------

.. code-block:: python

    result = qsospec.fit_global_lines(
        spectrum,
        global_config=qsospec.GlobalContinuumConfig(
            power_law=qsospec.PowerLawConfig(slope=-1.7),
            balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
                sync_with_hbeta=True,
            ),
        ),
        hbeta_config=qsospec.HbetaComplexConfig(
            fit_oiii_wings=True,
            heii_enabled=True,
        ),
        uncertainty_config=qsospec.UncertaintyConfig(
            covariance=True,
            monte_carlo_trials=50,
            random_seed=42,
        ),
        lya_nv_config=qsospec.LyaNVComplexConfig(
            nv_mode="equal_doublet",
        ),
    )

    # Write products
    files = qsospec.write_global_line_products(
        result,
        "outputs/my_object",
        qa_plot_config=qsospec.GlobalQAPlotConfig(
            show_host_context_in_overview=False,
        ),
    )
    for name, path in files.items():
        print(f"  {name}: {path}")

    # Access model grids
    import matplotlib.pyplot as plt
    plt.plot(result.continuum.wave_rest, result.continuum.model, label="Continuum")
    plt.plot(result.continuum.wave_rest,
             result.continuum.component_models["power_law"],
             "--", label="Power-law")
    plt.legend()


Global fitting with host decomposition
---------------------------------------

Requires ``qsospec[host]`` and a pPXF template NPZ file.

.. code-block:: python

    result = qsospec.fit_global_lines_workflow(
        "spectrum.fits",
        row_index=0,
        run_host_decomp=True,
        host_template_root="/path/to/templates/",
        host_template_file="templates.npz",
    )

    if result.host_decomp_enabled and result.host_sed is not None:
        print(f"Host fraction at 5100 Å: {result.host_sed.fraction_5100:.3f}")

    files = qsospec.write_global_line_products(
        result, "outputs/",
        qa_plot_config=qsospec.GlobalQAPlotConfig(
            show_host_context_in_overview=True,
        ),
    )


Batch fitting
-------------

.. code-block:: python

    # From Parquet files
    batch = qsospec.fit_batch(
        ["spectra-000.parquet"],
        "runs/sample",
        n_workers=4,
    )
    print(f"Completed: {batch.n_completed}, Failed: {batch.n_failed}")

    # From FITS files
    batch = qsospec.fit_batch(
        ["/data/sdss/*.fits"],
        "runs/sdss",
    )

    # Resume a run
    batch = qsospec.fit_batch(
        ["spectra-000.parquet"],
        "runs/sample",   # same directory, same config → resumes
    )


Inspecting a run
-----------------

.. code-block:: python

    run = qsospec.open_run("runs/sample")

    # Load a fitted model
    model = qsospec.load_model(run, "12345678901234567")
    print(model.summary())

    # Build a science catalog
    catalog = qsospec.build_science_catalog(
        run,
        {
            "log_l_hbeta_broad": {
                "section": "hbeta",
                "quantity": "log_l_hbeta_broad",
                "include_error": True,
            },
            "fwhm_hbeta_broad": {
                "section": "hbeta",
                "quantity": "fwhm_broad_kms",
                "include_error": True,
            },
            "log_l_5100": {
                "section": "continuum_sample",
                "quantity": "log_l_5100",
                "include_error": True,
            },
        },
    )
    print(catalog.head())

    # Render QA for selected objects
    qsospec.render_qa(
        run,
        warning_codes=["optional_line_fit_failed", "lya_fit_unreliable"],
        sample=10,
    )


Exploring recipes
-----------------

.. code-block:: python

    # List all built-in recipes
    for recipe in qsospec.recipes.list_complexes():
        info = qsospec.recipes.describe(recipe.id)
        print(f"{info['id']:12s} {info['label']}")

    # Inspect a recipe
    mgii = qsospec.recipes.get("mgii")
    for comp in mgii.components:
        print(f"  {comp.id}: {comp.role}, profile={comp.profile}, "
              f"lines={comp.line_ids}")

    # Build a custom narrow-line recipe
    custom = qsospec.recipes.generic_narrow_lines(
        line_ids=["neon_v_3426", "oiii_3133"],
        fit_window=(3000, 3600),
    )


Custom line complex
-------------------

.. code-block:: python

    from qsospec import ComponentRecipe, ComplexRecipe

    recipe = ComplexRecipe(
        id="my_complex",
        label="Custom complex",
        fit_window=(4500, 5100),
        components=[
            ComponentRecipe(
                id="broad_hb",
                line_ids=["hbeta_4863"],
                role="broad",
                profile="gaussian",
                multiplicity=2,
                fwhm_bounds_kms=(1000, 15000),
                velocity_bounds_kms=(-3000, 3000),
            ),
            ComponentRecipe(
                id="narrow_hb",
                line_ids=["hbeta_4863"],
                role="narrow",
                profile="gaussian",
                multiplicity=1,
                fwhm_bounds_kms=(50, 1500),
                velocity_bounds_kms=(-500, 500),
            ),
        ],
    )
