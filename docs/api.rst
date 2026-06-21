API Reference
=============

This page provides auto-generated API documentation for the main
public classes and functions.

.. currentmodule:: qsospec

Spectrum and metadata
---------------------

.. autosummary::
   :toctree: _generated

   Spectrum
   SpectrumMetadata

.. autofunction:: qsospec.resolve_spectrum_metadata


Configuration
-------------

.. autosummary::
   :toctree: _generated

   GlobalContinuumConfig
   GalacticExtinctionConfig
   PowerLawConfig
   BalmerPseudoContinuumConfig
   IronTemplateConfig
   HbetaComplexConfig
   HalphaComplexConfig
   MgIIComplexConfig
   LyaNVComplexConfig
   UncertaintyConfig
   LocalFitConfig
   LineComplexConfig
   GaussianComponent
   LorentzianComponent
   GlobalQAPlotConfig


Result objects
--------------

.. autosummary::
   :toctree: _generated

   FitResult
   LocalFitResult
   GlobalContinuumResult
   EmissionComplexResult
   WorkflowResult
   FitWarning
   BatchResult
   HostWorkflowResult


Fitting functions
-----------------

.. autofunction:: qsospec.fit_local
.. autofunction:: qsospec.fit_line_complex
.. autofunction:: qsospec.fit_global_continuum
.. autofunction:: qsospec.fit_hbeta_complex
.. autofunction:: qsospec.fit_mgii_complex
.. autofunction:: qsospec.fit_halpha_complex
.. autofunction:: qsospec.fit_global_lines
.. autofunction:: qsospec.fit_global_hbeta
.. autofunction:: qsospec.fit_global_lines_workflow
.. autofunction:: qsospec.fit_global_hbeta_workflow
.. autofunction:: qsospec.fit_with_optional_host_decomp
.. autofunction:: qsospec.fit_batch
.. autofunction:: qsospec.fit_object_to_store


I/O functions
-------------

.. autofunction:: qsospec.query_galactic_ebv
.. autofunction:: qsospec.f99_dereddening_factor
.. autofunction:: qsospec.correct_spectrum
.. autofunction:: qsospec.correct_spectrum_data
.. autofunction:: qsospec.preflight_galactic_extinction
.. autofunction:: qsospec.read_spectrum
.. autofunction:: qsospec.discover_fits_inputs
.. autofunction:: qsospec.read_input_manifest
.. autofunction:: qsospec.scan_parquet_spectra
.. autofunction:: qsospec.detect_fits_reader
.. autofunction:: qsospec.write_global_line_products
.. autofunction:: qsospec.write_global_hbeta_products
.. autofunction:: qsospec.open_run
.. autofunction:: qsospec.load_model
.. autofunction:: qsospec.finalize_run
.. autofunction:: qsospec.build_science_catalog
.. autofunction:: qsospec.compute_derived_quantities
.. autofunction:: qsospec.render_qa


Template functions
------------------

.. autofunction:: qsospec.list_iron_templates
.. autofunction:: qsospec.load_iron_template
.. autofunction:: qsospec.list_balmer_templates
.. autofunction:: qsospec.load_balmer_template


Recipe and line registries
---------------------------

.. autofunction:: qsospec.recipes.list_complexes
.. autofunction:: qsospec.recipes.get
.. autofunction:: qsospec.recipes.resolve
.. autofunction:: qsospec.recipes.describe
.. autofunction:: qsospec.recipes.generic_narrow_lines
.. autofunction:: qsospec.lines.list
.. autofunction:: qsospec.lines.get
.. autofunction:: qsospec.lines.resolve


Plotting
--------

.. autofunction:: qsospec.plot_line_result
.. autofunction:: qsospec.plot_local_result
.. autofunction:: qsospec.save_local_window_plots


Models
------

.. autofunction:: qsospec.models.gaussian
.. autofunction:: qsospec.models.gaussian_partials
.. autofunction:: qsospec.models.lorentzian
.. autofunction:: qsospec.models.lorentzian_partials
.. autofunction:: qsospec.models.continuum
.. autofunction:: qsospec.models.continuum_partials


Solvers
-------

.. autofunction:: qsospec.solvers.solve_variable_projection
.. autofunction:: qsospec.solvers.run_least_squares
