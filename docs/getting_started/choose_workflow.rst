Choose a fitting workflow
=========================

Use a high-level workflow to read a file, apply foreground correction and
prepare its rest-frame flux density. Use an array fitter when you want control
of preparation and intermediate models.

.. list-table:: Start from your task
   :header-rows: 1
   :widths: 28 35 37

   * - I want to…
     - Use
     - Follow
   * - Fit a file and inspect it now
     - ``fit_global_lines_workflow``
     - :doc:`quickstart`
   * - Save a single-object analysis
     - ``fit_object_to_store``
     - :doc:`../how_to/fit_file`
   * - Fit my own arrays
     - ``Spectrum.from_arrays``, ``prepare_spectrum``, ``fit_global_lines``
     - :doc:`../how_to/fit_arrays`
   * - Fit several independent line windows
     - ``fit_local``
     - :doc:`../user_guide/local_fitting`
   * - Process a sample
     - ``fit_batch``
     - :doc:`../user_guide/batch_fitting`
   * - Inspect a saved fit without running it again
     - ``open_run`` and ``load_model``
     - :doc:`../how_to/inspect_run`

Global or local continuum?
--------------------------

:doc:`Global fitting <../user_guide/global_fitting>` shares one AGN continuum
across the spectrum, then fits the covered line complexes. Start here for a
multi-line analysis. An emission complex is a group of neighboring lines
fitted together, such as Hβ and [O III].

:doc:`Local fitting <../user_guide/local_fitting>` fits a separate continuum
and line model in each selected window. Use it for a targeted line or when
a shared continuum is unsuitable. Measurements can differ because the
continuum definitions differ.

Save while fitting
------------------

Choose ``fit_object_to_store`` before starting if you want to reload the
analysis later. It returns the result in memory and writes the same run-bundle
format as ``fit_batch``. ``fit_global_lines_workflow`` and the array fitters
return an in-memory result; they do not create a run bundle.

The :doc:`results tutorial <../user_guide/results>` shows both routes. A run
bundle contains the measurements, fitted parameters, masks and assets needed
to reload the model without refitting.

Prepare arrays explicitly
-------------------------

``Spectrum.from_arrays`` declares units and wavelength/flux frames. Call
``prepare_spectrum`` before the low-level ``fit_global_lines``,
``fit_global_continuum`` or ``fit_local`` functions for ordinary observed-frame
arrays. The :doc:`preprocessing guide <../user_guide/preprocessing>` explains
the foreground correction and rest-frame conversion.

Add host fitting when needed
----------------------------

High-level file and saved workflows accept ``run_host_decomp=True``. This
requires pPXF and external stellar templates, and is intended for objects
with :math:`z < 1.2`. Start with :doc:`../how_to/fit_with_host`; inspect host
coverage and reliability before using host-derived measurements.
