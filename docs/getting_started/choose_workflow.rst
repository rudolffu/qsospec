Choose a workflow
=====================

Start with the task you want to complete:

- **Fit your first real spectrum:** use :func:`qsospec.fit_global_lines_workflow`
  as in the :doc:`quickstart`. It reads a file, prepares the spectrum, and
  returns the fit in memory.
- **Experiment with a prepared spectrum:** use :func:`qsospec.fit_global_lines`.
  This is useful when changing models or inspecting intermediate arrays.
- **Keep a single-object analysis for later:** use
  :func:`qsospec.fit_object_to_store`. It fits the object and writes a run
  bundle that can be reloaded; see :doc:`../user_guide/results`.
- **Process a sample:** use :func:`qsospec.fit_batch` with a configuration
  assessed on representative objects. It writes a resumable run bundle.
- **Fit independent wavelength windows:** use :func:`qsospec.fit_local` when
  you want separate local continua and line models.

Interfaces and inputs
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Interface
     - Input
     - Preparation and output
   * - ``fit_global_lines_workflow``
     - Spectrum file/row
     - Galactic correction and frame conversion; optional host fit;
       in-memory result.
   * - ``fit_global_lines``
     - Prepared ``Spectrum``
     - Fits continuum and lines; in-memory result. Call ``prepare_spectrum``
       first for observed-frame arrays.
   * - ``fit_object_to_store``
     - File, ``SpectrumData``, or ``Spectrum``
     - Prepares the input, fits it, and saves a run bundle and optional QA.
   * - ``fit_batch``
     - Parquet/FITS sample
     - Prepares each object and writes a resumable run bundle.
   * - ``fit_local``
     - Prepared ``Spectrum``
     - Independent window fits with local continua; in-memory result.

For array inputs, :doc:`../how_to/fit_arrays` names the variables you supply
and shows preparation. For input units and frames, see :doc:`spectral_basics`.
The complete workflow descriptions are in :doc:`../user_guide/index`.
