Choose a workflow
=================

.. list-table::
   :header-rows: 1
   :widths: 24 25 51

   * - Interface
     - Input
     - Behavior and intended use
   * - ``fit_global_lines``
     - ``Spectrum``
     - Caller-preprocessed; no host; in-memory result. Best for developing one
       global model.
   * - ``fit_local``
     - ``Spectrum``
     - Caller-preprocessed; no host; in-memory result. Best for targeted,
       independent windows.
   * - ``fit_global_lines_workflow``
     - Spectrum file/row
     - Galactic correction; optional host; in-memory result. Best for one
       file-based object.
   * - ``fit_object_to_store``
     - File, ``SpectrumData``, or ``Spectrum``
     - Galactic correction except caller-preprocessed ``Spectrum``; optional
       host; run bundle and QA. Best for reproducible single-object work.
   * - ``fit_batch``
     - Parquet/FITS sample
     - Galactic correction; optional host; resumable run bundle. Best for
       surveys and production samples.

Recommended path
----------------

- Start with :func:`qsospec.fit_global_lines` while developing a model.
- Use :func:`qsospec.fit_object_to_store` when outputs and QA must be archived.
- Use :func:`qsospec.fit_batch` after the single-object configuration is
  scientifically validated.
- Use local fitting only when independent windows are intentional.

The complete workflow descriptions are in :doc:`../user_guide/index`.
