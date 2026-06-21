Host decomposition
==================

Install ``qsospec[host]`` and provide a local pPXF E-MILES template bundle:

.. code-block:: python

   result = qsospec.fit_global_lines_workflow(
       "spectrum.fits",
       run_host_decomp=True,
       template_root="/path/to/ppxf_data",
       template_file="spectra_emiles_9.0.npz",
   )

Object-level gate
-----------------

``run_host_decomp=True`` is a request. pPXF runs only for finite
``redshift < 1.2``. At higher or missing redshift, fitting continues without
host subtraction and records ``host_decomp_skip_reason``.

Host fitting masks emission-line regions before fitting the stellar
continuum. Later successful |project_name| line fits take precedence in QA,
so pPXF masking does not imply that the line was omitted from the final
spectral model.

Results
-------

Inspect:

- ``result.host_decomp_enabled``
- ``result.host_fit`` and ``result.host_sed``
- ``result.host_model_on_quasar_grid``
- ``result.host_fit_mask`` and ``result.host_emission_mask``
- ``result.metadata["host_decomp_skip_reason"]``

Host fractions are shown only where the rest-frame data constrain the
requested wavelength. Host-refit Monte Carlo is available through
``UncertaintyConfig(refit_host_in_mc=True)`` and runs only when host
decomposition was enabled.
