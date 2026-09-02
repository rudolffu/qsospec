Host decomposition
==================

Install ``qsospec[host]`` and provide a local pPXF E-MILES template bundle
from `micappe/ppxf_data <https://github.com/micappe/ppxf_data>`__:

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

Host strategies
---------------

``HostDecompConfig(strategy="masked_simple")`` preserves the historical
masked pPXF model and is the default. The explicit
``agn_pseudocontinuum_masked`` strategy adds width-matched Fe II and Balmer
pseudo-continuum templates while determining the stellar weights, then
subtracts only the stellar model. See
:doc:`../how_to/agn_aware_ppxf_host_decomposition` for configuration,
provenance, coverage classes, fractions, and limitations.

Stellar-template resolution profiles
------------------------------------

E-MILES remains the default. Native XSL and exact object-specific preconvolved
XSL are optional profiles. All preserve the native input spectrum, and a
stellar template that is coarser than the data is retained with diagnostics
rather than excluded. See
:doc:`../how_to/stellar_template_resolution_profiles`.
