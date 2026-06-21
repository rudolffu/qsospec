Fit with host decomposition
===========================

Prerequisites
-------------

Install ``qsospec[host]``, configure dust maps, and obtain the expected pPXF
template NPZ bundle.

.. code-block:: python

   result = qsospec.fit_global_lines_workflow(
       "spectrum.fits",
       run_host_decomp=True,
       template_root="/path/to/ppxf_data",
       template_file="spectra_emiles_9.0.npz",
   )

Expected outputs
----------------

When ``redshift < 1.2``, ``host_decomp_enabled`` is true and the result
contains the pPXF fit, host SED, host model on the quasar grid, and host masks.

Common failures
---------------

- ``host_decomp_enabled`` is false: inspect ``host_decomp_skip_reason``.
- Template file missing: check ``template_root`` and ``template_file``.
- Too little optical coverage: widen the input coverage or skip host fitting.

Next: :doc:`../user_guide/host_decomposition`.
