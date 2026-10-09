Choose a stellar-template resolution profile
=============================================

The host workflow keeps native E-MILES as its default stellar population
library. Native XSL is an optional, higher-resolution alternative, and an
object-specific preconvolved XSL product can be used as an exact computational
cache. E-MILES and XSL are distinct SPS families: comparing them probes a
stellar-library systematic, while comparing native and preconvolved XSL is a
numerical-equivalence test.

Install the external template files from
`micappe/ppxf_data <https://github.com/micappe/ppxf_data>`__. They are not
bundled with qsospec.

Profiles
--------

``emiles_native``
   The default, using ``spectra_emiles_9.0.npz``. It retains E-MILES
   where the library is coarser than the data and records the mismatch.

``xsl_native``
   An explicit alternative using ``spectra_xsl_9.0.npz``. XSL is convolved at
   runtime toward each object's wavelength-dependent LSF where required.

``xsl_preconvolved``
   An externally generated fit-time product tied to one object, redshift,
   pPXF logarithmic grid, fit interval, native-XSL hash, and target LSF. The
   native source XSL file is still required to reconstruct the intrinsic
   template-grid HostSED.

``custom_native``
   A legacy or user-supplied native NPZ product. Its identity and available
   resolution metadata are retained explicitly.

One-sided resolution matching
------------------------------

qsospec preserves the input wavelength, flux, error/inverse variance, mask,
sampling, and object-specific LSF. It matches the template resolution to the
data using

.. math::

   \sigma_\mathrm{add}(\lambda) =
   \sqrt{\max\left[0,
   \sigma_\mathrm{data}^2(\lambda)-
   \sigma_\mathrm{template}^2(\lambda)\right]}.

Templates sharper than the data receive this additional convolution. A
coarser template is left unchanged, and the pixel remains in the pPXF fit.
The mismatch is recorded in the resolution diagnostics, and these pixels
are included in the fit. Host continuum and host fractions can remain usable,
while narrow absorption features, stellar kinematics, or population estimates
may be limited by the template resolution (``template_resolution_limited``). No
``native_data_sigma_floor`` or
deconvolution is applied.

Configuration
-------------

The default configuration uses E-MILES:

.. code-block:: python

   host_config = qsospec.HostDecompConfig()

Select native XSL explicitly:

.. code-block:: python

   host_config = qsospec.HostDecompConfig(
       template_profile="xsl_native",
       template_family="xsl",
       template_file="spectra_xsl_9.0.npz",
   )

Use an exact preconvolved product:

.. code-block:: python

   host_config = qsospec.HostDecompConfig(
       template_profile="xsl_preconvolved",
       template_family="xsl",
       template_product_kind="preconvolved",
       template_file="xsl_preconvolved_<cache-key>.npz",
       source_template_file="spectra_xsl_9.0.npz",
   )

Profile, family, product kind, and canonical filenames must agree. Setting
``preserve_native_data=False`` is rejected.

Build an exact XSL product
--------------------------

A preconvolved XSL library is specific to the object and its
wavelength-dependent LSF. Choose a cache directory when building it:

.. code-block:: bash

   PYTHONPATH=src python scripts/preconvolve_xsl_for_host_fit.py \
       --spectrum /path/to/input.parquet \
       --row-index 0 \
       --object-key survey:targetid \
       --redshift 0.5 \
       --template-root "$PPXF_TEMPLATE_ROOT" \
       --cache-root "$QSOSPEC_TEMPLATE_CACHE"

The builder hashes the native-XSL source, source ordering, target object,
redshift, fit range, exact pPXF grid, rest-frame target sigma, valid-resolution
mask, algorithm, and normalization convention. It writes atomically and
validates the product after reading it back. Products for another object, LSF,
grid, source hash, or template order are rejected.
Automatic cache generation is not enabled by default.

HostSED and final fitting
-------------------------

Fit-time and source-template matrices are separate. pPXF stellar weights are
applied to the native source SSP matrix, together with the fitting-time scales,
to reconstruct the broad optical/NIR HostSED used by Euclid host transfer. The
intrinsic SSP SED is reconstructed from the native source library.

After pPXF, qsospec subtracts only the stellar model evaluated on the native
input grid. The AGN components remain in the spectrum. The final continuum and
line fit uses the
native host-subtracted flux with the original errors, mask, sampling, and LSF.

Inspect diagnostics
-------------------

The host quality metadata includes the resolved profile and hashes, fractions
where the template is sharper/equal/coarser, mismatch amplitudes in Angstrom
and km/s, one-sided convolution fraction, preconvolution validation status,
and separate science assessments:

.. code-block:: python

   quality = result.metadata["host_fit_quality"]
   print(quality["stellar_template_profile"])
   print(quality["template_resolution_status"])
   print(quality["template_coarser_than_data_fraction_goodpixels"])
   print(result.metadata["host_continuum_reliable"])
   print(result.metadata["stellar_kinematics_resolution_status"])

The pPXF QA panel shows resolution diagnostics and pixel masks separately.
Coarser-template regions are included in the fit and are shown as fitted pixels.

Current limitations
-------------------

Exact preconvolution currently requires a sigma-like object-specific LSF;
banded resolution matrices are not supported. A missing data LSF is a
separate reliability limitation. Host reliability also depends on S/N,
coverage, AGN fraction, clipping, parameter bounds, and repeatability. E-MILES
is the default. Compare E-MILES and XSL on representative spectra before
adopting a different template profile.

