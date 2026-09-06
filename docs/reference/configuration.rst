Configuration reference
=======================

All configuration objects are immutable dataclasses. Construct a new object
for each scientific choice; run bundles hash the complete configuration.

Common presets
--------------

Default global fit:

.. code-block:: python

   global_config = qsospec.GlobalContinuumConfig()

Lyα-safe continuum windows:

.. code-block:: python

   global_config = qsospec.GlobalContinuumConfig.lya_safe()

Automatic single/broken power-law selection:

.. code-block:: python

   global_config = qsospec.GlobalContinuumConfig(
       power_law=qsospec.PowerLawConfig(mode="auto")
   )

The broken law is selected only with adequate wavelength leverage on both
sides of 4661 Å and a default BIC improvement of at least 10.

One full-range Verner et al. (2009) iron template:

.. code-block:: python

   global_config = qsospec.GlobalContinuumConfig.with_single_iron(
       "verner09"
   )

Pass ``IronTemplateConfig.verner09(fwhm_kms=4000)`` instead of the string to
customize its target width. This preset is exclusive: it disables the default
VW01 UV and Park22 optical components and fits one amplitude and one FWHM over
the Verner template's 2000–10000 Å support.

Optional additive polynomial (disabled by default for all surveys):

.. code-block:: python

   polynomial = qsospec.PolynomialContinuumConfig(
       enabled=None,  # explicitly opt into BIC-gated SDSS assessment
       degree=2,
       max_fraction=0.10,
       max_norm_fraction=0.10,
       auto_delta_bic=10.0,
   )
   config = qsospec.GlobalContinuumConfig(polynomial=polynomial)

The polynomial has no constant term and is additive to the global continuum.
The default ``enabled=False`` skips assessment, including for SDSS inputs.
Explicit ``enabled=None`` opts into assessment with ``survey="sdss"`` provenance;
survey metadata or a filename alone never activates it. The polynomial-free baseline slope is fixed,
and the candidate must pass the improvement score and numerical safeguards.
``enabled=True`` bypasses only the score requirement, not fractional bounds or
conditioning checks. ``enabled=False`` retains polynomial-free fitting.
See :doc:`../science/continuum_model` for the staged fit and covariance policy.

Known foreground E(B-V):

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(ebv_override=0.035)

Long-wavelength Galactic correction:

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(law="wang2019")

The Galactic-extinction provenance status is one of ``applied``,
``declared_corrected``, ``disabled``, or
``skipped_wavelength_out_of_range``. The last status means the configured
extinction law did not cover the observed wavelength grid and the configured
``wavelength_out_of_range`` policy allowed fitting to continue.

Continuum-only validation:

.. code-block:: python

   result = qsospec.fit_global_lines(
       spectrum,
       global_config=global_config,
       complexes=[],
   )

Configuration objects
---------------------

Exact fields, types, defaults, and validation are generated from the current
package:

- :class:`qsospec.GalacticExtinctionConfig`
- :class:`qsospec.GlobalContinuumConfig`
- :class:`qsospec.PowerLawConfig`
- :class:`qsospec.PolynomialContinuumConfig`
- :class:`qsospec.IronTemplateConfig`
- :class:`qsospec.BalmerPseudoContinuumConfig`
- :class:`qsospec.HbetaComplexConfig`
- :class:`qsospec.MgIIComplexConfig`
- :class:`qsospec.HalphaComplexConfig`
- :class:`qsospec.LyaNVComplexConfig`
- :class:`qsospec.UncertaintyConfig`
- :class:`qsospec.LocalFitConfig`
- :class:`qsospec.LineComplexConfig`
- :class:`qsospec.GlobalQAPlotConfig`
- :class:`qsospec.HostDecompConfig`
- :class:`qsospec.HostBroadLinePrefitConfig`
- :class:`qsospec.HostAgnPseudoContinuumConfig`
- :class:`qsospec.HostCoverageConfig`

The generated pages are grouped in :doc:`api/configuration`.

Selection precedence
--------------------

- Explicit caller configuration is preserved.
- If no global configuration is supplied and Lyα is fit-eligible, the
  Lyα-safe continuum preset is selected automatically.
- ``complexes=None`` means all covered auto-enabled recipes.
- ``complexes=[]`` means continuum only.
- Host decomposition and Galactic extinction are controlled independently.

See :doc:`../user_guide/preprocessing` for workflow order.

Host strategy preset
--------------------

The backward-compatible default is ``strategy="masked_simple"``. Enable the
AGN-aware masked pseudo-continuum basis explicitly:

.. code-block:: python

   host_config = qsospec.HostDecompConfig(
       strategy="agn_pseudocontinuum_masked"
   )

The nested prefit, pseudo-continuum, and coverage configurations are included
in run hashing. ``use_regularization=True`` and historical
``n_iterations != 1`` are rejected because they otherwise represented silent
no-op settings. See
:doc:`../how_to/agn_aware_ppxf_host_decomposition`.

Stellar-template profile preset
-------------------------------

The default resolves to ``emiles_native`` and
``spectra_emiles_9.0.npz``. Optional ``xsl_native`` and
``xsl_preconvolved`` profiles must be selected explicitly and preserve the
native science arrays. Profile/family/product conflicts are rejected. See
:doc:`../how_to/stellar_template_resolution_profiles`.
