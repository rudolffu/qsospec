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
