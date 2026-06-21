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

Known foreground E(B-V):

.. code-block:: python

   extinction = qsospec.GalacticExtinctionConfig(ebv_override=0.035)

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
