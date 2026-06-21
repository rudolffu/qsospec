Quick start
===========

This example fits a synthetic in-memory spectrum. It does not need external
dust maps because array APIs treat the input as already preprocessed.

.. doctest::

   >>> import numpy as np
   >>> import qsospec
   >>> wave = np.linspace(3500.0, 5500.0, 800)
   >>> flux = 2.0 * (wave / 4000.0) ** -1.2
   >>> err = np.full_like(wave, 0.05)
   >>> spectrum = qsospec.Spectrum.from_arrays(
   ...     wave, flux, err=err, z=0.0, wave_frame="observed"
   ... )
   >>> config = qsospec.GlobalContinuumConfig(
   ...     uv_iron=None,
   ...     optical_iron=None,
   ...     balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
   ...         enabled=False
   ...     ),
   ...     clip_passes=0,
   ... )
   >>> result = qsospec.fit_global_lines(
   ...     spectrum, global_config=config, complexes=[]
   ... )
   >>> result.continuum_success
   True

For ordinary science fitting, omit the simplified configuration and let
``fit_global_lines`` select covered recipes:

.. code-block:: python

   result = qsospec.fit_global_lines(spectrum)

   for recipe_id, status in result.complex_statuses.items():
       print(recipe_id, status)

Inspect ``result.continuum``, ``result.line_complexes``, ``result.warnings``,
and ``result.metadata``. See :doc:`../user_guide/results` and
:doc:`../user_guide/qa_plots`.
