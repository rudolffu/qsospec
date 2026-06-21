qsospec
=======

Fit UV, optical, and near-infrared quasar spectra with reproducible continuum,
emission-line, host-decomposition, QA, and batch workflows.

.. grid:: 2 2 4 4
   :gutter: 2

   .. grid-item-card:: Start fitting
      :link: getting_started/quickstart
      :link-type: doc

      Install |project_name| and fit a spectrum with the default global model.

   .. grid-item-card:: Choose a workflow
      :link: getting_started/choose_workflow
      :link-type: doc

      Compare array, file, host-decomposition, and batch interfaces.

   .. grid-item-card:: Interpret QA
      :link: user_guide/qa_plots
      :link-type: doc

      Read fitted, masked, unmodelled, and residual regions correctly.

   .. grid-item-card:: Browse recipes
      :link: reference/recipes
      :link-type: doc

      See the built-in line complexes and their current coverage rules.

Minimal array example
---------------------

Array APIs treat the input spectrum as already corrected for Galactic
extinction.

.. code-block:: python

   import qsospec

   spectrum = qsospec.Spectrum.from_arrays(
       wavelength,
       flux,
       err=uncertainty,
       z=redshift,
       wave_frame="observed",
   )
   result = qsospec.fit_global_lines(spectrum)

   print(result.continuum_success)
   print(result.complex_statuses)

What is included
----------------

- Global power-law, Fe II, and continuous Balmer pseudo-continuum fitting.
- Coverage-aware Lyα/N V, C IV, C III], Mg II, Balmer, optical, and NIR recipes.
- Optional pPXF host decomposition for objects with :math:`z < 1.2`.
- Galactic dereddening for file and batch workflows.
- QA plots that distinguish fitted, pPXF-masked, and unmodelled regions.
- Resumable Parquet run bundles for single objects and large samples.

.. toctree::
   :maxdepth: 2
   :caption: Start Here

   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: How-to Guides

   how_to/index

.. toctree::
   :maxdepth: 2
   :caption: Science Model

   science/index

.. toctree::
   :maxdepth: 3
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/index

Project links
-------------

`GitHub <https://github.com/rudolffu/qsospec>`__ ·
`PyPI <https://pypi.org/project/qsospec/>`__ ·
`Issues <https://github.com/rudolffu/qsospec/issues>`__ ·
`GPLv3 license <https://github.com/rudolffu/qsospec/blob/main/LICENSE>`__
