qsospec
=======

|Release|
|License| |Python|

.. |Release| image:: https://img.shields.io/badge/release-v0.1.0-blue
   :target: https://pypi.org/project/qsospec/
.. |License| image:: https://img.shields.io/badge/license-GPLv3-green
   :target: https://github.com/rudolffu/qsospec/blob/main/LICENSE
.. |Python| image:: https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue
   :target: https://pypi.org/project/qsospec/

:Repository: `github.com/rudolffu/qsospec <https://github.com/rudolffu/qsospec>`_

A Python package for fitting UV, optical, and near-infrared quasar spectra.
Provides array-based local and global fitting, recipe-driven emission
complexes, bundled iron and Balmer templates, optional pPXF host
subtraction, and resumable Parquet batch runs.

Installation
------------

Install from PyPI:

.. code-block:: bash

   python -m pip install qsospec

With optional pPXF host decomposition:

.. code-block:: bash

   python -m pip install "qsospec[host]"

Install from source (development):

.. code-block:: bash

   git clone https://github.com/rudolffu/qsospec.git
   cd qsospec
   python -m pip install -e ".[dev,host,docs]"

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/configuration
   user_guide/workflows
   user_guide/results_warnings

.. toctree::
   :maxdepth: 2
   :caption: Reference

   recipe_reference
   scientific_definitions
   api

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples

.. toctree::
   :maxdepth: 1
   :caption: Development

   run_bundles
   development_plan
