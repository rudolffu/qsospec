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

Configure Galactic dust maps
----------------------------

File-based and batch workflows apply Galactic dereddening by default using
the Planck Collaboration (2016) GNILC map.  Configure the external
``dustmaps`` data directory and fetch the maps after installation:

.. code-block:: python

   from dustmaps.config import config

   config["data_dir"] = "/path/to/dustmaps"

   from dustmaps import planck, sfd
   planck.fetch(which="GNILC")
   sfd.fetch()

The directory will contain ``planck/`` and ``sfd/`` subdirectories.  Missing
coordinates or map files raise an error while correction is enabled.  See
:doc:`user_guide/configuration` for map selection, disabling the correction,
and supplying an explicit E(B-V).

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

.. toctree::
   :maxdepth: 1
   :caption: Background

   references
