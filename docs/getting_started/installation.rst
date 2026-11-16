Install qsospec
===============

The |release| source tree requires Python 3.10 or later. Use the Python
installation in which you run your notebooks or analysis scripts; the commands
below install into that interpreter.

Choose a package version
------------------------

For the published PyPI package:

.. code-block:: bash

   python -m pip install qsospec

These pages follow the source tree, which can be ahead of the PyPI release.
For the APIs and defaults documented here, install this checkout:

.. code-block:: bash

   git clone https://github.com/rudolffu/qsospec.git
   cd qsospec
   python -m pip install -e .

If you already cloned the repository, run only the last command from its root.
The editable installation follows your source edits. Rerun it after package
version or dependency changes to refresh installed metadata.

Check the active installation
-----------------------------

.. code-block:: bash

   python -c "import sys, qsospec; print(sys.executable); print(qsospec.__version__); print(qsospec.__file__)"

Check that the version and source path match your intended installation. In a
notebook, run the same imports and ``print`` calls in a cell. If its interpreter
differs from your terminal, install using that notebook's interpreter and
restart its kernel.

Data needed for your first fit
------------------------------

Fe II and Balmer templates are bundled. The standard foreground correction
needs a separate :doc:`Planck GNILC dust-map download <dustmaps>`. Download it
once before the :doc:`quickstart`; the map is reused across objects.

The quickstart downloads one public SDSS spectrum and disables host fitting.
After the input and map are on disk, fitting works without downloading them
again.

Optional host-galaxy fitting
----------------------------

Install pPXF support with:

.. code-block:: bash

   python -m pip install "qsospec[host]"

For a source checkout, use ``python -m pip install -e ".[host]"`` instead.
Stellar templates are external to qsospec. The default host configuration uses
``spectra_emiles_9.0.npz`` from
`micappe/ppxf_data <https://github.com/micappe/ppxf_data>`__.
Follow :doc:`../how_to/fit_with_host` to choose the file and pass its directory.

Documentation and development
-----------------------------

From a checkout, add the tools you need:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Use ``.[dev,docs]`` for tests and lint, or ``.[dev,host,docs]`` when developing
host fitting too. The :doc:`../contributing/development` guide gives the build
and validation commands. Documentation uses the
`Furo theme <https://pradyunsg.me/furo/>`__.

Next steps
----------

Configure :doc:`dustmaps`, then fit the :doc:`SDSS example <quickstart>`.
For installation or import problems, see :doc:`troubleshooting`.
