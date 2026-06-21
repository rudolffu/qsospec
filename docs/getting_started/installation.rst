Installation
============

Install the core package from PyPI:

.. code-block:: bash

   python -m pip install qsospec

Install optional pPXF host decomposition:

.. code-block:: bash

   python -m pip install "qsospec[host]"

For development and documentation:

.. code-block:: bash

   git clone https://github.com/rudolffu/qsospec.git
   cd qsospec
   python -m pip install -e ".[dev,host,docs]"

|project_name| supports Python |python_versions|. Scientific templates for
Fe II and the Balmer pseudo-continuum are included in the package. Galactic
dust maps and pPXF stellar templates are external data and must be configured
separately.

Next steps
----------

- Configure the default foreground correction in :doc:`dustmaps`.
- Fit an in-memory spectrum in :doc:`quickstart`.
- See :doc:`choose_workflow` before processing files or samples.
