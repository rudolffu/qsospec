Installation
================

Install the core package from PyPI:

.. code-block:: bash

   python -m pip install qsospec

Python 3.10 or later is required. Check the installed version with:

.. code-block:: bash

   python -c "import qsospec; print(qsospec.__version__)"

The SDSS :doc:`quickstart` needs the core package and the Planck GNILC dust
map. The spectrum is downloaded once and cached. Fe II and Balmer templates
are included with the package; host stellar templates are needed only when
you enable host decomposition.

Optional host fitting
-------------------------

Install optional pPXF host decomposition:

.. code-block:: bash

   python -m pip install "qsospec[host]"

Download the E-MILES NPZ bundle from
`micappe/ppxf_data <https://github.com/micappe/ppxf_data>`__ and pass its local
directory through ``template_root``. See :doc:`../how_to/fit_with_host`.

Development installation
----------------------------

For a source checkout with tests and documentation tools:

.. code-block:: bash

   git clone https://github.com/rudolffu/qsospec.git
   cd qsospec
   python -m pip install -e ".[dev,host,docs]"

Next steps
--------------

- Configure the default foreground correction in :doc:`dustmaps`.
- Fit a real SDSS quasar spectrum in :doc:`quickstart`.
- See :doc:`choose_workflow` before processing files or samples.
