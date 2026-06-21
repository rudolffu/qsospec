Development guide
=================

Environment
-----------

.. code-block:: bash

   git clone https://github.com/rudolffu/qsospec.git
   cd qsospec
   python -m pip install -e ".[dev,host,docs]"

Validation
----------

Run tests and lint:

.. code-block:: bash

   pytest
   ruff check src tests

Build documentation:

.. code-block:: bash

   python -m sphinx -W -b html docs docs/_build/html
   python -m sphinx -W -b doctest docs docs/_build/doctest

Build distributions:

.. code-block:: bash

   python -m build
   python -m twine check dist/*

Design principles
-----------------

- Explicit spectrum, configuration, and result objects.
- Pure array models and residual functions inside optimizers.
- Variable projection for linear amplitudes and bounded nonlinear parameters.
- Analytic or semi-analytic derivatives with finite-difference validation.
- Plotting, storage, and host orchestration outside optimization loops.
- Scientific behavior changes accompanied by synthetic recovery and archive
  round-trip tests.

Documentation policy
--------------------

Each public workflow or configuration type has one canonical prose location.
Exact signatures/defaults belong in generated API pages. Examples must run
without private data, using synthetic arrays or explicit ``ebv_override=0``.
