Development guide
=====================

Environment
---------------

.. code-block:: bash

   git clone https://github.com/rudolffu/qsospec.git
   cd qsospec
   python -m pip install -e ".[dev,host,docs]"

Validation
--------------

Run tests and lint:

.. code-block:: bash

   pytest
   ruff check src tests

Run the offline examples and math checks, then build documentation:

.. code-block:: bash

   python -m pytest tests/test_docs_examples.py -m "not slow and not external_data and not plotting"
   python -m pytest tests/test_docs_math.py
   QSOSPEC_DOCS_OFFLINE=1 python -m sphinx -W --keep-going -b html docs docs/_build/html

Ordinary Python code blocks are not executed by an HTML build. Keep runnable
excerpts in the tested companion examples, using ``literalinclude`` where
appropriate. The real SDSS test is opt-in: set ``QSOSPEC_SDSS_EXAMPLE_FITS``
to the local tutorial file and configure Planck before running
``pytest tests/test_docs_examples.py -m external_data``.

Build distributions:

.. code-block:: bash

   python -m build
   python -m twine check dist/*

Design principles
---------------------

- Explicit spectrum, configuration, and result objects.
- Pure array models and residual functions inside optimizers.
- Variable projection for linear amplitudes and bounded nonlinear parameters.
- Analytic or semi-analytic derivatives with finite-difference validation.
- Plotting, storage, and host orchestration outside optimization loops.
- Scientific behavior changes accompanied by synthetic recovery and archive
  round-trip tests.

Documentation policy
------------------------

Each public workflow or configuration type has one canonical prose location.
Exact signatures/defaults belong in generated API pages. Core tutorials use
public data and give their input identity, assumptions, and output location.
Synthetic arrays belong in labeled tests. A zero-reddening override describes
a controlled test or a justified zero-extinction assumption.

Use ``$...$`` for inline math and ``$$...$$`` for display math in Markdown
and chat. In reStructuredText pages, use the ``:math:`` role and
``.. math::`` directive; dollar delimiters are not parsed as math there.
Check balanced LaTeX braces and inspect the rendered equations. A Sphinx
build can succeed even when MathJax cannot render an expression.

Scientific example comparisons
----------------------------------

The iron/Balmer examples compare model choices and resampling on synthetic
test inputs. A short bootstrap run checks execution; use more trials when
assessing interval tails:

.. code-block:: bash

   python examples/iron_balmer_uncertainty.py --output /tmp/iron-comparison.json
   python examples/iron_balmer_uncertainty.py --trials 20 --output /tmp/iron-bootstrap.json
   python -m pytest tests/test_qsospec_iron_templates.py tests/test_qsospec_global_workflow.py tests/test_qsospec_run_store.py tests/test_qsospec_host_workflow.py

For an archived real-spectrum comparison, extract the JSONL fields described
by the example scripts and write results to a new directory:

.. code-block:: bash

   python examples/compare_archived_iron_balmer.py --input /tmp/qsospec-three-spectra.jsonl --output-dir /tmp/new-real-comparison
   python examples/plot_iron_balmer_comparison.py --source /tmp/new-real-comparison --output /tmp/new-real-comparison-plots

Reader check
----------------

For a future session with a student unfamiliar with QSOSpec, ask them to:

1. Find and run the real SDSS example.
2. Explain its flux units and observed/rest-frame convention.
3. Predict which requested line lies outside its coverage.
4. Report broad Hβ FWHM with its available uncertainty, or explain a missing error.
5. State whether instrumental broadening is included.
6. Interpret a warning or missing measurement.
7. Reload the saved result and recover the same measurement.

This is a checklist for a future reader session. Executable example checks
and a documentation build assess different aspects of the tutorial.
