# Contributing to qsospec

Use [GitHub issues](https://github.com/rudolffu/qsospec/issues) to report a bug,
ask about a measurement, or discuss a proposed change. Include the package
version, a small reproducible example, and relevant configuration and warnings.
For scientific changes, describe the expected effect on fitted measurements.

From a source checkout, install development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

Add the `host` extra when working on pPXF. The
[development guide](https://qsospec.readthedocs.io/en/latest/contributing/development.html)
contains the test commands, documentation build, external-data checks and
scientific validation policy.

Keep user-facing examples aligned with the public API. Describe input units,
frames, prerequisites and expected outputs; link to the canonical science
or reference page for definitions. Update the changelog when behavior changes.
