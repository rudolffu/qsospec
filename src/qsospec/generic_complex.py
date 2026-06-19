"""Compatibility alias for :mod:`qsospec.fitting.complexes`."""

import sys

from .fitting import complexes as _implementation

sys.modules[__name__] = _implementation
