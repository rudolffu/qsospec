"""Compatibility alias for :mod:`qsospec.io.readers`."""

import sys

from .io import readers as _implementation

sys.modules[__name__] = _implementation
