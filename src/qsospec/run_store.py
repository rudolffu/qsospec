"""Compatibility alias for :mod:`qsospec.io.run_store`."""

import sys

from .io import run_store as _implementation

sys.modules[__name__] = _implementation
