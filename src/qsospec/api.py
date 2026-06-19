"""Compatibility alias for :mod:`qsospec.fitting.local`."""

import sys

from .fitting import local as _implementation

sys.modules[__name__] = _implementation
