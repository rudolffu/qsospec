"""Compatibility alias for :mod:`qsospec.fitting.global_fit`."""

import sys

from .fitting import global_fit as _implementation

sys.modules[__name__] = _implementation
