"""Compatibility alias for :mod:`qsospec.io.qa`."""

import sys

from .io import qa as _implementation

sys.modules[__name__] = _implementation
