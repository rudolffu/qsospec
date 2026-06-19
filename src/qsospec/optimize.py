"""Compatibility alias for :mod:`qsospec.solvers.least_squares`."""

import sys

from .solvers import least_squares as _implementation

sys.modules[__name__] = _implementation
