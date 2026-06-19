"""Compatibility alias for :mod:`qsospec.solvers.variable_projection`."""

import sys

from .solvers import variable_projection as _implementation

sys.modules[__name__] = _implementation
