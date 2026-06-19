"""Compatibility alias for :mod:`qsospec.io.products`."""

import sys

from .io import products as _implementation

sys.modules[__name__] = _implementation
