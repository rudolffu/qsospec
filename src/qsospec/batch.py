"""Compatibility alias for :mod:`qsospec.workflows.batch`."""

import sys

from .workflows import batch as _implementation

sys.modules[__name__] = _implementation
