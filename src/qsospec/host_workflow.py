"""Compatibility alias for :mod:`qsospec.workflows.host_workflow`."""

import sys

from .workflows import host_workflow as _implementation

sys.modules[__name__] = _implementation
