"""Structured warnings for neofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import warnings as _warnings


@dataclass(frozen=True)
class FitWarning:
    """Stable warning/status message for science-facing neofit results."""

    code: str
    message: str
    severity: str = "warning"
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "context": dict(self.context),
        }


def __getattr__(name: str):
    if name == "NeoFitWarning":
        _warnings.warn(
            "NeoFitWarning is deprecated; use FitWarning.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FitWarning
    raise AttributeError(name)
