"""Structured warnings for qsospec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class FitWarning:
    """Stable warning/status message for science-facing qsospec results."""

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
