"""Spectrum metadata and strict flux-unit handling for qsospec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


CGS_FLAMBDA_UNIT = "erg s^-1 cm^-2 Angstrom^-1"
_SURVEY_ALIASES = {
    "desi": "desi",
    "desidr1": "desi",
    "desi-dr1": "desi",
    "desi_dr1": "desi",
    "desiedr": "desi",
    "desi-edr": "desi",
    "desi_edr": "desi",
    "sdss": "sdss",
}
@dataclass
class SpectrumMetadata:
    """Wavelength and flux-density metadata kept outside numerical fitting."""

    wave_unit: str = "Angstrom"
    flux_unit: str = "relative"
    flux_scale: Optional[float] = None
    survey: Optional[str] = None
    source: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    galactic_extinction_corrected: bool = False
    galactic_extinction: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return {
            "wave_unit": self.wave_unit,
            "flux_unit": self.flux_unit,
            "flux_scale": self.flux_scale,
            "survey": self.survey,
            "source": self.source,
            "ra": self.ra,
            "dec": self.dec,
            "galactic_extinction_corrected": bool(
                self.galactic_extinction_corrected
            ),
            "galactic_extinction": dict(self.galactic_extinction),
            "notes": list(self.notes),
        }


def _normalize_survey(survey: Optional[str]) -> Optional[str]:
    if survey is None:
        return None
    key = str(survey).strip().lower().replace(" ", "").replace(".", "")
    if key not in _SURVEY_ALIASES:
        raise ValueError(f"Unknown qsospec survey preset: {survey!r}")
    return _SURVEY_ALIASES[key]


def _legacy_flux_fields(metadata: Mapping[str, Any]) -> tuple[str, Optional[float]]:
    """Translate schema-v1--v3 flux metadata."""

    scale = metadata.get("flux_density_scale_to_cgs")
    label = str(metadata.get("flux_density_unit", "input")).lower()
    if scale is not None:
        return "cgs", float(scale)
    if "erg" in label and "cm" in label:
        return "cgs", 1.0
    return "relative", None


def _metadata_from_base(metadata: Optional[Any]) -> SpectrumMetadata:
    if metadata is None:
        return SpectrumMetadata()
    if isinstance(metadata, SpectrumMetadata):
        return SpectrumMetadata(
            wave_unit=metadata.wave_unit,
            flux_unit=metadata.flux_unit,
            flux_scale=metadata.flux_scale,
            survey=metadata.survey,
            source=metadata.source,
            ra=metadata.ra,
            dec=metadata.dec,
            galactic_extinction_corrected=bool(
                metadata.galactic_extinction_corrected
            ),
            galactic_extinction=dict(metadata.galactic_extinction),
            notes=list(metadata.notes),
        )
    if isinstance(metadata, Mapping):
        flux_unit = metadata.get("flux_unit")
        flux_scale = metadata.get("flux_scale")
        if flux_unit is None:
            flux_unit, flux_scale = _legacy_flux_fields(metadata)
        return SpectrumMetadata(
            wave_unit=str(metadata.get("wave_unit", "Angstrom")),
            flux_unit=str(flux_unit),
            flux_scale=flux_scale,
            survey=metadata.get("survey"),
            source=metadata.get("source"),
            ra=metadata.get("ra"),
            dec=metadata.get("dec"),
            galactic_extinction_corrected=bool(
                metadata.get("galactic_extinction_corrected", False)
            ),
            galactic_extinction=dict(
                metadata.get("galactic_extinction", {})
            ),
            notes=list(metadata.get("notes", [])),
        )
    raise TypeError("metadata must be a SpectrumMetadata, mapping, or None.")


def resolve_spectrum_metadata(
    *,
    survey: Optional[str] = None,
    wave_unit: Optional[str] = None,
    flux_unit: Optional[str] = None,
    flux_scale: Optional[float] = None,
    source: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    galactic_extinction_corrected: Optional[bool] = None,
    galactic_extinction: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Any] = None,
) -> SpectrumMetadata:
    """Resolve metadata with explicit keywords taking highest priority."""

    resolved = _metadata_from_base(metadata)
    canonical_survey = _normalize_survey(survey)

    if canonical_survey in ("desi", "sdss"):
        resolved.wave_unit = "Angstrom"
        resolved.flux_unit = "cgs"
        resolved.flux_scale = 1e-17
        resolved.survey = canonical_survey

    if flux_unit is not None:
        normalized_flux_unit = str(flux_unit).strip().lower()
        if normalized_flux_unit not in ("cgs", "relative"):
            raise ValueError("flux_unit must be 'cgs' or 'relative'.")
        resolved.flux_unit = normalized_flux_unit
        if normalized_flux_unit == "cgs":
            resolved.flux_scale = 1.0 if flux_scale is None else float(
                flux_scale
            )
        else:
            if flux_scale is not None:
                raise ValueError(
                    "flux_scale is only valid when flux_unit='cgs'."
                )
            resolved.flux_scale = None
    elif flux_scale is not None:
        raise ValueError("flux_scale requires flux_unit='cgs'.")

    if wave_unit is not None:
        resolved.wave_unit = str(wave_unit)
    if resolved.flux_scale is not None and (
        not np.isfinite(resolved.flux_scale) or resolved.flux_scale <= 0
    ):
        raise ValueError("flux_scale must be finite and positive.")
    if source is not None:
        resolved.source = str(source)
    if ra is not None:
        resolved.ra = float(ra)
    if dec is not None:
        resolved.dec = float(dec)
    if galactic_extinction_corrected is not None:
        resolved.galactic_extinction_corrected = bool(
            galactic_extinction_corrected
        )
    if galactic_extinction is not None:
        resolved.galactic_extinction = dict(galactic_extinction)

    return resolved
