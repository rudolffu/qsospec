"""Canonical names and provenance for wavelength-specific host measurements."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

MEASUREMENT_VOCABULARY_VERSION = "2"

FINAL_HOST_FRACTION_DEFINITION_ID = (
    "host_fraction_ppxf_host_plus_qsospec_agn_v1"
)
PPXF_HOST_FRACTION_DEFINITION_ID = (
    "host_fraction_ppxf_host_over_ppxf_total_v1"
)
HOST_FRACTION_DELTA_DEFINITION_ID = (
    "host_fraction_delta_final_minus_ppxf_v1"
)

_COMPONENT_PREFIXES = {
    "host": "fHost",
    "agn": "fAGN",
    "total": "fTotal",
    "host_fraction": "fracHost",
}
_KNOWN_WAVELENGTH_SUFFIXES = {
    "1um": 10000.0,
    "1p6um": 16000.0,
    "2p2um": 22000.0,
}
_LEGACY_PPXF_PATTERNS = (
    (re.compile(r"^fHostFit_(.+)$"), "host"),
    (re.compile(r"^fAGNFit_(.+)$"), "agn"),
    (re.compile(r"^fTotalFit_(.+)$"), "total"),
    (re.compile(r"^fracHost_(?!pPXF_)(.+)$"), "host_fraction"),
)


def _wavelength_suffix(wavelength_angstrom: float | str) -> str:
    if isinstance(wavelength_angstrom, str):
        suffix = wavelength_angstrom.strip()
        if not suffix:
            raise ValueError("wavelength suffix must not be empty")
        return suffix
    value = float(wavelength_angstrom)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("wavelength_angstrom must be positive and finite")
    if value.is_integer():
        return str(int(value))
    return format(value, ".10g").replace(".", "p")


def wavelength_from_sample_name(quantity: str) -> float | None:
    """Return the encoded rest wavelength, if *quantity* is a host sample."""

    match = re.match(
        r"^(?:fHost|fAGN|fTotal|fracHost)(?:_pPXF)?_(.+)$",
        str(quantity),
    )
    if match is None:
        return None
    suffix = match.group(1)
    if suffix in _KNOWN_WAVELENGTH_SUFFIXES:
        return _KNOWN_WAVELENGTH_SUFFIXES[suffix]
    try:
        value = float(suffix.replace("p", "."))
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def final_host_sample_name(
    component: str,
    wavelength_angstrom: float | str,
) -> str:
    """Return a final/adopted qsospec host-sample quantity name."""

    try:
        prefix = _COMPONENT_PREFIXES[str(component)]
    except KeyError as error:
        raise ValueError(
            "component must be host, agn, total, or host_fraction"
        ) from error
    return f"{prefix}_{_wavelength_suffix(wavelength_angstrom)}"


def ppxf_host_sample_name(
    component: str,
    wavelength_angstrom: float | str,
) -> str:
    """Return a direct-pPXF fitted-grid host-sample quantity name."""

    try:
        prefix = _COMPONENT_PREFIXES[str(component)]
    except KeyError as error:
        raise ValueError(
            "component must be host, agn, total, or host_fraction"
        ) from error
    return f"{prefix}_pPXF_{_wavelength_suffix(wavelength_angstrom)}"


def delta_host_fraction_name(
    wavelength_angstrom: float | str,
) -> str:
    """Return the final-minus-pPXF host-fraction diagnostic name."""

    return (
        "deltaFracHost_final_pPXF_"
        f"{_wavelength_suffix(wavelength_angstrom)}"
    )


def is_final_host_fraction_name(quantity: str) -> bool:
    """Whether *quantity* is a concise final/adopted host fraction."""

    return bool(re.match(r"^fracHost_(?!pPXF_).+$", str(quantity)))


def is_ppxf_host_sample_name(quantity: str) -> bool:
    """Whether *quantity* is a canonical direct-pPXF local sample."""

    return bool(
        re.match(
            r"^(?:fHost|fAGN|fTotal|fracHost)_pPXF_.+$",
            str(quantity),
        )
    )


def canonicalize_legacy_measurement_name(
    section: str,
    quantity: str,
) -> tuple[str, dict[str, Any]]:
    """Map a legacy quantity using its mandatory measurement-section context."""

    section = str(section)
    quantity = str(quantity)
    if section == "host_sample":
        for pattern, component in _LEGACY_PPXF_PATTERNS:
            match = pattern.match(quantity)
            if match is not None:
                canonical = ppxf_host_sample_name(component, match.group(1))
                return canonical, {
                    "legacy_quantity_name": quantity,
                    "legacy_measurement_vocabulary_version": "1",
                    "canonical_measurement_vocabulary_version": (
                        MEASUREMENT_VOCABULARY_VERSION
                    ),
                    "legacy_name_mapped": True,
                }
    return quantity, {}


def canonicalize_host_fit_samples(
    samples: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Canonicalize a raw ``host_fit_samples`` mapping without changing values."""

    output: dict[str, Any] = {}
    mappings: dict[str, dict[str, Any]] = {}
    for quantity, value in samples.items():
        canonical, metadata = canonicalize_legacy_measurement_name(
            "host_sample", str(quantity)
        )
        if canonical in output:
            raise ValueError(
                "Canonical host-fit sample collision for "
                f"{canonical!r}; explicit source resolution is required."
            )
        output[canonical] = value
        if metadata:
            mappings[canonical] = metadata
    return output, mappings


def host_sample_measurement_descriptor(
    section: str,
    quantity: str,
    *,
    host_strategy_used: str | None = None,
) -> dict[str, Any] | None:
    """Return method, unit, and scientific provenance for a host sample."""

    section = str(section)
    quantity = str(quantity)
    wavelength = wavelength_from_sample_name(quantity)
    if wavelength is None:
        return None
    common = {
        "measurement_vocabulary_version": MEASUREMENT_VOCABULARY_VERSION,
        "wavelength_rest_angstrom": float(wavelength),
        "direct_coverage_required": True,
        "host_strategy_used": host_strategy_used,
    }
    suffix = quantity.rsplit("_", 1)[-1]
    if section == "continuum_sample":
        if quantity == final_host_sample_name("host", suffix):
            metadata = {
                **common,
                "definition_id": "ppxf_stellar_host_flux_density_v1",
                "host_source": "ppxf_stellar_model",
                "continuum_source": "final_qsospec_agn_continuum",
            }
            return {
                "unit": "input_flux_density",
                "method": "ppxf_component_interpolation",
                "metadata": metadata,
            }
        if quantity == final_host_sample_name("agn", suffix):
            metadata = {
                **common,
                "definition_id": "final_qsospec_agn_flux_density_v1",
                "host_source": (
                    "ppxf_stellar_model" if host_strategy_used else None
                ),
                "continuum_source": "final_qsospec_agn_continuum",
            }
            return {
                "unit": "input_flux_density",
                "method": "final_qsospec_continuum_interpolation",
                "metadata": metadata,
            }
        if quantity == final_host_sample_name("host_fraction", suffix):
            metadata = {
                **common,
                "definition_id": FINAL_HOST_FRACTION_DEFINITION_ID,
                "host_source": "ppxf_stellar_model",
                "continuum_source": "final_qsospec_agn_continuum",
                "numerator_quantity": final_host_sample_name("host", suffix),
                "denominator_quantities": [
                    final_host_sample_name("host", suffix),
                    final_host_sample_name("agn", suffix),
                ],
            }
            return {
                "unit": "dimensionless",
                "method": "ppxf_host_plus_qsospec_agn",
                "metadata": metadata,
            }
    if section == "host_sample" and is_ppxf_host_sample_name(quantity):
        component_match = re.match(
            r"^(fHost|fAGN|fTotal|fracHost)_pPXF_(.+)$", quantity
        )
        if component_match is None:
            return None
        prefix, suffix = component_match.groups()
        metadata = {
            **common,
            "host_source": "ppxf_stellar_model",
            "continuum_source": "ppxf_total_model",
        }
        if prefix == "fracHost":
            metadata.update(
                {
                    "definition_id": PPXF_HOST_FRACTION_DEFINITION_ID,
                    "numerator_quantity": ppxf_host_sample_name(
                        "host", suffix
                    ),
                    "denominator_quantities": [
                        ppxf_host_sample_name("total", suffix)
                    ],
                }
            )
            unit = "dimensionless"
        else:
            component = {
                "fHost": "host",
                "fAGN": "agn",
                "fTotal": "total",
            }[prefix]
            metadata["definition_id"] = (
                f"ppxf_{component}_flux_density_v1"
            )
            unit = "input_flux_density"
        return {
            "unit": unit,
            "method": "ppxf_component_interpolation",
            "metadata": metadata,
        }
    return None
