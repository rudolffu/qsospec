"""Unit-aware monochromatic and bolometric luminosity utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import Cosmology, Planck18

FLAMBDA_UNIT = u.erg / (u.s * u.cm**2 * u.AA)
LLAMBDA_UNIT = u.erg / (u.s * u.AA)


@dataclass(frozen=True)
class MonochromaticLuminosity:
    """Total and optional host-subtracted luminosity at one rest wavelength."""

    wavelength_rest: u.Quantity
    l_lambda: u.Quantity
    lambda_l_lambda: u.Quantity
    l_nu: u.Quantity
    l_lambda_error: u.Quantity | None
    lambda_l_lambda_error: u.Quantity | None
    l_nu_error: u.Quantity | None
    host_subtracted_lambda_l_lambda: u.Quantity | None
    host_subtracted_error: u.Quantity | None
    metadata: Mapping[str, Any]


def luminosity_distance(redshift: float, cosmology: Cosmology = Planck18) -> u.Quantity:
    """Return luminosity distance for one finite positive redshift."""

    if not np.isfinite(redshift) or redshift <= 0:
        raise ValueError("redshift must be finite and positive")
    return cosmology.luminosity_distance(float(redshift)).to(u.cm)


def monochromatic_luminosity(
    observed_f_lambda: u.Quantity,
    *,
    wavelength_rest: u.Quantity,
    redshift: float,
    observed_f_lambda_error: u.Quantity | None = None,
    host_fraction: float | None = None,
    host_fraction_error: float | None = None,
    cosmology: Cosmology = Planck18,
    extinction_correction_status: str = "unspecified",
) -> MonochromaticLuminosity:
    """Convert observed-frame ``f_lambda`` to rest-frame luminosity density.

    The supplied flux density must be evaluated at
    ``wavelength_rest * (1 + redshift)``. The conversion is
    ``L_lambda(rest) = 4 pi D_L^2 (1+z) f_lambda(observed)``.
    """

    flux = u.Quantity(observed_f_lambda).to(FLAMBDA_UNIT)
    wavelength = u.Quantity(wavelength_rest).to(u.AA)
    if not np.all(np.isfinite(flux.value)):
        raise ValueError("observed_f_lambda must be finite")
    if not np.isfinite(wavelength.value) or wavelength.value <= 0:
        raise ValueError("wavelength_rest must be finite and positive")
    distance = luminosity_distance(redshift, cosmology)
    l_lambda = (4.0 * np.pi * distance**2 * (1.0 + redshift) * flux).to(LLAMBDA_UNIT)
    lambda_l_lambda = (wavelength * l_lambda).to(u.erg / u.s)
    c_angstrom = c.to(u.AA / u.s)
    l_nu = (l_lambda * wavelength**2 / c_angstrom).to(u.erg / u.s / u.Hz)

    l_lambda_error = lambda_l_lambda_error = l_nu_error = None
    if observed_f_lambda_error is not None:
        flux_error = u.Quantity(observed_f_lambda_error).to(FLAMBDA_UNIT)
        if not np.all(np.isfinite(flux_error.value)) or np.any(flux_error.value < 0):
            raise ValueError("observed_f_lambda_error must be finite and non-negative")
        l_lambda_error = (
            4.0 * np.pi * distance**2 * (1.0 + redshift) * flux_error
        ).to(LLAMBDA_UNIT)
        lambda_l_lambda_error = (wavelength * l_lambda_error).to(u.erg / u.s)
        l_nu_error = (l_lambda_error * wavelength**2 / c_angstrom).to(u.erg / u.s / u.Hz)

    host_subtracted = host_subtracted_error = None
    if host_fraction is not None:
        if not np.isfinite(host_fraction) or not 0.0 <= host_fraction <= 1.0:
            raise ValueError("host_fraction must lie in [0, 1]")
        host_subtracted = lambda_l_lambda * (1.0 - host_fraction)
        terms = []
        if lambda_l_lambda_error is not None:
            terms.append(((1.0 - host_fraction) * lambda_l_lambda_error) ** 2)
        if host_fraction_error is not None:
            if not np.isfinite(host_fraction_error) or host_fraction_error < 0:
                raise ValueError("host_fraction_error must be finite and non-negative")
            terms.append((lambda_l_lambda * host_fraction_error) ** 2)
        if terms:
            host_subtracted_error = np.sqrt(sum(terms)).to(u.erg / u.s)
    elif host_fraction_error is not None:
        raise ValueError("host_fraction_error requires host_fraction")

    metadata = {
        "cosmology": cosmology.name,
        "redshift": float(redshift),
        "wavelength_rest_angstrom": float(wavelength.value),
        "input_flux_density_frame": "observed",
        "host_subtraction_status": "applied" if host_fraction is not None else "not_applied",
        "extinction_correction_status": str(extinction_correction_status),
    }
    return MonochromaticLuminosity(
        wavelength_rest=wavelength,
        l_lambda=l_lambda,
        lambda_l_lambda=lambda_l_lambda,
        l_nu=l_nu,
        l_lambda_error=l_lambda_error,
        lambda_l_lambda_error=lambda_l_lambda_error,
        l_nu_error=l_nu_error,
        host_subtracted_lambda_l_lambda=host_subtracted,
        host_subtracted_error=host_subtracted_error,
        metadata=metadata,
    )


def bolometric_luminosity(
    monochromatic: u.Quantity,
    *,
    correction: float,
    monochromatic_error: u.Quantity | None = None,
    correction_error: float | None = None,
    prescription: str,
) -> dict[str, Any]:
    """Apply one explicit bolometric correction with propagated uncertainty."""

    luminosity = u.Quantity(monochromatic).to(u.erg / u.s)
    if not np.isfinite(correction) or correction <= 0:
        raise ValueError("correction must be finite and positive")
    if not prescription.strip():
        raise ValueError("prescription must identify the bolometric correction")
    value = correction * luminosity
    variance_terms = []
    if monochromatic_error is not None:
        error = u.Quantity(monochromatic_error).to(u.erg / u.s)
        if not np.isfinite(error.value) or error.value < 0:
            raise ValueError("monochromatic_error must be finite and non-negative")
        variance_terms.append((correction * error) ** 2)
    if correction_error is not None:
        if not np.isfinite(correction_error) or correction_error < 0:
            raise ValueError("correction_error must be finite and non-negative")
        variance_terms.append((correction_error * luminosity) ** 2)
    uncertainty = np.sqrt(sum(variance_terms)).to(u.erg / u.s) if variance_terms else None
    return {
        "value": value,
        "error": uncertainty,
        "metadata": {
            "bolometric_correction": float(correction),
            "bolometric_correction_error": correction_error,
            "prescription": prescription,
        },
    }
