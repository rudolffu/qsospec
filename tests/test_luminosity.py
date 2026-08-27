from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from qsospec.luminosity import bolometric_luminosity, monochromatic_luminosity


def test_monochromatic_conversion_and_host_uncertainty() -> None:
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, name="test")
    flux = 2e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    error = 0.2e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    result = monochromatic_luminosity(
        flux,
        wavelength_rest=5100 * u.AA,
        redshift=1.0,
        observed_f_lambda_error=error,
        host_fraction=0.25,
        host_fraction_error=0.05,
        cosmology=cosmology,
        extinction_correction_status="applied",
    )
    distance = cosmology.luminosity_distance(1.0).to(u.cm)
    expected = (4 * np.pi * distance**2 * 2.0 * flux * 5100 * u.AA).to(u.erg / u.s)
    assert result.lambda_l_lambda.to_value(u.erg / u.s) == pytest.approx(expected.value)
    assert result.host_subtracted_lambda_l_lambda.value == pytest.approx(0.75 * expected.value)
    assert result.host_subtracted_error is not None


def test_bolometric_correction_is_explicit() -> None:
    result = bolometric_luminosity(
        1e45 * u.erg / u.s,
        correction=5.0,
        correction_error=0.5,
        prescription="example-5100",
    )
    assert result["value"].to_value(u.erg / u.s) == pytest.approx(5e45)
    with pytest.raises(ValueError, match="prescription"):
        bolometric_luminosity(1e45 * u.erg / u.s, correction=5.0, prescription="")
