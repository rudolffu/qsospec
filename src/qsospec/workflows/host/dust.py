"""Dust/extinction helpers for host-decomposition preprocessing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from astropy import units as u


def _f99_extinction(wave_obs: np.ndarray, ebv: float, rv: float) -> np.ndarray:
    """Return the legacy Fitzpatrick-like extinction curve in magnitudes."""

    x = 1.0 / (np.asarray(wave_obs, dtype=float) * u.AA).to(u.micron).value
    curve = np.zeros_like(x)
    optical = (x >= 0.3) & (x <= 1.1)
    y = x[optical] - 1.82
    curve[optical] = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    ultraviolet = (x > 1.1) & (x <= 3.3)
    y = x[ultraviolet] - 1.82
    curve[ultraviolet] = (
        1.0
        + 0.104 * y
        - 0.609 * y**2
        + 0.701 * y**3
        + 1.137 * y**4
        - 1.718 * y**5
        - 0.827 * y**6
        + 1.647 * y**7
        - 0.505 * y**8
    )
    far_uv = x > 3.3
    y = x[far_uv]
    curve[far_uv] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341)
    return curve * float(rv) * float(ebv)


def apply_galactic_dereddening(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    ebv: Optional[float] = None,
    rv: float = 3.1,
) -> np.ndarray:
    """Apply Galactic dereddening when an E(B-V) value is supplied."""

    if ebv is None or float(ebv) == 0.0:
        return np.asarray(flux, dtype=float)
    extinction = _f99_extinction(wave_obs, float(ebv), float(rv))
    return np.asarray(flux, dtype=float) * 10.0 ** (0.4 * extinction)
