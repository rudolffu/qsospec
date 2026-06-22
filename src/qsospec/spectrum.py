"""Spectrum container for qsospec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Any, Optional

import numpy as np

from .metadata import SpectrumMetadata, resolve_spectrum_metadata


@dataclass(frozen=True)
class Spectrum:
    """Array-only spectrum input for qsospec."""

    wave_obs: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    z: float
    metadata: SpectrumMetadata = field(default_factory=SpectrumMetadata)
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_arrays(
        cls,
        wave: np.ndarray,
        flux: np.ndarray,
        err: Optional[np.ndarray] = None,
        ivar: Optional[np.ndarray] = None,
        z: float = 0.0,
        wave_frame: str = "observed",
        mask: Optional[np.ndarray] = None,
        survey: Optional[str] = None,
        wave_unit: Optional[str] = None,
        flux_unit: Optional[str] = None,
        flux_scale: Optional[float] = None,
        source: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        galactic_extinction_corrected: bool = False,
        galactic_extinction: Optional[Mapping[str, Any]] = None,
        metadata: Optional[SpectrumMetadata] = None,
    ) -> "Spectrum":
        """Build a spectrum from plain arrays.

        ``wave_frame`` may be ``"observed"`` or ``"rest"`` and declares the
        frame of both wavelength and F_lambda. Internally the observed
        wavelength is stored and rest wavelength is derived from ``z``.
        """

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave.ndim != 1 or flux.ndim != 1:
            raise ValueError("wave and flux must be 1D arrays.")
        if wave.shape != flux.shape:
            raise ValueError("wave and flux must have the same shape.")
        if not np.isfinite(z):
            raise ValueError("z must be finite.")
        if 1.0 + float(z) <= 0:
            raise ValueError("z must satisfy 1 + z > 0.")
        if metadata is None and survey is None and flux_unit is None:
            raise ValueError(
                "flux_unit is required for array spectra; use 'cgs' for "
                "physical f_lambda or 'relative' for arbitrary/model spectra."
            )
        if ra is not None:
            ra = float(ra)
            if not np.isfinite(ra) or not 0.0 <= ra < 360.0:
                raise ValueError(
                    "ra must be finite and satisfy 0 <= ra < 360 degrees."
                )
        if dec is not None:
            dec = float(dec)
            if not np.isfinite(dec) or not -90.0 <= dec <= 90.0:
                raise ValueError(
                    "dec must be finite and satisfy -90 <= dec <= 90 degrees."
                )

        if err is None:
            if ivar is None:
                raise ValueError("Either err or ivar must be provided.")
            ivar = np.asarray(ivar, dtype=float)
            if ivar.shape != wave.shape:
                raise ValueError("ivar must have the same shape as wave.")
            err_arr = np.full_like(ivar, np.inf, dtype=float)
            good = ivar > 0
            err_arr[good] = 1.0 / np.sqrt(ivar[good])
        else:
            err_arr = np.asarray(err, dtype=float)
            if err_arr.shape != wave.shape:
                raise ValueError("err must have the same shape as wave.")

        mask_arr = None
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != wave.shape:
                raise ValueError("mask must have the same shape as wave.")

        frame = str(wave_frame).strip().lower()
        if frame == "observed":
            wave_obs = wave
        elif frame == "rest":
            wave_obs = wave * (1.0 + float(z))
        else:
            raise ValueError("wave_frame must be 'observed' or 'rest'.")

        corrected_metadata_value = (
            galactic_extinction_corrected
            if metadata is None or galactic_extinction_corrected
            else None
        )
        return cls(
            wave_obs=wave_obs.copy(),
            flux=flux.copy(),
            err=err_arr.copy(),
            z=float(z),
            metadata=resolve_spectrum_metadata(
                survey=survey,
                wave_unit=wave_unit,
                flux_unit=flux_unit,
                flux_scale=flux_scale,
                flux_frame=(frame if metadata is None else None),
                source=source,
                ra=ra,
                dec=dec,
                galactic_extinction_corrected=(
                    corrected_metadata_value
                ),
                galactic_extinction=galactic_extinction,
                metadata=metadata,
            ),
            mask=None if mask_arr is None else mask_arr.copy(),
        )

    @property
    def wave_rest(self) -> np.ndarray:
        """Rest-frame wavelength array."""

        return self.wave_obs / (1.0 + self.z)

    @property
    def valid_mask(self) -> np.ndarray:
        """Finite, positive-error pixels allowed for fitting."""

        valid = (
            np.isfinite(self.wave_obs)
            & np.isfinite(self.flux)
            & np.isfinite(self.err)
            & (self.wave_obs > 0)
            & (self.err > 0)
        )
        if self.mask is not None:
            valid &= self.mask
        return valid

    @property
    def wave_unit(self) -> str:
        """Wavelength unit label."""

        return self.metadata.wave_unit

    @property
    def flux_unit(self) -> str:
        """Flux unit kind: physical cgs or relative f_lambda."""

        return self.metadata.flux_unit

    @property
    def flux_scale(self) -> Optional[float]:
        """Multiplicative scale from input f_lambda to physical cgs."""

        return self.metadata.flux_scale

    @property
    def flux_frame(self) -> str:
        """Flux-density frame: ``observed`` or ``rest``."""

        return self.metadata.flux_frame

    @property
    def flux_density_unit(self) -> str:
        """Internal display label for the input f_lambda values."""

        if self.flux_unit == "cgs":
            return "erg s^-1 cm^-2 Angstrom^-1"
        return "relative f_lambda"

    @property
    def flux_density_scale_to_cgs(self) -> Optional[float]:
        """Internal compatibility alias for the physical cgs scale."""

        return self.flux_scale


def require_rest_frame_flux(spectrum: Spectrum) -> None:
    """Raise when a numerical fitter receives observed-frame flux density."""

    if spectrum.flux_frame != "rest":
        raise ValueError(
            "This fitter requires rest-frame F_lambda. Call "
            "qsospec.prepare_spectrum(spectrum) first, or construct model/"
            "composite arrays with wave_frame='rest'."
        )
