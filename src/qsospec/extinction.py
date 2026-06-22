"""Foreground Galactic-extinction queries and dereddening helpers."""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from dust_extinction.parameter_averages import F99

from .config import GalacticExtinctionConfig
from .metadata import SpectrumMetadata
from .spectrum import Spectrum
from .workflows.host.io import SpectrumData


_PLANCK_GNILC_FILENAME = (
    "COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits"
)
_PROVENANCE_KEY = "galactic_extinction"
_REST_FRAME_KEY = "rest_frame_conversion"


def _resolved_config(
    config: Optional[GalacticExtinctionConfig],
) -> GalacticExtinctionConfig:
    return config or GalacticExtinctionConfig()


def _canonical_map_name(name: str) -> str:
    value = str(name).strip().lower()
    return "planck" if value in ("planck", "planck16") else value


def _resolved_data_dir(config: GalacticExtinctionConfig) -> Optional[str]:
    if config.dustmaps_data_dir is not None:
        return str(Path(config.dustmaps_data_dir).expanduser().resolve())
    if not config.enabled or config.ebv_override is not None:
        return None
    from dustmaps.std_paths import data_dir

    return str(Path(data_dir()).expanduser().resolve())


def _config_provenance(config: GalacticExtinctionConfig) -> Dict[str, Any]:
    values = asdict(config)
    values["map_name"] = _canonical_map_name(config.map_name)
    values["dustmaps_data_dir"] = _resolved_data_dir(config)
    if (
        config.enabled
        and config.ebv_override is None
        and values["dustmaps_data_dir"] is not None
    ):
        map_root = Path(values["dustmaps_data_dir"])
        values["map_path"] = str(
            map_root / "planck" / _PLANCK_GNILC_FILENAME
            if values["map_name"] == "planck"
            else map_root / "sfd"
        )
    else:
        values["map_path"] = None
    return values


@lru_cache(maxsize=8)
def _dust_query(map_name: str, data_dir: Optional[str]):
    if map_name == "planck":
        from dustmaps.planck import PlanckGNILCQuery

        map_fname = None
        if data_dir is not None:
            map_fname = str(
                Path(data_dir)
                / "planck"
                / _PLANCK_GNILC_FILENAME
            )
        return PlanckGNILCQuery(map_fname=map_fname)
    if map_name == "sfd":
        from dustmaps.sfd import SFDQuery

        map_dir = (
            str(Path(data_dir) / "sfd")
            if data_dir is not None
            else None
        )
        return SFDQuery(map_dir=map_dir)
    raise ValueError(f"Unsupported Galactic dust map: {map_name!r}")


def preflight_galactic_extinction(
    config: Optional[GalacticExtinctionConfig] = None,
) -> None:
    """Validate that the configured external dust map can be opened."""

    cfg = _resolved_config(config)
    if not cfg.enabled or cfg.ebv_override is not None:
        return
    _dust_query(
        _canonical_map_name(cfg.map_name),
        _resolved_data_dir(cfg),
    )


def query_galactic_ebv(
    ra: Optional[float],
    dec: Optional[float],
    config: Optional[GalacticExtinctionConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return applied E(B-V) and query provenance for one ICRS coordinate."""

    cfg = _resolved_config(config)
    provenance = _config_provenance(cfg)
    provenance.update(
        {
            "requested": bool(cfg.enabled),
            "ra": None if ra is None else float(ra),
            "dec": None if dec is None else float(dec),
        }
    )
    if not cfg.enabled:
        provenance.update(
            {
                "applied": False,
                "status": "disabled",
                "raw_ebv": None,
                "applied_ebv": 0.0,
                "warning": None,
            }
        )
        return 0.0, provenance

    if cfg.ebv_override is not None:
        raw_ebv = float(cfg.ebv_override)
        source = "override"
    else:
        if ra is None or dec is None:
            raise ValueError(
                "Galactic extinction correction requires finite RA and Dec "
                "in degrees, or GalacticExtinctionConfig.ebv_override."
            )
        ra_value = float(ra)
        dec_value = float(dec)
        if (
            not np.isfinite(ra_value)
            or not np.isfinite(dec_value)
            or not 0.0 <= ra_value < 360.0
            or not -90.0 <= dec_value <= 90.0
        ):
            raise ValueError(
                "Galactic extinction correction requires finite ICRS "
                "coordinates with 0 <= RA < 360 and -90 <= Dec <= 90 degrees."
            )
        data_dir = _resolved_data_dir(cfg)
        query = _dust_query(_canonical_map_name(cfg.map_name), data_dir)
        coordinate = SkyCoord(
            ra=ra_value * u.deg,
            dec=dec_value * u.deg,
            frame="icrs",
        )
        raw_ebv = float(np.asarray(query(coordinate)).reshape(-1)[0])
        source = _canonical_map_name(cfg.map_name)

    if not np.isfinite(raw_ebv):
        raise ValueError("The Galactic dust-map query returned non-finite E(B-V).")
    warning = None
    map_ebv = raw_ebv
    if raw_ebv < 0:
        if not cfg.clip_negative_ebv:
            raise ValueError(
                "The Galactic dust-map query returned negative E(B-V) and "
                "clip_negative_ebv is disabled."
            )
        map_ebv = 0.0
        warning = "negative_ebv_clipped_to_zero"
    applied_ebv = (
        map_ebv * float(cfg.sfd_recalibration)
        if source == "sfd"
        else map_ebv
    )
    provenance.update(
        {
            "applied": True,
            "status": "applied",
            "source": source,
            "raw_ebv": raw_ebv,
            "applied_ebv": applied_ebv,
            "warning": warning,
        }
    )
    return float(applied_ebv), provenance


def f99_dereddening_factor(
    wave_obs: np.ndarray,
    ebv: float,
    rv: float = 3.1,
) -> np.ndarray:
    """Return the multiplicative F99 dereddening factor."""

    wave = np.asarray(wave_obs, dtype=float)
    if wave.ndim != 1:
        raise ValueError("Observed wavelengths must be a one-dimensional array.")
    if not np.all(np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError(
            "F99 Galactic extinction correction requires positive, finite "
            "observed wavelengths."
        )
    model = F99(Rv=float(rv))
    inverse_micron = (1.0 / (wave * u.AA)).to_value(1 / u.micron)
    if (
        np.any(inverse_micron < model.x_range[0])
        or np.any(inverse_micron > model.x_range[1])
    ):
        supported_min = 1.0e4 / model.x_range[1]
        supported_max = 1.0e4 / model.x_range[0]
        raise ValueError(
            "Observed wavelengths fall outside the F99 supported range "
            f"[{supported_min:.1f}, {supported_max:.1f}] Angstrom."
        )
    attenuation = np.asarray(
        model.extinguish(wave * u.AA, Ebv=float(ebv)),
        dtype=float,
    )
    return 1.0 / attenuation


def _same_correction(
    existing: Dict[str, Any],
    config: GalacticExtinctionConfig,
    ra: Optional[float],
    dec: Optional[float],
) -> bool:
    requested = _config_provenance(config)
    for key, value in requested.items():
        if existing.get(key) != value:
            return False
    if config.ebv_override is None:
        return existing.get("ra") == ra and existing.get("dec") == dec
    return True


def correct_spectrum_data(
    spectrum: SpectrumData,
    config: Optional[GalacticExtinctionConfig] = None,
) -> SpectrumData:
    """Apply the configured Galactic correction exactly once."""

    cfg = _resolved_config(config)
    metadata = dict(spectrum.metadata)
    existing = metadata.get(_PROVENANCE_KEY)
    if isinstance(existing, dict):
        if existing.get("status") in (
            "caller_preprocessed",
            "declared_corrected",
        ):
            return _prepare_spectrum_data_rest_frame(spectrum)
        if existing.get("status") == "disabled":
            if not cfg.enabled and _same_correction(
                existing, cfg, spectrum.ra, spectrum.dec
            ):
                return _prepare_spectrum_data_rest_frame(spectrum)
            metadata.pop(_PROVENANCE_KEY, None)
            spectrum = replace(spectrum, metadata=metadata)
            existing = None
    if isinstance(existing, dict):
        if _same_correction(existing, cfg, spectrum.ra, spectrum.dec):
            return _prepare_spectrum_data_rest_frame(spectrum)
        raise ValueError(
            "SpectrumData already contains a different Galactic-extinction "
            "correction and the raw arrays are unavailable."
        )

    ebv, provenance = query_galactic_ebv(spectrum.ra, spectrum.dec, cfg)
    if not cfg.enabled:
        metadata[_PROVENANCE_KEY] = provenance
        return _prepare_spectrum_data_rest_frame(
            replace(spectrum, metadata=metadata)
        )

    factor = f99_dereddening_factor(spectrum.wave_obs, ebv, cfg.rv)
    flux = np.asarray(spectrum.flux, dtype=float) * factor
    error = (
        None
        if spectrum.error is None
        else np.asarray(spectrum.error, dtype=float) * factor
    )
    ivar = (
        None
        if spectrum.ivar is None
        else np.asarray(spectrum.ivar, dtype=float) / factor**2
    )
    provenance.update(
        {
            "correction_factor_min": float(np.min(factor)),
            "correction_factor_max": float(np.max(factor)),
        }
    )
    metadata[_PROVENANCE_KEY] = provenance
    return _prepare_spectrum_data_rest_frame(
        replace(
            spectrum,
            flux=flux,
            error=error,
            ivar=ivar,
            metadata=metadata,
        )
    )


def _rest_frame_provenance(
    redshift: float,
    *,
    input_frame: str,
    status: str,
) -> Dict[str, Any]:
    factor = 1.0 + float(redshift)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Rest-frame conversion requires finite z with 1 + z > 0.")
    return {
        "status": status,
        "input_flux_frame": input_frame,
        "output_flux_frame": "rest",
        "redshift": float(redshift),
        "flux_error_factor": float(factor),
        "inverse_variance_factor": float(1.0 / factor**2),
    }


def _metadata_with_rest_frame(
    metadata: Dict[str, Any],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    output = dict(metadata)
    output["flux_frame"] = "rest"
    output[_REST_FRAME_KEY] = dict(provenance)
    nested = output.get("spectrum_metadata")
    if isinstance(nested, dict):
        nested = dict(nested)
        nested["flux_frame"] = "rest"
        nested[_REST_FRAME_KEY] = dict(provenance)
        output["spectrum_metadata"] = nested
    return output


def _prepare_spectrum_data_rest_frame(
    spectrum: SpectrumData,
) -> SpectrumData:
    metadata = dict(spectrum.metadata)
    frame = str(metadata.get("flux_frame", "observed")).lower()
    if spectrum.redshift is None or not np.isfinite(spectrum.redshift):
        raise ValueError("Rest-frame conversion requires a finite redshift.")
    if frame == "rest":
        existing = metadata.get(_REST_FRAME_KEY)
        if isinstance(existing, dict) and existing:
            expected = _rest_frame_provenance(
                float(spectrum.redshift),
                input_frame="observed",
                status="applied",
            )
            if (
                existing.get("output_flux_frame") != "rest"
                or not np.isclose(
                    float(existing.get("redshift", np.nan)),
                    expected["redshift"],
                )
                or not np.isclose(
                    float(existing.get("flux_error_factor", np.nan)),
                    expected["flux_error_factor"],
                )
            ):
                raise ValueError(
                    "SpectrumData contains conflicting rest-frame conversion "
                    "provenance."
                )
        return spectrum
    if frame != "observed":
        raise ValueError("SpectrumData flux_frame must be 'observed' or 'rest'.")
    if metadata.get(_REST_FRAME_KEY):
        raise ValueError(
            "SpectrumData is marked observed but already contains rest-frame "
            "conversion provenance."
        )
    provenance = _rest_frame_provenance(
        float(spectrum.redshift),
        input_frame="observed",
        status="applied",
    )
    factor = provenance["flux_error_factor"]
    return replace(
        spectrum,
        flux=np.asarray(spectrum.flux, dtype=float) * factor,
        error=(
            None
            if spectrum.error is None
            else np.asarray(spectrum.error, dtype=float) * factor
        ),
        ivar=(
            None
            if spectrum.ivar is None
            else np.asarray(spectrum.ivar, dtype=float) / factor**2
        ),
        metadata=_metadata_with_rest_frame(metadata, provenance),
    )


def _updated_spectrum_metadata(
    metadata: SpectrumMetadata,
    *,
    ra: Optional[float],
    dec: Optional[float],
    corrected: bool,
    provenance: Dict[str, Any],
) -> SpectrumMetadata:
    return replace(
        metadata,
        ra=ra,
        dec=dec,
        galactic_extinction_corrected=bool(corrected),
        galactic_extinction=dict(provenance),
        notes=list(metadata.notes),
    )


def _prepare_spectrum_rest_frame(spectrum: Spectrum) -> Spectrum:
    if spectrum.flux_frame == "rest":
        existing = spectrum.metadata.rest_frame_conversion
        if existing:
            expected = _rest_frame_provenance(
                spectrum.z,
                input_frame="observed",
                status="applied",
            )
            if (
                existing.get("output_flux_frame") != "rest"
                or not np.isclose(
                    float(existing.get("redshift", np.nan)),
                    expected["redshift"],
                )
                or not np.isclose(
                    float(existing.get("flux_error_factor", np.nan)),
                    expected["flux_error_factor"],
                )
            ):
                raise ValueError(
                    "Spectrum contains conflicting rest-frame conversion "
                    "provenance."
                )
        return spectrum
    if spectrum.flux_frame != "observed":
        raise ValueError("Spectrum flux_frame must be 'observed' or 'rest'.")
    if spectrum.metadata.rest_frame_conversion:
        raise ValueError(
            "Spectrum is marked observed but already contains rest-frame "
            "conversion provenance."
        )
    provenance = _rest_frame_provenance(
        spectrum.z,
        input_frame="observed",
        status="applied",
    )
    factor = provenance["flux_error_factor"]
    metadata = replace(
        spectrum.metadata,
        flux_frame="rest",
        rest_frame_conversion=dict(provenance),
        notes=list(spectrum.metadata.notes),
    )
    return replace(
        spectrum,
        flux=np.asarray(spectrum.flux, dtype=float) * factor,
        err=np.asarray(spectrum.err, dtype=float) * factor,
        metadata=metadata,
    )


def prepare_spectrum(
    spectrum: Spectrum,
    *,
    galactic_extinction_config: Optional[
        GalacticExtinctionConfig
    ] = None,
) -> Spectrum:
    """Prepare an in-memory spectrum for fitting.

    Array spectra are assumed to be uncorrected unless constructed with
    ``galactic_extinction_corrected=True``. Correction is performed in the
    observed frame, then flux and uncertainty are normalized to rest-frame
    F_lambda. Both operations are recorded and applied exactly once.
    """

    cfg = _resolved_config(galactic_extinction_config)
    ra = spectrum.metadata.ra
    dec = spectrum.metadata.dec
    existing = dict(spectrum.metadata.galactic_extinction)
    status = existing.get("status")

    if status in ("declared_corrected", "caller_preprocessed"):
        metadata = _updated_spectrum_metadata(
            spectrum.metadata,
            ra=ra,
            dec=dec,
            corrected=True,
            provenance=existing,
        )
        prepared = (
            spectrum
            if metadata == spectrum.metadata
            else replace(spectrum, metadata=metadata)
        )
        return _prepare_spectrum_rest_frame(prepared)

    if status == "applied":
        if not _same_correction(existing, cfg, ra, dec):
            raise ValueError(
                "Spectrum already contains a different Galactic-extinction "
                "correction and the raw arrays are unavailable."
            )
        metadata = _updated_spectrum_metadata(
            spectrum.metadata,
            ra=ra,
            dec=dec,
            corrected=True,
            provenance=existing,
        )
        prepared = (
            spectrum
            if metadata == spectrum.metadata
            else replace(spectrum, metadata=metadata)
        )
        return _prepare_spectrum_rest_frame(prepared)

    if spectrum.metadata.galactic_extinction_corrected:
        provenance = _config_provenance(cfg)
        provenance.update(
            {
                "requested": bool(cfg.enabled),
                "applied": False,
                "status": "declared_corrected",
                "ra": ra,
                "dec": dec,
                "raw_ebv": None,
                "applied_ebv": None,
                "warning": None,
            }
        )
        return _prepare_spectrum_rest_frame(
            replace(
                spectrum,
                metadata=_updated_spectrum_metadata(
                    spectrum.metadata,
                    ra=ra,
                    dec=dec,
                    corrected=True,
                    provenance=provenance,
                ),
            )
        )

    if not cfg.enabled:
        _, provenance = query_galactic_ebv(ra, dec, cfg)
        return _prepare_spectrum_rest_frame(
            replace(
                spectrum,
                metadata=_updated_spectrum_metadata(
                    spectrum.metadata,
                    ra=ra,
                    dec=dec,
                    corrected=False,
                    provenance=provenance,
                ),
            )
        )

    try:
        ebv, provenance = query_galactic_ebv(ra, dec, cfg)
    except ValueError as exc:
        if "requires finite RA and Dec" in str(exc):
            raise ValueError(
                "This array spectrum is marked as not corrected for Galactic "
                "extinction. Supply finite ra and dec to Spectrum.from_arrays, "
                "set GalacticExtinctionConfig(ebv_override=...), or set "
                "galactic_extinction_corrected=True when the supplied arrays "
                "have already been dereddened."
            ) from exc
        raise

    factor = f99_dereddening_factor(spectrum.wave_obs, ebv, cfg.rv)
    provenance.update(
        {
            "correction_factor_min": float(np.min(factor)),
            "correction_factor_max": float(np.max(factor)),
        }
    )
    return _prepare_spectrum_rest_frame(
        replace(
            spectrum,
            flux=np.asarray(spectrum.flux, dtype=float) * factor,
            err=np.asarray(spectrum.err, dtype=float) * factor,
            metadata=_updated_spectrum_metadata(
                spectrum.metadata,
                ra=ra,
                dec=dec,
                corrected=True,
                provenance=provenance,
            ),
        )
    )


def correct_spectrum(
    spectrum: Spectrum,
    *,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    config: Optional[GalacticExtinctionConfig] = None,
) -> Tuple[Spectrum, Dict[str, Any]]:
    """Return a corrected in-memory spectrum and correction provenance."""

    metadata = replace(
        spectrum.metadata,
        ra=spectrum.metadata.ra if ra is None else float(ra),
        dec=spectrum.metadata.dec if dec is None else float(dec),
        galactic_extinction_corrected=False,
        galactic_extinction={},
        notes=list(spectrum.metadata.notes),
    )
    corrected = prepare_spectrum(
        replace(spectrum, metadata=metadata),
        galactic_extinction_config=config,
    )
    return corrected, dict(corrected.metadata.galactic_extinction)
