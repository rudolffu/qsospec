"""Deterministic model reconstruction for QA and observed-frame comparisons."""
from __future__ import annotations

import numpy as np


def _complex_windows(recipe_id, fit):
    """Return geometric windows and the recipe's fallback evidence centroid."""
    from .complex_recipes import get

    try:
        recipe = get(recipe_id)
    except ValueError:
        # Custom recipes are not in the built-in registry. Their executed
        # windows are persisted by the generic fitter and the run store.
        metadata = getattr(fit, "metadata", {})
        if "fit_windows" not in metadata:
            raise ValueError(
                f"No registered or saved fitting windows for complex {recipe_id!r}"
            ) from None
        windows = metadata["fit_windows"]
        center = None
    else:
        windows = recipe.fit_windows or (recipe.fit_window,)
        center = float(np.mean(recipe.fit_window))
    validated = []
    for lo, hi in windows:
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"Invalid fitting window for complex {recipe_id!r}")
        validated.append((lo, hi))
    if center is None:
        center = (float(np.mean((min(lo for lo, _ in validated), max(hi for _, hi in validated))))
                  if validated else np.nan)
    return tuple(validated), center


def _reconstruct_line_model(result, spectrum):
    """Assemble continuum-subtracted complexes in native spectrum units.

    Each successful complex contributes only in its geometric windows, even
    when its saved local-continuum model is evaluated on the full grid. At an
    overlap the narrowest window wins, then recipe ID. Input validity and fit
    clipping masks do not truncate model curves. The host and global continuum
    are deliberately absent; callers add each once in their required context.
    """
    wave = np.asarray(spectrum.wave_obs, dtype=float)
    if wave.ndim != 1 or np.any(np.diff(wave) <= 0):
        raise ValueError("Observed grid must be strictly increasing")
    model = np.zeros_like(wave)
    failed = np.zeros_like(wave, dtype=bool)
    owner = np.full(wave.shape, "", dtype=object)
    windows, claims, evidence = [], [], []
    for recipe_id, fit in sorted(result.line_complexes.items()):
        rest_windows, fallback_center = _complex_windows(recipe_id, fit)
        for lo, hi in rest_windows:
            lo, hi = lo * (1 + spectrum.z), hi * (1 + spectrum.z)
            windows.append((lo, hi))
            region = (wave >= lo) & (wave <= hi)
            if not fit.success:
                failed |= region
                continue
            claims.append((hi - lo, recipe_id, lo, hi, fit))
        bad = any(
            any(code in warning.code.lower() for code in ("bound", "failed", "rank_deficient"))
            for warning in fit.warnings
        )
        for name, value in fit.metrics.items():
            if not name.endswith("_flux_cgs"):
                continue
            error = fit.metric_errors.get(name, np.nan)
            prefix = name[:-len("_flux_cgs")]
            center = fit.metrics.get(prefix + "_centroid", np.nan)
            if not np.isfinite(center):
                center = fallback_center
            evidence.append(dict(
                recipe_id=recipe_id, quantity=name,
                wavelength=float(center) * (1 + spectrum.z),
                snr=float(value / error) if error is not None and np.isfinite(error) and error > 0 else 0.,
                usable=bool(fit.success and not bad),
                narrow=("narrow" in name or "core" in name),
            ))
    for _, recipe_id, lo, hi, fit in sorted(claims, key=lambda item: (item[0], item[1], item[2])):
        mask = (wave >= lo) & (wave <= hi) & (owner == "")
        fit_wave = np.asarray(fit.wave_rest, dtype=float) * (1 + spectrum.z)
        fit_model = np.asarray(fit.model, dtype=float)
        if fit_wave.ndim != 1 or fit_wave.shape != fit_model.shape or np.any(np.diff(fit_wave) <= 0):
            raise ValueError(f"Invalid model wavelength grid for complex {recipe_id!r}")
        values = np.interp(wave, fit_wave, fit_model, left=np.nan, right=np.nan)
        model[mask] = values[mask]
        owner[mask] = recipe_id
    return dict(model=model, failed=failed, complex_owner=owner,
                windows=windows, line_evidence=evidence)


def reconstruct_observed_model(result):
    """Return total f_lambda in physical cgs on the original observed grid.

    Each line-complex owns its registered windows. In overlaps the narrowest
    window wins, then recipe ID; local continua are never summed twice. Outside
    those windows the global continuum plus host is the model. Failed registered
    windows are unsupported, even if an overlapping successful fit exists.
    Fit clipping masks do not participate in support or native validity.
    """
    spectrum = result.total_spectrum if result.total_spectrum is not None else result.spectrum
    wave = np.asarray(spectrum.wave_obs, dtype=float)
    scale = spectrum.flux_scale
    if scale is None:
        raise ValueError("Physical flux scale is required")
    factor = float(scale) / (1 + spectrum.z) if spectrum.metadata.flux_frame == "rest" else float(scale)
    continuum = np.asarray(result.continuum.model, dtype=float)
    if continuum.shape != wave.shape:
        raise ValueError("Continuum grid mismatch")
    host = result.host_model_on_quasar_grid
    host = np.zeros_like(continuum) if host is None else np.asarray(host, dtype=float)
    if host.shape != wave.shape:
        raise ValueError("Host grid mismatch")
    reconstructed = _reconstruct_line_model(result, spectrum)
    continuum_total = continuum + host
    model = continuum_total + reconstructed["model"]
    supported = ~reconstructed["failed"] & np.isfinite(model)
    return dict(
        wavelength=wave, flux=np.asarray(spectrum.flux) * factor,
        error=np.asarray(spectrum.err) * factor, valid=np.asarray(spectrum.valid_mask, bool),
        model=model * factor, continuum=continuum_total * factor,
        supported=supported, windows=reconstructed["windows"],
        line_evidence=reconstructed["line_evidence"], success=bool(result.continuum.success),
        complex_owner=reconstructed["complex_owner"],
        definition="observed_total_v1_narrowest_window_owner",
        flux_unit="erg s-1 cm-2 Angstrom-1",
    )
