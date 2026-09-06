"""Fixed-anchor local continua in linear flux space, with explicit support gates.

Statistical covariance assumes independent supplied pixel errors. A host-scale
perturbation is propagated as one correlated nuisance, not independent noise.
Neither that conditional term nor their sum includes template-shape uncertainty
or covariance between a host scale estimated from these pixels and their noise.
"""

from dataclasses import dataclass
import numpy as np
from astropy.cosmology import Planck18
from .extinction import galactic_dereddening_factor
from .models.continuum import continuum_partials


@dataclass(frozen=True)
class WindowConfig:
    lower: float
    upper: float
    anchor: float
    min_pixels: int = 10
    slope_min_pixels: int = 20
    slope_min_span_fraction: float = 0.5
    min_pixels_each_side: int = 2
    max_anchor_gap: float = 100.0

    def __post_init__(self):
        if not self.lower < self.anchor < self.upper:
            raise ValueError("anchor must be strictly inside its window")
        if self.min_pixels < 2 or self.max_anchor_gap <= 0:
            raise ValueError("invalid support policy")


def corrected_rest_arrays(
    wave_obs,
    flux_obs,
    ivar_obs,
    *,
    redshift,
    native_cgs_scale,
    ebv,
    already_corrected=False
):
    """Convert native observed arrays once; caller declares their input state.

    An already corrected input receives only frame/unit conversion. This API
    accepts native observed arrays, never already converted physical/rest arrays.
    """
    w, f, iv = (np.asarray(a, dtype=float) for a in (wave_obs, flux_obs, ivar_obs))
    if w.ndim != 1 or f.shape != w.shape or iv.shape != w.shape:
        raise ValueError("array shapes disagree")
    if (
        not np.isfinite(redshift)
        or redshift < 0
        or not np.isfinite(native_cgs_scale)
        or native_cgs_scale <= 0
    ):
        raise ValueError("invalid redshift or native unit scale")
    if not np.isfinite(ebv) or ebv < 0:
        raise ValueError("invalid foreground E(B-V)")
    factor = (
        np.ones(w.shape)
        if already_corrected
        else galactic_dereddening_factor(w, ebv, rv=3.1, law="f99")
    )
    scale = native_cgs_scale * (1 + redshift) * factor
    variance = np.full(iv.shape, np.inf)
    good = np.isfinite(iv) & (iv > 0)
    variance[good] = scale[good] ** 2 / iv[good]
    return w / (1 + redshift), f * scale, variance


def lambda_l_lambda(anchor, flux_rest, redshift):
    """Physical cgs rest f_lambda to erg/s using Planck18 luminosity distance."""
    return float(
        4
        * np.pi
        * Planck18.luminosity_distance(redshift).to_value("cm") ** 2
        * anchor
        * flux_rest
    )


def measure_window(
    wavelength,
    flux,
    variance,
    valid_mask,
    config,
    *,
    scale_perturbation=None,
    component="original_total"
):
    """Return a fixed-anchor WLS estimate and conditional covariance in cgs.

    ``valid_mask=True`` means usable. Zeros/bitmasks are a reader contract, not
    a positivity cut here. The anchor must have nearby valid pixels on both sides.
    """
    w, f, v = (np.asarray(a, float) for a in (wavelength, flux, variance))
    valid = np.asarray(valid_mask, bool)
    if w.ndim != 1 or any(a.shape != w.shape for a in (f, v, valid)):
        raise ValueError("array shapes disagree")
    order = np.argsort(w, kind="stable")
    w, f, v, valid = (a[order] for a in (w, f, v, valid))
    if not np.all(np.isfinite(w)) or np.any(np.diff(w) <= 0):
        raise ValueError("wavelength must be finite and unique")
    if component not in {"original_total", "stellar_subtracted"}:
        raise ValueError("unknown component")
    c = config
    span = c.upper - c.lower
    inside = (w >= c.lower) & (w <= c.upper)
    good = inside & valid & np.isfinite(f) & np.isfinite(v) & (v > 0)
    x = w[good]
    n = len(x)
    left = x[x < c.anchor]
    right = x[x > c.anchor]
    bracket = (
        len(left) >= c.min_pixels_each_side and len(right) >= c.min_pixels_each_side
    )
    gap = float(right[0] - left[-1]) if len(left) and len(right) else np.nan
    # Pixel footprints quantify missing coverage against the FULL requested window.
    if len(w) > 1:
        edges = np.r_[
            w[0] - (w[1] - w[0]) / 2, (w[:-1] + w[1:]) / 2, w[-1] + (w[-1] - w[-2]) / 2
        ]
        widths = np.maximum(
            0, np.minimum(edges[1:], c.upper) - np.maximum(edges[:-1], c.lower)
        )
        coverage = float(widths[good].sum() / span)
    else:
        coverage = 0.0
    out = dict(
        n_raw=int(inside.sum()),
        n_valid=n,
        valid_span=float(np.ptp(x)) if n else 0.0,
        window_coverage_fraction=coverage,
        anchor_bracketed=bool(bracket),
        anchor_gap=gap,
        effective_wavelength=np.nan,
        continuum_f_lambda_rest=np.nan,
        continuum_f_lambda_rest_error=np.nan,
        local_slope=np.nan,
        local_parameter_covariance=None,
        host_scale_parameter_covariance=None,
        host_scale_error=np.nan,
        total_error=np.nan,
        statistical_uncertainty_status="unavailable",
        host_scale_uncertainty_status=(
            "not_applicable" if component == "original_total" else "unavailable"
        ),
        template_shape_uncertainty_status=(
            "not_applicable"
            if component == "original_total"
            else "unavailable_no_ensemble"
        ),
        scale_pixel_cross_covariance_status=(
            "not_applicable"
            if component == "original_total"
            else "unavailable_same_pixels"
        ),
        measurement_status="unsupported",
        measurement_reasons="",
        local_model=None,
    )
    reasons = []
    if n < c.min_pixels:
        reasons.append("insufficient_valid_pixels")
    if not bracket:
        reasons.append("anchor_not_bracketed")
    if bracket and gap > c.max_anchor_gap:
        reasons.append("masked_anchor_gap")
    if reasons:
        out["measurement_reasons"] = ";".join(reasons)
        return out
    mode = (
        "linear"
        if n >= c.slope_min_pixels and np.ptp(x) >= c.slope_min_span_fraction * span
        else "constant"
    )
    design = continuum_partials(x, mode=mode, wave_ref=c.anchor, scale=span)
    # Normalize uncertainties to keep physical cgs normal matrices well scaled.
    err = np.sqrt(v[good])
    unit = np.median(err)
    y = f[good] / unit
    weight = (unit / err) ** 2
    normal = design.T @ (weight[:, None] * design)
    covariance = np.linalg.inv(normal) * unit**2
    response = np.linalg.solve(normal, design.T * weight)
    pars = response @ y * unit
    out.update(
        effective_wavelength=float(np.average(x, weights=weight)),
        continuum_f_lambda_rest=float(pars[0]),
        continuum_f_lambda_rest_error=float(np.sqrt(covariance[0, 0])),
        local_slope=float(pars[1] / span) if mode == "linear" else np.nan,
        local_parameter_covariance=covariance.tolist(),
        local_model=mode,
        statistical_uncertainty_status="available_independent_pixels",
        measurement_status="measured" if pars[0] > 0 else "nonpositive",
        measurement_reasons="" if pars[0] > 0 else "nonpositive_anchor_log_unavailable",
    )
    if component == "original_total":
        out["total_error"] = out["continuum_f_lambda_rest_error"]
    elif scale_perturbation is not None:
        u = np.asarray(scale_perturbation, float)
        if u.shape != w.shape:
            raise ValueError("scale perturbation shape disagrees")
        u = u[order][good]
        if np.isfinite(u).all():
            delta = response @ u
            scov = np.outer(delta, delta)
            out.update(
                host_scale_parameter_covariance=scov.tolist(),
                host_scale_error=float(abs(delta[0])),
                host_scale_uncertainty_status="available_conditional_rank_one",
            )
    return out
