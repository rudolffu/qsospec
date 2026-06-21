"""Output products and diagnostic plots for the global qsospec workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
import re
from typing import Dict, Mapping, Optional, Tuple, Union
import unicodedata

import numpy as np
import pandas as pd

from ..global_result import WorkflowResult
from .. import complex_recipes, lines


@dataclass(frozen=True)
class GlobalQAPlotConfig:
    """Rendering options for the global continuum and emission-line QA plot."""

    figure_width: float = 10.5
    figure_height: float = 8.0
    max_zoom_panels: int = 4
    show_smoothed_data: bool = True
    smooth_original_spectrum_for_display: bool = False
    smoothing_window_pixels: int = 7
    show_residual_panel: bool = True
    show_fit_regions: bool = True
    unmodelled_windows: Tuple[Tuple[float, float, str], ...] = (
        (1170.0, 1275.0, "Lyα"),
    )
    show_host_context_in_overview: bool = True
    object_name: Optional[str] = None
    object_label: Optional[str] = None
    show_coordinates: bool = True
    output_format: str = "png"
    write_other_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.figure_width <= 0 or self.figure_height <= 0:
            raise ValueError("QA figure dimensions must be positive.")
        if self.max_zoom_panels < 1:
            raise ValueError("max_zoom_panels must be at least one.")
        if self.smoothing_window_pixels < 1 or self.smoothing_window_pixels % 2 == 0:
            raise ValueError("smoothing_window_pixels must be a positive odd integer.")
        for window in self.unmodelled_windows:
            if len(window) != 3 or not float(window[0]) < float(window[1]):
                raise ValueError(
                    "Each unmodelled window must be (lower, upper, label)."
                )
        if self.output_format not in ("png", "pdf", "both"):
            raise ValueError("output_format must be 'png', 'pdf', or 'both'.")


_SCIENCE_PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "STIX Two Text",
        "DejaVu Serif",
    ],
    "font.size": 12.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.0,
    "figure.titlesize": 16.0,
    "mathtext.fontset": "stix",
}


def _science_plot_style(function):
    """Render one plot without mutating process-global Matplotlib settings."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        import matplotlib.pyplot as plt

        with plt.rc_context(_SCIENCE_PLOT_STYLE):
            return function(*args, **kwargs)

    return wrapped


def _normalized_file_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "object"


def _qa_object_name(
    result: WorkflowResult,
    config: GlobalQAPlotConfig,
) -> str:
    value = config.object_name
    if value in (None, ""):
        value = (
            result.metadata.get("object_name")
            or result.metadata.get("object_id")
            or result.metadata.get("targetid")
            or result.spectrum.metadata.source
            or "object"
        )
    return _normalized_file_label(value)


def _plot_formats(config: GlobalQAPlotConfig) -> Tuple[str, ...]:
    if config.output_format == "both":
        return ("png", "pdf")
    return (config.output_format,)


def _plot_paths(
    output_dir: Path,
    stem: str,
    config: GlobalQAPlotConfig,
) -> Dict[str, Path]:
    return {
        file_format: output_dir / f"{stem}.{file_format}"
        for file_format in _plot_formats(config)
    }


def _save_figure(fig, paths: Mapping[str, Path]) -> Dict[str, str]:
    saved = {}
    for file_format, path in paths.items():
        save_kwargs = {"dpi": 160} if file_format == "png" else {}
        fig.savefig(path, **save_kwargs)
        saved[file_format] = str(path)
    return saved


def _coerce_plot_paths(
    paths: Union[Path, Mapping[str, Path]],
) -> Dict[str, Path]:
    if isinstance(paths, Path):
        file_format = paths.suffix.lower().lstrip(".") or "png"
        return {file_format: paths}
    return dict(paths)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _percentile_limits(values, percentiles: Tuple[float, float] = (1.0, 99.0), pad: float = 0.08):
    arrays = [np.ravel(np.asarray(value, dtype=float)) for value in values if value is not None]
    if not arrays:
        return None
    data = np.concatenate(arrays)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    lo, hi = np.percentile(data, percentiles)
    width = hi - lo
    margin = width * pad if width > 0 else max(abs(lo) * pad, 1.0)
    return float(lo - margin), float(hi + margin)


def _flux_display_scale(spectrum) -> float:
    """Return the display-only scale into DESI-style 1e-17 cgs units."""

    if getattr(spectrum, "flux_unit", None) != "cgs":
        return 1.0
    scale = getattr(spectrum, "flux_scale", None)
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return 1.0
    return float(scale) / 1.0e-17


@_science_plot_style
def _plot_global(
    result: WorkflowResult,
    paths: Union[Path, Mapping[str, Path]],
    config: GlobalQAPlotConfig,
    window: Optional[Tuple[float, float]] = None,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    paths = _coerce_plot_paths(paths)

    spectrum = result.spectrum
    continuum = result.continuum
    wave = spectrum.wave_rest
    display_scale = _flux_display_scale(spectrum)
    valid = spectrum.valid_mask
    if window is not None:
        valid &= (wave >= window[0]) & (wave <= window[1])
    fig, ax = plt.subplots(
        figsize=(config.figure_width, 3.8),
        constrained_layout=True,
    )
    ax.plot(
        wave[valid],
        display_scale * spectrum.flux[valid],
        color="0.45",
        lw=0.65,
        label="host-subtracted data",
    )
    ax.plot(
        wave[valid],
        display_scale * continuum.model[valid],
        color="black",
        lw=1.8,
        label="full continuum",
    )
    iron_label_used = False
    balmer_label_used = False
    for name, component in continuum.component_models.items():
        color, linestyle = _CONTINUUM_STYLES.get(name, ("0.5", "-"))
        if name in ("uv_iron", "optical_iron"):
            label = "iron" if not iron_label_used else "_nolegend_"
            iron_label_used = True
        elif name in ("balmer_bound_free", "balmer_high_order_series"):
            label = (
                "Balmer pseudo-continuum"
                if not balmer_label_used
                else "_nolegend_"
            )
            balmer_label_used = True
        else:
            label = name.replace("_", " ")
        ax.plot(
            wave[valid],
            display_scale * component[valid],
            lw=0.75,
            ls=linestyle,
            color=color,
            label=label,
        )
    used = continuum.clip_mask & valid
    if np.any(used):
        ax.scatter(
            wave[used],
            display_scale * spectrum.flux[used],
            s=5,
            color="k",
            alpha=0.25,
            label="fit pixels",
        )
    upper = _rounded_model_upper_limit(
        display_scale * continuum.model[valid]
    )
    if upper is not None:
        ax.set_ylim(0.0, upper)
    if window is not None:
        ax.set_xlim(*window)
    ax.set_xlabel(r"Rest wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(
        _flux_density_axis_label(spectrum),
        fontsize=13,
    )
    _configure_qa_axis(ax)
    ax.legend(fontsize=9, ncol=3, framealpha=0.72)
    saved = _save_figure(fig, paths)
    plt.close(fig)
    return saved


_COMPLEX_WINDOWS = {
    recipe.id: recipe.fit_window
    for recipe in complex_recipes.list_complexes()
    if recipe.fit_window[1] > recipe.fit_window[0] + 1.0
}
_COMPLEX_WINDOWS.update({"hbeta": (4600.0, 5120.0), "halpha": (6400.0, 6800.0)})
_TCC_COLORS = {
    "data": "#a8a19d",  # 鼠毛
    "data_smooth": "#686b68",  # 石涅
    "total_model": "#28292b",  # 元青
    "continuum": "#ee781f",  # 金红
    "host": "#785034",  # 驼褐
    "powerlaw": "#b03766",  # 魏红
    "feii": "#7b5aa3",  # 青莲
    "balmer_cont": "#db9c4b",  # 库金
    "broad_total": "#014a8f",  # 空青
    "broad_component": "#30aecf",  # 法蓝
    "narrow": "#007d62",  # 孔雀绿
    "outflow": "#d80835",  # 朱砂
    "line_marker": "#6f9bc6",  # 挼蓝
    "unmodelled_span": "#e4ecf0",  # 卵白
    "masked_span": "#ddd4d3",  # 葭灰
}
_SPECIES_COLORS = {
    "MgII": _TCC_COLORS["broad_component"],
    "Hb": _TCC_COLORS["broad_component"],
    "HeII": _TCC_COLORS["outflow"],
    "OIII": _TCC_COLORS["narrow"],
    "Ha": _TCC_COLORS["outflow"],
    "NII": _TCC_COLORS["narrow"],
    "SII": _TCC_COLORS["host"],
}
_IRON_STYLE = (_TCC_COLORS["feii"], ":")
_BALMER_STYLE = (_TCC_COLORS["balmer_cont"], "-.")
_CONTINUUM_STYLES = {
    "power_law": (_TCC_COLORS["powerlaw"], "--"),
    "uv_iron": _IRON_STYLE,
    "optical_iron": _IRON_STYLE,
    "balmer_bound_free": _BALMER_STYLE,
    "balmer_high_order_series": _BALMER_STYLE,
}
_COMBINED_BROAD_STYLE = {
    "color": _TCC_COLORS["broad_total"],
    "linestyle": "-",
    "linewidth": 1.6,
}
_BROAD_COMPONENT_STYLE = {
    "color": _TCC_COLORS["broad_component"],
    "linestyle": "-",
    "linewidth": 0.9,
}
_NARROW_STYLE = {
    "color": _TCC_COLORS["narrow"],
    "linestyle": "-",
    "linewidth": 1.15,
}
_WING_STYLE = {
    "color": _TCC_COLORS["outflow"],
    "linestyle": "-",
    "linewidth": 0.9,
}
_HOST_STYLE = {
    "color": _TCC_COLORS["host"],
    "linestyle": "-",
    "linewidth": 0.9,
}
_MAJOR_EMISSION_LINES = (
    (1215.67, r"Ly$\alpha$"),
    (1549.06, r"C IV"),
    (1908.73, r"C III]"),
    (2798.75, r"Mg II"),
    (4862.68, r"H$\beta$"),
    (5008.24, r"[O III] 5008"),
    (6564.61, r"H$\alpha$"),
)
_ZOOM_EMISSION_LINES = {
    "mgii": ((2798.75, r"Mg II"),),
    "hbeta": (
        (4862.68, r"H$\beta$"),
        (4960.30, r"[O III] 4960"),
        (5008.24, r"[O III] 5008"),
    ),
    "halpha": (
        (6549.85, r"[N II] 6550"),
        (6564.61, r"H$\alpha$"),
        (6585.28, r"[N II] 6585"),
        (6718.29, r"[S II] 6718"),
        (6732.67, r"[S II] 6733"),
    ),
}
for _recipe in complex_recipes.list_complexes():
    if _recipe.qa_labels:
        _ZOOM_EMISSION_LINES[_recipe.id] = tuple(
            (
                lines.get(line_id).vacuum_wavelength,
                lines.get(line_id).label,
            )
            for line_id in _recipe.qa_labels
        )
_ZOOM_PRIORITY = (
    "lya_nv",
    "hbeta_oiii",
    "hbeta",
    "mgii",
    "halpha_nii_sii",
    "halpha",
)
_LINE_MARKER_STYLE = {
    "color": _TCC_COLORS["line_marker"],
    "linestyle": ":",
    "linewidth": 0.8,
    "alpha": 0.65,
}


def _species_from_component(name: str) -> str:
    for species in ("MgII", "HeII", "OIII", "NII", "SII", "Hb", "Ha"):
        if name.startswith(species):
            return species
    return "Hb"


def _line_groups(name: str, fit) -> Tuple[Tuple[str, np.ndarray, str, str], ...]:
    broad_names = [key for key in fit.component_models if "broad" in key and "wing" not in key]
    groups = []
    if broad_names:
        broad_sum = sum(
            (fit.component_models[key] for key in broad_names),
            np.zeros_like(fit.model),
        )
        try:
            label = f"{complex_recipes.get(name).label} broad"
        except ValueError:
            label = f"{name} broad"
        species = _species_from_component(broad_names[0])
        groups.append((label, broad_sum, species, "broad"))
    for component_name, component in fit.component_models.items():
        if component_name in broad_names:
            continue
        kind = "wing" if "wing" in component_name else "narrow"
        groups.append(
            (
                component_name.replace("_", " "),
                component,
                _species_from_component(component_name),
                kind,
            )
        )
    return tuple(groups)


def _broad_component_names(fit) -> Tuple[str, ...]:
    return tuple(
        name
        for name in fit.component_models
        if "broad" in name and "wing" not in name
    )


def _combined_broad_profile(fit) -> np.ndarray:
    return sum(
        (fit.component_models[name] for name in _broad_component_names(fit)),
        np.zeros_like(fit.model),
    )


def _select_zoom_complexes(
    line_complexes: Mapping[str, object],
    max_zoom_panels: int,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    available = [
        name
        for name, fit in line_complexes.items()
        if name in _COMPLEX_WINDOWS and bool(getattr(fit, "success", False))
        and not (
            name == "oii_nev_neiii_hgamma"
            and not bool(
                getattr(fit, "metadata", {}).get(
                    "qa_all_lines_covered", False
                )
            )
        )
    ]
    priority_order = [
        name for name in _ZOOM_PRIORITY if name in available
    ] + sorted(
        (name for name in available if name not in _ZOOM_PRIORITY),
        key=lambda name: _COMPLEX_WINDOWS[name][0],
    )
    selected = set(priority_order[:max_zoom_panels])
    displayed = tuple(
        sorted(selected, key=lambda name: _COMPLEX_WINDOWS[name][0])
    )
    omitted = tuple(
        sorted(
            (name for name in available if name not in selected),
            key=lambda name: _COMPLEX_WINDOWS[name][0],
        )
    )
    return displayed, omitted


def _masked_running_median(
    values: np.ndarray,
    valid: np.ndarray,
    window_pixels: int,
) -> np.ndarray:
    series = pd.Series(
        np.where(np.asarray(valid, dtype=bool), np.asarray(values, dtype=float), np.nan)
    )
    smoothed = series.rolling(
        window=window_pixels,
        center=True,
        min_periods=1,
    ).median().to_numpy(copy=True)
    smoothed[~np.asarray(valid, dtype=bool)] = np.nan
    return smoothed


def _format_reduced_chi2(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "n/a"


def _qa_overview_title(
    result: WorkflowResult,
    plot_config: Optional[GlobalQAPlotConfig] = None,
) -> str:
    config = plot_config or GlobalQAPlotConfig()
    object_name = (
        config.object_name
        if config.object_name not in (None, "")
        else result.metadata.get("object_id")
    )
    parts = []
    if object_name not in (None, ""):
        object_label = config.object_label or "Object"
        parts.append(f"{object_label} {object_name}".strip())
    redshift = result.metadata.get("redshift")
    if redshift is not None and np.isfinite(redshift):
        parts.append(f"z = {float(redshift):.4f}")
    return "   ".join(parts)


def _configure_qa_axis(axis) -> None:
    axis.minorticks_on()
    axis.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=11,
    )
    axis.tick_params(which="major", length=4.0)
    axis.tick_params(which="minor", length=2.2)


def _annotate_emission_lines(axis, lines, *, y_fraction: float) -> Tuple[str, ...]:
    labels = []
    x_min, x_max = axis.get_xlim()
    for line_wave, line_label in lines:
        if not x_min <= line_wave <= x_max:
            continue
        label_y_fraction = (
            min(y_fraction, 0.68)
            if line_label.startswith("[")
            else y_fraction
        )
        axis.axvline(
            line_wave,
            ymin=0.88,
            ymax=1.0,
            zorder=0.5,
            **_LINE_MARKER_STYLE,
        )
        axis.text(
            line_wave,
            label_y_fraction,
            line_label,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=_TCC_COLORS["line_marker"],
            alpha=0.82,
        )
        labels.append(line_label)
    return tuple(labels)


def _flux_density_axis_label(spectrum_or_unit) -> str:
    flux_unit = getattr(spectrum_or_unit, "flux_unit", None)
    if flux_unit == "cgs":
        return (
            r"$F_\lambda\ "
            r"[10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,"
            r"\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}]$"
        )
    if flux_unit == "relative":
        return r"$F_\lambda$ [relative units]"
    normalized = str(spectrum_or_unit).lower().replace("angstrom", "aa")
    if (
        "1e-17" in normalized
        and "erg" in normalized
        and "cm" in normalized
        and "aa" in normalized
    ):
        return (
            r"$F_\lambda\ "
            r"[10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,"
            r"\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}]$"
        )
    if "relative" in normalized:
        return r"$F_\lambda$ [relative units]"
    return rf"$F_\lambda$ [{spectrum_or_unit}]"


def _rounded_model_upper_limit(model_values: np.ndarray) -> Optional[float]:
    """Return a compact rounded upper limit with headroom above the fitted model."""

    values = np.asarray(model_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    peak = float(np.max(values))
    if peak <= 0:
        return None
    target = 1.2 * peak
    magnitude = 10.0 ** np.floor(np.log10(target))
    normalized = target / magnitude
    for nice_value in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= nice_value:
            return float(nice_value * magnitude)
    return float(10.0 * magnitude)


def _full_line_model(result: WorkflowResult) -> np.ndarray:
    return sum(
        (
            complex_result.model
            for complex_result in result.line_complexes.values()
            if complex_result.success
        ),
        np.zeros_like(result.continuum.model),
    )


def _has_host_context(result: WorkflowResult) -> bool:
    if result.total_spectrum is None or result.host_model_on_quasar_grid is None:
        return False
    host = np.asarray(result.host_model_on_quasar_grid, dtype=float)
    return host.shape == result.spectrum.flux.shape and np.any(np.isfinite(host))


def _host_fraction_annotation(result: WorkflowResult) -> str:
    samples = result.metadata.get("continuum_samples", {})
    valid_wave = result.spectrum.wave_rest[result.spectrum.valid_mask]
    entries = []
    for wavelength in (3000, 5100):
        if (
            valid_wave.size == 0
            or wavelength < float(np.min(valid_wave))
            or wavelength > float(np.max(valid_wave))
        ):
            continue
        fraction = samples.get(f"fracHost_{wavelength}")
        if fraction is not None and np.isfinite(fraction):
            entries.append(
                rf"$f_{{\rm host}}({wavelength}\,\mathrm{{\AA}})="
                f"{100.0 * float(fraction):.1f}\\%$"
            )
    return "\n".join(entries)


def _final_fit_masks(
    result: WorkflowResult,
    config: GlobalQAPlotConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final fitted, unmodelled, and pPXF-masked pixels."""

    wave = result.spectrum.wave_rest
    valid = result.spectrum.valid_mask
    fitted = np.asarray(result.continuum.clip_mask, dtype=bool) & valid
    successful_line_fit = np.zeros_like(fitted)
    unmodelled = np.zeros_like(fitted)
    lya_fit_successful = bool(
        "lya_nv" in result.line_complexes
        and result.line_complexes["lya_nv"].success
    )
    for name, fit in result.line_complexes.items():
        if fit.success:
            line_mask = np.asarray(fit.fit_mask, dtype=bool) & valid
            fitted |= line_mask
            successful_line_fit |= line_mask
        elif name in _COMPLEX_WINDOWS:
            lo, hi = _COMPLEX_WINDOWS[name]
            unmodelled |= valid & (wave >= lo) & (wave <= hi)
    for lo, hi, _ in config.unmodelled_windows:
        if lya_fit_successful and lo <= 1215.67 <= hi:
            continue
        unmodelled |= valid & (wave >= float(lo)) & (wave <= float(hi))
    unmodelled &= ~fitted
    ppxf_masked = np.zeros_like(fitted)
    if (
        result.host_fit_mask is not None
        and result.host_emission_mask is not None
    ):
        host_fit_mask = np.asarray(result.host_fit_mask, dtype=bool)
        host_emission_mask = np.asarray(result.host_emission_mask, dtype=bool)
        if host_fit_mask.shape == fitted.shape and host_emission_mask.shape == fitted.shape:
            ppxf_masked = (
                valid & host_fit_mask & host_emission_mask & ~fitted
            )
    return fitted, unmodelled, ppxf_masked


def _mask_intervals(
    wave: np.ndarray,
    mask: np.ndarray,
) -> Tuple[Tuple[float, float], ...]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, breaks)
    return tuple(
        (float(wave[group[0]]), float(wave[group[-1]]))
        for group in groups
        if group.size
    )


def _shade_mask_regions(
    axis,
    wave: np.ndarray,
    unmodelled: np.ndarray,
    ppxf_masked: np.ndarray,
    *,
    labels: bool,
) -> None:
    for index, (lo, hi) in enumerate(_mask_intervals(wave, ppxf_masked)):
        axis.axvspan(
            lo,
            hi,
            facecolor=_TCC_COLORS["masked_span"],
            alpha=0.42,
            linewidth=0.0,
            zorder=-10,
            label="masked in pPXF host fit" if labels and index == 0 else "_nolegend_",
        )
    for index, (lo, hi) in enumerate(_mask_intervals(wave, unmodelled)):
        axis.axvspan(
            lo,
            hi,
            facecolor=_TCC_COLORS["unmodelled_span"],
            alpha=0.45,
            hatch="////",
            edgecolor=_TCC_COLORS["masked_span"],
            linewidth=0.0,
            zorder=-9,
            label="not fitted / not modelled" if labels and index == 0 else "_nolegend_",
        )


def _deduplicated_legend_items(axis):
    items = {}
    for handle, label in zip(*axis.get_legend_handles_labels()):
        if label and label != "_nolegend_" and label not in items:
            items[label] = handle
    return list(items.values()), list(items)


@_science_plot_style
def _plot_host_context(
    result: WorkflowResult,
    paths: Union[Path, Mapping[str, Path]],
    *,
    config: GlobalQAPlotConfig,
) -> Dict[str, str]:
    """Plot the original spectrum, host decomposition, and final AGN model."""

    import matplotlib.pyplot as plt
    paths = _coerce_plot_paths(paths)

    if not _has_host_context(result):
        raise ValueError("A total spectrum and finite host model are required.")

    spectrum = result.spectrum
    total_spectrum = result.total_spectrum
    wave = spectrum.wave_rest
    display_scale = _flux_display_scale(spectrum)
    host = np.asarray(result.host_model_on_quasar_grid, dtype=float)
    line_model = _full_line_model(result)
    agn_model = result.continuum.model + line_model
    reconstructed_total = host + agn_model
    valid_fit = spectrum.valid_mask & np.isfinite(host)
    valid_total = (
        total_spectrum.valid_mask
        & np.isfinite(host)
        & np.isfinite(reconstructed_total)
    )
    smoothing_effective = bool(
        config.smooth_original_spectrum_for_display
        and wave.size > 4000
    )
    displayed_total_flux_native = (
        _masked_running_median(
            total_spectrum.flux,
            valid_total,
            config.smoothing_window_pixels,
        )
        if smoothing_effective
        else total_spectrum.flux
    )
    displayed_total_flux = display_scale * displayed_total_flux_native
    reconstructed_total_display = display_scale * reconstructed_total
    host_display = display_scale * host
    fit_flux_display = display_scale * spectrum.flux
    agn_model_display = display_scale * agn_model
    original_label = (
        "original spectrum\nsmoothed for display"
        if smoothing_effective
        else "original spectrum"
    )
    valid_wave = wave[valid_total | valid_fit]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(config.figure_width, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )
    top_axis, bottom_axis = axes

    top_axis.plot(
        wave[valid_total],
        displayed_total_flux[valid_total],
        color="0.48",
        lw=0.65,
        label=original_label,
    )
    top_axis.plot(
        wave[valid_total],
        reconstructed_total_display[valid_total],
        color="black",
        lw=1.7,
        label="host + final AGN model",
    )
    top_axis.plot(
        wave[valid_total],
        host_display[valid_total],
        **_HOST_STYLE,
        label="host galaxy",
    )
    top_upper = _rounded_model_upper_limit(
        reconstructed_total_display[valid_total]
    )
    if top_upper is not None:
        top_axis.set_ylim(0.0, top_upper)
    top_axis.set_ylabel(
        _flux_density_axis_label(spectrum),
        fontsize=13,
    )
    source_title = _qa_overview_title(result, config)
    top_axis.set_title(
        "Host decomposition and final-model context"
        + (f" — {source_title}" if source_title else ""),
        fontsize=12,
    )
    fraction_text = _host_fraction_annotation(result)
    if fraction_text:
        top_axis.text(
            0.01,
            0.96,
            fraction_text,
            transform=top_axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.72,
            },
        )
    top_axis.legend(
        fontsize=9,
        ncol=3,
        loc="best",
        framealpha=0.72,
        borderpad=0.35,
    )

    bottom_axis.plot(
        wave[valid_fit],
        fit_flux_display[valid_fit],
        color="0.48",
        lw=0.65,
        label="host-subtracted spectrum",
    )
    bottom_axis.plot(
        wave[valid_fit],
        agn_model_display[valid_fit],
        color="black",
        lw=1.7,
        label="final AGN + emission-line model",
    )
    bottom_upper = _rounded_model_upper_limit(
        agn_model_display[valid_fit]
    )
    if bottom_upper is not None:
        bottom_axis.set_ylim(0.0, bottom_upper)
    bottom_axis.set_ylabel(
        _flux_density_axis_label(spectrum),
        fontsize=13,
    )
    bottom_axis.set_xlabel(
        r"Rest wavelength [$\mathrm{\AA}$]",
        fontsize=13,
    )
    bottom_axis.legend(
        fontsize=9,
        ncol=2,
        loc="best",
        framealpha=0.72,
        borderpad=0.35,
    )

    if valid_wave.size:
        x_limits = (float(valid_wave.min()), float(valid_wave.max()))
        top_axis.set_xlim(*x_limits)
        result.metadata["host_context_xlim"] = list(x_limits)
    for axis in axes:
        _configure_qa_axis(axis)

    result.metadata["host_context_figure_size_inches"] = [
        float(config.figure_width),
        5.2,
    ]
    result.metadata["host_context_ymin"] = 0.0
    result.metadata["host_context_model_upper_limits"] = {
        "original_plus_host": top_upper,
        "host_subtracted": bottom_upper,
    }
    result.metadata["host_context_fraction_annotation"] = fraction_text
    result.metadata["host_context_original_spectrum_smoothed_for_display"] = (
        smoothing_effective
    )
    result.metadata["host_context_smoothing_requested"] = bool(
        config.smooth_original_spectrum_for_display
    )
    result.metadata["host_context_smoothing_effective"] = smoothing_effective
    result.metadata["host_context_flux_display_scale"] = display_scale
    saved = _save_figure(fig, paths)
    plt.close(fig)
    return saved


@_science_plot_style
def _plot_qa(
    result: WorkflowResult,
    paths: Optional[Union[Path, Mapping[str, Path]]] = None,
    plot_config: Optional[GlobalQAPlotConfig] = None,
    *,
    return_figure: bool = False,
):
    import matplotlib.pyplot as plt
    resolved_paths = None if paths is None else _coerce_plot_paths(paths)

    config = plot_config or GlobalQAPlotConfig()
    available, omitted = _select_zoom_complexes(
        result.line_complexes,
        config.max_zoom_panels,
    )
    result.metadata["qa_panel_count"] = (
        1 + int(config.show_residual_panel) + len(available)
    )
    result.metadata["qa_percentiles"] = [1.0, 99.8]
    result.metadata["qa_layout"] = (
        "overview_residual_complexes"
        if config.show_residual_panel
        else "overview_complexes"
    )
    result.metadata["qa_figure_size_inches"] = [
        float(config.figure_width),
        float(config.figure_height),
    ]
    result.metadata["qa_max_zoom_panels"] = int(config.max_zoom_panels)
    result.metadata["qa_displayed_complexes"] = list(available)
    result.metadata["qa_omitted_complexes"] = list(omitted)
    result.metadata["qa_smoothed_data_requested"] = bool(
        config.show_smoothed_data
    )
    result.metadata["qa_original_spectrum_smoothed_for_display"] = bool(
        config.smooth_original_spectrum_for_display
    )
    result.metadata["qa_show_fit_regions"] = bool(config.show_fit_regions)
    result.metadata["qa_show_residual_panel"] = bool(
        config.show_residual_panel
    )
    result.metadata["qa_unmodelled_windows"] = [
        [float(lo), float(hi), str(label)]
        for lo, hi, label in config.unmodelled_windows
    ]
    result.metadata["qa_plot_style"] = "qsospec_science_serif"
    result.metadata["qa_plot_style_rc"] = dict(_SCIENCE_PLOT_STYLE)
    result.metadata["qa_smoothing_window_pixels"] = int(
        config.smoothing_window_pixels
    )
    result.metadata["qa_tick_direction"] = "in"
    result.metadata["qa_minor_ticks"] = True
    result.metadata["qa_zoom_model_upper_limits"] = {}
    result.metadata["qa_zoom_ymin"] = {}
    result.metadata["qa_zoom_titles"] = {}
    result.metadata["qa_zoom_line_labels"] = {}
    ncols = max(len(available), 1)
    fig = plt.figure(
        figsize=(config.figure_width, config.figure_height),
        constrained_layout=True,
    )
    if config.show_residual_panel:
        grid = fig.add_gridspec(
            3,
            ncols,
            height_ratios=(1.0, 0.28, 0.78),
        )
        overview_axis = fig.add_subplot(grid[0, :])
        residual_axis = fig.add_subplot(
            grid[1, :],
            sharex=overview_axis,
        )
        zoom_row = 2
    else:
        grid = fig.add_gridspec(2, ncols, height_ratios=(1.0, 0.78))
        overview_axis = fig.add_subplot(grid[0, :])
        residual_axis = None
        zoom_row = 1
    zoom_axes = [
        fig.add_subplot(grid[zoom_row, index])
        for index in range(ncols)
    ]
    spectrum = result.spectrum
    wave = spectrum.wave_rest
    display_scale = _flux_display_scale(spectrum)
    valid = spectrum.valid_mask
    line_model = _full_line_model(result)
    full_model = result.continuum.model + line_model
    fitted_mask, unmodelled_mask, ppxf_masked = _final_fit_masks(
        result, config
    )
    if not config.show_fit_regions:
        unmodelled_mask[:] = False
        ppxf_masked[:] = False
    result.metadata["qa_n_fitted_pixels"] = int(np.count_nonzero(fitted_mask))
    result.metadata["qa_n_unmodelled_pixels"] = int(
        np.count_nonzero(unmodelled_mask)
    )
    result.metadata["qa_n_ppxf_masked_pixels"] = int(
        np.count_nonzero(ppxf_masked)
    )
    result.metadata["qa_host_mask_provenance"] = result.metadata.get(
        "host_mask_provenance",
        "exact"
        if result.host_fit_mask is not None
        and result.host_emission_mask is not None
        else "unavailable",
    )
    host_overview = bool(
        config.show_host_context_in_overview and _has_host_context(result)
    )
    result.metadata["qa_host_context_overview_requested"] = bool(
        config.show_host_context_in_overview
    )
    result.metadata["qa_host_context_overview_used"] = host_overview
    host_model = (
        np.asarray(result.host_model_on_quasar_grid, dtype=float)
        if host_overview
        else np.zeros_like(full_model)
    )
    overview_data_native = (
        np.asarray(result.total_spectrum.flux, dtype=float)
        if host_overview
        else spectrum.flux
    )
    overview_full_model_native = full_model + host_model
    overview_data = display_scale * overview_data_native
    overview_full_model = display_scale * overview_full_model_native
    full_model_display = display_scale * full_model
    fit_data_display = display_scale * spectrum.flux
    fit_error_display = display_scale * spectrum.err
    host_model_display = display_scale * host_model
    overview_valid = valid.copy()
    if host_overview:
        overview_valid &= (
            result.total_spectrum.valid_mask
            & np.isfinite(host_model)
            & np.isfinite(overview_data_native)
        )
    smoothing_requested = bool(
        config.show_smoothed_data
        or config.smooth_original_spectrum_for_display
    )
    smoothing_effective = bool(smoothing_requested and wave.size > 4000)
    smoothed_fit_data_native = (
        _masked_running_median(
            spectrum.flux,
            valid,
            config.smoothing_window_pixels,
        )
        if smoothing_effective
        else None
    )
    smoothed_overview_data_native = (
        _masked_running_median(
            overview_data_native,
            overview_valid,
            config.smoothing_window_pixels,
        )
        if smoothing_effective
        else None
    )
    smoothed_fit_data = (
        display_scale * smoothed_fit_data_native
        if smoothed_fit_data_native is not None
        else None
    )
    smoothed_overview_data = (
        display_scale * smoothed_overview_data_native
        if smoothed_overview_data_native is not None
        else None
    )
    replace_original_with_smoothed = bool(
        config.smooth_original_spectrum_for_display
        and smoothing_effective
    )
    show_smoothed_trace = bool(
        config.show_smoothed_data and smoothing_effective
    )
    result.metadata["qa_original_spectrum_smoothed_used"] = (
        replace_original_with_smoothed
    )
    result.metadata["qa_smoothed_data"] = show_smoothed_trace
    result.metadata["qa_smoothing_requested"] = smoothing_requested
    result.metadata["qa_smoothing_effective"] = smoothing_effective
    result.metadata["qa_smoothing_suppressed_short_spectrum"] = bool(
        smoothing_requested and wave.size <= 4000
    )
    result.metadata["qa_flux_display_scale"] = display_scale
    result.metadata["qa_flux_display_unit"] = (
        "1e-17 cgs" if spectrum.flux_unit == "cgs" else "relative"
    )

    def plot_observed(
        ax,
        panel_mask,
        *,
        data_values,
        smoothed_values,
        labels,
    ):
        if not replace_original_with_smoothed:
            ax.plot(
                wave[panel_mask],
                data_values[panel_mask],
                color=_TCC_COLORS["data"],
                lw=0.8,
                alpha=0.8,
                zorder=1,
                label="observed spectrum" if labels else "_nolegend_",
            )
        if smoothed_values is not None and (
            show_smoothed_trace or replace_original_with_smoothed
        ):
            ax.plot(
                wave[panel_mask],
                smoothed_values[panel_mask],
                color=_TCC_COLORS["data_smooth"],
                lw=0.9,
                alpha=0.9,
                zorder=2,
                label=(
                    "observed spectrum (smoothed for display)"
                    if labels else "_nolegend_"
                ),
            )

    def plot_continuum_components(ax, panel_mask, *, labels):
        iron_label_used = False
        balmer_label_used = False
        for component_name, component in result.continuum.component_models.items():
            if not np.any(np.abs(component[panel_mask]) > 0):
                continue
            color, linestyle = _CONTINUUM_STYLES.get(
                component_name, ("0.5", ":")
            )
            if component_name in ("uv_iron", "optical_iron"):
                label = (
                    "Fe II"
                    if labels and not iron_label_used
                    else "_nolegend_"
                )
                iron_label_used = True
            elif component_name in (
                "balmer_bound_free",
                "balmer_high_order_series",
            ):
                label = (
                    "Balmer pseudo-continuum"
                    if labels and not balmer_label_used
                    else "_nolegend_"
                )
                balmer_label_used = True
            else:
                label = (
                    component_name.replace("_", " ")
                    if labels else "_nolegend_"
                )
            ax.plot(
                wave[panel_mask],
                display_scale * component[panel_mask],
                color=color,
                ls=linestyle,
                lw=0.8,
                alpha=0.9,
                label=label,
                zorder=3,
            )

    plot_observed(
        overview_axis,
        overview_valid,
        data_values=overview_data,
        smoothed_values=smoothed_overview_data,
        labels=True,
    )
    overview_model_plot = np.where(
        overview_valid,
        overview_full_model,
        np.nan,
    )
    overview_axis.plot(
        wave,
        overview_model_plot,
        color=_TCC_COLORS["total_model"],
        lw=1.8,
        label="total model",
        zorder=6,
    )
    if host_overview:
        overview_axis.plot(
            wave[overview_valid],
            host_model_display[overview_valid],
            label="host galaxy",
            zorder=3,
            **_HOST_STYLE,
        )
    plot_continuum_components(
        overview_axis,
        overview_valid,
        labels=True,
    )

    overview_title = _qa_overview_title(result, config)
    result.metadata["qa_overview_title"] = overview_title
    overview_axis.set_title(overview_title, fontsize=12)
    overview_axis.set_ylabel(
        _flux_density_axis_label(spectrum),
        fontsize=13,
    )
    _configure_qa_axis(overview_axis)
    broad_label_used = False
    narrow_label_used = False
    wing_label_used = False
    for complex_name, fit in result.line_complexes.items():
        if not fit.success:
            continue
        fit = result.line_complexes[complex_name]
        component_mask = valid & np.asarray(fit.fit_mask, dtype=bool)
        combined = _combined_broad_profile(fit)
        overview_axis.plot(
            wave[component_mask],
            display_scale * combined[component_mask],
            label="broad-line model" if not broad_label_used else "_nolegend_",
            zorder=5,
            **_COMBINED_BROAD_STYLE,
        )
        broad_label_used |= bool(np.any(combined[component_mask] != 0))
        for label, component, species, kind in _line_groups(complex_name, fit):
            if kind == "broad":
                continue
            style = _WING_STYLE if kind == "wing" else _NARROW_STYLE
            if kind == "wing":
                legend_label = "outflow wing" if not wing_label_used else "_nolegend_"
                wing_label_used = True
            else:
                legend_label = "narrow-line model" if not narrow_label_used else "_nolegend_"
                narrow_label_used = True
            overview_axis.plot(
                wave[component_mask],
                display_scale * component[component_mask],
                label=legend_label,
                zorder=5,
                **style,
            )
    if config.show_fit_regions:
        _shade_mask_regions(
            overview_axis,
            wave,
            unmodelled_mask,
            ppxf_masked,
            labels=True,
        )
    valid_wave = wave[overview_valid]
    if valid_wave.size:
        overview_axis.set_xlim(float(valid_wave.min()), float(valid_wave.max()))
        result.metadata["qa_overview_xlim"] = [
            float(valid_wave.min()),
            float(valid_wave.max()),
        ]
        result.metadata["qa_major_emission_line_labels"] = list(
            _annotate_emission_lines(
                overview_axis,
                _MAJOR_EMISSION_LINES,
                y_fraction=0.82,
            )
        )
    valid_overview_wave = wave[overview_valid]
    lya_in_coverage = bool(
        valid_overview_wave.size
        and float(np.min(valid_overview_wave)) <= 1215.67
        and float(np.max(valid_overview_wave)) >= 1215.67
    )
    lya_model_fitted = any(
        (
            "lya" in str(name).lower()
            or "lyalpha" in str(name).lower()
        )
        and fit.success
        for name, fit in result.line_complexes.items()
    )
    if lya_in_coverage and not lya_model_fitted:
        data_limits = _percentile_limits(
            [overview_data[overview_valid]],
            percentiles=(1.0, 99.8),
        )
        overview_upper = data_limits[1] if data_limits is not None else None
        result.metadata["qa_overview_upper_policy"] = "data_only_percentile"
        result.metadata["qa_overview_upper_percentile"] = 99.8
    else:
        overview_upper = _rounded_model_upper_limit(
            overview_full_model[overview_valid]
        )
        result.metadata["qa_overview_upper_policy"] = "rounded_model"
        result.metadata["qa_overview_upper_percentile"] = None
    result.metadata["qa_lya_in_valid_coverage"] = lya_in_coverage
    result.metadata["qa_lya_model_fitted"] = lya_model_fitted
    if overview_upper is not None:
        overview_axis.set_ylim(
            0.0,
            max(overview_upper, 0.0),
        )
        result.metadata["qa_overview_ymin"] = 0.0
        result.metadata["qa_overview_model_upper_limit"] = overview_upper
        clipped = overview_valid & np.isfinite(overview_data) & (
            overview_data > overview_upper
        )
        if np.any(clipped):
            overview_axis.scatter(
                wave[clipped],
                np.full(np.count_nonzero(clipped), 0.985 * overview_upper),
                marker="^",
                s=12,
                facecolor=_TCC_COLORS["data_smooth"],
                edgecolor="none",
                alpha=0.7,
                zorder=8,
            )
        result.metadata["qa_overview_clipped_peak_count"] = int(
            np.count_nonzero(clipped)
        )
    host_state = (
        "decomposed with pPXF"
        if result.host_decomp_enabled
        or result.metadata.get("host_decomp_enabled", False)
        else "not decomposed"
    )
    overview_annotation_lines = [
        rf"$\chi^2_\nu(\mathrm{{cont., fitted\ pixels}})="
        f"{_format_reduced_chi2(result.continuum.reduced_chi2)}$",
        f"Host: {host_state}",
    ]
    lya_status = result.metadata.get("lya_coverage_status")
    if lya_status in ("red_side_only", "edge_truncated"):
        overview_annotation_lines.append(
            "Lyα: "
            + (
                "limited red-side fit"
                if lya_status == "red_side_only"
                else "edge-truncated; not fitted"
            )
        )
    if config.show_coordinates:
        ra = result.metadata.get("ra")
        dec = result.metadata.get("dec")
        if ra is not None and np.isfinite(ra):
            overview_annotation_lines.insert(0, f"RA = {float(ra):.5f}")
        if dec is not None and np.isfinite(dec):
            insertion = 1 if overview_annotation_lines[0].startswith("RA") else 0
            overview_annotation_lines.insert(
                insertion, f"Dec = {float(dec):+.5f}"
            )
    host_fraction_annotation = _host_fraction_annotation(result)
    if host_fraction_annotation:
        overview_annotation_lines.extend(host_fraction_annotation.splitlines())
    overview_annotation = "\n".join(overview_annotation_lines)
    overview_axis.text(
        0.01,
        0.97,
        overview_annotation,
        transform=overview_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.72,
        },
    )
    result.metadata["qa_overview_annotation"] = {
        "continuum_reduced_chi2": float(result.continuum.reduced_chi2),
        "host_state": host_state,
        "host_fractions": host_fraction_annotation,
        "ra": result.metadata.get("ra") if config.show_coordinates else None,
        "dec": result.metadata.get("dec") if config.show_coordinates else None,
    }
    handles, labels = _deduplicated_legend_items(overview_axis)
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=10,
            ncol=min(5, len(handles)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            framealpha=0.82,
            borderpad=0.35,
            handlelength=2.4,
        )
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.0, 1.0, 0.91))

    if residual_axis is not None:
        residual_mask = (
            fitted_mask
            & overview_valid
            & np.isfinite(overview_data)
            & np.isfinite(overview_full_model)
            & np.isfinite(spectrum.err)
            & (spectrum.err > 0)
        )
        normalized_residual = np.full_like(overview_data, np.nan, dtype=float)
        normalized_residual[residual_mask] = (
            overview_data[residual_mask]
            - overview_full_model[residual_mask]
        ) / fit_error_display[residual_mask]
        residual_axis.plot(
            wave,
            normalized_residual,
            color=_TCC_COLORS["total_model"],
            lw=0.55,
        )
        for value, linestyle, alpha in (
            (0.0, "-", 0.7),
            (3.0, "--", 0.45),
            (-3.0, "--", 0.45),
        ):
            residual_axis.axhline(
                value,
                color=_TCC_COLORS["data_smooth"],
                lw=0.7,
                ls=linestyle,
                alpha=alpha,
                zorder=0,
            )
        if config.show_fit_regions:
            _shade_mask_regions(
                residual_axis,
                wave,
                unmodelled_mask,
                ppxf_masked,
                labels=False,
            )
        residual_axis.set_ylim(-5.5, 5.5)
        residual_axis.set_ylabel(
            r"$\Delta/\sigma$",
            fontsize=11,
        )
        residual_axis.tick_params(axis="x", labelbottom=False)
        _configure_qa_axis(residual_axis)
        result.metadata["qa_residual_definition"] = "(data-model)/sigma"
        result.metadata["qa_residual_reference_lines"] = [0.0, -3.0, 3.0]
        result.metadata["qa_n_residual_pixels"] = int(
            np.count_nonzero(residual_mask)
        )

    for zoom_index, (axis, complex_name) in enumerate(zip(zoom_axes, available)):
        lo, hi = _COMPLEX_WINDOWS[complex_name]
        panel_mask = valid & (wave >= lo) & (wave <= hi)
        try:
            title = complex_recipes.get(complex_name).label
        except ValueError:
            title = complex_name
        fit = result.line_complexes[complex_name]
        title = (
            f"{title}  |  "
            rf"$\chi^2_\nu={_format_reduced_chi2(fit.reduced_chi2)}$"
        )
        if complex_name == "lya_nv":
            coverage_status = fit.metadata.get("lya_coverage_status")
            if coverage_status == "red_side_only":
                title = (
                    r"Ly$\alpha$ / N V"
                    + "\n"
                    + rf"$\chi^2_\nu={_format_reduced_chi2(fit.reduced_chi2)}$"
                    + "; limited"
                )
            elif not fit.metadata.get("lya_fit_reliable", False):
                title = (
                    r"Ly$\alpha$ / N V"
                    + "\n"
                    + rf"$\chi^2_\nu={_format_reduced_chi2(fit.reduced_chi2)}$"
                    + "; unreliable"
                )
        result.metadata.setdefault("qa_zoom_titles", {})[complex_name] = title
        plot_observed(
            axis,
            panel_mask,
            data_values=fit_data_display,
            smoothed_values=smoothed_fit_data,
            labels=False,
        )
        axis.plot(
            wave,
            np.where(panel_mask & valid, full_model_display, np.nan),
            color=_TCC_COLORS["total_model"],
            lw=1.8,
            label="_nolegend_",
            zorder=6,
        )
        plot_continuum_components(axis, panel_mask, labels=False)
        axis.set_title(title, fontsize=12)
        _configure_qa_axis(axis)
        combined = _combined_broad_profile(fit)
        axis.plot(
            wave[panel_mask],
            display_scale * combined[panel_mask],
            label="broad-line model",
            **_COMBINED_BROAD_STYLE,
        )
        broad_component_label_used = False
        broad_names = set(_broad_component_names(fit))
        for component_name, component in fit.component_models.items():
            if component_name in broad_names:
                if complex_name == "lya_nv":
                    component_label = (
                        "Lyα component"
                        if component_name.lower().startswith("lya")
                        else "N V component"
                    )
                else:
                    component_label = (
                        "broad components"
                        if not broad_component_label_used
                        else "_nolegend_"
                    )
                axis.plot(
                    wave[panel_mask],
                    display_scale * component[panel_mask],
                    label=component_label,
                    **_BROAD_COMPONENT_STYLE,
                )
                broad_component_label_used = True
                continue
            kind = "wing" if "wing" in component_name else "narrow"
            style = _WING_STYLE if kind == "wing" else _NARROW_STYLE
            axis.plot(
                wave[panel_mask],
                display_scale * component[panel_mask],
                label=(
                    "outflow wing"
                    if kind == "wing"
                    else "narrow lines"
                ),
                **style,
            )
        if complex_name == "lya_nv":
            excluded = (
                np.asarray(fit.excluded_mask, dtype=bool)
                if fit.excluded_mask is not None
                else np.zeros_like(panel_mask)
            )
            excluded &= panel_mask
            if np.any(excluded):
                axis.scatter(
                    wave[excluded],
                    fit_data_display[excluded],
                    marker="x",
                    s=18,
                    linewidths=0.8,
                    color=_TCC_COLORS["outflow"],
                    label="masked absorption",
                    zorder=8,
                )
        limits = axis.get_ylim()
        zoom_upper = _rounded_model_upper_limit(
            full_model_display[panel_mask]
        )
        axis.set_ylim(0.0, max(zoom_upper if zoom_upper is not None else limits[1], 0.0))
        result.metadata.setdefault("qa_zoom_model_upper_limits", {})[
            complex_name
        ] = zoom_upper
        result.metadata.setdefault("qa_zoom_ymin", {})[complex_name] = 0.0
        axis.set_xlim(lo, hi)
        axis.axhline(0.0, color="0.55", lw=0.65, zorder=0)
        result.metadata.setdefault("qa_zoom_line_labels", {})[complex_name] = list(
            _annotate_emission_lines(
                axis,
                _ZOOM_EMISSION_LINES.get(complex_name, ()),
                y_fraction=0.82,
            )
        )
        handles, labels = _deduplicated_legend_items(axis)
        allowed = {
            "broad components",
            "narrow lines",
            "outflow wing",
            "Lyα component",
            "N V component",
            "masked absorption",
        }
        local_items = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label in allowed
        ]
        if local_items:
            axis.legend(
                [item[0] for item in local_items],
                [item[1] for item in local_items],
                fontsize=8,
                loc="best",
                framealpha=0.72,
                borderpad=0.35,
            )
        if zoom_index == 0:
            axis.set_ylabel(
                _flux_density_axis_label(spectrum),
                fontsize=13,
            )
    if not available:
        zoom_axes[0].set_visible(False)
    fig.supxlabel(r"Rest wavelength [$\mathrm{\AA}$]", fontsize=13)
    result.metadata["qa_shared_axis_labels"] = False
    result.metadata["qa_y_label_policy"] = "overview_and_leftmost_zoom"
    if return_figure:
        return fig
    if resolved_paths is None:
        raise ValueError("QA output paths are required when saving a figure.")
    saved = _save_figure(fig, resolved_paths)
    plt.close(fig)
    return saved


def plot_qa_figure(
    result: WorkflowResult,
    plot_config: Optional[GlobalQAPlotConfig] = None,
):
    """Return an open Matplotlib QA figure without writing a file."""

    return _plot_qa(
        result,
        None,
        plot_config or GlobalQAPlotConfig(),
        return_figure=True,
    )


@_science_plot_style
def _plot_hbeta(
    result: WorkflowResult,
    paths: Union[Path, Mapping[str, Path]],
    config: GlobalQAPlotConfig,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    paths = _coerce_plot_paths(paths)

    fit = result.hbeta
    if fit is None:
        raise ValueError("Hβ diagnostic requested when Hβ was not fitted.")
    view = (fit.wave_rest >= 4600.0) & (fit.wave_rest <= 5120.0)
    display_scale = _flux_display_scale(result.spectrum)
    fig, ax = plt.subplots(
        figsize=(config.figure_width, 3.8),
        constrained_layout=True,
    )
    ax.plot(
        fit.wave_rest[view],
        display_scale * fit.flux_continuum_subtracted[view],
        color="0.45",
        lw=0.65,
        label="continuum-subtracted data",
    )
    ax.plot(
        fit.wave_rest[view],
        display_scale * fit.model[view],
        color="black",
        lw=1.8,
        label="full line model",
    )
    broad_label_used = False
    narrow_label_used = False
    wing_label_used = False
    for name, component in fit.component_models.items():
        if "broad" in name and "wing" not in name:
            style = _BROAD_COMPONENT_STYLE
            label = "broad components" if not broad_label_used else "_nolegend_"
            broad_label_used = True
        elif "wing" in name:
            style = _WING_STYLE
            label = "outflow wing" if not wing_label_used else "_nolegend_"
            wing_label_used = True
        else:
            style = _NARROW_STYLE
            label = "narrow line" if not narrow_label_used else "_nolegend_"
            narrow_label_used = True
        ax.plot(
            fit.wave_rest[view],
            display_scale * component[view],
            label=label,
            **style,
        )
    upper = _rounded_model_upper_limit(display_scale * fit.model[view])
    if upper is not None:
        ax.set_ylim(0.0, upper)
    ax.set_xlim(4600.0, 5120.0)
    ax.set_xlabel(r"Rest wavelength [$\mathrm{\AA}$]", fontsize=13)
    ax.set_ylabel(
        _flux_density_axis_label(result.spectrum),
        fontsize=13,
    )
    ax.set_title(
        "Hβ / [O III] diagnostic"
        + (
            f" — {_qa_overview_title(result, config)}"
            if _qa_overview_title(result, config)
            else ""
        ),
        fontsize=12,
    )
    _configure_qa_axis(ax)
    ax.legend(fontsize=9, ncol=3, framealpha=0.72)
    saved = _save_figure(fig, paths)
    plt.close(fig)
    return saved


def _measurement_row(fit) -> Dict[str, object]:
    row = {}
    row.update(fit.param_values)
    row.update({f"{key}_err": value for key, value in fit.param_errors.items()})
    row.update(fit.metrics)
    row.update({f"{key}_err": value for key, value in fit.metric_errors.items()})
    row.update(
        {
            "success": fit.success,
            "selected_model": fit.selected_model,
            "reduced_chi2": fit.reduced_chi2,
            "bic": fit.bic,
        }
    )
    return row


def _write_complex_products(out: Path, name: str, fit, files: Dict[str, str]) -> None:
    filenames = {
        "mgii": ("mgii_measurements.csv", "mgii_model.csv"),
        "hbeta": ("hbeta_oiii_measurements.csv", "hbeta_oiii_model.csv"),
        "hbeta_oiii": ("hbeta_oiii_measurements.csv", "hbeta_oiii_model.csv"),
        "halpha": ("halpha_nii_sii_measurements.csv", "halpha_nii_sii_model.csv"),
        "halpha_nii_sii": ("halpha_nii_sii_measurements.csv", "halpha_nii_sii_model.csv"),
    }
    measurement_name, model_name = filenames.get(
        name, (f"{name}_measurements.csv", f"{name}_model.csv")
    )
    measurement_path = out / measurement_name
    pd.DataFrame([_measurement_row(fit)]).to_csv(measurement_path, index=False)
    files[f"{name}_measurements_csv"] = str(measurement_path)
    legacy_name = {
        "hbeta_oiii": "hbeta",
        "halpha_nii_sii": "halpha",
    }.get(name)
    if legacy_name is not None:
        files[f"{legacy_name}_measurements_csv"] = str(measurement_path)

    if name in _COMPLEX_WINDOWS:
        lo, hi = _COMPLEX_WINDOWS[name]
    elif np.any(fit.fit_mask):
        lo = float(np.nanmin(fit.wave_rest[fit.fit_mask]))
        hi = float(np.nanmax(fit.wave_rest[fit.fit_mask]))
    else:
        lo, hi = float(fit.wave_rest.min()), float(fit.wave_rest.max())
    view = (fit.wave_rest >= lo) & (fit.wave_rest <= hi)
    model_grid = {
        "wave_rest": fit.wave_rest[view],
        "continuum_subtracted": fit.flux_continuum_subtracted[view],
        "err": fit.err[view],
        "model": fit.model[view],
        "fit_used": fit.fit_mask[view].astype(int),
    }
    for component_name, component in fit.component_models.items():
        model_grid[component_name] = component[view]
    model_path = out / model_name
    pd.DataFrame(model_grid).to_csv(model_path, index=False)
    files[f"{name}_model_csv"] = str(model_path)
    if legacy_name is not None:
        files[f"{legacy_name}_model_csv"] = str(model_path)


def write_global_line_products(
    result: WorkflowResult,
    output_dir: str,
    qa_plot_config: Optional[GlobalQAPlotConfig] = None,
) -> Dict[str, str]:
    """Write standard global-continuum and multi-complex products."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    plot_config = qa_plot_config or GlobalQAPlotConfig()
    file_object_name = _qa_object_name(result, plot_config)
    result.metadata["qa_file_object_name"] = file_object_name
    result.metadata["qa_output_format"] = plot_config.output_format
    result.metadata["write_other_diagnostics"] = bool(
        plot_config.write_other_diagnostics
    )

    summary_path = out / "qsospec_global_lines_summary.json"
    compatibility_summary_path = out / "qsospec_global_hbeta_summary.json"
    summary_payload = json.dumps(
        result.summary(), indent=2, sort_keys=True, default=_json_default
    )
    summary_path.write_text(
        summary_payload,
        encoding="utf-8",
    )
    compatibility_summary_path.write_text(
        summary_payload,
        encoding="utf-8",
    )
    files["summary_json"] = str(summary_path)
    files["compatibility_summary_json"] = str(compatibility_summary_path)

    continuum_row = {}
    continuum_row.update(result.continuum.param_values)
    continuum_row.update({f"{key}_err": value for key, value in result.continuum.param_errors.items()})
    continuum_row.update(result.metadata.get("continuum_samples", {}))
    for key in (
        "balmer_pseudocontinuum_implied_hbeta_flux_input",
        "balmer_pseudocontinuum_implied_hbeta_flux_cgs",
        "balmer_pseudocontinuum_fwhm_kms",
        "balmer_pseudocontinuum_velocity_kms",
        "balmer_pseudocontinuum_edge_flux_density_input",
        "balmer_pseudocontinuum_template_provenance",
        "balmer_pseudocontinuum_n_min",
        "balmer_pseudocontinuum_n_max",
        "balmer_pseudocontinuum_fwhm_source",
        "balmer_pseudocontinuum_fwhm_synced_to_hbeta",
        "balmer_pseudocontinuum_fwhm_warning_codes",
        "balmer_pseudocontinuum_fwhm_snr",
        "hbeta_sync_requested",
        "hbeta_sync_attempted",
        "hbeta_sync_converged",
        "hbeta_sync_iterations",
    ):
        if key in result.continuum.metadata:
            continuum_row[key] = result.continuum.metadata[key]
    continuum_row.update(
        {
            "success": result.continuum.success,
            "reduced_chi2": result.continuum.reduced_chi2,
            "balmer_template": result.continuum.metadata.get("balmer_template"),
        }
    )
    continuum_path = out / "global_continuum_measurements.csv"
    pd.DataFrame([continuum_row]).to_csv(continuum_path, index=False)
    files["continuum_measurements_csv"] = str(continuum_path)

    for complex_name, fit in result.line_complexes.items():
        _write_complex_products(out, complex_name, fit, files)

    spectrum = result.spectrum
    grid = {
        "wave_obs": spectrum.wave_obs,
        "wave_rest": spectrum.wave_rest,
        "flux_fit_input": spectrum.flux,
        "err": spectrum.err,
        "global_continuum": result.continuum.model,
        "continuum_subtracted": spectrum.flux - result.continuum.model,
        "fit_used_continuum": result.continuum.clip_mask.astype(int),
        "full_model": result.continuum.model + _full_line_model(result),
    }
    if result.total_spectrum is not None:
        grid["flux_total_before_host"] = result.total_spectrum.flux
    if result.host_model_on_quasar_grid is not None:
        grid["ppxf_host_model"] = result.host_model_on_quasar_grid
    for name, component in result.continuum.component_models.items():
        grid[f"continuum_{name}"] = component
    for complex_name, fit in result.line_complexes.items():
        grid[f"{complex_name}_model"] = fit.model
        grid[f"fit_used_{complex_name}"] = fit.fit_mask.astype(int)
        for component_name, component in fit.component_models.items():
            grid[f"line_{complex_name}_{component_name}"] = component
    grid_path = out / "qsospec_global_lines_full_grid.csv"
    pd.DataFrame(grid).to_csv(grid_path, index=False)
    files["full_grid_csv"] = str(grid_path)
    compatibility_grid_path = out / "qsospec_global_hbeta_full_grid.csv"
    pd.DataFrame(grid).to_csv(compatibility_grid_path, index=False)
    files["compatibility_full_grid_csv"] = str(compatibility_grid_path)

    main_qa_paths = _plot_paths(
        out,
        f"main_qa_{file_object_name}",
        plot_config,
    )
    main_qa_files = _plot_qa(
        result,
        main_qa_paths,
        plot_config,
    )
    for file_format, path in main_qa_files.items():
        files[f"main_qa_{file_format}"] = path
    primary_main_qa = main_qa_files.get(
        "png",
        next(iter(main_qa_files.values())),
    )
    files["main_qa"] = primary_main_qa
    files["global_plot"] = primary_main_qa
    files["qa_plot"] = primary_main_qa

    if plot_config.write_other_diagnostics and _has_host_context(result):
        host_context_files = _plot_host_context(
            result,
            _plot_paths(
                out,
                "diagnostic_global_host_context",
                plot_config,
            ),
            config=plot_config,
        )
        for file_format, path in host_context_files.items():
            files[f"host_context_plot_{file_format}"] = path
        files["host_context_plot"] = host_context_files.get(
            "png",
            next(iter(host_context_files.values())),
        )
        result.metadata["host_context_plot_created"] = True
    else:
        result.metadata["host_context_plot_created"] = False

    if plot_config.write_other_diagnostics:
        balmer_edge_files = _plot_global(
            result,
            _plot_paths(out, "diagnostic_balmer_edge", plot_config),
            plot_config,
            window=(3300.0, 4300.0),
        )
        for file_format, path in balmer_edge_files.items():
            files[f"balmer_edge_plot_{file_format}"] = path
        files["balmer_edge_plot"] = balmer_edge_files.get(
            "png",
            next(iter(balmer_edge_files.values())),
        )
        if result.hbeta is not None:
            hbeta_files = _plot_hbeta(
                result,
                _plot_paths(
                    out,
                    "diagnostic_hbeta_oiii",
                    plot_config,
                ),
                plot_config,
            )
            for file_format, path in hbeta_files.items():
                files[f"hbeta_plot_{file_format}"] = path
            files["hbeta_plot"] = hbeta_files.get(
                "png",
                next(iter(hbeta_files.values())),
            )
    result.output_files.update(files)
    summary_payload = json.dumps(
        result.summary(), indent=2, sort_keys=True, default=_json_default
    )
    summary_path.write_text(summary_payload, encoding="utf-8")
    compatibility_summary_path.write_text(summary_payload, encoding="utf-8")
    return files


def write_global_hbeta_products(
    result: WorkflowResult,
    output_dir: str,
    qa_plot_config: Optional[GlobalQAPlotConfig] = None,
) -> Dict[str, str]:
    """Compatibility wrapper retaining H-beta summary/full-grid paths."""

    files = write_global_line_products(result, output_dir, qa_plot_config)
    files["generic_summary_json"] = files["summary_json"]
    files["generic_full_grid_csv"] = files["full_grid_csv"]
    files["summary_json"] = files["compatibility_summary_json"]
    files["full_grid_csv"] = files["compatibility_full_grid_csv"]
    result.output_files.update(files)
    summary_payload = json.dumps(
        result.summary(), indent=2, sort_keys=True, default=_json_default
    )
    Path(files["summary_json"]).write_text(summary_payload, encoding="utf-8")
    Path(files["generic_summary_json"]).write_text(summary_payload, encoding="utf-8")
    return files
