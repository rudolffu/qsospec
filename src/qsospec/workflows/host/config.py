"""Configuration defaults for optional pPXF host decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_LINE_CENTERS = {
    "MgII": 2798.0,
    "NeV3426": 3426.0,
    "OII3727": 3727.0,
    "Hdelta": 4102.0,
    "Hgamma": 4341.0,
    "Hbeta": 4861.0,
    "OIII4959": 4959.0,
    "OIII5007": 5007.0,
    "HeI5876": 5876.0,
    "OI6300": 6300.0,
    "Halpha": 6563.0,
    "NII6548": 6548.0,
    "NII6583": 6583.0,
    "SII6716": 6716.0,
    "SII6731": 6731.0,
}


DEFAULT_LINE_MASK_WIDTHS = {
    "default": 800.0,
    "MgII": 1800.0,
    "Hdelta": 1400.0,
    "Hgamma": 1400.0,
    "Hbeta": 1800.0,
    "Halpha": 2200.0,
}


DEFAULT_BROAD_LINE_MASK_WIDTHS = {
    "default": 1200.0,
    "MgII": 3500.0,
    "Hdelta": 2600.0,
    "Hgamma": 2600.0,
    "Hbeta": 3500.0,
    "Halpha": 4500.0,
}

DEFAULT_OBSERVED_ARTIFACT_WINDOWS = (
    (5570.0, 5585.0),
    (5885.0, 5900.0),
    (6290.0, 6310.0),
    (6355.0, 6372.0),
    (6860.0, 6930.0),
    (7580.0, 7700.0),
)


AYDAR_2026_BROADENING_GRID_KMS = (
    1000.0,
    1200.0,
    1400.0,
    1600.0,
    1800.0,
    2000.0,
    2400.0,
    2800.0,
    3400.0,
    4000.0,
    4800.0,
    5800.0,
    7000.0,
    8400.0,
    10000.0,
    11800.0,
)


AYDAR_2026_POWERLAW_SLOPES_FLAMBDA = tuple(
    round(-3.0 + 0.1 * index, 10) for index in range(31)
)


@dataclass(frozen=True)
class HostBroadLinePrefitConfig:
    """Broad-Balmer nuisance fit used only to select host templates."""

    enabled: bool = True
    preferred_lines: Tuple[str, ...] = ("halpha", "hbeta")
    minimum_flux_snr: float = 3.0
    minimum_fwhm_snr: float = 3.0
    reject_parameter_bounds: bool = True
    fallback_policy: str = "masked_simple"
    fixed_fallback_fwhm_kms: float | None = None

    def __post_init__(self) -> None:
        allowed_lines = {"halpha", "hbeta"}
        if not self.preferred_lines or any(
            line not in allowed_lines for line in self.preferred_lines
        ):
            raise ValueError(
                "HostBroadLinePrefitConfig.preferred_lines must contain "
                "'halpha' and/or 'hbeta'."
            )
        if len(set(self.preferred_lines)) != len(self.preferred_lines):
            raise ValueError(
                "HostBroadLinePrefitConfig.preferred_lines must be unique."
            )
        for name, value in (
            ("minimum_flux_snr", self.minimum_flux_snr),
            ("minimum_fwhm_snr", self.minimum_fwhm_snr),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.fallback_policy not in {"masked_simple", "fixed_width", "fail"}:
            raise ValueError(
                "fallback_policy must be 'masked_simple', 'fixed_width', or 'fail'."
            )
        if self.fallback_policy == "fixed_width":
            value = self.fixed_fallback_fwhm_kms
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "fixed_fallback_fwhm_kms must be positive for fixed_width fallback."
                )


@dataclass(frozen=True)
class HostAgnPseudoContinuumConfig:
    """AGN template basis for the masked pseudo-continuum host strategy."""

    enabled: bool = True
    powerlaw_slopes: Tuple[float, ...] = AYDAR_2026_POWERLAW_SLOPES_FLAMBDA
    powerlaw_pivot_angstrom: float = 5100.0
    optical_feii_template: str = "bg92"
    uv_feii_template: str | None = None
    balmer_enabled: bool = True
    balmer_log10_ne: int = 9
    balmer_temperature_k: float = 15000.0
    balmer_tau_edge: float = 1.0
    width_grid_kms: Tuple[float, ...] = AYDAR_2026_BROADENING_GRID_KMS
    maximum_width_iterations: int = 2
    width_convergence_tolerance_kms: float = 0.0
    additive_polynomial_degree: int = -1
    multiplicative_polynomial_degree: int = 0
    global_fagn_warning_threshold: float = 0.8

    def __post_init__(self) -> None:
        slopes = tuple(float(value) for value in self.powerlaw_slopes)
        if not slopes or any(not math.isfinite(value) for value in slopes):
            raise ValueError("powerlaw_slopes must contain finite values.")
        if len(set(slopes)) != len(slopes):
            raise ValueError("powerlaw_slopes must be unique.")
        grid = tuple(float(value) for value in self.width_grid_kms)
        if (
            not grid
            or any(not math.isfinite(value) or value <= 0 for value in grid)
            or any(right <= left for left, right in zip(grid, grid[1:]))
        ):
            raise ValueError("width_grid_kms must be positive and strictly increasing.")
        if not math.isfinite(self.powerlaw_pivot_angstrom) or self.powerlaw_pivot_angstrom <= 0:
            raise ValueError("powerlaw_pivot_angstrom must be positive and finite.")
        if self.balmer_log10_ne not in {9, 10}:
            raise ValueError("balmer_log10_ne must be 9 or 10.")
        if self.maximum_width_iterations not in {1, 2}:
            raise ValueError("maximum_width_iterations must be 1 or 2.")
        if (
            not math.isfinite(self.width_convergence_tolerance_kms)
            or self.width_convergence_tolerance_kms < 0
        ):
            raise ValueError("width_convergence_tolerance_kms must be non-negative.")
        if self.additive_polynomial_degree < -1:
            raise ValueError("additive_polynomial_degree must be at least -1.")
        if self.multiplicative_polynomial_degree < 0:
            raise ValueError("multiplicative_polynomial_degree must be non-negative.")
        if not 0 < self.global_fagn_warning_threshold < 1:
            raise ValueError("global_fagn_warning_threshold must lie between zero and one.")


@dataclass(frozen=True)
class HostCoverageConfig:
    """Rest-frame leverage requirements for host-fit reliability."""

    full_optical_range: Tuple[float, float] = (3600.0, 7000.0)
    optical_core_range: Tuple[float, float] = (3600.0, 5500.0)
    blue_optical_range: Tuple[float, float] = (3600.0, 4500.0)
    minimum_valid_fraction: float = 0.7
    minimum_valid_pixels: int = 200
    endpoint_tolerance_angstrom: float = 50.0

    def __post_init__(self) -> None:
        for name, window in (
            ("full_optical_range", self.full_optical_range),
            ("optical_core_range", self.optical_core_range),
            ("blue_optical_range", self.blue_optical_range),
        ):
            if (
                len(window) != 2
                or not all(math.isfinite(value) for value in window)
                or window[1] <= window[0]
            ):
                raise ValueError(f"{name} must be a finite increasing interval.")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must lie in (0, 1].")
        if self.minimum_valid_pixels < 1:
            raise ValueError("minimum_valid_pixels must be positive.")
        if self.endpoint_tolerance_angstrom < 0:
            raise ValueError("endpoint_tolerance_angstrom must be non-negative.")


@dataclass
class HostDecompConfig:
    """Runtime config for the optional pPXF host-decomposition workflow."""

    strategy: str = "masked_simple"
    template_root: str = "~/tools/ppxf_data"
    template_file: str = "spectra_emiles_9.0.npz"
    template_family: str = "emiles"
    template_profile: str | None = None
    template_product_kind: str = "native"
    source_template_file: str | None = None
    resolution_matching_mode: str = "convolve_template_toward_data_only"
    template_coarser_action: str = "warn"
    preserve_native_data: bool = True
    preconvolved_validation: str = "exact"
    fit_range: Tuple[float, float] = (3600.0, 7000.0)
    line_mask_widths: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LINE_MASK_WIDTHS))
    broad_line_mask_widths: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BROAD_LINE_MASK_WIDTHS))
    observed_artifact_windows: List[Tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_OBSERVED_ARTIFACT_WINDOWS)
    )
    max_native_gap_pixels: float = 3.0
    systematic_error_floor_fraction: float = 0.02
    adaptive_broad_line_max_velocity: float = 10000.0
    adaptive_line_residual_sigma: float = 3.0
    residual_clip_sigma: float = 4.5
    residual_clip_iterations: int = 2
    residual_clip_dilation_pixels: int = 2
    max_noise_rescale: float = 5.0
    minimum_clean_fraction: float = 0.35
    minimum_clean_pixels: int = 200
    minimum_continuum_snr: float = 2.0
    maximum_clipped_fraction: float = 0.25
    polynomial_degree: int = 4
    multiplicative_polynomial_degree: int = 0
    use_regularization: bool = False
    agn_powerlaw_slopes: Sequence[float] = (-2.0, -1.5, -1.0, -0.5, 0.0)
    run_qsospec: bool = False
    n_iterations: int = 1
    euclid_scaling_mode: str = "free_scale"
    continuum_windows: List[Tuple[float, float]] = field(
        default_factory=lambda: [(4100.0, 4300.0), (5200.0, 5600.0), (6000.0, 6200.0)]
    )
    output_dir: str = "outputs/ppxf_qsospec"
    broad_line_prefit: HostBroadLinePrefitConfig = field(
        default_factory=HostBroadLinePrefitConfig
    )
    agn_pseudocontinuum: HostAgnPseudoContinuumConfig = field(
        default_factory=HostAgnPseudoContinuumConfig
    )
    coverage: HostCoverageConfig = field(default_factory=HostCoverageConfig)

    def __post_init__(self) -> None:
        if self.strategy not in {
            "masked_simple",
            "agn_pseudocontinuum_masked",
        }:
            raise ValueError(
                "HostDecompConfig.strategy must be 'masked_simple' or "
                "'agn_pseudocontinuum_masked'."
            )
        if self.template_profile not in {
            None,
            "emiles_native",
            "xsl_native",
            "xsl_preconvolved",
            "custom_native",
        }:
            raise ValueError(
                "template_profile must be emiles_native, xsl_native, "
                "xsl_preconvolved, custom_native, or None."
            )
        if self.template_product_kind not in {"native", "preconvolved"}:
            raise ValueError(
                "template_product_kind must be 'native' or 'preconvolved'."
            )
        if self.resolution_matching_mode not in {
            "convolve_template_toward_data_only",
            "object_specific_runtime",
            "preconvolved_exact",
        }:
            raise ValueError("Unsupported resolution_matching_mode.")
        resolved_profile = self.template_profile
        if resolved_profile is None:
            if self.template_product_kind == "preconvolved":
                resolved_profile = "xsl_preconvolved"
            elif Path(self.template_file).name == "spectra_xsl_9.0.npz":
                resolved_profile = "xsl_native"
            elif Path(self.template_file).name == "spectra_emiles_9.0.npz":
                resolved_profile = "emiles_native"
            else:
                resolved_profile = "custom_native"
        expected_matching_mode = {
            "xsl_native": "object_specific_runtime",
            "xsl_preconvolved": "preconvolved_exact",
        }.get(resolved_profile, "convolve_template_toward_data_only")
        if self.resolution_matching_mode != expected_matching_mode:
            # The historical/default value describes E-MILES. Resolve it to
            # the selected profile's contract; reject any other contradiction.
            if self.resolution_matching_mode == "convolve_template_toward_data_only":
                self.resolution_matching_mode = expected_matching_mode
            else:
                raise ValueError(
                    f"Template profile {resolved_profile!r} requires "
                    f"resolution_matching_mode={expected_matching_mode!r}."
                )
        if self.template_coarser_action not in {"warn", "ignore"}:
            raise ValueError("template_coarser_action must be 'warn' or 'ignore'.")
        if not self.preserve_native_data:
            raise ValueError(
                "Host decomposition requires preserve_native_data=True; "
                "qsospec never smooths the science spectrum to the template resolution."
            )
        if self.preconvolved_validation != "exact":
            raise ValueError("Only preconvolved_validation='exact' is supported.")
        if self.template_profile == "xsl_preconvolved":
            if self.template_product_kind != "preconvolved":
                raise ValueError(
                    "xsl_preconvolved requires template_product_kind='preconvolved'."
                )
            if not self.source_template_file:
                raise ValueError(
                    "xsl_preconvolved requires source_template_file naming the native XSL library."
                )
        if self.use_regularization:
            raise ValueError(
                "HostDecompConfig.use_regularization is not implemented; "
                "leave it False."
            )
        if self.n_iterations != 1:
            raise ValueError(
                "HostDecompConfig.n_iterations is deprecated and only 1 is "
                "accepted; use agn_pseudocontinuum.maximum_width_iterations."
            )
        if self.run_qsospec:
            raise ValueError(
                "HostDecompConfig.run_qsospec is deprecated; select the "
                "local or global workflow through the fitting API."
            )
        if self.euclid_scaling_mode != "free_scale":
            raise ValueError(
                "HostDecompConfig.euclid_scaling_mode is deprecated; use "
                "EuclidHostScaleConfig instead."
            )
        default_windows = [
            (4100.0, 4300.0),
            (5200.0, 5600.0),
            (6000.0, 6200.0),
        ]
        if self.continuum_windows != default_windows:
            raise ValueError(
                "HostDecompConfig.continuum_windows is deprecated; use "
                "EuclidHostScaleConfig for Euclid host scaling."
            )


def default_config() -> HostDecompConfig:
    """Return a fresh default host-decomposition config."""

    return HostDecompConfig()
