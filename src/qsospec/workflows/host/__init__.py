"""Optional pPXF host-decomposition support."""

from .agn_templates import (
    HostAgnTemplateBundle,
    HostAgnTemplateComponent,
    build_host_agn_template_bundle,
)
from .broad_line_prefit import (
    HostBroadLinePrefitResult,
    nearest_width_grid_value,
    run_host_broad_line_prefit,
)
from .config import (
    AYDAR_2026_BROADENING_GRID_KMS,
    AYDAR_2026_POWERLAW_SLOPES_FLAMBDA,
    HostAgnPseudoContinuumConfig,
    HostBroadLinePrefitConfig,
    HostCoverageConfig,
    HostDecompConfig,
    default_config,
)
from .coverage import HostCoverageResult, classify_host_coverage
from .euclid import (
    EuclidHostScaleConfig,
    EuclidHostScaleFit,
    euclid_nir_line_mask,
    fit_euclid_host_aperture_scale,
)
from .io import SpectrumData, inspect_spectrum, read_sparcli_spectrum
from .ppxf_host import (
    HostSED,
    PPXFHostFitResult,
    prepare_desi_for_host_decomp,
    prepare_spectrum_for_host_decomp,
    predict_host_sed,
    predict_host_sed_on_grid,
    run_ppxf_host_fit,
)
from .templates import PPXFTemplateLibrary, load_ppxf_npz_templates

__all__ = [
    "HostDecompConfig",
    "HostAgnPseudoContinuumConfig",
    "HostAgnTemplateBundle",
    "HostAgnTemplateComponent",
    "HostBroadLinePrefitConfig",
    "HostBroadLinePrefitResult",
    "HostCoverageConfig",
    "HostCoverageResult",
    "AYDAR_2026_BROADENING_GRID_KMS",
    "AYDAR_2026_POWERLAW_SLOPES_FLAMBDA",
    "EuclidHostScaleConfig",
    "EuclidHostScaleFit",
    "HostSED",
    "PPXFHostFitResult",
    "PPXFTemplateLibrary",
    "SpectrumData",
    "default_config",
    "build_host_agn_template_bundle",
    "classify_host_coverage",
    "inspect_spectrum",
    "euclid_nir_line_mask",
    "fit_euclid_host_aperture_scale",
    "load_ppxf_npz_templates",
    "predict_host_sed",
    "predict_host_sed_on_grid",
    "prepare_desi_for_host_decomp",
    "prepare_spectrum_for_host_decomp",
    "read_sparcli_spectrum",
    "run_ppxf_host_fit",
    "nearest_width_grid_value",
    "run_host_broad_line_prefit",
]
