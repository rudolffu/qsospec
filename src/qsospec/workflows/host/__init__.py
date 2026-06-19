"""Optional pPXF host-decomposition support."""

from .config import HostDecompConfig, default_config
from .io import SpectrumData, inspect_spectrum, read_sparcli_spectrum
from .ppxf_host import (
    HostSED,
    PPXFHostFitResult,
    prepare_desi_for_host_decomp,
    predict_host_sed,
    predict_host_sed_on_grid,
    run_ppxf_host_fit,
)
from .templates import PPXFTemplateLibrary, load_ppxf_npz_templates

__all__ = [
    "HostDecompConfig",
    "HostSED",
    "PPXFHostFitResult",
    "PPXFTemplateLibrary",
    "SpectrumData",
    "default_config",
    "inspect_spectrum",
    "load_ppxf_npz_templates",
    "predict_host_sed",
    "predict_host_sed_on_grid",
    "prepare_desi_for_host_decomp",
    "read_sparcli_spectrum",
    "run_ppxf_host_fit",
]
