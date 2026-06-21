"""Standalone quasar spectral fitting with NumPy and SciPy."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

try:
    __version__ = _version("qsospec")
except _PackageNotFoundError:  # source checkout without installation
    __version__ = "0.1.0"

from . import lines, recipes
from .lines import LineDefinition
from .complex_recipes import ComponentRecipe, ComplexRecipe
from .fitting.local import fit_line_complex, fit_local
from .workflows.batch import BatchResult, fit_batch, fit_object_to_store
from .config import (
    BalmerPseudoContinuumConfig,
    GalacticExtinctionConfig,
    GaussianComponent,
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    IronTemplateConfig,
    LyaNVComplexConfig,
    LineComplexConfig,
    LocalFitConfig,
    LorentzianComponent,
    MgIIComplexConfig,
    PowerLawConfig,
    UncertaintyConfig,
)
from .extinction import (
    correct_spectrum,
    correct_spectrum_data,
    f99_dereddening_factor,
    preflight_galactic_extinction,
    query_galactic_ebv,
)
from .fitting.global_fit import (
    fit_global_continuum,
    fit_global_hbeta,
    fit_global_lines,
    fit_halpha_complex,
    fit_hbeta_complex,
    fit_mgii_complex,
)
from .io.products import (
    GlobalQAPlotConfig,
    write_global_hbeta_products,
    write_global_line_products,
)
from .global_result import (
    EmissionComplexResult,
    GlobalContinuumResult,
    HbetaComplexResult,
    WorkflowResult,
)
from .workflows.host_workflow import (
    HostWorkflowResult,
    fit_global_hbeta_workflow,
    fit_global_lines_workflow,
    fit_with_optional_host_decomp,
)
from .metadata import SpectrumMetadata, resolve_spectrum_metadata
from .io.qa import render_qa
from .io.readers import (
    SpectrumInput,
    detect_fits_reader,
    discover_fits_inputs,
    read_input_manifest,
    read_spectrum,
    scan_parquet_spectra,
)
from .plotting import plot_line_result, plot_local_result, save_local_window_plots
from .result import FitResult, LocalFitResult
from .io.run_store import (
    RunStore,
    build_science_catalog,
    compute_derived_quantities,
    finalize_run,
    load_model,
    open_run,
)
from .spectrum import Spectrum
from .templates import (
    BalmerSeriesTemplate,
    IronTemplate,
    balmer_bound_free_shape,
    evaluate_balmer_pseudocontinuum,
    evaluate_balmer_pseudocontinuum_with_derivatives,
    list_balmer_templates,
    list_iron_templates,
    load_balmer_template,
    load_iron_template,
)
from .warnings import FitWarning

__all__ = [
    "BalmerPseudoContinuumConfig",
    "BalmerSeriesTemplate",
    "BatchResult",
    "ComponentRecipe",
    "ComplexRecipe",
    "FitResult",
    "EmissionComplexResult",
    "GaussianComponent",
    "GalacticExtinctionConfig",
    "GlobalContinuumConfig",
    "GlobalContinuumResult",
    "GlobalQAPlotConfig",
    "HalphaComplexConfig",
    "HbetaComplexConfig",
    "HbetaComplexResult",
    "IronTemplate",
    "IronTemplateConfig",
    "LineComplexConfig",
    "LineDefinition",
    "LyaNVComplexConfig",
    "LocalFitConfig",
    "LocalFitResult",
    "LorentzianComponent",
    "MgIIComplexConfig",
    "FitWarning",
    "HostWorkflowResult",
    "WorkflowResult",
    "PowerLawConfig",
    "RunStore",
    "Spectrum",
    "SpectrumInput",
    "SpectrumMetadata",
    "UncertaintyConfig",
    "balmer_bound_free_shape",
    "build_science_catalog",
    "compute_derived_quantities",
    "correct_spectrum",
    "correct_spectrum_data",
    "detect_fits_reader",
    "discover_fits_inputs",
    "finalize_run",
    "fit_batch",
    "f99_dereddening_factor",
    "fit_global_continuum",
    "fit_global_hbeta",
    "fit_global_hbeta_workflow",
    "fit_global_lines",
    "fit_global_lines_workflow",
    "fit_halpha_complex",
    "fit_hbeta_complex",
    "fit_mgii_complex",
    "fit_line_complex",
    "fit_local",
    "fit_object_to_store",
    "fit_with_optional_host_decomp",
    "evaluate_balmer_pseudocontinuum",
    "evaluate_balmer_pseudocontinuum_with_derivatives",
    "list_balmer_templates",
    "list_iron_templates",
    "lines",
    "load_balmer_template",
    "load_iron_template",
    "load_model",
    "open_run",
    "preflight_galactic_extinction",
    "query_galactic_ebv",
    "plot_line_result",
    "plot_local_result",
    "recipes",
    "read_input_manifest",
    "read_spectrum",
    "render_qa",
    "resolve_spectrum_metadata",
    "save_local_window_plots",
    "scan_parquet_spectra",
    "write_global_hbeta_products",
    "write_global_line_products",
]
