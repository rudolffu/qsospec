"""Result containers for the qsospec global continuum and H-beta workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .spectrum import Spectrum
from .warnings import FitWarning


@dataclass
class GlobalContinuumResult:
    """Global continuum fit evaluated on the full input grid."""

    success: bool
    status: int
    message: str
    param_values: Dict[str, float]
    param_errors: Dict[str, float]
    covariance: Optional[np.ndarray]
    chi2: float
    dof: int
    reduced_chi2: float
    wave_rest: np.ndarray
    model: np.ndarray
    component_models: Dict[str, np.ndarray]
    fit_mask: np.ndarray
    clip_mask: np.ndarray
    warnings: List[FitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimizer_result: Optional[Any] = None

    def warning_codes(self) -> List[str]:
        return [warning.code for warning in self.warnings]

    def summary(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "status": int(self.status),
            "message": self.message,
            "param_values": dict(self.param_values),
            "param_errors": dict(self.param_errors),
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "reduced_chi2": float(self.reduced_chi2),
            "n_fit_pixels": int(np.count_nonzero(self.fit_mask)),
            "n_clipped_pixels": int(np.count_nonzero(self.fit_mask & ~self.clip_mask)),
            "warning_codes": self.warning_codes(),
            "metadata": dict(self.metadata),
        }


@dataclass
class EmissionComplexResult:
    """One continuum-subtracted emission-line complex fit."""

    success: bool
    status: int
    message: str
    selected_model: str
    param_values: Dict[str, float]
    param_errors: Dict[str, float]
    covariance: Optional[np.ndarray]
    metrics: Dict[str, float]
    metric_errors: Dict[str, float]
    chi2: float
    dof: int
    reduced_chi2: float
    bic: float
    wave_rest: np.ndarray
    flux_continuum_subtracted: np.ndarray
    err: np.ndarray
    model: np.ndarray
    component_models: Dict[str, np.ndarray]
    fit_mask: np.ndarray
    warnings: List[FitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimizer_result: Optional[Any] = None
    excluded_mask: Optional[np.ndarray] = None

    def warning_codes(self) -> List[str]:
        return [warning.code for warning in self.warnings]

    def summary(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "status": int(self.status),
            "message": self.message,
            "selected_model": self.selected_model,
            "param_values": dict(self.param_values),
            "param_errors": dict(self.param_errors),
            "metrics": dict(self.metrics),
            "metric_errors": dict(self.metric_errors),
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "reduced_chi2": float(self.reduced_chi2),
            "bic": float(self.bic),
            "n_fit_pixels": int(np.count_nonzero(self.fit_mask)),
            "warning_codes": self.warning_codes(),
            "metadata": dict(self.metadata),
        }


HbetaComplexResult = EmissionComplexResult


@dataclass
class WorkflowResult:
    """One optional-host, global-continuum, multi-complex workflow."""

    spectrum: Spectrum
    continuum_initial: GlobalContinuumResult
    continuum: GlobalContinuumResult
    hbeta_initial: Optional[HbetaComplexResult] = None
    hbeta: Optional[HbetaComplexResult] = None
    mgii: Optional[EmissionComplexResult] = None
    halpha: Optional[EmissionComplexResult] = None
    line_complexes: Dict[str, EmissionComplexResult] = field(default_factory=dict)
    complex_statuses: Dict[str, str] = field(default_factory=dict)
    host_decomp_enabled: bool = False
    total_spectrum: Optional[Spectrum] = None
    host_fit: Optional[Any] = None
    host_sed: Optional[Any] = None
    host_model_on_quasar_grid: Optional[np.ndarray] = None
    host_fit_mask: Optional[np.ndarray] = None
    host_emission_mask: Optional[np.ndarray] = None
    host_warnings: List[str] = field(default_factory=list)
    monte_carlo: Dict[str, Any] = field(default_factory=dict)
    warnings: List[FitWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_files: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.line_complexes:
            self.line_complexes = {}
            if self.hbeta is not None:
                self.line_complexes["hbeta_oiii"] = self.hbeta
            if self.mgii is not None:
                self.line_complexes["mgii"] = self.mgii
            if self.halpha is not None:
                self.line_complexes["halpha_nii_sii"] = self.halpha

    @property
    def continuum_success(self) -> bool:
        return bool(self.continuum.success)

    @property
    def legacy_hbeta_success(self) -> bool:
        """Deprecated Hβ-oriented success verdict."""

        return bool(
            self.continuum.success
            and self.hbeta is not None
            and self.hbeta.success
        )

    def warning_codes(self) -> List[str]:
        codes = [warning.code for warning in self.warnings]
        codes.extend(self.continuum.warning_codes())
        for result in self.line_complexes.values():
            codes.extend(result.warning_codes())
        return codes

    @property
    def qa_path(self) -> Optional[str]:
        """Primary saved QA path, when the workflow wrote one."""

        return self.output_files.get("main_qa")

    def plot_qa(self, plot_config=None):
        """Return an open Matplotlib QA figure for notebook use."""

        from .io.products import plot_qa_figure

        return plot_qa_figure(self, plot_config)

    def show_qa(self, plot_config=None):
        """Display and return the QA figure in an interactive session."""

        import matplotlib.pyplot as plt

        figure = self.plot_qa(plot_config)
        plt.show()
        return figure

    def summary(self) -> Dict[str, Any]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            package_version = version("qsospec")
        except PackageNotFoundError:
            package_version = "0.1.0"
        power_law_parameters = {
            name: value
            for name, value in self.continuum.param_values.items()
            if name.startswith("power_law.")
        }
        continuum_samples = self.metadata.get("continuum_samples", {})
        return {
            "package_version": package_version,
            "object_id": self.metadata.get("object_id"),
            "redshift": float(self.spectrum.z),
            "flux_unit": self.spectrum.flux_unit,
            "flux_scale": self.spectrum.flux_scale,
            "galactic_extinction": self.metadata.get(
                "galactic_extinction", {}
            ),
            "continuum_success": self.continuum_success,
            "continuum_reduced_chi2": float(
                self.continuum.reduced_chi2
            ),
            "host_decomp_enabled": bool(self.host_decomp_enabled),
            "host": {
                "status": self.metadata.get("host_ppxf_status"),
                "reduced_chi2": self.metadata.get(
                    "host_ppxf_reduced_chi2"
                ),
                "template_file": self.metadata.get("host_template_file"),
                "fractions": {
                    name: value
                    for name, value in continuum_samples.items()
                    if name.startswith("fracHost_")
                },
            },
            "power_law_mode": self.metadata.get(
                "power_law_mode_selected",
                self.continuum.metadata.get("power_law_mode_selected"),
            ),
            "power_law_parameters": power_law_parameters,
            "power_law_selection": {
                "reason": self.metadata.get("power_law_selection_reason"),
                "single_bic": self.metadata.get("power_law_single_bic"),
                "double_bic": self.metadata.get("power_law_double_bic"),
                "delta_bic": self.metadata.get("power_law_delta_bic"),
            },
            "complex_statuses": dict(self.complex_statuses),
            "warning_codes": self.warning_codes(),
            "output_files": {
                key: value
                for key, value in self.output_files.items()
                if key in ("run_directory", "manifest", "main_qa")
            },
        }
