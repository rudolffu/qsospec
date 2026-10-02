"""Contracts for the 0.2 API cleanup, without changing scientific objectives."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

import qsospec
from qsospec.config import DEFAULT_GLOBAL_MODEL_ID, RegionalIronConfig
from qsospec.fitting.global_fit import _ContinuumContext
from qsospec.templates import load_iron_template
from qsospec.templates.iron import resolve_iron_width
from qsospec.uncertainties import (
    workflow_measurements,
    apply_bootstrap_errors,
    measurement_key,
    canonicalize_matched_draws,
)
from qsospec.workflows.host.config import resolve_host_runtime_config
from qsospec.workflows.batch import _configuration


@pytest.mark.parametrize(
    "template,mode,coordinate",
    [
        ("park22", "legacy", "kernel"),
        ("verner09", "legacy", "target"),
        ("verner09", "kernel", "kernel"),
        ("verner09", "target", "target"),
    ],
)
def test_iron_coordinate_and_rendering(template, mode, coordinate):
    cfg = qsospec.IronTemplateConfig(template, width_mode=mode, fwhm_kms=3200.0, fwhm_bounds=(0.0, 10000.0))
    wave = np.linspace(2000.0, 6500.0, 350)
    spectrum = qsospec.Spectrum.from_arrays(
        wave, np.ones_like(wave), err=np.ones_like(wave), wave_frame="rest", flux_unit="relative"
    )
    global_cfg = qsospec.GlobalContinuumConfig.with_single_iron(
        cfg,
        power_law=qsospec.PowerLawConfig(mode="single"),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        continuum_windows=((2000.0, 6500.0),),
    )
    ctx = _ContinuumContext(spectrum, global_cfg)
    index = ctx.index["full_iron.fwhm_kms"]
    assert ctx.initial[index] == 3200.0
    expected = resolve_iron_width(load_iron_template(template), 3200.0, mode)
    assert expected["requested_width_mode"] == coordinate
    assert ctx.lower[index] == (900.0 if coordinate == "target" else 0.0)
    metadata = qsospec.fit_global_continuum(spectrum, global_cfg).metadata["iron_templates"]["full_iron"]
    assert metadata["configuration_width_mode"] == mode
    assert metadata["native_width_status"] == load_iron_template(template).native_width_status
    assert metadata["convolution_fwhm_kms"] == metadata["kernel_fwhm_kms"]
    assert metadata["requested_configuration"]["width_mode"] == mode
    if template == "park22":
        assert metadata["target_fwhm_kms"] is None
        assert metadata["effective_width_status"] == "unavailable"


def test_invalid_targets_and_log_kernel_bounds():
    with pytest.raises(ValueError, match="justified native"):
        qsospec.IronTemplateConfig.park22(width_mode="target")
    with pytest.raises(ValueError, match="sharpen"):
        qsospec.IronTemplateConfig.verner09(width_mode="target", fwhm_kms=800.0)
    with pytest.raises(ValueError, match="upper bound"):
        qsospec.IronTemplateConfig.verner09(width_mode="target", fwhm_bounds=(500.0, 850.0))
    cfg = qsospec.IronTemplateConfig.verner09(width_mode="kernel", fwhm_kms=0.0, fwhm_bounds=(0.0, 500.0))
    assert cfg.fwhm_kms == 0.0
    with pytest.raises(ValueError, match="positive lower"):
        qsospec.GlobalContinuumConfig(uv_iron=cfg, iron_width_coupling="soft")
    with pytest.raises(ValueError, match="positive lower"):
        qsospec.GlobalContinuumConfig(
            uv_iron=qsospec.IronTemplateConfig.verner09(width_mode="target", fwhm_bounds=(900.0, 10000.0)),
            iron_width_coupling="soft",
        )


@pytest.mark.parametrize(
    "value,canonical",
    [
        ("off", "off"),
        ("soft", "soft"),
        ("hard", "hard"),
        ("require", "require"),
        ("none", "off"),
        ("never", "off"),
        ("auto", "hard"),
        ("hard_legacy", "hard"),
    ],
)
def test_hgamma_canonical_config(value, canonical):
    config = qsospec.BalmerPseudoContinuumConfig(sync_with_hgamma=value)
    assert config.sync_with_hgamma == canonical
    assert asdict(config)["sync_with_hgamma"] == canonical


def test_model_presets_and_polynomial_modes():
    default = qsospec.GlobalContinuumConfig()
    assert default.model_id == DEFAULT_GLOBAL_MODEL_ID == "global_v2"
    assert default.regional_iron.enabled and default.iron_width_coupling == "independent"
    assert default.balmer_pseudocontinuum.sync_with_hgamma == "soft"
    assert default.polynomial.mode == "off"
    assert (default.uv_iron.template, default.optical_iron.template) == ("vw01", "park22")
    legacy = qsospec.GlobalContinuumConfig.legacy_v1()
    assert legacy.model_id == "global_v1"
    assert not legacy.regional_iron.enabled
    assert legacy.balmer_pseudocontinuum.sync_with_hgamma == "hard"
    for mode in ("off", "auto", "on"):
        assert asdict(qsospec.PolynomialContinuumConfig(mode=mode))["mode"] == mode
    with pytest.raises(ValueError, match="off.*auto.*on"):
        qsospec.PolynomialContinuumConfig(mode="yes")


@pytest.mark.parametrize(
    "changes",
    [
        {"amp": np.nan},
        {"amp": -1},
        {"fixed_kernel_fwhm_kms": -1},
        {"uv_interval": (3500.0, 3400.0)},
        {"optical_interval": (3350.0, 3600.0)},
    ],
)
def test_regional_validation(changes):
    with pytest.raises(ValueError):
        RegionalIronConfig(**changes)


def _effective_configuration(host_config=None, **overrides):
    kwargs = dict(
        run_host_decomp=True,
        template_root=None,
        template_file=None,
        host_fit_range=None,
        host_config=host_config,
        galactic_extinction_config=qsospec.GalacticExtinctionConfig(),
        global_config=None,
        hbeta_config=qsospec.HbetaComplexConfig(),
        mgii_config=qsospec.MgIIComplexConfig(),
        halpha_config=qsospec.HalphaComplexConfig(),
        lya_nv_config=qsospec.LyaNVComplexConfig(),
        uncertainty_config=qsospec.UncertaintyConfig(),
        complexes=None,
    )
    kwargs.update(overrides)
    return _configuration(**kwargs)


def test_resolved_host_precedence_and_hash_input_equivalence():
    default = qsospec.HostDecompConfig()
    assert resolve_host_runtime_config() == default
    custom = qsospec.HostDecompConfig(
        template_root="/data/templates", template_file="custom.npz", fit_range=(4000.0, 6000.0)
    )
    assert resolve_host_runtime_config(custom) == custom
    resolved = resolve_host_runtime_config(
        custom,
        template_root=default.template_root,
        template_file=default.template_file,
        host_fit_range=default.fit_range,
    )
    assert resolved == default  # explicitly supplied literal default still wins
    assert _effective_configuration(custom) == _effective_configuration(
        template_root=custom.template_root, template_file=custom.template_file, host_fit_range=custom.fit_range
    )
    assert _effective_configuration(custom)["global_model_id"] == "global_v2"


def test_matched_measurements_are_scoped_and_map_back_without_collisions():
    spectrum = qsospec.Spectrum.from_arrays(
        [4800.0, 4900.0, 5100.0], [1.0, 1.0, 1.0], err=[0.1, 0.1, 0.1], wave_frame="rest", flux_unit="relative"
    )
    fits = {
        key: SimpleNamespace(success=True, metrics={"shared_flux": value}, metadata={}, metric_errors={})
        for key, value in [("first", 2.0), ("second", 7.0)]
    }
    result = SimpleNamespace(
        spectrum=spectrum,
        metadata={"continuum_samples": {"fAGN_5100": 1.0}},
        continuum=SimpleNamespace(param_values={"power_law.slope": -1.0}, metadata={}),
        line_complexes=fits,
    )
    values = workflow_measurements(result)
    assert values["line:first:shared_flux"] == 2.0
    assert values["line:second:shared_flux"] == 7.0
    assert "shared_flux" not in values
    assert values["continuum_sample:fAGN_5100"] == 1.0
    assert values["continuum_param:power_law.slope"] == -1.0
    assert "derived:ws22_log_l1700" in values
    result.monte_carlo = {
        "errors": {
            "line:first:shared_flux": 0.2,
            "line:second:shared_flux": 0.7,
            "continuum_sample:fAGN_5100": 0.1,
            "shared_flux": 99.0,
        },
        "percentiles": {"line:first:shared_flux": {"p50": 2.0}},
        "method": "bootstrap",
    }
    apply_bootstrap_errors(result)
    assert fits["first"].metric_errors["shared_flux"] == 0.2
    assert fits["second"].metric_errors["shared_flux"] == 0.7
    assert result.metadata["continuum_sample_errors"]["fAGN_5100"] == 0.1
    assert fits["first"].metadata["measurement_intervals"]["shared_flux"] == {"p50": 2.0}
    with pytest.raises(ValueError, match="explicit recipe"):
        measurement_key("line", "shared_flux")


def test_old_draw_reader_preserves_qualified_values_and_drops_ambiguous_aliases():
    old = {
        "errors": {"first:shared_flux": 0.2, "shared_flux": 99.0, "fAGN_5100": 0.1},
        "draws": [{"trial_id": 0, "values": {"first:shared_flux": 2.0, "shared_flux": 99.0, "fAGN_5100": 1.0}}],
        "covariance_measurement_names": ["first:shared_flux", "shared_flux"],
        "measurement_covariance": [[1.0, 0.0], [0.0, 2.0]],
    }
    restored = canonicalize_matched_draws(old, ["first"], ["fAGN_5100"])
    assert restored["measurement_key_schema"] == "qualified_v1"
    assert restored["errors"] == {"line:first:shared_flux": 0.2, "continuum_sample:fAGN_5100": 0.1}
    assert restored["measurement_covariance"] == [[1.0]]
    assert old["errors"]["shared_flux"] == 99.0


def test_established_root_api_and_specialist_modules():
    # Public symbols from the PR base, fixed independently of the new namespace.
    established = (
        "BalmerPseudoContinuumConfig",
        "BalmerAnchorRatios",
        "BalmerSeriesTemplate",
        "BatchResult",
        "ComponentRecipe",
        "ComplexRecipe",
        "FitResult",
        "EmissionComplexResult",
        "EuclidHostScaleConfig",
        "EuclidHostScaleFit",
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
        "fit_euclid_host_aperture_scale",
        "f99_dereddening_factor",
        "galactic_dereddening_factor",
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
        "euclid_nir_line_mask",
        "list_balmer_templates",
        "list_iron_templates",
        "load_balmer_anchor_ratios",
        "lines",
        "load_balmer_template",
        "load_iron_template",
        "load_model",
        "open_run",
        "prepare_spectrum",
        "preflight_galactic_extinction",
        "query_galactic_ebv",
        "plot_line_result",
        "plot_local_result",
        "recipes",
        "read_input_manifest",
        "read_spectrum",
        "render_qa",
        "resolve_qa_plot_config",
        "resolve_spectrum_metadata",
        "save_local_window_plots",
        "scan_parquet_spectra",
        "write_global_hbeta_products",
        "write_global_line_products",
        "wang2019_extinction_mag",
    )
    assert all(hasattr(qsospec, name) for name in established)
    for name in (
        "PolynomialContinuumConfig",
        "RegionalIronConfig",
        "HostDecompConfig",
        "HostBroadLinePrefitConfig",
        "HostAgnPseudoContinuumConfig",
        "HostCoverageConfig",
        "BatchResumePlan",
        "plan_batch_resume",
        "load_model_by_key",
    ):
        assert name in qsospec.__all__ and hasattr(qsospec, name)
    from qsospec.halpha_classification import fit_halpha_model_grid
    from qsospec.resolution import SpectralResolution
    from qsospec.line_peaks import recover_line_peaks
    from qsospec.systemic_redshift import estimate_systemic_redshift
    from qsospec.observed_model import reconstruct_observed_model
    from qsospec.uncertainties import measure_selected_profile

    assert all(
        callable(item)
        for item in (
            fit_halpha_model_grid,
            SpectralResolution,
            recover_line_peaks,
            estimate_systemic_redshift,
            reconstruct_observed_model,
            measure_selected_profile,
        )
    )
