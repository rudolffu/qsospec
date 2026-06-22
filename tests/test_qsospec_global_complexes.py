"""Tests for Mg II/H-alpha global line complexes and QA products."""

from dataclasses import replace

import numpy as np
import pytest

import qsospec
from qsospec.workflows.host.io import SpectrumData
from qsospec.fitting import global_fit
from qsospec.io import products as global_io
from qsospec.workflows import host_workflow
from qsospec.fitting.global_fit import (
    C_KMS,
    _HalphaContext,
    _MgIIContext,
    _gaussian_area_profile,
)
from qsospec.io.products import (
    _BROAD_COMPONENT_STYLE,
    _COMBINED_BROAD_STYLE,
    _CONTINUUM_STYLES,
    _HOST_STYLE,
    _NARROW_STYLE,
    _TCC_COLORS,
    _WING_STYLE,
    _annotate_emission_lines,
    _configure_qa_axis,
    _flux_density_axis_label,
    _flux_display_scale,
    _final_fit_masks,
    _line_groups,
    _host_fraction_annotation,
    _has_host_context,
    _input_spectrum_label,
    _masked_running_median,
    _percentile_limits,
    _plot_qa,
    _qa_overview_title,
    _rounded_model_upper_limit,
    _select_zoom_complexes,
)
from qsospec.global_result import GlobalContinuumResult


def _continuum_result(spectrum, model):
    return GlobalContinuumResult(
        success=True,
        status=1,
        message="known",
        param_values={},
        param_errors={},
        covariance=None,
        chi2=0.0,
        dof=1,
        reduced_chi2=0.0,
        wave_rest=spectrum.wave_rest.copy(),
        model=model.copy(),
        component_models={"power_law": model.copy()},
        fit_mask=spectrum.valid_mask.copy(),
        clip_mask=spectrum.valid_mask.copy(),
    )


def _centered_difference(function, value, step=0.01):
    return (function(value + step) - function(value - step)) / (2.0 * step)


@pytest.mark.parametrize(
    "context",
    [
        _MgIIContext(qsospec.MgIIComplexConfig(), 100.0),
        _HalphaContext(qsospec.HalphaComplexConfig(), 150.0),
    ],
)
def test_optional_complex_design_derivatives_match_centered_differences(context):
    wave = np.linspace(2700.0, 2900.0, 700) if isinstance(context, _MgIIContext) else np.linspace(6400.0, 6800.0, 900)
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    design, derivatives = context.separable_design(nonlinear, wave, True)
    assert design.shape[1] == len(context.linear_names)
    for index, derivative in enumerate(derivatives):

        def evaluate(value):
            trial = nonlinear.copy()
            trial[index] = value
            return context.separable_design(trial, wave, False)[0]

        finite = _centered_difference(evaluate, nonlinear[index])
        assert derivative == pytest.approx(finite, rel=5.0e-5, abs=1.0e-11)


def test_mgii_recovers_two_broad_components_and_metrics():
    wave = np.linspace(2680.0, 2920.0, 1200)
    continuum = np.full_like(wave, 2.0)
    line = _gaussian_area_profile(wave, 70.0, 2798.75 * np.exp(-100.0 / C_KMS), 2200.0)
    line += _gaussian_area_profile(wave, 30.0, 2798.75 * np.exp(200.0 / C_KMS), 7000.0)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        survey="desi",
    )
    result = qsospec.fit_mgii_complex(spectrum, _continuum_result(spectrum, continuum))

    assert result.success
    assert result.metadata["optimizer_used"] == "variable_projection"
    assert result.param_values["MgII_broad1.flux"] == pytest.approx(70.0, rel=1.0e-3)
    assert result.param_values["MgII_broad2.flux"] == pytest.approx(30.0, rel=1.0e-3)
    assert result.metrics["MgII_broad_flux_input"] == pytest.approx(100.0, rel=1.0e-3)
    assert result.covariance.shape == (9, 9)


def test_halpha_recovers_tied_narrow_lines_and_fixed_nii_ratio():
    wave = np.linspace(6350.0, 6850.0, 1800)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 80.0, 6564.61, 2200.0)
    line += _gaussian_area_profile(wave, 30.0, 6564.61, 4200.0)
    line += _gaussian_area_profile(wave, 10.0, 6564.61, 9000.0)
    for flux, center in (
        (12.0, 6564.61),
        (25.0, 6585.28),
        (25.0 / 2.96, 6549.85),
        (8.0, 6718.29),
        (6.0, 6732.67),
    ):
        line += _gaussian_area_profile(wave, flux, center, 320.0)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        survey="desi",
    )
    result = qsospec.fit_halpha_complex(spectrum, _continuum_result(spectrum, continuum))

    assert result.success
    assert result.metrics["Ha_broad_flux_input"] == pytest.approx(120.0, rel=1.0e-3)
    assert result.metrics["Ha_narrow_flux_input"] == pytest.approx(12.0, rel=1.0e-3)
    assert result.metrics["NII6585_flux_input"] / result.metrics["NII6549_flux_input"] == pytest.approx(
        2.96, rel=1.0e-8
    )
    assert result.metrics["SII6718_flux_input"] == pytest.approx(8.0, rel=1.0e-3)
    assert result.metrics["SII6733_flux_input"] == pytest.approx(6.0, rel=1.0e-3)
    centers = {
        name: np.trapezoid(wave * component, wave) / np.trapezoid(component, wave)
        for name, component in result.component_models.items()
        if name in {"Ha_narrow", "NII6549", "NII6585", "SII6718", "SII6733"}
    }
    velocities = {
        name: np.log(
            centers[name]
            / {
                "Ha_narrow": 6564.61,
                "NII6549": 6549.85,
                "NII6585": 6585.28,
                "SII6718": 6718.29,
                "SII6733": 6732.67,
            }[name]
        )
        * C_KMS
        for name in centers
    }
    assert max(velocities.values()) - min(velocities.values()) < 0.1


@pytest.mark.parametrize(
    ("wave_range", "covered"),
    [
        ((2700.0, 2900.0), True),
        ((2740.0, 2860.0), False),
        ((2800.0, 3000.0), False),
    ],
)
def test_mgii_coverage_rules(wave_range, covered):
    wave = np.linspace(*wave_range, 300)
    continuum = np.ones_like(wave)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        flux_unit="relative",
    )
    result = qsospec.fit_mgii_complex(spectrum, _continuum_result(spectrum, continuum))
    assert result.success is covered
    assert ("line_complex_not_covered" in result.warning_codes()) is (not covered)


def test_variable_projection_matches_legacy_for_optional_complexes():
    wave = np.linspace(6350.0, 6850.0, 1500)
    continuum = np.full_like(wave, 1.5)
    line = _gaussian_area_profile(wave, 100.0, 6564.61, 2400.0)
    line += _gaussian_area_profile(wave, 15.0, 6564.61, 350.0)
    line += _gaussian_area_profile(wave, 20.0, 6585.28, 350.0)
    line += _gaussian_area_profile(wave, 20.0 / 2.96, 6549.85, 350.0)
    line += _gaussian_area_profile(wave, 7.0, 6718.29, 350.0)
    line += _gaussian_area_profile(wave, 6.0, 6732.67, 350.0)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.02),
        wave_frame="rest",
        flux_unit="relative",
    )
    known = _continuum_result(spectrum, continuum)
    optimized = qsospec.fit_halpha_complex(
        spectrum,
        known,
        qsospec.HalphaComplexConfig(optimizer_method="variable_projection"),
    )
    legacy = qsospec.fit_halpha_complex(
        spectrum,
        known,
        qsospec.HalphaComplexConfig(optimizer_method="legacy_joint"),
    )
    assert optimized.chi2 <= legacy.chi2 + 1.0e-4
    assert optimized.metrics["Ha_broad_flux_input"] == pytest.approx(legacy.metrics["Ha_broad_flux_input"], rel=5.0e-3)
    assert optimized.metrics["Ha_broad_fwhm_kms"] == pytest.approx(legacy.metrics["Ha_broad_fwhm_kms"], abs=5.0)


def _global_spectrum(wave):
    continuum = 2.0 * (wave / 3000.0) ** -1.2
    line = _gaussian_area_profile(wave, 60.0, 2798.75, 2500.0)
    line += _gaussian_area_profile(wave, 80.0, 4862.68, 2500.0)
    line += _gaussian_area_profile(wave, 100.0, 6564.61, 2500.0)
    return qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        survey="desi",
    )


def _simple_global_config():
    return qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
    )


def test_global_workflow_fits_only_covered_complexes_and_writes_qa(tmp_path):
    full = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 3000)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    partial = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(1875.0, 5130.0, 2500)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    full.metadata.update({"object_id": "synthetic-qa", "redshift": 1.23456})
    full.host_decomp_enabled = True
    files = qsospec.write_global_line_products(full, str(tmp_path))

    assert set(full.line_complexes) == {
        "mgii",
        "hbeta_oiii",
        "halpha_nii_sii",
        "oii_nev_neiii_hgamma",
    }
    assert set(partial.line_complexes) == {
        "mgii",
        "hbeta_oiii",
        "oii_nev_neiii_hgamma",
    }
    assert partial.halpha is None
    assert partial.complex_statuses["halpha_nii_sii"] == "not_covered"
    assert full.continuum_success
    assert all(fit.success for fit in full.line_complexes.values())
    assert files["qa_plot"] == files["global_plot"]
    assert "host_context_plot" not in files
    assert full.metadata["host_context_plot_created"] is False
    assert full.metadata["qa_panel_count"] == 6
    assert full.metadata["qa_percentiles"] == [1.0, 99.8]
    assert full.metadata["qa_layout"] == "overview_residual_complexes"
    assert full.metadata["qa_figure_size_inches"] == [10.5, 8.0]
    assert full.metadata["qa_displayed_complexes"] == [
        "mgii",
        "oii_nev_neiii_hgamma",
        "hbeta_oiii",
        "halpha_nii_sii",
    ]
    assert full.metadata["qa_omitted_complexes"] == []
    assert full.metadata["qa_smoothed_data_requested"] is True
    assert full.metadata["qa_smoothed_data"] is False
    assert full.metadata["qa_smoothing_suppressed_short_spectrum"] is True
    assert full.metadata["qa_minor_ticks"] is True
    assert full.metadata["qa_tick_direction"] == "in"
    assert full.metadata["qa_overview_title"] == ("Object synthetic-qa   z = 1.2346")
    assert full.metadata["qa_overview_annotation"] == {
        "continuum_reduced_chi2": full.continuum.reduced_chi2,
        "host_state": "decomposed with pPXF",
        "zoom_spectrum": "input",
        "host_fractions": "",
        "ra": None,
        "dec": None,
    }
    assert full.metadata["qa_overview_xlim"] == pytest.approx(
        [full.spectrum.wave_rest.min(), full.spectrum.wave_rest.max()]
    )
    assert full.metadata["qa_overview_ymin"] == 0.0
    assert full.metadata["qa_overview_model_upper_limit"] >= np.max(
        full.continuum.model
        + sum(
            (fit.model for fit in full.line_complexes.values()),
            np.zeros_like(full.continuum.model),
        )
    )
    assert set(full.metadata["qa_zoom_model_upper_limits"]) == {
        "mgii",
        "oii_nev_neiii_hgamma",
        "hbeta_oiii",
        "halpha_nii_sii",
    }
    for complex_name, upper_limit in full.metadata["qa_zoom_model_upper_limits"].items():
        lo, hi = {
            "mgii": (2700.0, 2900.0),
            "oii_nev_neiii_hgamma": (3380.0, 4425.0),
            "hbeta_oiii": (4640.0, 5100.0),
            "halpha_nii_sii": (6400.0, 6800.0),
        }[complex_name]
        mask = full.spectrum.valid_mask & (full.spectrum.wave_rest >= lo) & (full.spectrum.wave_rest <= hi)
        complex_model = full.continuum.model + sum(
            (fit.model for fit in full.line_complexes.values()),
            np.zeros_like(full.continuum.model),
        )
        assert upper_limit >= np.max(complex_model[mask])
    assert set(full.metadata["qa_zoom_ymin"].values()) == {0.0}
    for name in (
        "mgii",
        "oii_nev_neiii_hgamma",
        "hbeta_oiii",
        "halpha_nii_sii",
    ):
        assert f"{full.line_complexes[name].reduced_chi2:.2f}" in full.metadata["qa_zoom_titles"][name]
    assert set(full.metadata["qa_zoom_line_labels"]["mgii"]) == {"Mg II"}
    assert set(full.metadata["qa_zoom_line_labels"]["hbeta_oiii"]) == {
        "Hβ",
        "[O III] 4960",
        "[O III] 5008",
    }
    assert set(full.metadata["qa_zoom_line_labels"]["halpha_nii_sii"]) == {
        "[N II] 6550",
        "Hα",
        "[N II] 6585",
        "[S II] 6718",
        "[S II] 6733",
    }
    assert set(full.metadata["qa_major_emission_line_labels"]) == {
        "Mg II",
        r"H$\beta$",
        "[O III] 5008",
        r"H$\alpha$",
    }
    assert {
        "mgii_measurements_csv",
        "halpha_nii_sii_measurements_csv",
    } <= set(files)
    assert files["summary_json"].endswith("qsospec_global_lines_summary.json")
    assert files["compatibility_summary_json"].endswith("qsospec_global_hbeta_summary.json")
    compatibility = qsospec.write_global_hbeta_products(
        full,
        str(tmp_path / "compatibility"),
        qa_plot_config=qsospec.GlobalQAPlotConfig(show_smoothed_data=True),
    )
    assert full.metadata["qa_smoothed_data"] is False
    assert compatibility["summary_json"].endswith("qsospec_global_hbeta_summary.json")
    assert compatibility["generic_summary_json"].endswith("qsospec_global_lines_summary.json")


def test_optional_fit_failure_preserves_legacy_success(monkeypatch):
    spectrum = _global_spectrum(np.linspace(2600.0, 7000.0, 2500))

    def fail(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(global_fit, "fit_halpha_complex", fail)
    result = qsospec.fit_global_lines(
        spectrum,
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    assert result.continuum_success
    assert result.legacy_hbeta_success
    assert result.halpha is not None
    assert not result.halpha.success
    assert "optional_line_fit_failed" in result.warning_codes()


def test_global_monte_carlo_includes_covered_optional_complexes():
    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 1800)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
        uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=1, random_seed=4),
    )
    percentiles = result.monte_carlo["percentiles"]
    assert result.monte_carlo["continuum_success_count"] == 1
    assert result.monte_carlo["complex_success_counts"]["hbeta_oiii"] == 1
    assert "MgII_broad_fwhm_kms" in percentiles
    assert "Hb_broad_fwhm_kms" in percentiles
    assert "Ha_broad_fwhm_kms" in percentiles


def test_host_refit_monte_carlo_includes_optional_complexes(monkeypatch):
    spectrum = _global_spectrum(np.linspace(2600.0, 7000.0, 1600))
    spectrum_data = SpectrumData(
        wave_obs=spectrum.wave_obs,
        flux=spectrum.flux,
        error=spectrum.err,
        redshift=0.0,
        object_id="synthetic",
    )

    def fake_host_subtraction(data, **kwargs):
        fit_spectrum = qsospec.Spectrum.from_arrays(
            data.wave_obs,
            data.flux,
            err=data.uncertainty(),
            wave_frame="rest",
            survey="desi",
        )
        host = np.zeros_like(data.flux)
        return fit_spectrum, fit_spectrum, None, None, host, data.flux.copy(), []

    monkeypatch.setattr(host_workflow, "_host_subtracted_spectrum", fake_host_subtraction)
    result = host_workflow._run_host_refit_mc(
        spectrum_data,
        n_trials=1,
        seed=3,
        redshift=0.0,
        template_root="unused",
        template_file="unused",
        host_fit_range=(3600.0, 7000.0),
        host_config=None,
        source="synthetic",
        global_config=_simple_global_config(),
        hbeta_config=qsospec.HbetaComplexConfig(fit_oiii_wings=False),
        mgii_config=qsospec.MgIIComplexConfig(),
        halpha_config=qsospec.HalphaComplexConfig(),
    )
    assert result["continuum_success_count"] == 1
    assert result["complex_success_counts"]["hbeta_oiii"] == 1
    assert "MgII_broad_fwhm_kms" in result["percentiles"]
    assert "Ha_broad_fwhm_kms" in result["percentiles"]


def test_qa_percentiles_and_component_styles():
    values = np.arange(1001, dtype=float)
    lo, hi = _percentile_limits([values], percentiles=(1.0, 99.8), pad=0.0)
    assert lo == pytest.approx(np.percentile(values, 1.0))
    assert hi == pytest.approx(np.percentile(values, 99.8))
    assert _rounded_model_upper_limit(np.array([0.0, 11.0])) == 15.0
    assert _rounded_model_upper_limit(np.array([0.0, 4.0])) == 5.0

    wave = np.linspace(6350.0, 6850.0, 1200)
    continuum = np.ones_like(wave)
    line = _gaussian_area_profile(wave, 50.0, 6564.61, 2200.0)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        continuum + line,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        flux_unit="relative",
    )
    fit = qsospec.fit_halpha_complex(spectrum, _continuum_result(spectrum, continuum))
    kinds = [kind for _, _, _, kind in _line_groups("halpha", fit)]
    assert kinds.count("broad") == 1
    assert set(kinds) <= {"broad", "narrow", "wing"}
    assert _COMBINED_BROAD_STYLE["color"] != _BROAD_COMPONENT_STYLE["color"]
    assert _BROAD_COMPONENT_STYLE["color"] == "#30aecf"
    assert _COMBINED_BROAD_STYLE["linestyle"] == "-"
    assert _BROAD_COMPONENT_STYLE["linestyle"] == "-"
    assert _NARROW_STYLE["linestyle"] == "-"
    assert _CONTINUUM_STYLES["balmer_bound_free"][0] == "#db9c4b"
    assert _CONTINUUM_STYLES["balmer_bound_free"][0] != _NARROW_STYLE["color"]
    assert _WING_STYLE["color"] == "#d80835"
    assert _WING_STYLE["linestyle"] == "-"
    assert _WING_STYLE["linewidth"] == pytest.approx(0.9)
    assert _HOST_STYLE["linestyle"] == "-"
    assert _HOST_STYLE["linewidth"] == pytest.approx(0.9)
    assert _WING_STYLE["color"] != _CONTINUUM_STYLES["uv_iron"][0]
    assert _CONTINUUM_STYLES["uv_iron"] == _CONTINUUM_STYLES["optical_iron"]
    assert _CONTINUUM_STYLES["power_law"][0] == "#b03766"
    assert _CONTINUUM_STYLES["balmer_bound_free"] == _CONTINUUM_STYLES["balmer_high_order_series"]
    assert {style[1] for style in _CONTINUUM_STYLES.values()} == {"-", "--", "-."}
    assert _TCC_COLORS["total_model"] == "#28292b"
    assert _TCC_COLORS["unmodelled_span"] == "#e4ecf0"
    label = _flux_density_axis_label("1e-17 erg cm^-2 s^-1 Angstrom^-1")
    assert "F_\\lambda" in label
    assert "\\mathrm{\\AA}" in label
    assert _flux_density_axis_label("relative f_lambda") == (
        r"$F_\lambda$ [relative units]"
    )


def test_qa_flux_display_scales_are_display_only():
    wave = np.linspace(4000.0, 5000.0, 20)
    physical = qsospec.Spectrum.from_arrays(
        wave,
        np.full_like(wave, 2.0e-17),
        err=np.full_like(wave, 1.0e-18),
        wave_frame="rest",
        flux_unit="cgs",
        flux_scale=1.0,
    )
    desi_scaled = qsospec.Spectrum.from_arrays(
        wave,
        np.full_like(wave, 2.0),
        err=np.full_like(wave, 0.1),
        wave_frame="rest",
        flux_unit="cgs",
        flux_scale=1.0e-17,
    )
    other_scaled = qsospec.Spectrum.from_arrays(
        wave,
        np.full_like(wave, 2.0),
        err=np.full_like(wave, 0.1),
        wave_frame="rest",
        flux_unit="cgs",
        flux_scale=2.5e-16,
    )
    relative = qsospec.Spectrum.from_arrays(
        wave,
        np.full_like(wave, 2.0),
        err=np.full_like(wave, 0.1),
        wave_frame="rest",
        flux_unit="relative",
    )
    original = physical.flux.copy()
    assert _flux_display_scale(physical) == pytest.approx(1.0e17)
    assert _flux_display_scale(desi_scaled) == pytest.approx(1.0)
    assert _flux_display_scale(other_scaled) == pytest.approx(25.0)
    assert _flux_display_scale(relative) == pytest.approx(1.0)
    np.testing.assert_array_equal(physical.flux, original)
    assert "F_\\lambda" in _flux_density_axis_label(physical)
    assert "10^{-17}" in _flux_density_axis_label(physical)
    assert _flux_density_axis_label(relative) == (
        r"$F_\lambda$ [relative units]"
    )


def test_qa_applies_display_scale_without_mutating_fit_arrays(
    tmp_path,
    monkeypatch,
):
    import matplotlib.pyplot as plt

    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(3400.0, 7000.0, 1200)),
        _simple_global_config(),
        complexes=[],
    )
    result.spectrum.metadata.flux_unit = "cgs"
    result.spectrum.metadata.flux_scale = 2.5e-16
    original_flux = result.spectrum.flux.copy()
    original_model = result.continuum.model.copy()
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    _plot_qa(
        result,
        tmp_path / "scaled.png",
        qsospec.GlobalQAPlotConfig(show_smoothed_data=False),
    )
    figure = plt.gcf()
    observed = next(
        line
        for line in figure.axes[0].lines
        if line.get_label() == "Input spectrum"
    )
    expected_scale = 25.0
    np.testing.assert_allclose(
        observed.get_ydata(),
        expected_scale * original_flux[result.spectrum.valid_mask],
    )
    np.testing.assert_array_equal(result.spectrum.flux, original_flux)
    np.testing.assert_array_equal(result.continuum.model, original_model)
    assert result.metadata["qa_flux_display_scale"] == pytest.approx(
        expected_scale
    )
    real_close(figure)


def test_qa_plot_config_and_selection_contract(monkeypatch):
    assert qsospec.GlobalQAPlotConfig() == qsospec.GlobalQAPlotConfig(
        figure_width=10.5,
        figure_height=8.0,
        max_zoom_panels=4,
        show_smoothed_data=True,
        smooth_original_spectrum_for_display=False,
        smoothing_window_pixels=7,
        show_residual_panel=True,
        show_fit_regions=True,
        unmodelled_windows=((1170.0, 1275.0, "Lyα"),),
    )
    with pytest.raises(ValueError):
        qsospec.GlobalQAPlotConfig(smoothing_window_pixels=4)

    successful = type("SuccessfulFit", (), {"success": True})()
    monkeypatch.setitem(global_io._COMPLEX_WINDOWS, "civ", (1450.0, 1700.0))
    displayed, omitted = _select_zoom_complexes(
        {
            "civ": successful,
            "halpha": successful,
            "mgii": successful,
            "hbeta": successful,
        },
        3,
    )
    assert displayed == ("mgii", "hbeta", "halpha")
    assert omitted == ("civ",)

    fully_covered_blue = type(
        "BlueFit",
        (),
        {"success": True, "metadata": {"qa_all_lines_covered": True}},
    )()
    partially_covered_blue = type(
        "BlueFit",
        (),
        {"success": True, "metadata": {"qa_all_lines_covered": False}},
    )()
    assert _select_zoom_complexes({"oii_nev_neiii_hgamma": fully_covered_blue}, 4)[0] == ("oii_nev_neiii_hgamma",)
    assert _select_zoom_complexes({"oii_nev_neiii_hgamma": partially_covered_blue}, 4)[0] == ()


def test_qa_fit_masks_residuals_and_region_precedence(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 2200)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    failed_mgii = replace(result.mgii, success=False)
    result.mgii = failed_mgii
    result.line_complexes = {
        **result.line_complexes,
        "mgii": failed_mgii,
    }
    host_fit_mask = np.ones_like(result.spectrum.valid_mask)
    host_emission_mask = (result.spectrum.wave_rest >= 5600.0) & (result.spectrum.wave_rest <= 5650.0)
    result.host_fit_mask = host_fit_mask
    result.host_emission_mask = host_emission_mask
    result.metadata["host_mask_provenance"] = "exact"
    config = qsospec.GlobalQAPlotConfig()
    fitted, unmodelled, ppxf_masked = _final_fit_masks(result, config)

    assert np.all(~fitted[(result.spectrum.wave_rest >= 2700.0) & (result.spectrum.wave_rest <= 2900.0)])
    assert np.any(unmodelled[(result.spectrum.wave_rest >= 2700.0) & (result.spectrum.wave_rest <= 2900.0)])
    assert np.any(ppxf_masked)
    assert not np.any(ppxf_masked & fitted)
    assert not np.any(unmodelled & fitted)

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    _plot_qa(result, tmp_path / "regions.png", config)
    figure = plt.gcf()
    overview = figure.axes[0]
    residual = figure.axes[1]
    total_model_line = next(line for line in overview.lines if line.get_label() == "total model")
    assert np.all(
        np.isfinite(
            total_model_line.get_ydata()[result.spectrum.valid_mask]
        )
    )
    residual_line = residual.lines[0]
    residual_values = residual_line.get_ydata()
    expected = (
        result.spectrum.flux[fitted]
        - (
            result.continuum.model[fitted]
            + sum(
                (fit.model[fitted] for fit in result.line_complexes.values() if fit.success),
                np.zeros(np.count_nonzero(fitted)),
            )
        )
    ) / result.spectrum.err[fitted]
    np.testing.assert_allclose(residual_values[fitted], expected)
    assert np.all(np.isnan(residual_values[~fitted]))
    assert result.metadata["qa_n_unmodelled_pixels"] > 0
    assert result.metadata["qa_n_ppxf_masked_pixels"] > 0
    assert result.metadata["qa_host_mask_provenance"] == "exact"
    real_close(figure)


def test_lya_overview_uses_unsmoothed_data_only_percentile(tmp_path):
    wave = np.linspace(1100.0, 2000.0, 1800)
    flux = 2.0 * (wave / 1500.0) ** -1.1
    flux += 400.0 * np.exp(-0.5 * ((wave - 1215.67) / 2.0) ** 2)
    valid = np.ones_like(wave, dtype=bool)
    valid[np.argmin(np.abs(wave - 1215.67))] = False
    flux[~valid] = 1.0e8
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.05),
        mask=valid,
        wave_frame="rest",
        flux_unit="relative",
    )
    result = qsospec.fit_global_lines(
        spectrum,
        qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
            continuum_windows=((1100.0, 2000.0),),
            mask_windows=(),
            clip_passes=0,
        ),
        complexes=[],
    )
    _plot_qa(
        result,
        tmp_path / "lya.png",
        qsospec.GlobalQAPlotConfig(show_smoothed_data=True),
    )
    expected = _percentile_limits(
        [spectrum.flux[spectrum.valid_mask]],
        percentiles=(1.0, 99.8),
    )[1]
    assert result.metadata["qa_overview_upper_policy"] == "data_only_percentile"
    assert result.metadata["qa_overview_upper_percentile"] == 99.8
    assert result.metadata["qa_overview_model_upper_limit"] == pytest.approx(expected)


def test_qa_title_smoothing_and_tick_helpers():
    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 1800)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    assert _qa_overview_title(result) == ""
    result.metadata.update(
        {
            "object_id": "abc",
            "redshift": 0.75,
            "ra": 151.123456,
            "dec": -2.345678,
            "galactic_extinction": {
                "status": "applied",
                "applied_ebv": 0.12345,
            },
        }
    )
    assert _qa_overview_title(result) == (
        "Object abc   z = 0.7500\n"
        "RA = 151.12346   Dec = -2.34568   $E(B-V) = 0.1235$"
    )
    assert _input_spectrum_label(result) == (
        "Input spectrum\nMW extinction corrected"
    )
    assert _input_spectrum_label(result, smoothed=True) == (
        "Input spectrum\n"
        "MW extinction corrected\nsmoothed for display"
    )
    assert (
        _qa_overview_title(
            result,
            qsospec.GlobalQAPlotConfig(
                object_name="My Quasar",
                object_label="Source",
                show_coordinates=False,
            ),
        )
        == "Source My Quasar   z = 0.7500"
    )

    smoothed = _masked_running_median(
        np.array([1.0, 100.0, 3.0, 5.0, 7.0]),
        np.array([True, False, True, True, True]),
        3,
    )
    assert smoothed == pytest.approx([1.0, np.nan, 4.0, 5.0, 6.0], nan_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    figure, axis = plt.subplots()
    _configure_qa_axis(axis)
    assert axis.xaxis.get_tick_params(which="major")["direction"] == "in"
    assert axis.xaxis.get_tick_params(which="major")["top"] is True
    assert axis.yaxis.get_tick_params(which="major")["right"] is True
    assert not isinstance(axis.xaxis.get_minor_locator(), NullLocator)
    assert not isinstance(axis.yaxis.get_minor_locator(), NullLocator)
    axis.set_xlim(4800.0, 5050.0)
    _annotate_emission_lines(
        axis,
        (
            (4862.68, r"H$\beta$"),
            (4960.30, "[O III] 4960"),
            (5008.24, "[O III] 5008"),
        ),
        y_fraction=0.82,
    )
    text_positions = {text.get_text(): text.get_position()[1] for text in axis.texts}
    assert text_positions[r"H$\beta$"] == pytest.approx(0.82)
    assert text_positions["[O III] 4960"] == pytest.approx(0.68)
    assert text_positions["[O III] 5008"] == pytest.approx(0.68)
    assert all(text.get_color() == _TCC_COLORS["line_marker"] for text in axis.texts)
    axis.set_xlim(3650.0, 3800.0)
    _annotate_emission_lines(
        axis,
        ((3728.47, "[O II] 3728"),),
        y_fraction=0.82,
    )
    assert axis.texts[-1].get_position()[1] == pytest.approx(0.68)
    plt.close(figure)


def test_qa_fixed_dimensions_smoothing_and_legends(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    full = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 4001)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    variants = {
        "one": replace(
            full,
            mgii=None,
            halpha=None,
            line_complexes={"hbeta": full.hbeta},
            metadata={},
        ),
        "two": replace(
            full,
            halpha=None,
            line_complexes={"mgii": full.mgii, "hbeta": full.hbeta},
            metadata={},
        ),
        "three": replace(full, metadata={}),
    }
    pixel_sizes = set()
    for name, result in variants.items():
        path = tmp_path / f"{name}.png"
        _plot_qa(result, path)
        image = plt.imread(path)
        pixel_sizes.add(image.shape[:2])
    assert pixel_sizes == {(1280, 1680)}
    assert variants["one"].metadata["qa_overview_annotation"] == {
        "continuum_reduced_chi2": variants["one"].continuum.reduced_chi2,
        "host_state": "not decomposed",
        "zoom_spectrum": "input",
        "host_fractions": "",
        "ra": None,
        "dec": None,
    }

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    smoothed_path = tmp_path / "smoothed.png"
    _plot_qa(
        variants["three"],
        smoothed_path,
        qsospec.GlobalQAPlotConfig(show_smoothed_data=True),
    )
    figure = plt.gcf()
    overview_axis = figure.axes[0]
    assert figure._supxlabel.get_text() == r"Rest wavelength [$\mathrm{\AA}$]"
    assert figure._supylabel is None
    assert all(axis.get_xlabel() == "" for axis in figure.axes)
    assert figure.axes[1].get_ylabel() == r"$\Delta/\sigma$"
    assert "F_\\lambda" in figure.axes[0].get_ylabel()
    assert "F_\\lambda" in figure.axes[2].get_ylabel()
    assert all(axis.get_ylabel() == "" for axis in figure.axes[3:])
    overview_labels = overview_axis.get_legend_handles_labels()[1]
    assert overview_labels.count(
        "Input spectrum"
    ) == 1
    assert overview_labels.count(
        "Input spectrum\nsmoothed for display"
    ) == 1
    assert overview_labels.count("Fe II") <= 1
    assert overview_labels.count("broad-line model") == 1
    assert overview_labels.count("narrow-line model") == 1
    assert variants["three"].metadata["qa_residual_definition"] == ("(data-model)/sigma")
    assert variants["three"].metadata["qa_n_residual_pixels"] > 0
    assert variants["three"].metadata["qa_smoothed_data"] is True
    assert variants["three"].metadata["qa_shared_axis_labels"] is False
    assert variants["three"].metadata["qa_y_label_policy"] == (
        "overview_and_leftmost_zoom"
    )
    real_close(figure)


@pytest.mark.parametrize(
    ("pixel_count", "expect_smoothed"),
    ((4000, False), (4001, True)),
)
def test_qa_smoothing_pixel_threshold(
    tmp_path,
    monkeypatch,
    pixel_count,
    expect_smoothed,
):
    import matplotlib.pyplot as plt

    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, pixel_count)),
        _simple_global_config(),
        complexes=[],
    )
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    _plot_qa(
        result,
        tmp_path / f"smoothing-{pixel_count}.png",
        qsospec.GlobalQAPlotConfig(
            show_smoothed_data=True,
            smooth_original_spectrum_for_display=True,
        ),
    )
    figure = plt.gcf()
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert (
        "Input spectrum\nsmoothed for display"
        in labels
    ) is expect_smoothed
    assert (
        "Input spectrum" in labels
    ) is (not expect_smoothed)
    assert result.metadata["qa_smoothing_effective"] is expect_smoothed
    assert result.metadata[
        "qa_smoothing_suppressed_short_spectrum"
    ] is (not expect_smoothed)
    real_close(figure)


def test_host_context_companion_plot(tmp_path, monkeypatch):
    import matplotlib
    import matplotlib.pyplot as plt

    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 4001)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    host = 0.45 * (result.spectrum.wave_rest / 5100.0) ** -0.4
    result.host_decomp_enabled = True
    result.host_model_on_quasar_grid = host
    result.total_spectrum = qsospec.Spectrum.from_arrays(
        result.spectrum.wave_rest,
        result.spectrum.flux + host,
        err=result.spectrum.err,
        wave_frame="rest",
        flux_unit="relative",
    )
    result.metadata.update(
        {
            "object_id": "host-test",
            "redshift": 0.5,
            "ra": 151.123456,
            "dec": -2.345678,
            "host_decomp_enabled": True,
            "continuum_samples": {
                "fracHost_3000": 0.2,
                "fracHost_5100": 0.3,
            },
        }
    )

    assert _has_host_context(result)
    assert "20.0\\%" in _host_fraction_annotation(result)
    assert "30.0\\%" in _host_fraction_annotation(result)
    files = qsospec.write_global_line_products(
        result,
        str(tmp_path),
        qsospec.GlobalQAPlotConfig(write_other_diagnostics=True),
    )

    assert files["host_context_plot"].endswith("diagnostic_global_host_context.png")
    assert result.metadata["host_context_plot_created"] is True
    assert result.metadata["host_context_figure_size_inches"] == [10.5, 5.2]
    assert result.metadata["host_context_ymin"] == 0.0
    assert result.metadata["host_context_fraction_annotation"]
    image = plt.imread(files["host_context_plot"])
    assert image.shape[:2] == (832, 1680)

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: None)
    original_font_family = list(matplotlib.rcParams["font.family"])
    qa_path = tmp_path / "qa_with_host_overview.png"
    _plot_qa(
        result,
        qa_path,
        qsospec.GlobalQAPlotConfig(
            show_host_context_in_overview=True,
            smooth_original_spectrum_for_display=True,
            smoothing_window_pixels=7,
        ),
    )
    figure = plt.gcf()
    overview_labels = figure.axes[0].get_legend_handles_labels()[1]
    assert (
        "Input spectrum\nsmoothed for display"
        in overview_labels
    )
    assert "Input spectrum" not in overview_labels
    assert "host galaxy" in overview_labels
    assert "total model" in overview_labels
    assert "continuum model (extrapolated)" not in overview_labels
    for axis in figure.axes[2:]:
        zoom_labels = axis.get_legend_handles_labels()[1]
        assert "Input spectrum" not in zoom_labels
        assert "host galaxy" not in zoom_labels
    assert result.metadata["qa_host_context_overview_requested"] is True
    assert result.metadata["qa_host_context_overview_used"] is True
    assert result.metadata["qa_overview_annotation"]["zoom_spectrum"] == (
        "host-subtracted"
    )
    assert any(
        "Zoom panels: host-subtracted spectrum" in text.get_text()
        for text in figure.axes[0].texts
    )
    assert "RA = 151.12346" in figure.axes[0].get_title()
    assert "Dec = -2.34568" in figure.axes[0].get_title()
    assert result.metadata["qa_original_spectrum_smoothed_for_display"] is True
    assert result.metadata["qa_original_spectrum_smoothed_used"] is True
    assert result.metadata["qa_smoothing_suppressed_short_spectrum"] is False
    assert result.metadata["qa_plot_style"] == "qsospec_science_serif"
    assert figure.axes[0].title.get_fontfamily() == ["serif"]
    assert list(matplotlib.rcParams["font.family"]) == original_font_family
    original_line = next(
        line
        for line in figure.axes[0].lines
        if line.get_label()
        == "Input spectrum\nsmoothed for display"
    )
    expected_smoothed = _masked_running_median(
        result.total_spectrum.flux,
        result.total_spectrum.valid_mask,
        7,
    )
    np.testing.assert_allclose(
        original_line.get_ydata(),
        expected_smoothed[result.total_spectrum.valid_mask],
    )
    assert "20.0\\%" in result.metadata["qa_overview_annotation"]["host_fractions"]
    assert "30.0\\%" in result.metadata["qa_overview_annotation"]["host_fractions"]
    real_close(figure)


def test_host_context_overview_falls_back_without_host(tmp_path):
    result = qsospec.fit_global_lines(
        _global_spectrum(np.linspace(2600.0, 7000.0, 1800)),
        _simple_global_config(),
        qsospec.HbetaComplexConfig(fit_oiii_wings=False),
    )
    _plot_qa(
        result,
        tmp_path / "qa_without_host.png",
        qsospec.GlobalQAPlotConfig(
            show_host_context_in_overview=True,
            smooth_original_spectrum_for_display=True,
        ),
    )
    assert result.metadata["qa_host_context_overview_requested"] is True
    assert result.metadata["qa_host_context_overview_used"] is False
    assert result.metadata["qa_original_spectrum_smoothed_used"] is False
    assert result.metadata["qa_smoothing_suppressed_short_spectrum"] is True
