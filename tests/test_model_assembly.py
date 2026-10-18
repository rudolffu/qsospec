"""Combined QA/CSV models use geometric windows, never local extrapolation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import qsospec
from qsospec.global_result import EmissionComplexResult, GlobalContinuumResult, WorkflowResult
from qsospec.io import products
from qsospec.observed_model import _reconstruct_line_model, reconstruct_observed_model


@pytest.fixture
def windowed_result():
    wave = np.arange(3300., 13501., 5.)
    valid = ~np.isin(wave, [4900., 10000.])
    def spectrum(flux):
        return qsospec.Spectrum.from_arrays(
            wave, np.full(wave.shape, flux), err=np.ones_like(wave), z=.5,
            wave_frame="rest", mask=valid, flux_unit="relative")
    spectrum_fit = spectrum(8.)
    continuum = GlobalContinuumResult(
        success=True, status=1, message="known synthetic model", param_values={"amp": 4.},
        param_errors={"amp": .1}, covariance=np.array([[.01]]), chi2=1., dof=1,
        reduced_chi2=1., wave_rest=wave, model=np.full_like(wave, 4.),
        component_models={"power_law": np.full_like(wave, 4.)},
        fit_mask=valid.copy(), clip_mask=valid.copy())
    def fit(values, name):
        values = np.asarray(values, dtype=float)
        return EmissionComplexResult(
            success=True, status=1, message="known synthetic model", selected_model=name,
            param_values={"amplitude": 3.}, param_errors={"amplitude": .2}, covariance=np.array([[.04]]),
            metrics={"synthetic_flux_cgs": 3.}, metric_errors={"synthetic_flux_cgs": .2},
            chi2=1., dof=1, reduced_chi2=1., bic=1., wave_rest=wave,
            flux_continuum_subtracted=spectrum_fit.flux-continuum.model,
            err=np.ones_like(wave), model=values,
            component_models={"local_continuum_intercept": values}, fit_mask=valid.copy())
    complexes = {
        "hbeta_oiii": fit(3. - .01*(wave-4870.), "hbeta_oiii"),
        "paschen_nir": fit(np.ones_like(wave), "paschen_nir"),
        "padelta": fit(np.full_like(wave, 3.), "padelta"),
    }
    return WorkflowResult(
        spectrum=spectrum_fit, total_spectrum=spectrum(10.),
        continuum_initial=continuum, continuum=continuum,
        host_model_on_quasar_grid=np.full_like(wave, 2.),
        line_complexes=complexes, complex_statuses={name: "fit" for name in complexes})


def _snapshot(result):
    """Scientific arrays and quantities, excluding renderer bookkeeping."""
    return deepcopy(dict(
        spectrum=vars(result.spectrum), total_spectrum=vars(result.total_spectrum),
        host=result.host_model_on_quasar_grid, continuum=vars(result.continuum),
        complexes={name: vars(fit) for name, fit in result.line_complexes.items()}))


def _assert_same(before, after):
    if isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            _assert_same(before[key], after[key])
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after)
    else:
        assert before == after


def test_windows_stop_local_continuum_extrapolation_and_preserve_masked_pixels(windowed_result):
    result = windowed_result
    before = _snapshot(result)
    wave = result.spectrum.wave_rest
    actual = products._full_line_model(result)
    naive = sum(fit.model for fit in result.line_complexes.values())
    assert naive[wave == 8000.][0] < -20.
    assert actual[wave == 8000.][0] == 0.
    assert actual[wave == 4900.][0] == pytest.approx(2.7)
    assert not result.spectrum.valid_mask[wave == 4900.][0]
    # Narrow Pa delta owns the overlap; the umbrella recipe has disjoint windows.
    assert actual[wave == 10000.][0] == 3.
    assert not result.line_complexes["padelta"].fit_mask[wave == 10000.][0]
    assert actual[wave == 10300.][0] == 0.
    assert actual[wave == 10800.][0] == 1.
    result.line_complexes = dict(reversed(list(result.line_complexes.items())))
    np.testing.assert_array_equal(products._full_line_model(result), actual)
    _assert_same(before, _snapshot(result))


@pytest.mark.parametrize("frame", ["observed", "rest"])
@pytest.mark.parametrize("scale", [1.e-17, 2.5e-19])
def test_host_once_native_and_physical_models_agree(windowed_result, frame, scale):
    result = windowed_result
    for name in ("spectrum", "total_spectrum"):
        spectrum = getattr(result, name)
        setattr(result, name, replace(spectrum, metadata=replace(
            spectrum.metadata, flux_unit="cgs", flux_scale=scale, flux_frame=frame)))
    before = _snapshot(result)
    line = products._full_line_model(result)
    agn = result.continuum.model + line
    total = agn + result.host_model_on_quasar_grid
    physical = reconstruct_observed_model(result)
    factor = scale/(1.+result.spectrum.z) if frame == "rest" else scale
    np.testing.assert_allclose(total*factor, physical["model"], rtol=1.e-14)
    np.testing.assert_allclose((result.continuum.model+result.host_model_on_quasar_grid)*factor,
                               physical["continuum"], rtol=1.e-14)
    np.testing.assert_allclose(
        (result.total_spectrum.flux-total)/result.total_spectrum.err,
        (result.spectrum.flux-agn)/result.spectrum.err, atol=4.e-15, rtol=0.)
    _assert_same(before, _snapshot(result))


def test_relative_qa_needs_no_physical_flux_scale(windowed_result):
    assert windowed_result.spectrum.flux_scale is None
    assert np.isfinite(products._full_line_model(windowed_result)).all()
    with pytest.raises(ValueError, match="Physical flux scale"):
        reconstruct_observed_model(windowed_result)


def test_equal_width_overlap_tie_and_custom_disjoint_windows(windowed_result):
    result = windowed_result
    template = result.line_complexes["padelta"]
    result.line_complexes = {
        "custom_z": replace(template, model=np.full_like(template.model, 7.),
                            metadata={"fit_windows": ((4800., 4900.), (5000., 5100.))}),
        "custom_a": replace(template, model=np.full_like(template.model, 2.),
                            metadata={"fit_windows": ((4800., 4900.),)}),
    }
    assembled = _reconstruct_line_model(result, result.spectrum)
    wave = result.spectrum.wave_rest
    assert assembled["model"][wave == 4850.][0] == 2.
    assert assembled["complex_owner"][wave == 4850.][0] == "custom_a"
    assert assembled["model"][wave == 4950.][0] == 0.
    assert assembled["model"][wave == 5050.][0] == 7.
    del result.line_complexes["custom_a"].metadata["fit_windows"]
    with pytest.raises(ValueError, match="No registered or saved fitting windows"):
        products._full_line_model(result)


def test_failed_overlap_is_unsupported_and_never_adds_failed_model(windowed_result):
    result = windowed_result
    result.line_complexes["padelta"].success = False
    reconstruction = _reconstruct_line_model(result, result.spectrum)
    wave = result.spectrum.wave_rest
    assert reconstruction["model"][wave == 10000.][0] == 1.
    assert reconstruction["failed"][wave == 10000.][0]
    assert not reconstruction["failed"][wave == 10800.][0]
    result.spectrum = replace(result.spectrum, metadata=replace(
        result.spectrum.metadata, flux_unit="cgs", flux_scale=1.e-17))
    result.total_spectrum = result.spectrum
    physical = reconstruct_observed_model(result)
    assert not physical["supported"][wave == 10000.][0]
    assert physical["supported"][wave == 10800.][0]


def test_missing_model_values_are_gaps_and_not_extrapolated(windowed_result):
    result = windowed_result
    fit = result.line_complexes["hbeta_oiii"]
    subset = (fit.wave_rest >= 4700.) & (fit.wave_rest <= 5000.)
    fit.wave_rest = fit.wave_rest[subset]
    fit.model = fit.model[subset].copy()
    fit.model[fit.wave_rest == 4900.] = np.nan
    line = products._full_line_model(result)
    wave = result.spectrum.wave_rest
    assert np.isnan(line[wave == 4650.][0])
    assert np.isnan(line[wave == 4900.][0])
    assert np.isnan(line[wave == 5050.][0])
    assert line[wave == 4600.][0] == 0.
    assert line[wave == 5150.][0] == 0.


@pytest.mark.parametrize("show_host", [False, True])
def test_qa_overview_zoom_and_residual_use_matching_host_context(windowed_result, show_host):
    import matplotlib.pyplot as plt

    result = windowed_result
    before = _snapshot(result)
    line = products._full_line_model(result)
    agn = result.continuum.model+line
    expected = agn+result.host_model_on_quasar_grid if show_host else agn
    figure = result.plot_qa(qsospec.GlobalQAPlotConfig(show_host_context_in_overview=show_host))
    try:
        overview, residual, *zooms = figure.axes
        plotted = next(item for item in overview.lines if item.get_label() == "total model")
        np.testing.assert_allclose(plotted.get_ydata(), expected)
        for axis in zooms:
            black = [item for item in axis.lines if item.get_color() == products._TCC_COLORS["total_model"]]
            assert black
            y = black[0].get_ydata()
            selected = np.isfinite(y)
            np.testing.assert_allclose(y[selected], agn[selected])
        y = residual.lines[0].get_ydata()
        selected = np.isfinite(y)
        data = result.total_spectrum.flux if show_host else result.spectrum.flux
        np.testing.assert_allclose(y[selected], ((data-expected)/result.spectrum.err)[selected])
        assert np.isnan(y[~result.spectrum.valid_mask]).all()
        _assert_same(before, _snapshot(result))
    finally:
        plt.close(figure)


def test_legacy_csv_full_model_remains_host_subtracted_and_raw_arrays_survive(windowed_result, tmp_path):
    result = windowed_result
    before = _snapshot(result)
    line = products._full_line_model(result)
    files = qsospec.write_global_line_products(result, str(tmp_path))
    frame = pd.read_csv(files["full_grid_csv"])
    np.testing.assert_allclose(frame.full_model, result.continuum.model+line)
    np.testing.assert_allclose(frame.ppxf_host_model, result.host_model_on_quasar_grid)
    np.testing.assert_allclose(frame.flux_total_before_host, result.total_spectrum.flux)
    for name, fit in result.line_complexes.items():
        np.testing.assert_allclose(frame[f"{name}_model"], fit.model)
    other = pd.read_csv(files["compatibility_full_grid_csv"])
    pd.testing.assert_frame_equal(frame, other)
    _assert_same(before, _snapshot(result))


def test_registered_evidence_center_retains_recipe_envelope(windowed_result, monkeypatch):
    from qsospec import complex_recipes

    original = complex_recipes.get
    def changed_envelope(recipe_id):
        recipe = original(recipe_id)
        return replace(recipe, fit_window=(4000., 8000.)) if recipe_id == "hbeta_oiii" else recipe
    monkeypatch.setattr(complex_recipes, "get", changed_envelope)
    reconstruction = _reconstruct_line_model(windowed_result, windowed_result.spectrum)
    evidence = next(row for row in reconstruction["line_evidence"] if row["recipe_id"] == "hbeta_oiii")
    assert evidence["wavelength"] == 6000.*(1.+windowed_result.spectrum.z)
    wave = windowed_result.spectrum.wave_rest
    assert reconstruction["model"][wave == 8000.][0] == 0.


def test_host_context_companion_adds_host_once(windowed_result, monkeypatch, tmp_path):
    result = windowed_result
    before = _snapshot(result)
    captured = []
    def save(figure, paths):
        captured.append(figure)
        return {name: str(path) for name, path in paths.items()}
    monkeypatch.setattr(products, "_save_figure", save)
    products._plot_host_context(result, tmp_path/"host.png", config=qsospec.GlobalQAPlotConfig())
    top, bottom = captured[0].axes
    total = next(line for line in top.lines if line.get_label() == "host + final AGN model")
    agn = next(line for line in bottom.lines if line.get_label() == "final AGN + emission-line model")
    valid = result.spectrum.valid_mask
    expected_agn = result.continuum.model+products._full_line_model(result)
    np.testing.assert_allclose(agn.get_ydata(), expected_agn[valid])
    np.testing.assert_allclose(total.get_ydata(), (expected_agn+result.host_model_on_quasar_grid)[valid])
    _assert_same(before, _snapshot(result))
