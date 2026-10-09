"""Plotting smoke tests for qsospec results."""

import numpy as np
import pytest

import qsospec


pytestmark = pytest.mark.plotting


@pytest.fixture
def masked_global_result():
    """Known full-grid models with flagged pixels and unusable uncertainties."""
    from qsospec.global_result import GlobalContinuumResult, EmissionComplexResult, WorkflowResult

    wave = np.linspace(2700.0, 2900.0, 160)
    continuum_model = 2.0 * (wave / 2800.0) ** -1.2
    line_model = 0.5 * np.exp(-0.5 * ((wave - 2798.) / 15.) ** 2)
    flux = continuum_model + line_model
    err = np.full_like(wave, 0.05)
    err[20] = 0.
    flux[21] = np.nan
    flux[22] = np.inf
    good = np.ones(wave.size, dtype=bool)
    good[70:75] = False
    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, mask=good, wave_frame="rest", flux_unit="relative")
    continuum = GlobalContinuumResult(
        success=True, status=1, message="synthetic fixture", param_values={}, param_errors={}, covariance=None,
        chi2=0., dof=1, reduced_chi2=0., wave_rest=wave, model=continuum_model,
        component_models={"power_law": continuum_model}, fit_mask=spectrum.valid_mask.copy(),
        clip_mask=spectrum.valid_mask.copy(),
    )
    line = EmissionComplexResult(
        success=True, status=1, message="synthetic fixture", selected_model="mgii", param_values={},
        param_errors={}, covariance=None, metrics={}, metric_errors={}, chi2=0., dof=1, reduced_chi2=0., bic=0.,
        wave_rest=wave, flux_continuum_subtracted=flux-continuum_model, err=err, model=line_model,
        component_models={"MgII_broad1": line_model}, fit_mask=spectrum.valid_mask.copy(),
    )
    return WorkflowResult(spectrum=spectrum, continuum_initial=continuum, continuum=continuum,
                          line_complexes={"mgii": line}, complex_statuses={"mgii": "fit"})


def test_global_qa_marks_invalid_pixels_and_draws_full_model(masked_global_result):
    import matplotlib.pyplot as plt

    result = masked_global_result
    original_model = result.continuum.model.copy()
    original_mask = result.spectrum.valid_mask.copy()
    default = result.plot_qa()
    assert not any(c.get_label() == "excluded input pixels" for c in default.axes[0].collections)
    data_trace = next(line for line in default.axes[0].lines if line.get_label() == "Input spectrum")
    finite = np.isfinite(result.spectrum.flux)
    np.testing.assert_allclose(data_trace.get_ydata()[finite], result.spectrum.flux[finite])
    plt.close(default)
    figure = result.plot_qa(mark_excluded_pixels=True)
    overview, residual, zoom = figure.axes
    invalid = ~original_mask & np.isfinite(result.spectrum.flux)
    markers = next(c for c in overview.collections if c.get_label() == "excluded input pixels")
    np.testing.assert_allclose(markers.get_offsets()[:, 0], result.spectrum.wave_rest[invalid])
    np.testing.assert_allclose(markers.get_offsets()[:, 1], result.spectrum.flux[invalid])
    assert markers.get_edgecolors()[0, 0] > markers.get_edgecolors()[0, 1]  # red crosses
    expected = result.continuum.model + result.line_complexes["mgii"].model
    model_trace = next(line for line in overview.lines if line.get_label() == "total model")
    np.testing.assert_allclose(model_trace.get_ydata(), expected)
    assert np.isfinite(model_trace.get_ydata()[invalid]).all()
    assert any(np.array_equal(line.get_ydata(), expected) for line in zoom.lines)
    assert any(len(c.get_offsets()) == invalid.sum() for c in zoom.collections)
    assert np.isnan(residual.lines[0].get_ydata()[~original_mask]).all()
    assert result.metadata["qa_n_residual_pixels"] == original_mask.sum()
    assert result.metadata["qa_n_marked_excluded_pixels"] == invalid.sum() == 6
    np.testing.assert_array_equal(result.spectrum.valid_mask, original_mask)
    np.testing.assert_array_equal(result.continuum.model, original_model)
    plt.close(figure)

    hidden = result.plot_qa(qsospec.GlobalQAPlotConfig(mark_excluded_pixels=True), mark_excluded_pixels=False)
    assert not any(c.get_label() == "excluded input pixels" for c in hidden.axes[0].collections)
    assert result.metadata["qa_n_marked_excluded_pixels"] == 0
    plt.close(hidden)


def test_archived_and_saved_qa_use_same_masked_pixel_renderer(masked_global_result, tmp_path):
    import matplotlib.pyplot as plt
    from qsospec.io.run_store import RunStore, workflow_payload

    result = masked_global_result
    store = RunStore.create(str(tmp_path / "run"), configuration={}, model_storage="arrays")
    store.write_payload(workflow_payload(result, run_id=store.run_id, object_key="masked", object_id="masked",
                                        input_record={"source": "synthetic fixture"}))
    figure = store.plot_qa("masked", mark_excluded_pixels=True)
    assert any(c.get_label() == "excluded input pixels" for c in figure.axes[0].collections)
    model = next(line for line in figure.axes[0].lines if line.get_label() == "total model")
    assert np.isfinite(model.get_ydata()).all()
    plt.close(figure)
    qsospec.write_global_line_products(result, str(tmp_path / "qa"),
                                      qa_plot_config=qsospec.GlobalQAPlotConfig(mark_excluded_pixels=True))
    assert result.metadata["qa_n_marked_excluded_pixels"] == 6
    assert result.metadata["qa_n_model_pixels"] == len(result.spectrum.wave_rest)


def test_plot_local_result_writes_png(tmp_path):
    wave = np.linspace(4700.0, 5100.0, 160)
    flux = 1.0 + 5.0 * np.exp(-0.5 * ((wave - 4861.33) / 20.0) ** 2)
    err = np.full_like(wave, 0.08)
    spec = qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=err,
        z=0.0,
        wave_frame="rest",
        survey="desi",
    )
    config = qsospec.LocalFitConfig(windows=[qsospec.recipes.local_hbeta()])
    result = qsospec.fit_local(spec, config)

    combined = qsospec.plot_local_result(result, tmp_path / "combined.png")
    per_window = qsospec.save_local_window_plots(result, tmp_path / "windows")

    assert (tmp_path / "combined.png").exists()
    assert combined.endswith("combined.png")
    assert "Hb_OIII" in per_window
    assert (tmp_path / "windows" / "Hb_OIII_qsospec.png").exists()
