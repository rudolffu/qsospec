"""Plotting smoke tests for qsospec results."""

import numpy as np

import qsospec


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
