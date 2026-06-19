"""Structured warning behavior for qsospec."""

import numpy as np

import qsospec


def test_line_center_near_edge_warning():
    wave = np.linspace(4800.0, 4920.0, 80)
    flux = np.ones_like(wave)
    err = np.ones_like(wave) * 0.1
    spec = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0)
    config = qsospec.LocalFitConfig(windows=[qsospec.recipes.local_hbeta()], edge_buffer=80.0)

    result = qsospec.fit_local(spec, config)

    assert "line_center_near_edge" in result.warning_codes()


def test_all_pixels_invalid_warning():
    wave = np.linspace(4800.0, 4920.0, 80)
    flux = np.ones_like(wave)
    err = np.ones_like(wave) * -1.0
    spec = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0)
    config = qsospec.LocalFitConfig(windows=[qsospec.recipes.local_hbeta()])

    result = qsospec.fit_local(spec, config)

    assert not result.success
    assert "all_pixels_invalid" in result.warning_codes()
