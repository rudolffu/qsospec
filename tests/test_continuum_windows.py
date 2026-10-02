import numpy as np
import pytest
from qsospec.continuum_windows import (
    WindowConfig,
    measure_window,
    corrected_rest_arrays,
)

C = WindowConfig(11480, 12480, 12000)


def fixture():
    w = np.arange(11480.0, 12481.0, 10.0)
    f = (2 + (w - 12000) / 1000) * 1e-17
    return w, f, np.full(len(w), 1e-36), np.ones(len(w), bool)


def test_fixed_anchor_and_order():
    w, f, v, m = fixture()
    a = measure_window(w, f, v, m, C)
    b = measure_window(w[::-1], f[::-1], v[::-1], m[::-1], C)
    assert a == b
    assert a["continuum_f_lambda_rest"] == pytest.approx(2e-17, abs=1e-30)
    assert a["effective_wavelength"] != C.anchor
    assert a["local_slope"] == pytest.approx(1e-20, abs=1e-32)


def test_negative_pixels_and_nonpositive_anchor():
    w, f, v, m = fixture()
    f[:] = -1e-17
    a = measure_window(w, f, v, m, C)
    assert a["n_valid"] == len(w) and a["measurement_status"] == "nonpositive"
    assert a["continuum_f_lambda_rest"] < 0


def test_gap_and_edge():
    w, f, v, m = fixture()
    m[abs(w - 12000) < 100] = False
    assert "masked_anchor_gap" in measure_window(w, f, v, m, C)["measurement_reasons"]
    m = w > 12000
    assert not measure_window(w, f, v, m, C)["anchor_bracketed"]


def test_missing_uncertainty_and_correlated_scale():
    w, f, v, m = fixture()
    a = measure_window(w, f, v, m, C, component="stellar_subtracted")
    assert np.isnan(a["total_error"]) and a["host_scale_parameter_covariance"] is None
    a = measure_window(
        w,
        f,
        v,
        m,
        C,
        component="stellar_subtracted",
        scale_perturbation=np.ones(len(w)) * 3e-18,
    )
    assert a["host_scale_error"] == pytest.approx(3e-18, abs=1e-30)
    assert np.isnan(a["total_error"])


def test_frame_units_and_no_double_correction():
    from qsospec.extinction import galactic_dereddening_factor

    w = np.linspace(12000, 18000, 30)
    f = np.ones(30) * 2
    iv = np.ones(30) * 4
    z = 0.3
    fac = galactic_dereddening_factor(w, 0.04, rv=3.1, law="f99")
    a = corrected_rest_arrays(w, f, iv, redshift=z, native_cgs_scale=1e-16, ebv=0.04)
    b = corrected_rest_arrays(
        w,
        f * fac,
        iv / fac**2,
        redshift=z,
        native_cgs_scale=1e-16,
        ebv=0.04,
        already_corrected=True,
    )
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        a[2], 1 / iv * fac**2 * (1 + z) ** 2 * 1e-32, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(1 / a[2], iv / fac**2 / (1 + z) ** 2 / 1e-32, rtol=1e-14)
