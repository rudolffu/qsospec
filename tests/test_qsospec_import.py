"""Standalone import and compatibility checks."""


def test_qsospec_public_imports_work():
    import qsospec
    from qsospec import Spectrum, fit_line_complex

    assert qsospec.Spectrum is Spectrum
    assert fit_line_complex is qsospec.fit_line_complex
    assert qsospec.__version__ == "0.1.1"
    assert hasattr(qsospec, "BalmerPseudoContinuumConfig")
    assert not hasattr(qsospec, "BalmerContinuumConfig")
    assert not hasattr(qsospec, "BalmerSeriesConfig")


def test_bundled_data_loads_from_qsospec_namespace():
    import qsospec

    assert qsospec.load_iron_template("vw01_uv").wave_rest.size > 0
    assert qsospec.load_balmer_template().n_upper.size > 0
