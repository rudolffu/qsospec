"""Standalone import and compatibility checks."""


def test_qsospec_public_imports_work():
    import qsospec
    from qsospec import Spectrum, fit_line_complex

    assert qsospec.Spectrum is Spectrum
    assert fit_line_complex is qsospec.fit_line_complex
    assert qsospec.__version__ == "0.1.0"


def test_deprecated_neofit_names_warn():
    import pytest
    import qsospec

    with pytest.deprecated_call(match="NeoFitWorkflowResult"):
        workflow_type = qsospec.NeoFitWorkflowResult
    with pytest.deprecated_call(match="NeoFitHostWorkflowResult"):
        host_type = qsospec.NeoFitHostWorkflowResult
    with pytest.deprecated_call(match="NeoFitWarning"):
        warning_type = qsospec.NeoFitWarning

    assert workflow_type is qsospec.WorkflowResult
    assert host_type is qsospec.HostWorkflowResult
    assert warning_type is qsospec.FitWarning


def test_bundled_data_loads_from_qsospec_namespace():
    import qsospec

    assert qsospec.load_iron_template("vw01_uv").wave_rest.size > 0
    assert qsospec.load_balmer_template().n_upper.size > 0
