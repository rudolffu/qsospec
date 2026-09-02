from types import SimpleNamespace

import numpy as np
import pytest

import qsospec


def _template_library(tmp_path):
    wave = np.linspace(3200.0, 23000.0, 64)
    templates = np.column_stack(
        [
            np.ones_like(wave),
            (wave / 5100.0) ** -1.0,
            np.exp(-0.5 * ((wave - 5000.0) / 900.0) ** 2),
        ]
    )
    path = tmp_path / "tiny_emiles.npz"
    np.savez(path, wave=wave, templates=templates)
    return qsospec.workflows.host.load_ppxf_npz_templates(
        template_root=str(tmp_path),
        template_file=path.name,
        write_report=False,
    )


def test_compact_state_reconstructs_exact_stellar_host_sed(tmp_path):
    library = _template_library(tmp_path)
    fit = SimpleNamespace(
        templates=library,
        stellar_template_scales=np.array([1.0, 2.0, 0.5]),
        stellar_weights=np.array([0.7, 0.2, 0.1]),
        preprocessed=SimpleNamespace(normalization=3.5, redshift=0.42),
        warnings=["fit_warning"],
        strategy_used="agn_pseudocontinuum_masked",
        host_reconstruction_state={},
    )

    original = qsospec.workflows.host.predict_host_sed(fit)
    reconstructed = qsospec.reconstruct_host_sed_from_state(
        fit.host_reconstruction_state,
        template_root=str(tmp_path),
    )

    np.testing.assert_array_equal(reconstructed.wave_rest, original.wave_rest)
    np.testing.assert_allclose(
        reconstructed.host_flux, original.host_flux, rtol=0.0, atol=0.0
    )
    assert reconstructed.samples == original.samples
    assert reconstructed.flags == original.flags
    assert "fit_warning" in reconstructed.warnings
    assert reconstructed.provenance["reconstructed_without_ppxf"] is True


def test_compact_state_rejects_template_hash_mismatch(tmp_path):
    library = _template_library(tmp_path)
    fit = SimpleNamespace(
        templates=library,
        stellar_template_scales=np.ones(3),
        stellar_weights=np.ones(3),
        preprocessed=SimpleNamespace(normalization=1.0, redshift=0.1),
        warnings=[],
        strategy_used="masked_simple",
        host_reconstruction_state={},
    )
    qsospec.workflows.host.predict_host_sed(fit)
    state = dict(fit.host_reconstruction_state)
    state["template_file_sha256"] = "0" * 64

    with pytest.raises(qsospec.HostSEDReconstructionError) as caught:
        qsospec.reconstruct_host_sed_from_state(
            state,
            template_root=str(tmp_path),
        )
    assert caught.value.code == "template_hash_mismatch"


def test_reconstructed_host_sed_never_extrapolates(tmp_path):
    library = _template_library(tmp_path)
    fit = SimpleNamespace(
        templates=library,
        stellar_template_scales=np.ones(3),
        stellar_weights=np.ones(3),
        preprocessed=SimpleNamespace(normalization=1.0, redshift=0.1),
        warnings=[],
        strategy_used="masked_simple",
        host_reconstruction_state={},
    )
    qsospec.workflows.host.predict_host_sed(fit)
    sed = qsospec.reconstruct_host_sed_from_state(
        fit.host_reconstruction_state,
        template_root=str(tmp_path),
    )
    values, warnings = qsospec.workflows.host.predict_host_sed_on_grid(
        sed, np.array([3000.0, 5100.0, 24000.0])
    )
    assert np.isnan(values[0]) and np.isnan(values[-1])
    assert np.isfinite(values[1])
    assert warnings == ["host_sed_grid_outside_template_coverage"]
