"""Coverage, fitting, archive, and QA tests for Lyα/N V."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import qsospec
from qsospec.fitting.complexes import (
    GenericComplexContext,
    classify_lya_coverage,
    fit_lya_nv_complex,
    lya_absorption_mask,
)
from qsospec.global_result import GlobalContinuumResult
from qsospec.io.run_store import RunStore, workflow_payload


def _spectrum(lower, upper, count=900, mask=None):
    wave = np.linspace(lower, upper, count)
    return qsospec.Spectrum.from_arrays(
        wave,
        2.0 * (wave / 1250.0) ** -1.2,
        err=np.full_like(wave, 0.03),
        mask=mask,
        wave_frame="rest",
        flux_unit="relative",
    )


@pytest.mark.parametrize(
    ("lower", "upper", "status"),
    [
        (1170.0, 1290.0, "full"),
        (1208.0, 1290.0, "red_side_only"),
        (1220.0, 1290.0, "edge_truncated"),
        (1270.0, 1290.0, "not_covered"),
    ],
)
def test_lya_coverage_states(lower, upper, status):
    coverage = classify_lya_coverage(_spectrum(lower, upper))
    assert coverage.status == status
    assert coverage.edge_truncated is (status == "edge_truncated")


def test_lya_coverage_uses_sampled_valid_fraction():
    spectrum = _spectrum(1150.0, 1290.0)
    valid = spectrum.valid_mask.copy()
    valid[::2] = False
    masked = qsospec.Spectrum.from_arrays(
        spectrum.wave_rest,
        spectrum.flux,
        err=spectrum.err,
        mask=valid,
        wave_frame="rest",
        flux_unit="relative",
    )
    coverage = classify_lya_coverage(masked)
    assert coverage.valid_pixel_fraction == pytest.approx(0.5, rel=0.01)
    assert coverage.status == "edge_truncated"


def test_lya_safe_continuum_preset_and_explicit_override():
    safe = qsospec.GlobalContinuumConfig.lya_safe()
    assert (1150.0, 1170.0) not in safe.continuum_windows
    assert (1275.0, 1290.0) in safe.continuum_windows
    assert all(not (lo < 1275.0 and hi > 1170.0) for lo, hi in safe.continuum_windows)

    spectrum = _spectrum(1150.0, 2050.0, 1800)
    automatic = qsospec.fit_global_lines(
        spectrum,
        complexes=["lya_nv"],
        uncertainty_config=qsospec.UncertaintyConfig(covariance=False),
    )
    explicit_config = qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        continuum_windows=((1150.0, 1170.0), (1275.0, 1290.0)),
        mask_windows=(),
        clip_passes=0,
    )
    explicit = qsospec.fit_global_lines(
        spectrum,
        explicit_config,
        complexes=["lya_nv"],
        uncertainty_config=qsospec.UncertaintyConfig(covariance=False),
    )
    assert automatic.metadata["continuum_preset"] == "lya_safe"
    assert (1150.0, 1170.0) not in automatic.continuum.metadata["continuum_windows"]
    assert explicit.metadata["continuum_preset"] == "explicit"
    assert explicit.continuum.metadata["continuum_windows"] == [
        (1150.0, 1170.0),
        (1275.0, 1290.0),
    ]


def test_lya_recipe_modes_and_width_only_tie():
    default = qsospec.recipes.get("lya_nv")
    assert default.auto_enabled
    assert default.backend == "lya_adapter"
    assert default.components[0].multiplicity == 2
    assert default.components[1].line_ids == ("nv_blend",)

    equal = qsospec.complex_recipes.lya_nv_recipe(qsospec.LyaNVComplexConfig(nv_mode="equal_doublet"))
    assert equal.components[1].line_ids == ("nv_1239", "nv_1243")

    tied = qsospec.complex_recipes.lya_nv_recipe(
        qsospec.LyaNVComplexConfig(
            lya_num_broad_gaussians=2,
            nv_num_broad_gaussians=2,
            nv_fwhm_bands_kms=((1200.0, 5000.0), (5000.0, 20000.0)),
            tie_nv_width_to_lya=True,
        )
    )
    context = GenericComplexContext(
        tied,
        tuple(component.id for component in tied.components),
        50.0,
    )
    width_names = [name for name in context.names if name.endswith(".fwhm_kms")]
    velocity_names = [name for name in context.names if name.endswith(".velocity_kms")]
    assert len(width_names) == 2
    assert len(velocity_names) == 4


def test_lya_absorption_mask_rejects_only_narrow_runs():
    wave = np.linspace(1150.0, 1290.0, 1401)
    residual = np.zeros_like(wave)
    residual[300:304] = -5.0
    residual[700:900] = -5.0
    mask = lya_absorption_mask(
        wave,
        residual,
        np.ones_like(wave, dtype=bool),
        qsospec.LyaNVComplexConfig(absorption_dilation_pixels=1),
    )
    assert np.all(mask[299:305])
    assert not np.any(mask[700:900])


def _synthetic_lya_result(
    *,
    lower=1150.0,
    add_absorption=True,
    covariance=True,
    config=None,
):
    config = config or qsospec.LyaNVComplexConfig()
    recipe = qsospec.complex_recipes.lya_nv_recipe(config)
    wave = np.linspace(lower, 1400.0, 1800)
    continuum_model = 2.0 * (wave / 1250.0) ** -1.2
    context = GenericComplexContext(
        recipe,
        tuple(component.id for component in recipe.components),
        80.0,
    )
    theta = context.initial.copy()
    flux_values = [90.0, 45.0, 25.0]
    for name, value in zip(context.linear_names, flux_values):
        theta[context.index[name]] = value
    line_model = context.model(theta, wave)
    flux = continuum_model + line_model
    if add_absorption:
        flux -= 1.5 * np.exp(-0.5 * ((wave - 1192.0) / 0.35) ** 2)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        flux,
        err=np.full_like(wave, 0.03),
        wave_frame="rest",
        flux_unit="relative",
    )
    continuum = GlobalContinuumResult(
        True,
        1,
        "known",
        {},
        {},
        None,
        0.0,
        1,
        0.0,
        wave.copy(),
        continuum_model,
        {"power_law": continuum_model},
        spectrum.valid_mask.copy(),
        spectrum.valid_mask.copy(),
    )
    return (
        fit_lya_nv_complex(
            spectrum,
            continuum,
            recipe,
            config,
            compute_covariance=covariance,
        ),
        spectrum,
        continuum,
    )


def test_lya_full_recovery_absorption_mask_and_metrics():
    result, _, _ = _synthetic_lya_result()
    assert result.success
    assert result.metadata["lya_coverage_status"] == "full"
    assert result.metadata["lya_absorption_masked_fraction"] > 0
    assert np.any(result.excluded_mask)
    assert result.metrics["lya_1216_broad_flux_input"] == pytest.approx(135.0, rel=0.08)
    assert np.isfinite(result.metrics["lya_1216_broad_fwhm_kms"])
    assert np.isfinite(result.metrics["nv_blend_broad_ew_rest"])


def test_lya_red_side_is_limited_and_edge_is_skipped():
    limited, _, _ = _synthetic_lya_result(
        lower=1208.0,
        add_absorption=False,
    )
    assert limited.success
    assert limited.metadata["lya_coverage_status"] == "red_side_only"
    assert limited.metadata["lya_fit_status"] == "limited"
    assert limited.metadata["lya_fit_reliable"] is False

    edge_spectrum = _spectrum(1220.0, 1400.0)
    continuum = GlobalContinuumResult(
        True,
        1,
        "known",
        {},
        {},
        None,
        0.0,
        1,
        0.0,
        edge_spectrum.wave_rest.copy(),
        edge_spectrum.flux.copy(),
        {"power_law": edge_spectrum.flux.copy()},
        edge_spectrum.valid_mask.copy(),
        edge_spectrum.valid_mask.copy(),
    )
    edge = fit_lya_nv_complex(
        edge_spectrum,
        continuum,
        qsospec.complex_recipes.lya_nv_recipe(),
    )
    assert not edge.success
    assert edge.metadata["lya_fit_status"] == "edge_truncated"
    assert not np.any(edge.model)


def test_lya_unreliable_warning_outcomes():
    no_covariance, _, _ = _synthetic_lya_result(covariance=False)
    assert no_covariance.metadata["lya_fit_reliable"] is False
    assert "lya_reliability_covariance_unavailable" in (no_covariance.warning_codes())

    absorption_dominated, _, _ = _synthetic_lya_result(
        config=qsospec.LyaNVComplexConfig(reliable_max_absorption_fraction=0.0)
    )
    assert absorption_dominated.metadata["lya_fit_reliable"] is False
    assert "lya_absorption_dominated" in absorption_dominated.warning_codes()


def test_lya_schema_v4_round_trip(tmp_path):
    fit, spectrum, continuum = _synthetic_lya_result()
    workflow = qsospec.WorkflowResult(
        spectrum=spectrum,
        total_spectrum=spectrum,
        continuum_initial=continuum,
        continuum=continuum,
        line_complexes={"lya_nv": fit},
        complex_statuses={"lya_nv": fit.metadata["lya_fit_status"]},
        metadata={
            "object_id": "lya-object",
            "lya_coverage_status": "full",
        },
    )
    store = RunStore.create(
        str(tmp_path / "run"),
        configuration={"complexes": ["lya_nv"]},
    )
    store.write_payload(
        workflow_payload(
            workflow,
            run_id=store.run_id,
            object_key="lya-object",
            object_id="lya-object",
            input_record={
                "source": "memory",
                "row_index": 0,
                "reader": "memory",
                "metadata": {},
            },
        )
    )
    loaded = qsospec.load_model(store, "lya-object")
    loaded_fit = loaded.line_complexes["lya_nv"]
    assert store.manifest["schema_version"] == "4"
    assert loaded_fit.metadata["lya_coverage_status"] == "full"
    np.testing.assert_array_equal(
        loaded_fit.excluded_mask,
        fit.excluded_mask,
    )

    model_path = next((store.path / "data" / "models").glob("*.parquet"))
    old_rows = pq.read_table(model_path).to_pylist()
    for item in old_rows[0]["complexes"]:
        item.pop("excluded_mask", None)
        item.pop("metadata", None)
    pq.write_table(pa.Table.from_pylist(old_rows), model_path)
    store.manifest["schema_version"] = "2"
    store._write_manifest()
    legacy = qsospec.load_model(str(store.path), "lya-object")
    legacy_fit = legacy.line_complexes["lya_nv"]
    assert legacy_fit.metadata == {"recipe_id": "lya_nv"}
    assert not np.any(legacy_fit.excluded_mask)


def test_lya_qa_zoom_marks_absorption(tmp_path):
    fit, spectrum, continuum = _synthetic_lya_result()
    workflow = qsospec.WorkflowResult(
        spectrum=spectrum,
        total_spectrum=spectrum,
        continuum_initial=continuum,
        continuum=continuum,
        line_complexes={"lya_nv": fit},
        complex_statuses={"lya_nv": "fit"},
        metadata={
            "object_id": "lya-qa",
            "redshift": 2.0,
            "lya_coverage_status": "full",
        },
    )
    files = qsospec.write_global_line_products(workflow, str(tmp_path))
    assert files["qa_plot"].endswith(".png")
    assert workflow.metadata["qa_displayed_complexes"] == ["lya_nv"]
    assert workflow.metadata["qa_lya_model_fitted"] is True
