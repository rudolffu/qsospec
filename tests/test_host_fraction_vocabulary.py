import json
from types import SimpleNamespace

import numpy as np
import pytest

import qsospec
from qsospec.global_result import GlobalContinuumResult, WorkflowResult
from qsospec.io.run_store import (
    RunStore,
    canonicalize_measurement_rows,
    workflow_payload,
)
from qsospec.measurement_vocabulary import (
    FINAL_HOST_FRACTION_DEFINITION_ID,
    HOST_FRACTION_DELTA_DEFINITION_ID,
    PPXF_HOST_FRACTION_DEFINITION_ID,
)
from qsospec.workflows.host.ppxf_host import fitted_host_fraction_samples


def _metadata_dict(row):
    return {
        item["key"]: json.loads(item["value"])
        for item in row.get("metadata", [])
    }


def _workflow(*, legacy_host_names=False):
    wave = np.linspace(3900.0, 5200.0, 80)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        np.full_like(wave, 4.0),
        err=np.full_like(wave, 0.1),
        z=0.0,
        wave_frame="rest",
        flux_unit="relative",
        galactic_extinction_corrected=True,
    )
    continuum = GlobalContinuumResult(
        success=True,
        status=1,
        message="synthetic",
        param_values={"power_law.norm": 1.0},
        param_errors={},
        covariance=None,
        chi2=1.0,
        dof=79,
        reduced_chi2=1.0 / 79.0,
        wave_rest=wave,
        model=np.ones_like(wave),
        component_models={"power_law": np.ones_like(wave)},
        fit_mask=np.ones_like(wave, dtype=bool),
        clip_mask=np.ones_like(wave, dtype=bool),
    )
    host_samples = {
        "fHost_pPXF_5100": 3.0,
        "fAGN_pPXF_5100": 2.0,
        "fTotal_pPXF_5100": 5.0,
        "fracHost_pPXF_5100": 0.6,
    }
    if legacy_host_names:
        host_samples = {
            "fHostFit_5100": 3.0,
            "fAGNFit_5100": 2.0,
            "fTotalFit_5100": 5.0,
            "fracHost_5100": 0.6,
        }
    return WorkflowResult(
        spectrum=spectrum,
        continuum_initial=continuum,
        continuum=continuum,
        metadata={
            "object_id": "vocabulary-object",
            "host_strategy_used": "agn_pseudocontinuum_masked",
            "continuum_samples": {
                "fHost_5100": 3.0,
                "fAGN_5100": 1.0,
                "fracHost_5100": 0.75,
            },
            "host_fit_samples": host_samples,
        },
    )


@pytest.mark.parametrize(
    "strategy", ["masked_simple", "agn_pseudocontinuum_masked"]
)
def test_direct_ppxf_sample_producer_uses_v2_names_and_values(strategy):
    wave = np.array([3900.0, 4000.0, 5100.0, 5200.0])
    fit = SimpleNamespace(
        preprocessed=SimpleNamespace(wave_rest=wave),
        host_model=np.full_like(wave, 3.0),
        agn_model=np.full_like(wave, 2.0),
        total_model=np.full_like(wave, 5.0),
        strategy_used=strategy,
    )

    samples = fitted_host_fraction_samples(fit)

    assert samples["fHost_pPXF_5100"] == pytest.approx(3.0)
    assert samples["fAGN_pPXF_5100"] == pytest.approx(2.0)
    assert samples["fTotal_pPXF_5100"] == pytest.approx(5.0)
    assert samples["fracHost_pPXF_5100"] == pytest.approx(0.6)
    assert not any("Fit_" in name for name in samples)
    assert "fracHost_5100" not in samples
    assert np.isnan(samples["fracHost_pPXF_8000"])


def test_naming_helpers_and_section_aware_legacy_mapping():
    assert qsospec.final_host_sample_name("host_fraction", 5100) == (
        "fracHost_5100"
    )
    assert qsospec.ppxf_host_sample_name("host_fraction", 5100) == (
        "fracHost_pPXF_5100"
    )
    assert qsospec.canonicalize_legacy_measurement_name(
        "host_sample", "fHostFit_5100"
    )[0] == "fHost_pPXF_5100"
    assert qsospec.canonicalize_legacy_measurement_name(
        "host_sample", "fracHost_5100"
    )[0] == "fracHost_pPXF_5100"
    assert qsospec.canonicalize_legacy_measurement_name(
        "continuum_sample", "fracHost_5100"
    ) == ("fracHost_5100", {})


def test_run_store_rows_definitions_units_delta_and_manifest(tmp_path):
    result = _workflow()
    store = RunStore.create(
        str(tmp_path / "new-run"), configuration={"test": "vocabulary-v2"}
    )
    payload = workflow_payload(
        result,
        run_id=store.run_id,
        object_key="memory:vocabulary-object",
        object_id="vocabulary-object",
        input_record={"source": "memory", "reader": "memory"},
    )
    store.write_payload(payload)

    assert store.manifest["measurement_vocabulary_version"] == "2"
    rows = store.read_measurements().to_pylist()
    by_quantity = {row["quantity"]: row for row in rows}
    final = by_quantity["fracHost_5100"]
    direct = by_quantity["fracHost_pPXF_5100"]
    delta = by_quantity["deltaFracHost_final_pPXF_5100"]

    assert final["section"] == "continuum_sample"
    assert final["value"] == pytest.approx(0.75)
    assert final["unit"] == "dimensionless"
    assert final["method"] == "ppxf_host_plus_qsospec_agn"
    assert _metadata_dict(final)["definition_id"] == (
        FINAL_HOST_FRACTION_DEFINITION_ID
    )
    assert direct["section"] == "host_sample"
    assert direct["value"] == pytest.approx(0.6)
    assert direct["unit"] == "dimensionless"
    assert direct["method"] == "ppxf_component_interpolation"
    assert _metadata_dict(direct)["definition_id"] == (
        PPXF_HOST_FRACTION_DEFINITION_ID
    )
    assert by_quantity["fHost_pPXF_5100"]["unit"] == (
        "input_flux_density"
    )
    assert delta["value"] == pytest.approx(0.15)
    assert delta["unit"] == "dimensionless"
    assert _metadata_dict(delta)["definition_id"] == (
        HOST_FRACTION_DELTA_DEFINITION_ID
    )


def test_historical_raw_and_canonical_reads_and_wide_catalog(tmp_path):
    result = _workflow(legacy_host_names=True)
    store = RunStore.create(
        str(tmp_path / "legacy-run"), configuration={"test": "legacy-v1"}
    )
    payload = workflow_payload(
        result,
        run_id=store.run_id,
        object_key="memory:vocabulary-object",
        object_id="vocabulary-object",
        input_record={"source": "memory", "reader": "memory"},
    )
    store.write_payload(payload)

    raw_quantities = set(
        store.read_measurements(canonical=False).column("quantity").to_pylist()
    )
    canonical_rows = store.read_measurements(canonical=True).to_pylist()
    canonical_quantities = {row["quantity"] for row in canonical_rows}
    assert "fHostFit_5100" in raw_quantities
    assert "fracHost_5100" in raw_quantities
    assert "fHost_pPXF_5100" not in raw_quantities
    assert {"fracHost_5100", "fracHost_pPXF_5100"}.issubset(
        canonical_quantities
    )
    mapped = next(
        row
        for row in canonical_rows
        if row["quantity"] == "fracHost_pPXF_5100"
    )
    assert _metadata_dict(mapped)["legacy_quantity_name"] == (
        "fracHost_5100"
    )

    catalog = qsospec.build_science_catalog(
        store,
        {
            "fracHost_5100": {
                "section": "continuum_sample",
                "quantity": "fracHost_5100",
                "include_error": False,
            },
            "fracHost_pPXF_5100": {
                "section": "host_sample",
                "quantity": "fracHost_pPXF_5100",
                "include_error": False,
            },
        },
    )
    assert catalog.loc[0, "fracHost_5100"] == pytest.approx(0.75)
    assert catalog.loc[0, "fracHost_pPXF_5100"] == pytest.approx(0.6)

    loaded = qsospec.load_model(store, "vocabulary-object")
    assert loaded.metadata["host_fit_samples"]["fracHost_pPXF_5100"] == (
        pytest.approx(0.6)
    )
    assert "fracHost_5100" not in loaded.metadata["host_fit_samples"]


def test_canonicalization_rejects_mixed_legacy_and_v2_collision():
    common = {
        "object_key": "one",
        "section": "host_sample",
        "value": 0.6,
        "metadata": [],
    }
    with pytest.raises(ValueError, match="Canonical host-measurement collision"):
        canonicalize_measurement_rows(
            [
                {**common, "quantity": "fracHost_5100"},
                {**common, "quantity": "fracHost_pPXF_5100"},
            ]
        )


def test_v1_vocabulary_run_is_readable_but_not_resumable(tmp_path):
    path = tmp_path / "historical-run"
    configuration = {"test": "historical-vocabulary"}
    RunStore.create(str(path), configuration=configuration)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("measurement_vocabulary_version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    opened = RunStore.open(str(path))
    assert "measurement_vocabulary_version" not in opened.manifest
    with pytest.raises(ValueError, match="Cannot resume a run written with"):
        RunStore.create(str(path), configuration=configuration, resume=True)
