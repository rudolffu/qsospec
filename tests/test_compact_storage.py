"""Numerical reconstruction and transactional persistence contracts."""

import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import qsospec
from qsospec.io.run_store import RunStore, workflow_payload, _from_key_values
from qsospec.templates.balmer import load_balmer_template, evaluate_balmer_series_with_derivatives
from qsospec.templates.balmer_cache import balmer_cache
from qsospec.model_evaluation import continuum_state, evaluate_continuum, encode, decode
from qsospec.fitting.global_fit import _ContinuumContext
from qsospec.io import indexed_store


def workflow():
    wave = np.linspace(3500.0, 5600.0, 300)
    flux = 3 * (wave / 4000) ** -1.2
    spectrum = qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.1), z=0, wave_frame="rest", flux_unit="relative"
    )
    config = qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
    )
    return qsospec.fit_global_lines(spectrum, config, complexes=[])


def write(store, result, key):
    return store.write_payload(
        workflow_payload(
            result,
            run_id=store.run_id,
            object_key=key,
            object_id=key,
            input_record=dict(source="memory", reader="memory"),
        )
    )


def test_cache_exact_eviction_mutation_and_isolation():
    template = load_balmer_template(log10_ne=9)
    wave = np.linspace(3500, 4500, 400)
    baseline = evaluate_balmer_series_with_derivatives(template, wave, 2000, 0)
    with balmer_cache(max_entries=1, max_bytes=100000) as cache:
        a = evaluate_balmer_series_with_derivatives(template, wave, 2000, 0)
        b = evaluate_balmer_series_with_derivatives(template, wave.copy(), 2000, 0)
        assert a is b and cache.hits == 1
        for expected, actual in zip(baseline, a):
            np.testing.assert_array_equal(expected, actual)
            with pytest.raises(ValueError):
                actual.setflags(write=True)
        evaluate_balmer_series_with_derivatives(template, wave, 2001, 0)
        assert len(cache.entries) == 1
        evaluate_balmer_series_with_derivatives(template, wave + 1, 2001, 0)
        assert cache.misses == 3
        original = template.rel_flux_hbeta.copy()
        try:
            template.rel_flux_hbeta = original * 2
            evaluate_balmer_series_with_derivatives(template, wave, 2001, 0)
            assert cache.misses == 4
        finally:
            template.rel_flux_hbeta = original
    with balmer_cache(max_bytes=1) as other:
        evaluate_balmer_series_with_derivatives(template, wave, 2000, 0)
        assert not other.entries and other.hits == 0


def test_cache_fit_outputs_exact():
    with balmer_cache(enabled=False):
        a = workflow()
    with balmer_cache():
        b = workflow()
    assert a.continuum.param_values == b.continuum.param_values
    np.testing.assert_array_equal(a.continuum.model, b.continuum.model)
    np.testing.assert_array_equal(a.continuum.covariance, b.continuum.covariance)


@pytest.mark.parametrize("power_mode", ["single", "double"])
@pytest.mark.parametrize("iron", [False, True])
def test_continuum_recipe_uses_same_evaluator(power_mode, iron):
    result = workflow()
    cfg = qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(mode=power_mode),
        uv_iron=None,
        optical_iron=qsospec.IronTemplateConfig(template="bg92") if iron else None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=True),
        clip_passes=0,
    )
    ctx = _ContinuumContext(result.spectrum, cfg)
    theta = ctx.initial
    parameters = dict(zip(ctx.names, theta))
    assets = {}
    recovered = decode(encode(continuum_state(ctx), assets), assets.__getitem__)
    original = ctx.components(theta, ctx.wave)
    restored = evaluate_continuum(recovered, parameters, ctx.wave)
    for name in original:
        np.testing.assert_array_equal(original[name], restored[name])


def test_default_parameter_storage_reload_conversion_and_catalog(tmp_path, monkeypatch):
    result = workflow()
    store = RunStore.create(str(tmp_path / "run"), configuration={})
    write(store, result, "a")
    row = store.object_row_by_key("models", "a")
    assert store.manifest["schema_version"] == "8"
    assert all(c["values"] is None for c in row["components"])
    assert all(c["definition"] for c in row["components"])
    loaded = qsospec.load_model(store, "a")
    np.testing.assert_array_equal(result.continuum.model, loaded.continuum.model)
    assert loaded.spectrum.z == result.spectrum.z
    monkeypatch.setattr(store, "read_asset", lambda *a: pytest.fail("scalar reads must not load assets"))
    assert qsospec.build_science_catalog(store).shape[0] == 1
    copy = qsospec.convert_run(store, tmp_path / "copy")
    np.testing.assert_array_equal(qsospec.load_model(copy, "a").continuum.model, result.continuum.model)
    with pytest.raises(FileExistsError):
        qsospec.convert_run(store, tmp_path / "copy")
    arrays = qsospec.convert_run(copy, tmp_path / "arrays", model_storage="arrays")
    assert all(c["values"] is not None for c in arrays.object_row_by_key("models", "a")["components"])


def test_array_fallback_reports_mismatch(tmp_path):
    result = workflow()
    result.continuum.component_models["power_law"][10] += 1
    store = RunStore.create(str(tmp_path / "run"), configuration={})
    write(store, result, "a")
    row = store.object_row_by_key("models", "a")
    outcome = _from_key_values(row["workflow_metadata"])["model_storage_components"][0]
    assert outcome["reason"] == "reconstruction_mismatch"
    assert row["components"][0]["values"] is not None


def test_packed_lookup_replacement_resume_and_rollover(tmp_path):
    store = RunStore.create(str(tmp_path / "run"), configuration={}, shard_objects=2)
    result = workflow()
    for key in ("a", "b", "c"):
        write(store, result, key)
    store.compact()
    assert len(list((store.path / "data/models").glob("*.parquet"))) == 2
    for key in ("a", "b", "c"):
        assert qsospec.load_model(store, key).spectrum.z == 0
    result.continuum.component_models["power_law"] *= 2
    write(store, result, "b")
    assert store.read_table("objects").num_rows == 3
    assert store.read_table("models").num_rows == 3
    np.testing.assert_array_equal(
        qsospec.load_model(store, "b").continuum.model, result.continuum.component_models["power_law"]
    )
    reopened = RunStore.open(str(store.path))
    assert reopened.reconcile_expected_keys(["a", "b", "c", "missing"])["completed_keys"] == {"a", "b", "c"}
    reopened.compact()
    assert reopened.read_table("models").num_rows == 3


def test_failed_generation_keeps_previous_and_staging(tmp_path, monkeypatch):
    store = RunStore.create(str(tmp_path / "run"), configuration={})
    result = workflow()
    write(store, result, "a")
    original = qsospec.load_model(store, "a").continuum.model.copy()
    payload = workflow_payload(result, run_id=store.run_id, object_key="a", object_id="a", input_record={})
    staging = store.stage_payload(payload)
    real = indexed_store._float_paths
    calls = []

    def fail(table):
        calls.append(1)
        if len(calls) > 1:
            raise OSError("injected publication failure")
        return real(table)

    monkeypatch.setattr(indexed_store, "_float_paths", fail)
    with pytest.raises(OSError, match="injected"):
        store.promote(staging)
    assert staging.exists()
    np.testing.assert_array_equal(qsospec.load_model(RunStore.open(str(store.path)), "a").continuum.model, original)
    monkeypatch.setattr(indexed_store, "_float_paths", real)
    store.promote(staging)
    assert not staging.exists()


def test_portable_assets_integrity(tmp_path):
    store = RunStore.create(str(tmp_path / "source"), configuration={})
    asset = np.arange(20.0).reshape(4, 5)
    token = qsospec.model_evaluation.array_identity(asset)
    indexed_store.write_asset(store, token, asset)
    shutil.move(store.path, tmp_path / "moved")
    moved = RunStore.open(str(tmp_path / "moved"))
    np.testing.assert_array_equal(moved.read_asset(token), asset)
    path = moved.path / "assets" / f"{token}.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        moved.read_asset(token)
    path.unlink()
    with pytest.raises(ValueError, match="Missing reconstruction asset"):
        moved.read_asset(token)


def test_forward_lsf_components_are_parameter_models(tmp_path):
    from test_adaptive_oiii import synthetic
    from dataclasses import replace
    from qsospec.resolution import SpectralResolution
    from qsospec.fitting.global_fit import fit_hbeta_complex
    from qsospec.global_result import WorkflowResult, GlobalContinuumResult

    resolution = SpectralResolution("sigma_lambda", values=np.array([2.0]))
    spectrum, cont, cfg = synthetic([(90, 60, 90)], resolution=resolution, z=0.5)
    fit = fit_hbeta_complex(spectrum, cont, replace(cfg, fit_oiii_wings=False))
    continuum = GlobalContinuumResult(
        True,
        1,
        "known",
        {},
        {},
        None,
        0,
        1,
        0,
        spectrum.wave_rest,
        cont.model,
        {"power_law": cont.model},
        spectrum.valid_mask,
        spectrum.valid_mask,
    )
    result = WorkflowResult(spectrum, continuum, continuum, hbeta=fit)
    store = RunStore.create(str(tmp_path / "run"), configuration={})
    write(store, result, "lsf")
    row = store.object_row_by_key("models", "lsf")
    assert all(c["definition"] for c in row["components"] if c["section"] == "complex")
    loaded = qsospec.load_model(store, "lsf")
    for name, values in fit.component_models.items():
        np.testing.assert_allclose(loaded.hbeta.component_models[name], values, rtol=1e-10, atol=1e-12)


def test_legacy_conversion_preserves_measurements_and_redshift(tmp_path):
    result = workflow()
    store = RunStore.create(str(tmp_path / "legacy"), configuration={}, model_storage="arrays")
    # Explicit fixture for a historical per-object layout.
    store.manifest.update(indexed_storage=False, schema_version="7")
    store.manifest["cosmology"] = {"name": "Planck18"}
    store._write_manifest(reconcile=False)
    write(store, result, "a")
    (store.path / "qa" / "example.txt").write_text("saved QA")
    before = {str(f.relative_to(store.path)): f.read_bytes() for f in store.path.rglob("*") if f.is_file()}
    converted = qsospec.convert_run(store, tmp_path / "converted")
    assert converted.manifest["cosmology"] == store.manifest["cosmology"]
    assert converted.manifest["source_manifest"]["schema_version"] == "7"
    assert (converted.path / "qa" / "example.txt").read_text() == "saved QA"
    after = {str(f.relative_to(store.path)): f.read_bytes() for f in store.path.rglob("*") if f.is_file()}
    assert before == after
    assert json.dumps(converted.read_table("measurements").to_pylist(), sort_keys=True) == json.dumps(
        store.read_table("measurements").to_pylist(), sort_keys=True
    )
    assert qsospec.load_model(converted, "a").spectrum.z == result.spectrum.z
    assert all(c["values"] is not None for c in converted.object_row_by_key("models", "a")["components"])


def test_restage_compact_records_without_materialized_arrays(tmp_path):
    source = RunStore.create(str(tmp_path / "source"), configuration={})
    write(source, workflow(), "a")
    destination = RunStore.create(str(tmp_path / "destination"), configuration={})
    for path in (source.path / "assets").glob("*.parquet"):
        indexed_store.write_asset(destination, path.stem, source.read_asset(path.stem))
    payload = {name: source.read_table(name).to_pylist() for name in source.manifest["tables"]}
    destination.write_payload({name: rows for name, rows in payload.items() if rows})
    np.testing.assert_array_equal(
        qsospec.load_model(source, "a").continuum.model, qsospec.load_model(destination, "a").continuum.model
    )


def test_snapshot_blocks_partial_cross_table_updates(tmp_path, monkeypatch):
    import threading
    import time
    import qsospec.io.run_store as module

    source = RunStore.create(str(tmp_path / "run"), configuration={})
    old = workflow()
    write(source, old, "a")
    new = workflow()
    new.continuum.param_values["power_law.norm"] *= 2
    new.continuum.component_models["power_law"] *= 2
    new.continuum.model *= 2
    started, finished, release = threading.Event(), threading.Event(), threading.Event()
    errors = []
    original = module._measurement_maps
    paused = False

    def slow(*args, **kwargs):
        nonlocal paused
        if not paused:
            paused = True
            started.set()
            assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_measurement_maps", slow)

    def writer():
        try:
            assert started.wait(10)
            write(RunStore.open(str(source.path)), new, "a")
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=writer)
    thread.start()

    def unlock():
        assert started.wait(10)
        time.sleep(0.1)
        assert not finished.is_set()
        release.set()

    unlock_thread = threading.Thread(target=unlock)
    unlock_thread.start()
    loaded = qsospec.load_model(source, "a")
    thread.join(10)
    unlock_thread.join(10)
    assert not errors and finished.is_set()
    np.testing.assert_array_equal(loaded.continuum.model, old.continuum.model)
    assert loaded.continuum.param_values == old.continuum.param_values
    latest = qsospec.load_model(source, "a")
    np.testing.assert_array_equal(latest.continuum.model, new.continuum.model)


def test_storage_policy_and_byte_limit(tmp_path):
    store = RunStore.create(str(tmp_path / "run"), configuration={}, shard_bytes=1)
    for key in ("a", "b"):
        write(store, workflow(), key)
    store.compact()
    assert len(list((store.path / "data/models").glob("*.parquet"))) == 2
    with pytest.raises(ValueError, match="Cannot change model_storage"):
        RunStore.create(str(store.path), configuration={}, model_storage="arrays")
    with pytest.raises(ValueError, match="outside the source"):
        qsospec.convert_run(store, store.path / "nested")


def test_unknown_evaluator_fails_without_refitting(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from qsospec.io.run_store import SCHEMAS

    store = RunStore.create(str(tmp_path / "run"), configuration={})
    write(store, workflow(), "a")
    row = store.object_row_by_key("models", "a")
    recipe = json.loads(row["components"][0]["definition"])
    recipe["version"] = 999
    row["components"][0]["definition"] = json.dumps(recipe)
    path = store.object_shard_path("models", "a")
    pq.write_table(pa.Table.from_pylist([row], schema=SCHEMAS["models"]), path)
    monkeypatch.setattr(qsospec, "fit_global_lines", lambda *a, **k: pytest.fail("reload must not fit"))
    with pytest.raises(ValueError, match="Unsupported model evaluator version"):
        qsospec.load_model(store, "a")


def test_fixed_hgamma_balmer_reconstruction(tmp_path):
    from qsospec.fitting.global_fit import _fit_global_continuum_with_fixed_balmer_amplitude

    result = workflow()
    config = qsospec.GlobalContinuumConfig(uv_iron=None, optical_iron=None, clip_passes=0)
    continuum = _fit_global_continuum_with_fixed_balmer_amplitude(
        result.spectrum, config, amplitude=15.0, fwhm_kms=2500.0, velocity_kms=40.0, compute_covariance=True
    )
    result.continuum = continuum
    result.continuum_initial = continuum
    store = RunStore.create(str(tmp_path / "run"), configuration={})
    write(store, result, "a")
    assert all(c["definition"] for c in store.object_row_by_key("models", "a")["components"])
    loaded = qsospec.load_model(store, "a")
    for key, values in continuum.component_models.items():
        np.testing.assert_array_equal(values, loaded.continuum.component_models[key])
