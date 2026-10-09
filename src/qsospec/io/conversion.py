"""Explicit, validated copy conversion of scientific run bundles."""

from pathlib import Path
import shutil
from copy import deepcopy
from .run_store import RunStore, TABLE_NAMES, _from_key_values, _key_values, load_model_by_key, finalize_run


def convert_run(source, destination, *, model_storage="parameters"):
    """Copy a run to a new schema-8 bundle without fitting or editing its source."""
    original = source if isinstance(source, RunStore) else RunStore.open(str(source))
    target = Path(destination).expanduser()
    if target.resolve().is_relative_to(original.path.resolve()):
        raise ValueError("Conversion destination must be outside the source run")
    if target.exists():
        raise FileExistsError(f"Conversion requires a new destination: {target}")
    store = RunStore.create(
        str(target),
        configuration=original.manifest.get("configuration", {}),
        run_id=original.run_id,
        model_storage=model_storage,
    )
    store.manifest["configuration_hash"] = original.configuration_hash
    store.manifest["cosmology"] = deepcopy(original.manifest.get("cosmology"))
    store.manifest["source_manifest"] = deepcopy(original.manifest)
    store.manifest["status"] = "converting"
    store.manifest["conversion"] = dict(
        source=str(original.path.resolve()), source_schema=original.manifest["schema_version"], status="in_progress"
    )
    store._write_manifest(reconcile=False)
    keys = sorted(
        {
            str(k)
            for name in TABLE_NAMES
            for k in original.read_table(name, columns=["object_key"])["object_key"].to_pylist()
        }
    )
    copied_assets = set()
    try:
        for key in keys:
            import pyarrow.dataset as ds

            payload = {
                name: (
                    original.read_table(name, filter_expression=ds.field("object_key") == key)
                    if not original.manifest.get("indexed_storage") and name == "inputs"
                    else original.read_object_table(name, key)
                ).to_pylist()
                for name in TABLE_NAMES
            }
            payload = {name: rows for name, rows in payload.items() if rows}
            if payload.get("models"):
                row = payload["models"][0]
                # Copy each immutable asset once, including root-only model recipes.
                import json

                def copy_assets(value):
                    if isinstance(value, dict):
                        if set(value) == {"asset"}:
                            from .indexed_store import write_asset

                            token = value["asset"]
                            if token not in copied_assets:
                                write_asset(store, token, original.read_asset(token))
                                copied_assets.add(token)
                        else:
                            for item in value.values():
                                copy_assets(item)
                    elif isinstance(value, list):
                        for item in value:
                            copy_assets(item)

                if model_storage == "parameters":
                    for component in row["components"]:
                        if component.get("definition"):
                            copy_assets(json.loads(component["definition"]))
                    copy_assets(_from_key_values(row["workflow_metadata"]).get("model_storage_root_definitions", {}))
                loaded = load_model_by_key(original, key)
                row["_evaluation_states"] = {
                    name: {"parameters": fit.param_values} for name, fit in loaded.line_complexes.items()
                }
                # Recover only profile definitions; do not remeasure or change metrics.
                from ..line_peaks import _recover_native_definitions

                for complex_row in row["complexes"]:
                    fit = loaded.line_complexes[complex_row["recipe_id"]]
                    meta = _from_key_values(complex_row["metadata"])
                    if not meta.get("peak_model"):
                        recovered = _recover_native_definitions(fit)
                        if recovered:
                            definitions, bounds = recovered
                            meta["peak_model"] = dict(
                                version=1, components=definitions, bounds=list(bounds), z=loaded.spectrum.z
                            )
                            meta["storage_reconstruction_provenance"] = "validated_legacy_definition"
                            complex_row["metadata"] = _key_values(meta)
                # Replace compact source arrays with verified reconstruction before re-saving.
                from ..model_evaluation import restore_row

                row = restore_row(deepcopy(row), original.read_asset)
                payload["models"] = [row]
            store.write_payload(payload, update_manifest=False)
        if (original.path / "qa").is_dir():
            shutil.copytree(original.path / "qa", store.path / "qa", dirs_exist_ok=True)
        finalize_run(store)
        store.manifest["conversion"]["status"] = "complete"
        store._write_manifest(reconcile=False)
        return store
    except BaseException:
        store.manifest["conversion"]["status"] = "failed"
        store.manifest["status"] = "conversion_failed"
        store._write_manifest(reconcile=False)
        raise
