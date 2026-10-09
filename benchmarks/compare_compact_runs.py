"""Compare prepared DESI cache/storage runs, including portable assets and reload costs.

Run audit_desi_fit.py twice, with --model-storage arrays --disable-balmer-cache
and with its defaults; pass those output directories here. Does not fit spectra.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import tracemalloc
import numpy as np
from qsospec.io.run_store import RunStore, load_model, finalize_run, _from_key_values
from qsospec.io.indexed_store import write_asset


def bundle_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def merge_profiles(source, destination):
    output = RunStore.create(
        str(destination),
        configuration={"benchmark": True},
        model_storage=RunStore.open(str(source / "fit_0")).manifest["model_storage"],
    )
    for path in sorted(source.glob("fit_*/manifest.json")):
        store = RunStore.open(str(path.parent))
        for asset in (store.path / "assets").glob("*.parquet"):
            write_asset(output, asset.stem, store.read_asset(asset.stem))
        payload = {name: store.read_table(name).to_pylist() for name in store.manifest["tables"]}
        output.write_payload({name: rows for name, rows in payload.items() if rows}, update_manifest=False)
    finalize_run(output)
    return output


def reload_benchmark(path, key):
    # Cold means a fresh process/evaluator cache, not an OS disk-cache purge.
    tracemalloc.start()
    start = perf_counter()
    model = load_model(str(path), key)
    cold = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    start = perf_counter()
    load_model(str(path), key)
    warm = perf_counter() - start
    return dict(
        cold_process_seconds=cold,
        warm_seconds=warm,
        traced_peak_bytes=peak,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        components=sum(len(f.component_models) for f in model.line_complexes.values()),
    )


def compare(a, b):
    count = 0
    for left, right in [
        (a.continuum, b.continuum),
        *[(a.line_complexes[k], b.line_complexes[k]) for k in a.line_complexes],
    ]:
        assert json.dumps(left.param_values, sort_keys=True) == json.dumps(right.param_values, sort_keys=True)
        assert json.dumps(left.param_errors, sort_keys=True) == json.dumps(right.param_errors, sort_keys=True)
        if left.covariance is not None:
            np.testing.assert_array_equal(left.covariance, right.covariance)
        np.testing.assert_allclose(left.model, right.model, rtol=1e-10, atol=1e-12)
        for key in left.component_models:
            np.testing.assert_allclose(left.component_models[key], right.component_models[key], rtol=1e-10, atol=1e-12)
        if hasattr(left, "metrics"):
            assert json.dumps(left.metrics, sort_keys=True) == json.dumps(right.metrics, sort_keys=True)
        count += 1
    assert a.spectrum.z == b.spectrum.z
    np.testing.assert_array_equal(a.spectrum.flux, b.spectrum.flux)
    for name in a.host_component_models:
        np.testing.assert_allclose(a.host_component_models[name], b.host_component_models[name], rtol=1e-10, atol=1e-12)
    if a.host_model_on_quasar_grid is not None:
        np.testing.assert_allclose(a.host_model_on_quasar_grid, b.host_model_on_quasar_grid, rtol=1e-10, atol=1e-12)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reload", type=Path)
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.reload:
        print(json.dumps(reload_benchmark(args.reload, args.key)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = merge_profiles(args.arrays, args.output / "arrays")
    compact = merge_profiles(args.parameters, args.output / "parameters")
    result = {
        "runtime_arrays_cache_disabled": json.loads((args.arrays / "runtime.json").read_text())["spectra"],
        "runtime_parameters_cache_enabled": json.loads((args.parameters / "runtime.json").read_text())["spectra"],
        "storage": {},
        "reload": {},
        "comparisons": [],
    }
    for store, label in [(arrays, "arrays"), (compact, "parameters")]:
        result["storage"][label] = {
            "total_bytes": bundle_bytes(store.path),
            "model_bytes": bundle_bytes(store.path / "data/models"),
            "asset_bytes": bundle_bytes(store.path / "assets"),
            "files": sum(p.is_file() for p in store.path.rglob("*")),
            "model_files": sum(p.is_file() for p in (store.path / "data/models").glob("*.parquet")),
        }
        result["reload"][label] = {}
        for key in sorted(store.completed_keys()):
            response = subprocess.check_output(
                [sys.executable, __file__, "--reload", str(store.path), "--key", key], env=os.environ, text=True
            )
            result["reload"][label][key] = json.loads(response)
    for key in sorted(arrays.completed_keys()):
        count = compare(load_model(arrays, key), load_model(compact, key))
        outcomes = _from_key_values(compact.object_row_by_key("models", key)["workflow_metadata"])[
            "model_storage_components"
        ]
        result["comparisons"].append(
            {
                "object_key": key,
                "comparison_groups": count,
                "components": len(outcomes),
                "fallbacks": [x for x in outcomes if x["storage"] == "array_fallback"],
            }
        )
    result["storage"]["saving_including_assets_percent"] = 100 * (
        1 - result["storage"]["parameters"]["total_bytes"] / result["storage"]["arrays"]["total_bytes"]
    )
    result["limitations"] = [
        "Three prepared archived DESI spectra; no stored LSF in these inputs.",
        "Cold reload uses a fresh process, but the OS filesystem cache is not purged.",
        "Reload timings include tracemalloc instrumentation; fit times include cProfile.",
        "Process peak RSS includes interpreter and dependencies; traced peak covers reload allocations.",
        "Shared stellar templates impose a one-time cost; small-run totals need not be smaller.",
    ]
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
