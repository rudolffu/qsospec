"""Six-source production performance/parity audit; preserves original bundles.

PYTHONPATH=src python benchmarks/benchmark_fit_performance.py
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import qsospec
from audit_desi_fit import prepared_data
from qsospec.fitting import global_fit, adaptive_oiii, complexes
from qsospec.io.run_store import RunStore, workflow_payload
from qsospec.workflows import host_workflow
from qsospec.workflows.host.ppxf_host import _require_ppxf
from qsospec.io.products import GlobalQAPlotConfig

TARGETS = (
    "39633422498402170",
    "39627911673676639",
    "39633437694364022",
    "39627935757370833",
    "39633493944175166",
    "39633448867988026",
)
MODES = ("cold_uncached", "cold_cached", "warm_cached")


def snapshot(result):
    fits = {"continuum": result.continuum, **result.line_complexes}
    arrays = {
        "spectrum.wave": result.spectrum.wave_rest,
        "spectrum.flux": result.spectrum.flux,
        "spectrum.error": result.spectrum.err,
        "spectrum.mask": result.spectrum.valid_mask,
        "host.model": result.host_model_on_quasar_grid,
    }
    values, errors, decisions, stats = {}, {}, {}, {}
    for name, fit in fits.items():
        arrays[name + ".model"] = fit.model
        arrays[name + ".covariance"] = fit.covariance
        arrays[name + ".mask"] = fit.fit_mask
        for key, model in fit.component_models.items():
            arrays[name + ".component." + key] = model
        for kind, table in [
            ("parameters", fit.param_values),
            ("errors", fit.param_errors),
            ("metrics", getattr(fit, "metrics", {})),
            ("metric_errors", getattr(fit, "metric_errors", {})),
        ]:
            for key, value in table.items():
                values[name + "." + kind + "." + key] = value
        for key, value in fit.param_errors.items():
            errors[name + ".parameters." + key] = value
        for key, value in getattr(fit, "metric_errors", {}).items():
            errors[name + ".metrics." + key] = value
        decisions[name] = dict(
            success=fit.success,
            status=fit.status,
            selected=getattr(fit, "selected_model", None),
            candidates=[
                (c["component_count"], c["accepted"], c["rejection_reasons"])
                for c in fit.metadata.get("candidate_selection", [])
            ],
            rank_warning="covariance_rank_deficient" in fit.warning_codes(),
        )
        stats[name] = fit.metadata.get("fit_performance", {})
    return dict(
        arrays=arrays,
        values=values,
        errors=errors,
        decisions=decisions,
        stats=stats,
        warnings=result.warning_codes(),
        z=result.spectrum.z,
    )


def compare(cold, other, error):
    exact = True
    changed = []
    max_model = 0.0
    max_shift = 0.0
    max_cov = 0.0
    for key, first in cold["arrays"].items():
        second = other["arrays"].get(key)
        if first is None or second is None:
            equal = first is None and second is None
        else:
            equal = first.shape == second.shape and np.array_equal(first, second, equal_nan=True)
            if first.shape != second.shape:
                max_model = np.inf
            elif key.endswith(".covariance"):
                diagonal = np.sqrt(np.maximum(np.abs(np.diag(first)), np.finfo(float).tiny))
                scale = diagonal[:, None] * diagonal[None, :]
                if np.all(np.isfinite(first)) and np.all(np.isfinite(second)):
                    max_cov = max(
                        max_cov,
                        float(
                            np.linalg.norm((first - second) / scale)
                            / max(np.linalg.norm(first / scale), np.finfo(float).eps)
                        ),
                    )
                elif not equal:
                    max_cov = np.inf
            elif key.endswith(".model") or ".component." in key:
                finite = np.isfinite(first) & np.isfinite(second) & np.isfinite(error) & (error > 0)
                if np.any(finite):
                    max_model = max(max_model, float(np.max(np.abs(first[finite] - second[finite]) / error[finite])))
                if not np.array_equal(np.isfinite(first), np.isfinite(second)):
                    max_model = np.inf
        exact &= equal
        if not equal:
            changed.append(key)
    for key, first in cold["values"].items():
        second = other["values"].get(key, np.nan)
        equal = np.array_equal(first, second, equal_nan=True)
        exact &= equal
        if not equal:
            changed.append(key)
        uncertainty = cold["errors"].get(key, np.nan)
        if np.isfinite(uncertainty) and uncertainty > 0 and np.isfinite(first):
            max_shift = max(max_shift, float(abs(first - second) / uncertainty) if np.isfinite(second) else np.inf)
    decisions_equal = cold["decisions"] == other["decisions"]
    warnings_equal = cold["warnings"] == other["warnings"]
    z_equal = cold["z"] == other["z"]
    gate = decisions_equal and warnings_equal and z_equal and max_model < 0.01 and max_shift < 0.05 and max_cov <= 0.01
    return dict(
        exact=bool(exact and decisions_equal and warnings_equal and z_equal),
        max_model_pixel_sigma=max_model,
        max_identifiable_shift_sigma=max_shift,
        max_normalized_covariance_change=max_cov,
        decisions_equal=decisions_equal,
        warnings_equal=warnings_equal,
        redshift_equal=z_equal,
        warm_gate=bool(gate),
        changed_groups=changed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/fit_performance_20261009"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path("validation/desi_performance_20261009/archived_run/data/models").glob("*.parquet"))
    inputs = {}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        row = pq.read_table(p).to_pylist()[0]
        for target in TARGETS:
            if target in row["object_key"]:
                inputs[target] = (p, row)
    assert set(inputs) == set(TARGETS)
    host_dict = json.load(open("validation/compact_storage_20261009/compact_final/runtime.json"))["config"]
    for key, cls in [
        ("broad_line_prefit", qsospec.HostBroadLinePrefitConfig),
        ("agn_pseudocontinuum", qsospec.HostAgnPseudoContinuumConfig),
        ("coverage", qsospec.HostCoverageConfig),
    ]:
        host_dict[key] = cls(**host_dict[key])
    host_dict["continuum_windows"] = [tuple(w) for w in host_dict["continuum_windows"]]
    host = qsospec.HostDecompConfig(**host_dict)
    _require_ppxf()
    references = {}
    records = []
    with threadpool_limits(limits=2):
        for target in TARGETS:
            path, row = inputs[target]
            for repeat in range(args.repeats):
                order = MODES[repeat % 3 :] + MODES[: repeat % 3]
                for mode in order:
                    performance = qsospec.FitPerformanceConfig(
                        cache_evaluations=mode != "cold_uncached", warm_starts=mode == "warm_cached"
                    )
                    timings = []
                    candidates = []
                    original_generic = adaptive_oiii.fit_generic_complex

                    def timed_generic(*a, **kw):
                        start = perf_counter()
                        fit = original_generic(*a, **kw)
                        timings.append(dict(complex=a[2].id, seconds=perf_counter() - start))
                        m = fit.metadata.get("multistart", {})
                        candidates.append(
                            dict(
                                nfev=m.get("total_nfev", 0),
                                calls=m.get("total_optimizer_calls", 0),
                                warm=m.get("warm_start", {}).get("used", False),
                                retry=m.get("warm_start", {}).get("retry_reasons", []),
                            )
                        )
                        return fit

                    # Time complete public line fitters and non-adaptive generic fits.
                    from contextlib import ExitStack

                    with ExitStack() as stack:
                        stack.enter_context(patch.object(adaptive_oiii, "fit_generic_complex", timed_generic))
                        for owner, name in [
                            (global_fit, "fit_hbeta_complex"),
                            (global_fit, "fit_mgii_complex"),
                            (global_fit, "fit_halpha_complex"),
                            (global_fit, "fit_lya_nv_complex"),
                            (global_fit, "fit_generic_complex"),
                        ]:
                            original = getattr(owner, name)

                            def timed(*a, _original=original, _name=name, **kw):
                                start = perf_counter()
                                fit = _original(*a, **kw)
                                timings.append(dict(complex=_name, seconds=perf_counter() - start))
                                return fit

                            stack.enter_context(patch.object(owner, name, timed))
                        start = perf_counter()
                        result = host_workflow._run_global_fit_with_optional_host(
                            prepared_data(row),
                            source="prepared_archived_desi",
                            input_path=str(path),
                            run_host_decomp=True,
                            host_config=host,
                            global_config=qsospec.GlobalContinuumConfig(performance=performance),
                            hbeta_config=qsospec.HbetaComplexConfig(),
                            uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=0),
                            galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
                        )
                        seconds = perf_counter() - start
                    state = snapshot(result)
                    if target not in references:
                        assert mode == "cold_uncached"
                        references[target] = state
                    item = dict(
                        target=target,
                        z=row["redshift"],
                        mode=mode,
                        repeat=repeat,
                        seconds=seconds,
                        comparison=compare(references[target], state, result.spectrum.err),
                        per_complex=timings,
                        candidates=candidates,
                        performance_statistics=state["stats"],
                    )
                    if repeat == 0:
                        directory = args.output / target / mode
                        store = RunStore.create(
                            str(directory / "run"),
                            configuration={"performance": performance},
                            model_storage="parameters",
                        )
                        start = perf_counter()
                        payload = workflow_payload(
                            result,
                            run_id=store.run_id,
                            object_key=row["object_key"],
                            object_id=row["object_id"],
                            input_record={
                                "source": str(path),
                                "row_index": 0,
                                "reader": "archived_model",
                                "metadata": {},
                            },
                        )
                        store.write_payload(payload)
                        item["persistence_seconds"] = perf_counter() - start
                        from qsospec import load_model

                        start = perf_counter()
                        loaded = load_model(store, row["object_key"])
                        item["reload_seconds"] = perf_counter() - start
                        reloaded = snapshot(loaded)
                        item["reload_comparison"] = compare(state, reloaded, result.spectrum.err)
                        item["reload_exact"] = item["reload_comparison"]["exact"]
                        # Parameter reconstruction follows schema-8 rtol/atol, not bit identity.
                        item["reload_parity"] = (
                            all(
                                (a is None and reloaded["arrays"].get(k) is None)
                                or (
                                    a is not None
                                    and reloaded["arrays"].get(k) is not None
                                    and np.allclose(a, reloaded["arrays"][k], rtol=1e-10, atol=1e-12, equal_nan=True)
                                )
                                for k, a in state["arrays"].items()
                            )
                            and all(
                                np.array_equal(v, reloaded["values"].get(k, np.nan), equal_nan=True)
                                for k, v in state["values"].items()
                            )
                            and state["warnings"] == reloaded["warnings"]
                            and state["decisions"] == reloaded["decisions"]
                            and state["z"] == reloaded["z"]
                        )
                        fig = result.plot_qa(GlobalQAPlotConfig(object_name=target, object_label="DESI"))
                        fig.savefig(directory / "qa.png", dpi=160)
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    records.append(item)
                    report = dict(
                        records=records, source_hashes=hashes, repeats=args.repeats, blas_threads=2, profiled=False
                    )
                    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
                    print(
                        target,
                        mode,
                        repeat,
                        round(seconds, 2),
                        "exact",
                        item["comparison"]["exact"],
                        "warm_gate",
                        item["comparison"]["warm_gate"],
                        flush=True,
                    )
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cached = [x for x in records if x["mode"] == "cold_cached"]
    warm = [x for x in records if x["mode"] == "warm_cached"]
    control = [x for x in records if x["mode"] == "cold_cached"]
    rollout = dict(
        exact_cache_parity=all(x["comparison"]["exact"] for x in cached),
        warm_science_pass=all(x["comparison"]["warm_gate"] for x in warm),
        warm_median_seconds=statistics.median(x["seconds"] for x in warm),
        cold_cached_median_seconds=statistics.median(x["seconds"] for x in control),
        reload_parity=all(x.get("reload_parity", True) for x in records),
    )
    rollout["enable_warm_default"] = (
        rollout["warm_science_pass"]
        and rollout["warm_median_seconds"] < rollout["cold_cached_median_seconds"]
        and rollout["exact_cache_parity"]
        and rollout["reload_parity"]
    )
    (args.output / "rollout.json").write_text(json.dumps(rollout, indent=2) + "\n")


if __name__ == "__main__":
    main()
