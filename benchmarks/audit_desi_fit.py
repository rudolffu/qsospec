"""Profile prepared DESI spectra without modifying their archived run.

All new fits and profiling outputs are written under --output. Archived inputs
are already extinction corrected; the benchmark does not deredden them again.
"""

import argparse
import cProfile
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import pstats
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

import qsospec
from qsospec.fitting.line_lsf import resolution_from_dict
from qsospec.io.run_store import RunStore, workflow_payload, _from_key_values
from qsospec.workflows import host_workflow
from qsospec.workflows.host.io import SpectrumData


def host_config(path, template_root):
    config = json.loads(Path(path).read_text())["host_decomposition"]
    agn = dict(config["agn_pseudocontinuum"])
    grid = agn.pop("powerlaw_slopes_f_lambda")
    agn["powerlaw_slopes"] = tuple(
        round(value, 10) for value in np.arange(grid["minimum"], grid["maximum"] + grid["step"] / 2, grid["step"])
    )
    return qsospec.HostDecompConfig(
        template_root=str(template_root),
        strategy=config["strategy"],
        fit_range=tuple(config["fit_range_angstrom"]),
        broad_line_prefit=qsospec.HostBroadLinePrefitConfig(**config["broad_line_prefit"]),
        agn_pseudocontinuum=qsospec.HostAgnPseudoContinuumConfig(**agn),
        coverage=qsospec.HostCoverageConfig(**config["coverage"]),
    )


def prepared_data(row):
    metadata = _from_key_values(row["spectrum_metadata"])
    resolution = resolution_from_dict(metadata.pop("spectral_resolution", None))
    z = row["redshift"]
    # Run bundles store F_lambda in the rest frame. SpectrumData accepts observed.
    scale = 1 + z if metadata.get("flux_frame") == "rest" else 1.0
    metadata["archived_rest_frame_conversion"] = metadata.pop("rest_frame_conversion", None)
    metadata["archived_galactic_extinction"] = metadata.get("galactic_extinction")
    metadata["galactic_extinction"] = {"status": "caller_preprocessed", "source": "archived_corrected_spectrum"}
    metadata.update(flux_frame="observed", galactic_extinction_corrected=True)
    total = row["total_flux"] if row["total_flux"] is not None else row["flux"]
    return SpectrumData(
        wave_obs=np.asarray(row["wave_rest"]) * (1 + z),
        flux=np.asarray(total) / scale,
        error=np.asarray(row["error"]) / scale,
        mask=None if row["input_mask"] is None else ~np.asarray(row["input_mask"], dtype=bool),
        redshift=z,
        object_id=row["object_id"],
        ra=metadata.get("ra"),
        dec=metadata.get("dec"),
        metadata=metadata,
        resolution=resolution,
    )


def profile_rows(profiler):
    entries = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        entries.append(
            dict(
                file=filename,
                line=line,
                function=name,
                calls=calls,
                primitive_calls=primitive,
                self_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--template-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oiii-mode", choices=("adaptive", "legacy"), default="adaptive")
    parser.add_argument(
        "--cache-balmer",
        action="store_true",
        help="Test an exact, bounded per-spectrum cache; production remains unchanged.",
    )
    parser.add_argument("--sample-index", type=int, help="Run just one of the three selected examples.")
    parser.add_argument("--disable-balmer-cache", action="store_true")
    parser.add_argument("--model-storage", choices=["parameters", "arrays"], default="parameters")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep one-time pPXF/Matplotlib imports out of steady-state fit timing.
    from qsospec.workflows.host.ppxf_host import _require_ppxf

    setup_start = perf_counter()
    _require_ppxf()
    setup_seconds = perf_counter() - setup_start
    files = sorted((args.run / "data/models").glob("*.parquet"))
    candidates = [(path, pq.read_table(path).to_pylist()[0]) for path in files]
    # Deterministic low-z host, intermediate-z host, and high-z no-host examples.
    selected = [
        min(candidates, key=lambda item: item[1]["redshift"]),
        min(candidates, key=lambda item: abs(item[1]["redshift"] - 0.75)),
        max(candidates, key=lambda item: item[1]["redshift"]),
    ]
    selected = list({row["object_key"]: (path, row) for path, row in selected}.values())
    if args.sample_index is not None:
        selected = [selected[args.sample_index]]
    host = host_config(args.config, args.template_root)
    report = dict(
        python=platform.python_version(),
        qsospec=qsospec.__version__,
        config=asdict(host),
        sample_policy="minimum z, nearest z=0.75, maximum z among 16 sorted shards",
        source_run=str(args.run.resolve()),
        profiled_times=True,
        oiii_mode=args.oiii_mode,
        oiii_mode_scope="host_broad_line_prefit_and_final_fit",
        ppxf_import_seconds=setup_seconds,
        resolution_note="Archived rows lacking an LSF use flagged observed-profile fitting; no resolution is fabricated.",
        model_storage=args.model_storage,
        production_balmer_cache=not args.disable_balmer_cache,
        spectra=[],
    )
    for index, (path, row) in enumerate(selected):
        timings = []

        def timed_global(*a, **kw):
            start = perf_counter()
            result = original(*a, **kw)
            timings.append(perf_counter() - start)
            return result

        original = host_workflow.fit_global_lines
        data = prepared_data(row)
        from qsospec.templates import balmer
        from qsospec.workflows.host import broad_line_prefit

        original_prefit = broad_line_prefit.run_host_broad_line_prefit

        def prefit_with_mode(*a, **kw):
            kw["hbeta_config"] = qsospec.HbetaComplexConfig(oiii_profile_mode=args.oiii_mode)
            return original_prefit(*a, **kw)

        original_balmer = balmer.evaluate_balmer_series_with_derivatives
        cache = OrderedDict()
        cache_stats = dict(hits=0, misses=0, maximum_entries=128, peak_array_bytes=0)

        def cached_balmer(template, wave, width, velocity):
            grid = np.asarray(wave, dtype=float)
            key = (id(template), grid.shape, grid.tobytes(), float(width), float(velocity))
            if key in cache:
                cache_stats["hits"] += 1
                cache.move_to_end(key)
                return cache[key]
            cache_stats["misses"] += 1
            values = original_balmer(template, wave, width, velocity)
            for array in values:
                array.flags.writeable = False
            cache[key] = values
            if len(cache) > 128:
                cache.popitem(last=False)
            cache_stats["peak_array_bytes"] = max(
                cache_stats["peak_array_bytes"], sum(a.nbytes for values in cache.values() for a in values)
            )
            return values

        print("Profiling", row["object_key"], "z=", row["redshift"], flush=True)
        profiler = cProfile.Profile()
        start = perf_counter()
        from qsospec.templates.balmer_cache import balmer_cache

        with ExitStack() as patches:
            production_cache = patches.enter_context(balmer_cache(enabled=not args.disable_balmer_cache))
            patches.enter_context(patch.object(host_workflow, "fit_global_lines", timed_global))
            patches.enter_context(patch.object(broad_line_prefit, "run_host_broad_line_prefit", prefit_with_mode))
            if args.cache_balmer:
                patches.enter_context(patch.object(balmer, "evaluate_balmer_series_with_derivatives", cached_balmer))
            profiler.enable()
            result = host_workflow._run_global_fit_with_optional_host(
                data,
                source="prepared_archived_desi",
                input_path=str(path),
                run_host_decomp=True,
                host_config=host,
                global_config=qsospec.GlobalContinuumConfig(),
                hbeta_config=qsospec.HbetaComplexConfig(oiii_profile_mode=args.oiii_mode),
                uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=0),
                galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
            )
            profiler.disable()
        fit_seconds = perf_counter() - start
        profiler.dump_stats(str(args.output / f"fit_{index}.prof"))
        entries = profile_rows(profiler)
        store = RunStore.create(
            str(args.output / f"fit_{index}"), configuration={"benchmark": True}, model_storage=args.model_storage
        )
        start = perf_counter()
        payload = workflow_payload(
            result,
            run_id=store.run_id,
            object_key=row["object_key"],
            object_id=row["object_id"],
            input_record={"source": str(path), "row_index": 0, "reader": "archived_model", "metadata": {}},
        )
        payload_seconds = perf_counter() - start
        start = perf_counter()
        store.write_payload(payload)
        write_seconds = perf_counter() - start
        item = dict(
            object_key=row["object_key"],
            z=row["redshift"],
            pixels=len(data.flux),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fit_seconds=fit_seconds,
            production_cache_stats=None
            if not production_cache
            else {
                "hits": production_cache.hits,
                "misses": production_cache.misses,
                "array_bytes": production_cache.bytes,
            },
            final_global_fit_calls=len(timings),
            final_global_fit_seconds=sum(timings),
            payload_seconds=payload_seconds,
            write_seconds=write_seconds,
            bundle_bytes=sum(p.stat().st_size for p in store.path.rglob("*.parquet")),
            host_enabled=result.host_decomp_enabled,
            host_quality=None if result.host_fit is None else result.host_fit.quality_metrics,
            fit_success=result.continuum.success,
            complex_statuses=result.complex_statuses,
            balmer_cache=cache_stats if args.cache_balmer else None,
            top_self=sorted(entries, key=lambda r: r["self_seconds"], reverse=True)[:40],
            top_cumulative=sorted(entries, key=lambda r: r["cumulative_seconds"], reverse=True)[:60],
        )
        report["spectra"].append(item)
        (args.output / "runtime.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("Finished", round(fit_seconds, 2), "s; writing", round(write_seconds, 3), "s", flush=True)


if __name__ == "__main__":
    main()
