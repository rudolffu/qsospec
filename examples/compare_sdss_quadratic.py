"""Compare identical frozen SDSS inputs with/without the anchored quadratic.

Run on the machine holding the recorded input shards and dust/host templates::

    python examples/compare_sdss_quadratic.py \
        --sample /path/to/sample.csv \
        --reference-manifest /path/to/comparison_manifest.json \
        --input-root /path/to/input/qsospec \
        --output runs/sdss_quadratic_comparison

This requires a new output directory and never updates the reference run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd

import qsospec


def _restore_value(annotation, value):
    if value is None:
        return None
    origin, args = get_origin(annotation), get_args(annotation)
    if isinstance(value, dict) and isinstance(annotation, type) and is_dataclass(annotation):
        return _restore_config(annotation, value)
    if origin in (list, tuple) and isinstance(value, list):
        restored = [_restore_value(args[0] if len(args) < 2 or args[-1] is Ellipsis else args[i], item)
                    for i, item in enumerate(value)]
        return tuple(restored) if origin is tuple else restored
    for kind in args:
        if isinstance(value, dict) and isinstance(kind, type) and is_dataclass(kind):
            return _restore_config(kind, value)
        if isinstance(value, list) and get_origin(kind) in (list, tuple):
            return _restore_value(kind, value)
    return value


def _restore_config(cls, values):
    """Restore nested public dataclasses from the recorded JSON configuration."""
    annotations = get_type_hints(cls)
    restored = {}
    allowed = {field.name for field in fields(cls)}
    if set(values) - allowed:
        raise ValueError(f"Unknown recorded {cls.__name__} fields: {set(values) - allowed}")
    for name, value in values.items():
        restored[name] = _restore_value(annotations[name], value)
    return cls(**restored)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(sample_path, reference_manifest, input_root, output):
    sample = pd.read_csv(sample_path, dtype={"object_id": str, "spectrum_key": str})
    reference = json.loads(Path(reference_manifest).read_text())
    root, output = Path(input_root), Path(output)
    if len(sample) != 10 or sample.spectrum_key.nunique() != 10:
        raise ValueError("Expected the frozen sample of ten distinct spectrum keys.")
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing runs are immutable.")
    recorded_hashes = {Path(key).name: value for key, value in reference["source_inputs_sha256"].items()}
    input_hashes = {}
    for filename in sample.input_file.unique():
        path = root / filename
        input_hashes[filename] = _hash(path)  # Preflight before creating a run.
        if input_hashes[filename] != recorded_hashes[filename]:
            raise ValueError(f"Input hash differs from the reference: {filename}")
    global_config = _restore_config(qsospec.GlobalContinuumConfig, reference["global_config"])
    host = _restore_config(qsospec.HostDecompConfig, reference["host_config"])
    uncertainty = _restore_config(qsospec.UncertaintyConfig, reference["uncertainty"])
    baseline_config = replace(global_config, polynomial=replace(global_config.polynomial, enabled=False))
    quadratic_config = replace(global_config, polynomial=qsospec.PolynomialContinuumConfig(enabled=None))
    output.mkdir(parents=True)
    source_root = Path(qsospec.__file__).parent
    manifest = {
        "package_version": qsospec.__version__,
        "reference_manifest_sha256": _hash(reference_manifest),
        "sample_sha256": _hash(sample_path),
        "source_hashes": {str(p.relative_to(source_root)): _hash(p) for p in sorted(source_root.rglob("*.py"))},
        "input_hashes": input_hashes,
        "baseline_config": asdict(baseline_config),
        "quadratic_config": asdict(quadratic_config),
        "host_config": asdict(host),
        "uncertainty_config": asdict(uncertainty),
    }
    (output / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2))
    sample.to_csv(output / "sample.csv", index=False)
    records, failures = [], []
    for row in sample.itertuples(index=False):
        for label, config in (("baseline", baseline_config), ("quadratic", quadratic_config)):
            start = perf_counter()
            try:
                result = qsospec.fit_object_to_store(
                    str(root / row.input_file), str(output / label),
                    row_index=int(row.physical_row_index), redshift=float(row.z_optical_fit),
                    object_id=row.spectrum_key, run_host_decomp=True,
                    host_config=host, template_root=host.template_root,
                    global_config=config, uncertainty_config=uncertainty,
                    qa_plot_config=qsospec.GlobalQAPlotConfig(output_format="both", max_zoom_panels=4),
                )
                cont = result.continuum
                records.append({
                    "sample_index": row.sample_index, "object_id": row.object_id,
                    "spectrum_key": row.spectrum_key, "variant": label,
                    "seconds": perf_counter() - start, "continuum_chi2": cont.chi2,
                    "continuum_reduced_chi2": cont.reduced_chi2,
                    "slope": cont.param_values.get("power_law.slope", np.nan),
                    "red_slope": cont.param_values.get("power_law.red_slope", np.nan),
                    "polynomial_status": cont.metadata.get("polynomial_status"),
                    "polynomial_fraction_max": cont.metadata.get("polynomial_fraction_max"),
                    "polynomial_fraction_rms": cont.metadata.get("polynomial_fraction_rms"),
                    "polynomial_delta_bic": cont.metadata.get("polynomial_delta_bic"),
                    "polynomial_condition": cont.metadata.get("polynomial_combined_condition"),
                    "parameters": cont.param_values,
                    "complex_statuses": result.metadata.get("complex_status", {}),
                    "line_metrics": {name: fit.metrics for name, fit in result.line_complexes.items()},
                    "qa_path": str(result.qa_path),
                })
            except Exception as exc:
                failures.append({"spectrum_key": row.spectrum_key, "variant": label, "error": str(exc)})
            (output / "comparison_results.json").write_text(json.dumps({"records": records, "failures": failures}, indent=2))
    scalar = pd.DataFrame(records).drop(columns=["parameters", "line_metrics", "complex_statuses"], errors="ignore")
    scalar.to_csv(output / "comparison_metrics.csv", index=False)
    for label in ("baseline", "quadratic"):
        store = qsospec.open_run(output / label)
        store.read_table("measurements").to_pandas().to_csv(output / f"{label}_measurements.csv", index=False)
    print(f"Completed {len(records)}/20 fits; failures={len(failures)}; output={output}")
    if failures:
        raise RuntimeError("Comparison has failures; inspect comparison_results.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, dest="sample_path")
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))
