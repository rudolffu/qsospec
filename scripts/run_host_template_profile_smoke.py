"""Run bounded E-MILES/native-XSL/exact-cache host-profile validation.

This is a validation utility, not a production batch runner.  It writes every
derived XSL product and run bundle below an explicit external output root.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import qsospec
from qsospec.io.readers import scan_parquet_spectra
from qsospec.workflows.host.ppxf_host import prepare_spectrum_for_host_decomp
from qsospec.workflows.host.preconvolved_templates import (
    build_preconvolved_xsl_product,
)
from qsospec.workflows.host.templates import load_ppxf_npz_templates

PROFILES = ("emiles_native", "xsl_native", "xsl_preconvolved")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--template-root", required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=PROFILES
    )
    return parser


def _native_config(profile: str, template_root: str) -> qsospec.HostDecompConfig:
    filename = (
        "spectra_emiles_9.0.npz"
        if profile == "emiles_native"
        else "spectra_xsl_9.0.npz"
    )
    family = "emiles" if profile == "emiles_native" else "xsl"
    return qsospec.HostDecompConfig(
        strategy="agn_pseudocontinuum_masked",
        template_root=template_root,
        template_file=filename,
        template_family=family,
        template_profile=profile,
    )


def _fit_cached_one(payload: dict[str, Any]) -> dict[str, Any]:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    descriptor = qsospec.SpectrumInput(
        source=payload["source"],
        row_index=int(payload["row_index"]),
        object_id=payload["object_id"],
        redshift=float(payload["redshift"]),
        reader="parquet",
        explicit_object_key=payload["object_key"],
    )
    config = qsospec.HostDecompConfig(
        strategy="agn_pseudocontinuum_masked",
        template_root=payload["template_root"],
        template_file=payload["cache_file"],
        template_family="xsl",
        template_profile="xsl_preconvolved",
        template_product_kind="preconvolved",
        source_template_file="spectra_xsl_9.0.npz",
        resolution_matching_mode="preconvolved_exact",
    )
    started = perf_counter()
    result = qsospec.fit_object_to_store(
        descriptor,
        payload["run_directory"],
        run_host_decomp=True,
        host_config=config,
        write_qa=False,
        resume=False,
    )
    return {
        "row_index": int(payload["row_index"]),
        "object_key": payload["object_key"],
        "object_id": payload["object_id"],
        "elapsed_seconds": perf_counter() - started,
        "host_ppxf_status": result.metadata.get("host_ppxf_status"),
        "host_fit_reliable": result.metadata.get("host_fit_reliable"),
        "template_profile": result.metadata.get("host_template_profile"),
        "template_resolution_status": result.metadata.get(
            "host_fit_quality", {}
        ).get("template_resolution_status"),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.rows < 1 or args.rows > 96:
        raise ValueError("--rows must be between 1 and the bounded 96-row pilot")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4")
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Validation output already exists and is immutable: {output_root}"
        )
    output_root.mkdir(parents=True)
    source = str(Path(args.spectrum).expanduser().resolve())
    template_root = str(Path(args.template_root).expanduser().resolve())
    selected = list(
        scan_parquet_spectra(
            source,
            row_indices=list(range(args.rows)),
            batch_size=max(args.rows, 16),
        )
    )
    if len(selected) != args.rows:
        raise ValueError(f"Expected {args.rows} inputs, found {len(selected)}")

    summary: dict[str, Any] = {
        "source": source,
        "rows": args.rows,
        "profiles": {},
    }
    for profile in args.profiles:
        started = perf_counter()
        if profile != "xsl_preconvolved":
            result = qsospec.fit_batch(
                [source],
                str(output_root / profile),
                row_indices=list(range(args.rows)),
                n_workers=args.workers,
                task_size=1,
                run_host_decomp=True,
                host_config=_native_config(profile, template_root),
                resume=False,
                show_progress=True,
                progress_total=args.rows,
            )
            summary["profiles"][profile] = {
                "run_directory": result.run_directory,
                "elapsed_seconds": perf_counter() - started,
            }
            continue

        native_xsl = load_ppxf_npz_templates(
            template_root=template_root,
            template_file="spectra_xsl_9.0.npz",
            template_family="xsl",
            template_profile="xsl_native",
            write_report=False,
        )
        payloads = []
        for descriptor, spectrum in selected:
            cache_path = (
                output_root
                / "xsl_cache"
                / f"row_{int(descriptor.row_index):03d}.npz"
            )
            # The public workflow does not request host decomposition at
            # z >= 1.2.  Keep those objects in the bounded run for input/final
            # fitting parity, but do not manufacture an unused XSL cache.
            if float(spectrum.redshift) < 1.2:
                prepared = prepare_spectrum_for_host_decomp(
                    spectrum,
                    redshift=spectrum.redshift,
                    fit_range=(3600.0, 7000.0),
                )
                prepared.metadata["spectral_resolution"] = spectrum.resolution
                prepared.metadata["object_id"] = descriptor.object_id
                prepared.metadata["host_preconvolution_object_key"] = (
                    descriptor.object_key
                )
                product = build_preconvolved_xsl_product(
                    native_xsl,
                    prepared,
                    output_path=cache_path,
                    object_key=descriptor.object_key,
                    fit_range=(3600.0, 7000.0),
                )
                cache_file = product.path
            else:
                cache_file = str(cache_path)
            payloads.append(
                {
                    "source": source,
                    "row_index": descriptor.row_index,
                    "object_key": descriptor.object_key,
                    "object_id": descriptor.object_id,
                    "redshift": descriptor.redshift,
                    "template_root": template_root,
                    "cache_file": cache_file,
                    "run_directory": str(
                        output_root
                        / profile
                        / f"object_{int(descriptor.row_index):03d}"
                    ),
                }
            )
        records = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_fit_cached_one, item) for item in payloads]
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda item: item["row_index"])
        summary["profiles"][profile] = {
            "run_directory": str(output_root / profile),
            "cache_directory": str(output_root / "xsl_cache"),
            "elapsed_seconds": perf_counter() - started,
            "objects": records,
        }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
