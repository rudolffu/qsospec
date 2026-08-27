"""Run He I/Pa-gamma injection/recovery on real Euclid RGS noise patterns.

The calibration samples archived rest-frame wavelength, error, and mask
patterns, injects tied narrow He I/Pa-gamma plus optional tied broad emission,
and refits with the locked point-source R=480 N0/B1 pair.  It never promotes a
physical class: independent higher-resolution decompositions remain required.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qsospec
from qsospec import lines
from qsospec.narrow_line_calibration import (
    selection_mask,
    stratified_recovery,
    summarize_recovery,
    threshold_sweep,
)

C_KMS = 299792.458
RUN_NAME = "production_no_balmer_no_host_v1"
CLASSIFICATION_NAME = "narrow_line_r480_v2"
CALIBRATION_NAME = "injection_recovery_hei_pgamma_v1"
DESIGN_VERSION = "balanced_narrow_broad_v2"
_WORKER_STORE = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_root() -> Path:
    data_root = os.environ.get("MLSPECZ_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set MLSPECZ_DATA_ROOT or pass --output-root")
    return Path(data_root) / "outputs/qsospec/dr1_identified_gold_rgs_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--mode",
        choices=("smoke", "production"),
        required=True,
        help="Require an explicit choice before a large injection run.",
    )
    parser.add_argument("--n-injections", type=int, default=2000)
    parser.add_argument("--n-base-patterns", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4801200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _decode_key_values(values: Any) -> dict[str, object]:
    output: dict[str, object] = {}
    if values is None:
        return output
    for item in values:
        value = item["value"]
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            value = item["value"]
        output[str(item["key"])] = value
    return output


def _gaussian_area_profile(
    wave: np.ndarray,
    flux: float,
    center: float,
    fwhm_kms: float,
) -> np.ndarray:
    sigma = (fwhm_kms / 2.354820045) * center / C_KMS
    return flux * np.exp(-0.5 * ((wave - center) / sigma) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma
    )


def _line_flux_sigma(
    wave: np.ndarray,
    error: np.ndarray,
    valid: np.ndarray,
    center: float,
    fwhm_kms: float,
) -> float:
    basis = _gaussian_area_profile(wave, 1.0, center, fwhm_kms)
    information = np.sum((basis[valid] / error[valid]) ** 2)
    return float(1.0 / np.sqrt(information)) if information > 0 else np.nan


def injection_design(
    n_injections: int,
    base_ids: list[str],
    *,
    seed: int,
) -> pd.DataFrame:
    """Construct a deterministic balanced design spanning required dimensions."""

    if n_injections <= 0 or not base_ids:
        raise ValueError("n_injections and base_ids must be non-empty")
    rng = np.random.default_rng(seed)
    levels = {
        "truth_hei_snr": np.array([5.0, 7.0, 10.0, 20.0, 40.0]),
        "truth_narrow_fwhm_intrinsic_kms": np.array([0.0, 300.0, 600.0, 900.0, 1150.0]),
        "truth_broad_fwhm_intrinsic_kms": np.array([1400.0, 2200.0, 4000.0, 7000.0, 9500.0]),
        # Balance true narrow and broad injections so raw purity is not driven
        # by an arbitrary simulation prevalence.  The broad half still spans
        # contributions from deliberately difficult to dominant.
        "truth_broad_fraction": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.10, 0.25, 0.50, 0.80]
        ),
        "truth_pgamma_to_hei_ratio": np.array([0.0, 0.03, 0.10, 0.30, 0.60]),
        "truth_effective_resolving_power": np.array([320.0, 400.0, 480.0, 560.0, 640.0]),
        "truth_narrow_velocity_kms": np.array([-250.0, 0.0, 250.0]),
        "truth_broad_velocity_kms": np.array([-500.0, 0.0, 500.0]),
    }
    rows: list[dict[str, object]] = []
    permutations = {
        name: rng.permutation(np.resize(values, n_injections))
        for name, values in levels.items()
    }
    source_order = rng.permutation(np.resize(np.asarray(base_ids, dtype=object), n_injections))
    seeds = rng.integers(0, 2**32 - 1, n_injections, dtype=np.uint32)
    for index in range(n_injections):
        row = {
            "injection_id": index,
            "injection_design_version": DESIGN_VERSION,
            "source_object_id": str(source_order[index]),
            "noise_seed": int(seeds[index]),
        }
        row.update({name: float(values[index]) for name, values in permutations.items()})
        row["truth_has_broad"] = row["truth_broad_fraction"] > 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def inject_and_fit(loaded: Any, design: dict[str, object]) -> dict[str, object]:
    """Inject one realization into an archived real error/mask pattern."""

    spectrum = loaded.spectrum
    continuum = loaded.continuum
    wave = np.asarray(spectrum.wave_rest, dtype=float)
    error = np.asarray(spectrum.err, dtype=float)
    valid = np.asarray(spectrum.valid_mask, dtype=bool)
    local = (wave >= 10550.0) & (wave <= 11150.0)
    valid_local = valid & local & np.isfinite(error) & (error > 0)
    if np.count_nonzero(valid_local) < 30:
        raise ValueError("Base spectrum lacks 30 valid He I/Pa-gamma pixels")

    effective_r = float(design["truth_effective_resolving_power"])
    instrument = C_KMS / effective_r
    narrow_width = float(
        np.hypot(design["truth_narrow_fwhm_intrinsic_kms"], instrument)
    )
    broad_width = float(
        np.hypot(design["truth_broad_fwhm_intrinsic_kms"], instrument)
    )
    narrow_velocity = float(design["truth_narrow_velocity_kms"])
    broad_velocity = float(design["truth_broad_velocity_kms"])
    hei_center = lines.get("hei_10833").vacuum_wavelength * (
        1.0 + narrow_velocity / C_KMS
    )
    pgamma_center = lines.get("pagamma").vacuum_wavelength * (
        1.0 + narrow_velocity / C_KMS
    )
    flux_sigma = _line_flux_sigma(
        wave, error, valid_local, hei_center, narrow_width
    )
    hei_narrow_flux = float(design["truth_hei_snr"]) * flux_sigma
    ratio = float(design["truth_pgamma_to_hei_ratio"])
    pgamma_narrow_flux = ratio * hei_narrow_flux
    broad_fraction = float(design["truth_broad_fraction"])
    hei_broad_flux = (
        broad_fraction / (1.0 - broad_fraction) * hei_narrow_flux
        if 0.0 < broad_fraction < 1.0 else 0.0
    )
    pgamma_broad_flux = ratio * hei_broad_flux
    hei_broad_center = lines.get("hei_10833").vacuum_wavelength * (
        1.0 + broad_velocity / C_KMS
    )
    pgamma_broad_center = lines.get("pagamma").vacuum_wavelength * (
        1.0 + broad_velocity / C_KMS
    )
    injected = (
        _gaussian_area_profile(wave, hei_narrow_flux, hei_center, narrow_width)
        + _gaussian_area_profile(wave, pgamma_narrow_flux, pgamma_center, narrow_width)
        + _gaussian_area_profile(wave, hei_broad_flux, hei_broad_center, broad_width)
        + _gaussian_area_profile(wave, pgamma_broad_flux, pgamma_broad_center, broad_width)
    )
    rng = np.random.default_rng(int(design["noise_seed"]))
    noise = np.zeros_like(wave)
    noise[valid] = rng.normal(0.0, error[valid])
    simulated = qsospec.Spectrum(
        wave_obs=spectrum.wave_obs.copy(),
        flux=continuum.model + injected + noise,
        err=error.copy(),
        z=spectrum.z,
        metadata=spectrum.metadata,
        mask=None if spectrum.mask is None else spectrum.mask.copy(),
    )
    pair = qsospec.fit_hei_pgamma_model_pair(simulated, continuum)
    if pair is None:
        raise RuntimeError("An explicitly covered injection became not covered")
    record = pair.to_record(int(design["injection_id"]))
    record.update(design)
    record["source_redshift"] = float(spectrum.z)
    record["snr_stratum"] = str(
        pd.cut(
            [float(design["truth_hei_snr"])],
            bins=[-np.inf, 7.0, 15.0, np.inf],
            labels=["low", "medium", "high"],
            right=False,
        )[0]
    )
    record["redshift_stratum"] = str(
        pd.cut(
            [float(spectrum.z)],
            bins=[-np.inf, 0.30, 0.48, np.inf],
            labels=["low", "middle", "high"],
            right=False,
        )[0]
    )
    record["resolution_stratum"] = str(
        pd.cut(
            [effective_r],
            bins=[-np.inf, 440.0, 520.0, np.inf],
            labels=["extended", "point_source", "higher_resolution"],
            right=False,
        )[0]
    )
    record["truth_hei_narrow_flux"] = hei_narrow_flux
    record["truth_pgamma_narrow_flux"] = pgamma_narrow_flux
    record["truth_hei_broad_flux"] = hei_broad_flux
    record["truth_pgamma_broad_flux"] = pgamma_broad_flux
    return record


def _initialize_worker(run_directory: str) -> None:
    global _WORKER_STORE
    _WORKER_STORE = qsospec.open_run(run_directory)


def _fit_source_group(payload):
    source_id, designs = payload
    loaded = qsospec.load_model(_WORKER_STORE, str(source_id))
    rows = []
    for design in designs:
        try:
            rows.append(inject_and_fit(loaded, design))
        except Exception as error:  # noqa: BLE001 - recovery failures are outputs
            rows.append(
                {
                    **design,
                    "fit_status": "fit_failed",
                    "evidence_status": "fit_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return rows


def _select_base_ids(objects: pd.DataFrame, count: int) -> list[str]:
    redshift = pd.to_numeric(objects.get("redshift"), errors="coerce")
    if redshift.isna().all() and "metadata" in objects:
        redshift = objects["metadata"].map(
            lambda value: float(_decode_key_values(value).get("redshift", np.nan))
        )
    eligible = objects[
        np.isfinite(redshift)
        & (10550.0 * (1.0 + redshift) >= 12047.0)
        & (11150.0 * (1.0 + redshift) <= 18734.0)
    ].copy()
    eligible["_redshift"] = redshift.loc[eligible.index]
    eligible = eligible.sort_values(["_redshift", "object_id"], kind="stable")
    if not len(eligible):
        raise ValueError("No archived objects have theoretical He I coverage")
    indices = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    return eligible.iloc[indices]["object_id"].astype(str).tolist()


def main() -> None:
    args = parse_args()
    if args.n_injections <= 0 or args.n_base_patterns <= 0 or args.workers <= 0:
        raise ValueError("Injection count, base-pattern count, and workers must be positive")
    root = args.output_root or default_root()
    run_dir = args.run_directory or root / "runs" / RUN_NAME
    base_output = (
        args.output_directory
        or root / "classification" / CLASSIFICATION_NAME / "calibration" / CALIBRATION_NAME
    )
    output_dir = base_output / "smoke" if args.mode == "smoke" else base_output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "injection_results.parquet"
    if output_path.exists() and not args.force:
        results = pd.read_parquet(output_path)
        if (
            "injection_design_version" not in results
            or not results["injection_design_version"].eq(DESIGN_VERSION).all()
        ):
            raise RuntimeError(
                "Existing injection results use an obsolete/unversioned design; "
                "rerun explicitly with --force."
            )
    else:
        store = qsospec.open_run(str(run_dir))
        objects = store.read_table("objects").to_pandas()
        n_injections = 64 if args.mode == "smoke" else args.n_injections
        base_ids = _select_base_ids(objects, min(args.n_base_patterns, n_injections))
        design = injection_design(n_injections, base_ids, seed=args.seed)
        payloads = [
            (source_id, group.to_dict("records"))
            for source_id, group in design.groupby("source_object_id", sort=False)
        ]
        if args.workers == 1:
            _initialize_worker(str(run_dir))
            nested = [_fit_source_group(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_initialize_worker,
                initargs=(str(run_dir),),
            ) as executor:
                nested = list(executor.map(_fit_source_group, payloads))
        results = pd.DataFrame([row for rows in nested for row in rows]).sort_values(
            "injection_id", kind="stable"
        )
        results.to_parquet(output_path, index=False)

    complete = results[results["fit_status"].eq("complete")].copy()
    sweep = threshold_sweep(complete)
    sweep.to_parquet(output_dir / "threshold_sweep.parquet", index=False)
    provisional, _ = selection_mask(complete)
    strata = stratified_recovery(complete, provisional)
    strata.to_parquet(output_dir / "stratified_recovery.parquet", index=False)
    summary = {
        "created_at": utc_now(),
        "mode": args.mode,
        "run_directory": str(run_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "random_seed": args.seed,
        "injection_design_version": DESIGN_VERSION,
        "n_requested": 64 if args.mode == "smoke" else args.n_injections,
        "n_results": len(results),
        "n_complete": len(complete),
        "n_failed": int((~results["fit_status"].eq("complete")).sum()),
        **summarize_recovery(complete, provisional),
        "calibration_scope": "hei_pgamma_real_rgs_error_mask_injection",
        "dimensions": [
            "redshift",
            "He I S/N",
            "intrinsic narrow width",
            "broad width",
            "broad fraction",
            "Pa-gamma/He I ratio",
            "effective resolving power/source extent",
        ],
        "material_passport_version": "dr1_narrow_hei_pgamma_r480_plan_v1",
        "verification_status": "UNVERIFIED",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        "# He I/Pa-gamma injection/recovery calibration\n\n"
        f"Completed {len(complete):,} of {len(results):,} injections. The locked "
        f"rule has estimated injection purity {summary['estimated_purity']!r}, "
        f"completeness {summary['estimated_completeness']!r}, and broad-source "
        f"false-narrow rate {summary['broad_false_narrow_rate']!r}.\n\n"
        "No label is promoted from these injections. Independent higher-resolution "
        "line decompositions are required even if the 95% injection-purity target "
        "is met.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
