"""Isolated speed prototypes; patches are process-local, never production edits.

Run with PYTHONPATH=src: python benchmarks/evaluate_fit_speed.py --repeats 2
Outputs live under ignored validation/. Uses prepared, already corrected spectra.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import qsospec
from audit_desi_fit import prepared_data
from qsospec.fitting import complexes, adaptive_oiii
from qsospec.solvers import variable_projection as vp
from qsospec.templates.balmer_cache import balmer_cache
from qsospec.workflows import host_workflow
from qsospec.workflows.host.ppxf_host import _require_ppxf

ORIGINAL_DESIGN = complexes.GenericComplexContext.separable_design
ORIGINAL_JAC = vp._VariableProjectionProblem.jacobian
ORIGINAL_GENERIC = adaptive_oiii.fit_generic_complex


def lean_design(self, nonlinear, wave, need_derivatives):
    # Same accumulation order, parameter definitions, and profile arithmetic.
    if not hasattr(self, "_speed_definition"):
        linear, names = self.linear_names, self.nonlinear_names
        by_flux = {}
        for instance in self.instances:
            ident, component = instance[:2]
            key = f"{component.fixed_ratio_to}.flux" if component.fixed_ratio_to is not None else f"{ident}.flux"
            by_flux.setdefault(key, []).append(instance)
        self._speed_definition = linear, names, by_flux
    linear, names, by_flux = self._speed_definition
    values = dict(zip(names, map(float, nonlinear)))
    zero = np.zeros_like(wave)
    columns = []
    derivative_columns = [[] for _ in names] if need_derivatives else None
    for name in linear:
        derivatives = {}
        if name == "continuum.constant":
            basis = np.ones_like(wave)
        elif name == "continuum.slope":
            basis = wave - self.pivot
        else:
            basis = np.zeros_like(wave)
            for instance in by_flux.get(name, ()):
                profile = self._instance_basis(instance, values, wave)
                basis += profile[0]
                if need_derivatives:
                    for key, derivative in (
                        (instance[3] + ".velocity_kms", profile[1]),
                        (instance[4] + ".fwhm_kms", profile[2]),
                    ):
                        derivatives[key] = derivatives.get(key, zero) + derivative
        columns.append(basis)
        if need_derivatives:
            for i, key in enumerate(names):
                derivative_columns[i].append(derivatives.get(key, zero))
    return np.column_stack(columns), None if not need_derivatives else tuple(
        np.column_stack(c) for c in derivative_columns
    )


def cached_design(self, nonlinear, wave, need_derivatives):
    # One-entry cache within one context, exact arguments; no interpolation.
    previous = getattr(self, "_speed_pair", None)
    if (
        previous is None
        or not np.array_equal(previous[0], nonlinear)
        or not np.array_equal(previous[1], wave)
        or previous[2] is not self.line_lsf
    ):
        design, derivatives = lean_design(self, nonlinear, wave, True)
        previous = (nonlinear.copy(), wave.copy(), self.line_lsf, design, derivatives)
        self._speed_pair = previous
    return previous[3], previous[4] if need_derivatives else None


def batched_jacobian(self, nonlinear):
    state = self.state(nonlinear, True)
    design = state.design / self.err[:, None]
    free = state.linear_active_mask == 0
    derivatives = np.asarray(state.derivatives) / self.err[None, :, None]
    direct = derivatives @ state.linear  # (parameter, pixel)
    if np.any(free):
        free_design = design[:, free]
        inverse = np.linalg.pinv(free_design.T @ free_design)
        rhs = derivatives[:, :, free].transpose(0, 2, 1) @ state.residual - direct @ free_design
        coefficients = rhs @ inverse.T
        jacobian = (-direct - coefficients @ free_design.T).T
    else:
        jacobian = -direct.T
    if state.prior.size:
        jacobian = np.vstack((jacobian, self.prior_jacobian(nonlinear)))
    if not np.all(np.isfinite(jacobian)):
        raise vp.VariableProjectionError("Invalid prototype Jacobian")
    return jacobian


def snapshot(result):
    arrays = {
        "host": result.host_model_on_quasar_grid,
        "continuum.model": result.continuum.model,
        "continuum.covariance": result.continuum.covariance,
    }
    scalars = {"continuum.params": result.continuum.param_values, "continuum.errors": result.continuum.param_errors}
    decisions = []
    for name, fit in result.line_complexes.items():
        arrays[name + ".model"] = fit.model
        arrays[name + ".covariance"] = fit.covariance
        scalars[name + ".params"] = fit.param_values
        scalars[name + ".metrics"] = fit.metrics
        for candidate in fit.metadata.get("candidate_selection", []):
            decisions.append(
                (name, candidate["component_count"], candidate["accepted"], candidate["rejection_reasons"])
            )
    return {
        "arrays": arrays,
        "scalars": deepcopy(scalars),
        "decisions": decisions,
        "z": result.spectrum.z,
        "warnings": result.warning_codes(),
    }


def compare(reference, candidate):
    exact, close, mismatches = True, True, []
    max_model_sigma = 0.0
    for key, first in reference["arrays"].items():
        second = candidate["arrays"].get(key)
        if first is None or second is None:
            equal = first is None and second is None
            exact &= equal
            close &= equal
            continue
        first, second = np.asarray(first), np.asarray(second)
        equal = first.shape == second.shape and np.array_equal(first, second, equal_nan=True)
        tolerant = first.shape == second.shape and np.allclose(first, second, rtol=1e-10, atol=1e-12, equal_nan=True)
        exact &= equal
        close &= tolerant
        if not tolerant:
            mismatches.append(key)
        if key.endswith(".model") and first.shape == second.shape:
            max_model_sigma = max(max_model_sigma, float(np.nanmax(np.abs(first - second) / ERROR)))
    max_param_shift_sigma = 0.0
    largest_parameter_shift = None
    for group, first in reference["scalars"].items():
        second = candidate["scalars"].get(group, {})
        for key, value in first.items():
            other = second.get(key, np.nan)
            equal = bool(np.array_equal(value, other, equal_nan=True))
            tolerant = bool(np.allclose(value, other, rtol=1e-10, atol=1e-12, equal_nan=True))
            exact &= equal
            close &= tolerant
            if not tolerant:
                mismatches.append(group + "." + key)
            if group.endswith(".params"):
                error = PARAM_ERRORS.get(group + "." + key, np.nan)
                if np.isfinite(error) and error > 0:
                    shift = float(abs(value - other) / error)
                    if shift > max_param_shift_sigma:
                        max_param_shift_sigma = shift
                        largest_parameter_shift = dict(
                            parameter=group + "." + key, baseline=value, prototype=other, baseline_error=error
                        )
    return dict(
        exact=bool(exact),
        reconstruction_tolerance=bool(close),
        changed_groups=mismatches,
        max_model_change_pixel_sigma=max_model_sigma,
        max_parameter_change_error_units=max_param_shift_sigma,
        largest_parameter_shift=largest_parameter_shift,
        decisions_equal=reference["decisions"] == candidate["decisions"],
        warnings_equal=reference["warnings"] == candidate["warnings"],
        redshift_equal=reference["z"] == candidate["z"],
    )


def main():
    global ERROR, PARAM_ERRORS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--modes", nargs="+", default=["baseline", "overhead", "paired_derivatives", "jacobian", "warm", "combined"]
    )
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("validation/fit_speed_20261009"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        pq.read_table(p).to_pylist()[0]
        for p in sorted(Path("validation/desi_performance_20261009/archived_run/data/models").glob("*.parquet"))
    ]
    selected = [
        min(rows, key=lambda x: x["redshift"]),
        min(rows, key=lambda x: abs(x["redshift"] - 0.75)),
        max(rows, key=lambda x: x["redshift"]),
    ]
    row = selected[args.sample_index]
    host_dict = json.load(open("validation/compact_storage_20261009/compact_final/runtime.json"))["config"]
    host_dict["broad_line_prefit"] = qsospec.HostBroadLinePrefitConfig(**host_dict["broad_line_prefit"])
    host_dict["agn_pseudocontinuum"] = qsospec.HostAgnPseudoContinuumConfig(**host_dict["agn_pseudocontinuum"])
    host_dict["coverage"] = qsospec.HostCoverageConfig(**host_dict["coverage"])
    host_dict["continuum_windows"] = [tuple(w) for w in host_dict["continuum_windows"]]
    host = qsospec.HostDecompConfig(**host_dict)
    _require_ppxf()
    reference = None
    records = []
    modes = args.modes
    with threadpool_limits(limits=2):
        for repeat in range(args.repeats):
            # Reverse order on second sweep to reduce systematic drift.
            order = modes if repeat % 2 == 0 else list(reversed(modes))
            for mode in order:
                warm_cache = {}
                calls = []
                jac_checks = []

                def generic(*a, **kw):
                    recipe = a[2]
                    cache_key = tuple(c.id for c in recipe.components)
                    if mode in ("warm_first", "paired_warm_first"):
                        cache_key = (cache_key, a[0].flux.tobytes())
                    previous = warm_cache.get(cache_key)
                    if mode in ("warm", "combined", "warm_first", "paired_warm_first") and previous is not None:
                        starts = deepcopy(kw["start_values"])
                        starts[0] = previous
                        kw.update(start_values=starts)
                        if mode in ("warm", "combined"):
                            kw["initial_values"] = previous
                    result = ORIGINAL_GENERIC(*a, **kw)
                    warm_cache[cache_key] = dict(result.param_values)
                    starts = result.metadata.get("multistart", {}).get("starts", [])
                    calls.append(
                        dict(
                            recipe=recipe.id,
                            chi2=result.chi2,
                            bic=result.bic,
                            nfev=sum(s["nfev"] for s in starts),
                            starts=len(starts),
                            warmed=previous is not None
                            and mode in ("warm", "combined", "warm_first", "paired_warm_first"),
                        )
                    )
                    return result

                def checked_jac(self, theta):
                    value = batched_jacobian(self, theta)
                    if len(jac_checks) < 20:
                        original = ORIGINAL_JAC(self, theta)
                        jac_checks.append(
                            dict(
                                max_absolute=float(np.max(np.abs(original - value))),
                                close=bool(np.allclose(original, value, rtol=1e-10, atol=1e-12)),
                            )
                        )
                    return value

                with ExitStack() as stack:
                    stack.enter_context(balmer_cache())
                    stack.enter_context(patch.object(adaptive_oiii, "fit_generic_complex", generic))
                    if mode == "overhead":
                        stack.enter_context(
                            patch.object(complexes.GenericComplexContext, "separable_design", lean_design)
                        )
                    if mode in ("paired_derivatives", "combined", "paired_warm_first"):
                        stack.enter_context(
                            patch.object(complexes.GenericComplexContext, "separable_design", cached_design)
                        )
                    if mode in ("jacobian", "combined"):
                        stack.enter_context(patch.object(vp._VariableProjectionProblem, "jacobian", checked_jac))
                    start = perf_counter()
                    result = host_workflow._run_global_fit_with_optional_host(
                        prepared_data(row),
                        source="prepared_archived_desi",
                        input_path="read_only_benchmark",
                        run_host_decomp=True,
                        host_config=host,
                        global_config=qsospec.GlobalContinuumConfig(),
                        hbeta_config=qsospec.HbetaComplexConfig(),
                        uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=0),
                        galactic_extinction_config=qsospec.GalacticExtinctionConfig(enabled=False),
                    )
                    seconds = perf_counter() - start
                current = snapshot(result)
                if reference is None:
                    reference = current
                    ERROR = result.spectrum.err
                    PARAM_ERRORS = {}
                    for name, fit in [("continuum", result.continuum), *result.line_complexes.items()]:
                        PARAM_ERRORS.update({name + ".params." + k: v for k, v in fit.param_errors.items()})
                records.append(
                    dict(
                        mode=mode,
                        repeat=repeat,
                        seconds=seconds,
                        candidates=calls,
                        jacobian_checks=jac_checks,
                        comparison=compare(reference, current),
                    )
                )
                report = dict(
                    object_key=row["object_key"],
                    z=row["redshift"],
                    blas_threads=2,
                    profiled=False,
                    original_runs_modified=False,
                    production_changes=False,
                    records=records,
                )
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    mode,
                    repeat,
                    round(seconds, 3),
                    "nfev",
                    sum(c["nfev"] for c in calls),
                    "exact",
                    records[-1]["comparison"]["exact"],
                    "decisions",
                    records[-1]["comparison"]["decisions_equal"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
