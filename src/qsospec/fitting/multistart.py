"""Deterministic shared multi-start searches with failed-start bookkeeping."""

import numpy as np
from scipy.optimize import OptimizeResult


def _solve_multistart(
    context,
    wave,
    flux,
    error,
    config,
    *,
    n_starts=1,
    max_starts=None,
    seed=1729,
    initial_values=None,
    start_values=None,
    expand_search=None,
    first_start=None,
):
    from .global_fit import _solve_once_with_fallback

    if not isinstance(n_starts, int) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer")
    max_starts = n_starts if max_starts is None else max_starts
    if not isinstance(max_starts, int) or max_starts < n_starts:
        raise ValueError("max_starts must be an integer >= n_starts")
    for name, value in (initial_values or {}).items():
        if name not in context.index or not np.isfinite(value):
            raise ValueError(f"Invalid initial parameter: {name}")
        i = context.index[name]
        context.initial[i] = np.clip(value, context.lower[i], context.upper[i])
    base = context.initial.copy()
    rng = np.random.default_rng(seed)
    starts = []
    solutions = []
    expansion = []
    budget = n_starts
    for trial in range(max_starts):
        if trial == budget:
            eligible = [i for i, s in enumerate(starts) if s["success"] and np.isfinite(s["chi2"])]
            best = min(eligible or range(len(starts)), key=lambda i: starts[i]["chi2"])
            expansion = list(expand_search(solutions[best][0], starts) or ()) if expand_search else []
            if not expansion:
                break
            if first_start is not None:
                break
            budget = max_starts
        initial = base.copy()
        if trial:
            for name in context.nonlinear_names:
                i = context.index[name]
                lo, hi = context.lower[i], context.upper[i]
                if trial == 1 and name.endswith("velocity_kms"):
                    initial[i] = np.clip(lo + hi - initial[i], lo, hi)
                elif name.endswith("fwhm_kms"):
                    initial[i] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
                else:
                    initial[i] = rng.uniform(lo, hi)
        if start_values and trial < len(start_values):
            # Explicit starts are perturbations of the same warm baseline.
            initial = base.copy()
            for name, value in start_values[trial].items():
                if name not in context.index or not np.isfinite(value):
                    raise ValueError(f"Invalid start parameter: {name}")
                i = context.index[name]
                initial[i] = np.clip(value, context.lower[i], context.upper[i])
        if trial == 0 and first_start is not None:
            for name, value in first_start.items():
                if name not in context.index or not np.isfinite(value):
                    raise ValueError(f"Invalid warm parameter: {name}")
                i = context.index[name]
                initial[i] = np.clip(value, context.lower[i], context.upper[i])
        context.initial = initial
        try:
            solution, method, fallback = _solve_once_with_fallback(context, wave, flux, error, initial, config)
            score = float(np.sum(((flux - context.model(solution.x, wave)) / error) ** 2))
            if not np.isfinite(score):
                score = float("inf")
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            solution = OptimizeResult(
                x=initial, success=False, status=-1, message=str(exc), jac=np.zeros((len(wave), len(initial))), nfev=0
            )
            score = float("inf")
            method = "failed"
            fallback = str(exc)
        starts.append(
            dict(
                start=trial,
                success=bool(solution.success and np.isfinite(score)),
                chi2=score,
                initial=dict(zip(context.names, initial.tolist())),
                parameters=dict(zip(context.names, solution.x.tolist())),
                status=int(solution.status),
                message=str(solution.message),
                nfev=int(getattr(solution, "nfev", 0)),
                optimizer=method,
                fallback=fallback,
            )
        )
        solutions.append((solution, method, fallback))
    valid = [i for i, s in enumerate(starts) if s["success"] and np.isfinite(s["chi2"])]
    best = min(valid or range(len(starts)), key=lambda i: starts[i]["chi2"])
    context.initial = base
    return (
        *solutions[best],
        dict(
            seed=int(seed),
            selected_start=best,
            starts=starts,
            expansion_reasons=expansion,
            successful_starts=len(valid),
            search_failed=not bool(valid),
        ),
    )


def solve_multistart(context, wave, flux, error, config, *, warm_values=None, **kwargs):
    """Retain the cold alternatives; discard an inadequate warm attempt."""
    from .performance import current_session

    result = _solve_multistart(context, wave, flux, error, config, first_start=warm_values, **kwargs)
    record = result[3]
    reasons = list(record["expansion_reasons"])
    if record["search_failed"]:
        reasons.append("unsuccessful_search")
    attempts = []
    session = current_session()
    if warm_values is not None:
        if session:
            session.statistics["warm_searches"] += 1
        attempts.append(dict(kind="warm", record=record))
        if reasons:
            if session:
                session.statistics["cold_retries"] += 1
            result = _solve_multistart(context, wave, flux, error, config, **kwargs)
            attempts.append(dict(kind="cold_retry", record=result[3]))
    metadata = dict(result[3])
    metadata["warm_start"] = dict(
        used=warm_values is not None,
        retry_reasons=reasons if warm_values is not None else [],
        attempts=attempts,
        provenance="previous_successful_compatible_workflow_candidate" if warm_values is not None else None,
    )
    all_starts = [s for attempt in attempts for s in attempt["record"]["starts"]] if attempts else metadata["starts"]
    metadata["total_nfev"] = sum(s["nfev"] for s in all_starts)
    metadata["total_optimizer_calls"] = len(all_starts)
    return (*result[:3], metadata)
