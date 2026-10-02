"""Deterministic shared multi-start searches with failed-start bookkeeping."""
import numpy as np
from scipy.optimize import OptimizeResult


def solve_multistart(context, wave, flux, error, config, *, n_starts=1,
                     max_starts=None, seed=1729, initial_values=None,
                     start_values=None, expand_search=None):
    from .global_fit import _solve_once_with_fallback
    if not isinstance(n_starts, int) or n_starts<1:
        raise ValueError('n_starts must be a positive integer')
    max_starts = n_starts if max_starts is None else max_starts
    if not isinstance(max_starts, int) or max_starts<n_starts:
        raise ValueError('max_starts must be an integer >= n_starts')
    for name,value in (initial_values or {}).items():
        if name not in context.index or not np.isfinite(value):
            raise ValueError(f'Invalid initial parameter: {name}')
        i=context.index[name]; context.initial[i]=np.clip(value,context.lower[i],context.upper[i])
    base=context.initial.copy(); rng=np.random.default_rng(seed)
    starts=[];solutions=[]; expansion=[]; budget=n_starts
    for trial in range(max_starts):
        if trial==budget:
            eligible=[i for i,s in enumerate(starts) if s['success'] and np.isfinite(s['chi2'])]
            best=min(eligible or range(len(starts)),key=lambda i:starts[i]['chi2'])
            expansion=list(expand_search(solutions[best][0],starts) or ()) if expand_search else []
            if not expansion:break
            budget=max_starts
        initial=base.copy()
        if trial:
            for name in context.nonlinear_names:
                i=context.index[name];lo,hi=context.lower[i],context.upper[i]
                if trial==1 and name.endswith('velocity_kms'):
                    initial[i]=np.clip(lo+hi-initial[i],lo,hi)
                elif name.endswith('fwhm_kms'):
                    initial[i]=np.exp(rng.uniform(np.log(lo),np.log(hi)))
                else:initial[i]=rng.uniform(lo,hi)
        if start_values and trial<len(start_values):
            # Explicit starts are perturbations of the same warm baseline.
            initial=base.copy()
            for name,value in start_values[trial].items():
                if name not in context.index or not np.isfinite(value):
                    raise ValueError(f'Invalid start parameter: {name}')
                i=context.index[name];initial[i]=np.clip(value,context.lower[i],context.upper[i])
        context.initial=initial
        try:
            solution,method,fallback=_solve_once_with_fallback(context,wave,flux,error,initial,config)
            score=float(np.sum(((flux-context.model(solution.x,wave))/error)**2))
            if not np.isfinite(score):score=float('inf')
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            solution=OptimizeResult(x=initial,success=False,status=-1,message=str(exc),
                jac=np.zeros((len(wave),len(initial))),nfev=0)
            score=float('inf');method='failed';fallback=str(exc)
        starts.append(dict(start=trial,success=bool(solution.success and np.isfinite(score)),
            chi2=score,initial=dict(zip(context.names,initial.tolist())),
            parameters=dict(zip(context.names,solution.x.tolist())),
            status=int(solution.status),message=str(solution.message),
            nfev=int(getattr(solution,'nfev',0)),optimizer=method,fallback=fallback))
        solutions.append((solution,method,fallback))
    valid=[i for i,s in enumerate(starts) if s['success'] and np.isfinite(s['chi2'])]
    best=min(valid or range(len(starts)),key=lambda i:starts[i]['chi2'])
    context.initial=base
    return (*solutions[best],dict(seed=int(seed),selected_start=best,starts=starts,
        expansion_reasons=expansion,successful_starts=len(valid),search_failed=not bool(valid)))
