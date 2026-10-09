"""Workflow-local performance controls, exact solver caches and line seeds."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import wraps
from hashlib import sha256
import inspect
import json

import numpy as np

from ..config import FitPerformanceConfig


@dataclass
class PerformanceSession:
    config: FitPerformanceConfig
    seeds: dict = field(default_factory=dict)
    statistics: dict = field(
        default_factory=lambda: dict(
            hits=0,
            misses=0,
            bypasses=0,
            peak_cache_bytes=0,
            evaluator_calls=0,
            warm_searches=0,
            cold_retries=0,
            optimizer_calls=0,
            linear_solves=0,
        )
    )


_CURRENT = ContextVar("qsospec_fit_performance", default=None)


def count_statistic(name):
    session = _CURRENT.get()
    if session is not None:
        session.statistics[name] += 1


def current_session():
    return _CURRENT.get()


def performance_scope(*, workflow=False):
    """Respect explicit overrides and isolate every workflow/uncertainty trial."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            explicit = bound.arguments.get("performance")
            if explicit is not None and not isinstance(explicit, FitPerformanceConfig):
                raise TypeError("performance must be a FitPerformanceConfig instance")
            active = current_session()
            config = bound.arguments.get("global_config", bound.arguments.get("config"))
            configured = getattr(config, "performance", None)
            settings = explicit or (configured if workflow or active is None else None)
            if settings is not None and not isinstance(settings, FitPerformanceConfig):
                raise TypeError("performance must be a FitPerformanceConfig instance")
            own = workflow or active is None or (explicit is not None and explicit != active.config)
            session = (
                PerformanceSession(settings or (active.config if active else FitPerformanceConfig())) if own else active
            )
            token = _CURRENT.set(session) if own else None
            before = dict(session.statistics)
            try:
                result = function(*args, **kwargs)
                if result is not None and hasattr(result, "metadata"):
                    result.metadata["fit_performance"] = dict(
                        settings=asdict(session.config),
                        statistics={
                            key: value - before[key] if key != "peak_cache_bytes" else value
                            for key, value in session.statistics.items()
                        },
                        cache_scope="solver",
                        warm_start_scope="workflow",
                        implementation_version="1",
                    )
                return result
            finally:
                if token is not None:
                    _CURRENT.reset(token)

        return wrapped

    return decorate


class ExactEvaluationCache:
    """One retained evaluation per solver, including scaled derivative arrays.

    The evaluator's compiled definition is immutable during a solve. A state
    token detects replacement of its definition, parameter order or LSF.
    """

    def __init__(self, evaluator, wave, *, state_token=None, derivatives=True):
        self.evaluator = evaluator
        self.wave = wave
        self.state_token = state_token or (lambda: None)
        self.derivatives = derivatives
        self.session = current_session()
        self.config = self.session.config if self.session else FitPerformanceConfig()
        self.entry = None
        self.statistics = dict(hits=0, misses=0, bypasses=0, peak_cache_bytes=0, evaluator_calls=0)

    def count(self, key, value=1):
        if key == "peak_cache_bytes":
            self.statistics[key] = max(self.statistics[key], value)
            if self.session:
                self.session.statistics[key] = max(self.session.statistics[key], value)
        else:
            self.statistics[key] += value
            if self.session:
                self.session.statistics[key] += value

    def __call__(self, nonlinear, need_derivatives):
        state = self.state_token()
        entry = self.entry
        if (
            entry is not None
            and entry[0] == state
            and entry[1].shape == np.asarray(nonlinear).shape
            and entry[1].dtype == np.asarray(nonlinear).dtype
            and entry[1].tobytes() == np.asarray(nonlinear).tobytes()
            and entry[2].shape == np.asarray(self.wave).shape
            and entry[2].dtype == np.asarray(self.wave).dtype
            and entry[2].tobytes() == np.asarray(self.wave).tobytes()
            and (not need_derivatives or entry[4] is not None)
        ):
            self.count("hits")
            return entry[3], entry[4] if need_derivatives else None
        self.count("misses")
        self.count("evaluator_calls")
        if not self.config.cache_evaluations:
            return self.evaluator(nonlinear, need_derivatives)
        design, derivatives = self.evaluator(nonlinear, need_derivatives or self.derivatives)
        arrays = (design,) + tuple(derivatives or ())
        size = sum(array.nbytes for array in arrays) + np.asarray(nonlinear).nbytes + np.asarray(self.wave).nbytes
        self.entry = None
        if size > self.config.evaluation_cache_max_bytes:
            self.count("bypasses")
            return design, derivatives if need_derivatives else None

        def immutable(array):
            array = np.asarray(array)
            return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

        theta, grid = immutable(nonlinear), immutable(self.wave)
        design = immutable(design)
        derivatives = tuple(immutable(array) for array in derivatives) if derivatives is not None else None
        self.entry = state, theta, grid, design, derivatives
        self.count("peak_cache_bytes", size)
        return design, derivatives if need_derivatives else None


def line_seed_key(spectrum, context, mask, optimizer_config=None):
    """Content identity of the input and candidate; continuum is deliberately omitted."""
    h = sha256()
    for value in (
        spectrum.wave_rest,
        spectrum.flux,
        spectrum.err,
        spectrum.valid_mask,
        mask,
        context.lower,
        context.upper,
    ):
        array = np.ascontiguousarray(value)
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(array.tobytes())
    definition = dict(
        optimizer=asdict(optimizer_config)
        if is_dataclass(optimizer_config)
        else vars(optimizer_config)
        if optimizer_config is not None
        else None,
        recipe=asdict(context.recipe),
        names=context.names,
        compiled_instances=context._compiled_instances,
        z=spectrum.z,
        metadata=spectrum.metadata.to_dict(),
        lsf=getattr(context.line_lsf, "metadata", None),
    )
    # Resolution is also archived in metadata; include its complete descriptor.
    from .line_lsf import resolution_to_dict

    definition["resolution"] = resolution_to_dict(spectrum.resolution)
    h.update(json.dumps(definition, sort_keys=True, default=str).encode())
    return h.hexdigest()
