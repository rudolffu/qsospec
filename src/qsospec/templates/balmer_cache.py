"""Exact, bounded, workflow-local Balmer evaluation reuse."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import numpy as np

_ACTIVE = ContextVar("qsospec_balmer_cache", default=None)


class BalmerCache:
    def __init__(self, max_entries=128, max_bytes=16 * 1024**2):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("Cache limits must be nonnegative")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def evaluate(self, function, template, wave, width, velocity):
        grid = np.ascontiguousarray(wave, dtype=float)
        token = hashlib.sha256()
        for array in (template.wavelength_vacuum, template.rel_flux_hbeta):
            array = np.ascontiguousarray(array)
            token.update(str((array.dtype.str, array.shape)).encode())
            token.update(array.tobytes())
        key = (
            token.digest(),
            grid.shape,
            hashlib.sha256(grid.tobytes()).digest(),
            np.float64(width).tobytes(),
            np.float64(velocity).tobytes(),
        )
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        values = function(template, grid, width, velocity)
        # Immutable backing bytes prevent callers re-enabling writes.
        values = tuple(np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape) for x in values)
        size = sum(x.nbytes for x in values)
        if self.max_entries and size <= self.max_bytes:
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                _, old = self.entries.popitem(last=False)
                self.bytes -= sum(x.nbytes for x in old)
            self.entries[key] = values
            self.bytes += size
        return values


@contextmanager
def balmer_cache(enabled=True, max_entries=128, max_bytes=16 * 1024**2):
    """Create an isolated cache, or explicitly disable caching for a workflow."""
    cache = BalmerCache(max_entries, max_bytes) if enabled else False
    token = _ACTIVE.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE.reset(token)


def cache_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _ACTIVE.get() is not None:
            return function(*args, **kwargs)
        with balmer_cache():
            return function(*args, **kwargs)

    return wrapped


def cached_series(function):
    @wraps(function)
    def wrapped(template, wave_rest, fwhm_kms, velocity_kms):
        cache = _ACTIVE.get()
        if not cache:
            return function(template, wave_rest, fwhm_kms, velocity_kms)
        return cache.evaluate(function, template, wave_rest, fwhm_kms, velocity_kms)

    return wrapped
