"""Linear, flux-conserving Gaussian LSF quadrature for emission lines only.

The input kernel is specified in observed Angstrom. Columns integrate source
flux into native pixel edges using Gaussian CDF differences. Masked pixels are
selected *after* constructing the operator, so masks never widen their neighbors.
"""
from functools import lru_cache
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr
from ..resolution import SpectralResolution


def resolution_to_dict(resolution):
    if resolution is None:
        return None
    from dataclasses import asdict
    return {k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in asdict(resolution).items()}


def resolution_from_dict(value):
    return SpectralResolution(**value) if value else None


def pixel_edges(wave):
    wave = np.asarray(wave, float)
    if wave.size < 2 or not np.all(np.isfinite(wave)) or np.any(np.diff(wave) <= 0):
        raise ValueError('LSF requires increasing finite wavelengths')
    mid = (wave[:-1] + wave[1:]) / 2
    return np.r_[wave[0] - (mid[0]-wave[0]), mid, wave[-1] + (wave[-1]-mid[-1])]


class GaussianLineLSF:
    def __init__(self, descriptor, native_wave=None):
        self.descriptor = descriptor
        self.native_wave = None if native_wave is None else np.asarray(native_wave)
        self.resolution = resolution_from_dict(descriptor['resolution'])
        self.z = descriptor['z']
        lo, hi = descriptor['source_bounds_rest']
        n = int(np.ceil((hi-lo)/descriptor['step_rest']))
        self.step = (hi-lo)/n
        self.source_wave = lo + (np.arange(n)+.5)*self.step
        self.sigma = self.resolution.sigma_lambda(self.source_wave*(1+self.z))/(1+self.z)
        self._operators = {}

    def operator(self, wave, *, integrated=False):
        wave = np.asarray(wave, float)
        key = (integrated, wave.tobytes())
        if key in self._operators:
            return self._operators[key]
        # Fit subsets retain edges from the original, unmasked detector grid.
        if integrated:
            edges = pixel_edges(self.native_wave)
            ids = np.searchsorted(self.native_wave, wave)
            if np.any(ids >= len(self.native_wave)) or not np.array_equal(self.native_wave[ids], wave):
                raise ValueError('Pixel-integrated evaluation requires native centers')
            left, right = edges[ids], edges[ids+1]
        else:
            left = right = wave
        rows, cols, values = [], [], []
        radius = 8*float(np.max(self.sigma))
        for i, (a,b) in enumerate(zip(left,right)):
            start = np.searchsorted(self.source_wave, a-radius)
            stop = np.searchsorted(self.source_wave, b+radius)
            x = self.source_wave[start:stop]
            sig = self.sigma[start:stop]
            if integrated:
                value = (ndtr((b-x)/sig)-ndtr((a-x)/sig))*self.step/(b-a)
            else:
                value = np.exp(-.5*((a-x)/sig)**2)*self.step/(np.sqrt(2*np.pi)*sig)
            keep = value > 1.e-18
            rows.extend(np.full(np.count_nonzero(keep), i))
            cols.extend(np.arange(start,stop)[keep]); values.extend(value[keep])
        matrix = csr_matrix((values,(rows,cols)), shape=(len(wave),len(self.source_wave)))
        # Retain grids used repeatedly by the solver, not every peak-refinement point.
        if len(wave)>4 and len(self._operators)<6:
            self._operators[key] = matrix
        return matrix

    def profile(self, wave, rest_center, velocity, fwhm, profile, *, integrated=False):
        from .complexes import _profile
        source = np.column_stack(_profile(self.source_wave, rest_center, velocity, fwhm, profile))
        result = self.operator(wave, integrated=integrated) @ source
        return tuple(result[:,i] for i in range(3))


@lru_cache(maxsize=8)
def _saved_operator(encoded):
    return GaussianLineLSF(json.loads(encoded))


def saved_lsf(descriptor):
    return _saved_operator(json.dumps(descriptor, sort_keys=True))


def build_line_lsf(spectrum, fit_window, *, min_fwhm_kms=70.):
    """Return a supported operator, or an explicit observed-profile fallback."""
    resolution = spectrum.resolution
    metadata = {'status': 'observed_profile', 'reason': 'missing_resolution',
                'wavelength_frame': 'observed', 'applied_to': 'emission_lines_only'}
    if resolution is None or resolution.mode == 'missing':
        return None, metadata
    metadata.update(resolution.metadata())
    if resolution.mode == 'banded_matrix':
        metadata['reason'] = 'unsupported_banded_matrix'
        return None, metadata
    try:
        wave = spectrum.wave_rest
        edges = pixel_edges(wave)
        raw = resolution_to_dict(resolution)
        if raw['wavelength'] is None and np.size(raw['values']) != 1:
            if np.size(raw['values']) != wave.size:
                raise ValueError('resolution_array_grid_mismatch')
            raw['wavelength'] = spectrum.wave_obs.tolist()
            resolution = resolution_from_dict(raw)
        if resolution.wavelength is not None:
            rw = np.asarray(resolution.wavelength, float)
            if not np.all(np.isfinite(rw)) or np.any(np.diff(rw)<=0):
                raise ValueError('invalid_resolution_grid')
        sigma = resolution.sigma_lambda(spectrum.wave_obs)/(1+spectrum.z)
        in_window = (wave>=fit_window[0]) & (wave<=fit_window[1])
        if not np.any(in_window) or np.any(~np.isfinite(sigma[in_window])):
            raise ValueError('incomplete_resolution_coverage')
        if np.any(sigma[in_window]<=0):
            raise ValueError('invalid_resolution_values')
        # Integrate only where the supplied resolution is defined; never extrapolate it.
        lo, hi = edges[0], edges[-1]
        if resolution.coverage:
            lo = max(lo, resolution.coverage[0]/(1+spectrum.z))
            hi = min(hi, resolution.coverage[1]/(1+spectrum.z))
        padding = 8*float(np.max(sigma[in_window]))
        if lo>fit_window[0]-padding or hi<fit_window[1]+padding:
            raise ValueError('incomplete_resolution_coverage')
        valid_sigma = sigma[np.isfinite(sigma)&(sigma>0)]
        # >= five quadrature samples across the narrowest allowed intrinsic sigma.
        step = min(np.min(np.diff(wave))/4, lo*min_fwhm_kms/299792.458/2.35482/5,
                   float(valid_sigma.min())/4, .25)
        if (hi-lo)/step>500000:
            raise ValueError('unsupported_resolution_sampling')
        descriptor = dict(version=1, resolution=raw, z=float(spectrum.z),
                          source_bounds_rest=[float(lo),float(hi)], step_rest=float(step))
        operator = GaussianLineLSF(descriptor, wave)
        if not np.all(np.isfinite(operator.sigma)) or np.any(operator.sigma<=0):
            raise ValueError('invalid_resolution_values')
        metadata.update(status='forward_modeled', reason='', descriptor=descriptor,
                        profile_parameters='intrinsic', primary_measurements='observed',
                        pixel_integration='gaussian_cdf', source_quadrature='midpoint',
                        approximation='approximate_input_resolution' if resolution.status=='approximate' else None)
        return operator, metadata
    except (ValueError, TypeError, IndexError) as exc:
        metadata['reason'] = str(exc)
        return None, metadata
