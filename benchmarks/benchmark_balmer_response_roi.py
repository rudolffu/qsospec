"""Repeated synthetic likelihood/Jacobian timings; no fits or production data.

Both paths use identical signed native response operators, factors, model
families and frozen rows. Their bounded linear solutions, residuals and
reduced Jacobians are checked bit-for-bit before any timing is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from scipy import sparse

import qsospec
from qsospec.fitting import balmer_local as native
from qsospec.solvers.variable_projection import _VariableProjectionProblem


class _OriginalFullDomainContext(native._ResponseContext):
    def _apply(self, evaluate, *, fit_only=False):
        result = super()._apply(evaluate)
        return result[self.fit_indices] if fit_only else result


def _input(line, count, family):
    wave_obs = np.linspace(3600., 9800., 7751)
    z = .07358
    bands = ((0, 3101), (2500, 5601), (4950, len(wave_obs)))
    support = np.zeros(len(wave_obs))
    for lo, hi in bands:
        support[lo:hi] += 1
    operators = []
    for name, (lo, hi) in zip(("b", "r", "z"), bands):
        n = hi-lo
        offsets = np.arange(-5, 6)
        weights = np.exp(-.5*(offsets/1.2)**2)
        weights /= weights.sum()
        weights[[0, -1]] = -.002
        response = sparse.diags(weights, offsets, shape=(n, n), format="csr")
        transmission = np.exp(-.2*(wave_obs[lo:hi]/5000.)**-1.2)
        operators.append(qsospec.BandResolutionOperator(name, wave_obs[lo:hi],
            np.arange(lo, hi), response, output_weights=1/support[lo:hi],
            input_factor=transmission, output_factor=1/transmission))
    spectrum = qsospec.Spectrum.from_arrays(wave_obs/(1+z), np.ones(len(wave_obs)),
        err=np.full(len(wave_obs), .05), z=z, wave_frame="rest", flux_unit="relative")
    config = qsospec.BalmerLocalConfig(line=line, broad_count=count, nuisance_family=family)
    mask, sii, _ = native._coverage(spectrum, config)
    recipe = native._recipe(config, sii)
    roi = native._ResponseContext(recipe, 100., spectrum, mask, operators)
    full = _OriginalFullDomainContext(recipe, 100., spectrum, mask, operators)
    _, bounds, theta, _ = roi.separable_initial_and_bounds()
    truth = roi.initial.copy()
    truth[roi.index["continuum.constant"]] = 2.
    truth[roi.index["continuum.slope"]] = .0001
    truth[roi.index["narrow.fwhm_kms"]] = 320.
    prefix = "Hb" if line == "hbeta" else "Ha"
    if count:
        truth[roi.index[prefix+"_broad1.fwhm_kms"]] = 2400.
    if family == "shared_outflow":
        truth[roi.index["outflow.fwhm_kms"]] = 1400.
    synthetic_flux = roi.model(truth, spectrum.wave_rest)
    synthetic_flux += np.random.default_rng(612).normal(0., spectrum.err)
    def problem(context):
        return _VariableProjectionProblem(synthetic_flux[mask], spectrum.err[mask], bounds,
            lambda value, derivatives: context.separable_design(value, spectrum.wave_rest[mask], derivatives))
    return roi, full, problem(roi), problem(full), theta


def _seconds_per_call(function, points, rounds):
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        for point in points:
            function(point)
        samples.append((time.perf_counter()-started)/len(points))
    return dict(median_seconds=float(np.median(samples)), min_seconds=float(min(samples)),
                max_seconds=float(max(samples)), rounds=rounds, calls_per_round=len(points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.rounds < 1:
        parser.error("repeats must be >=2 and rounds >=1")
    records = []
    for line in ("hbeta", "halpha"):
        for count in (0, 1):
            for family in ("core", "shared_outflow"):
                roi, full, roi_problem, full_problem, theta = _input(line, count, family)
                points = [theta+np.sin(index+np.arange(len(theta)))*.5
                          for index in range(args.repeats)]
                for point in points[:3]:
                    np.testing.assert_array_equal(roi_problem.residual(point), full_problem.residual(point))
                    np.testing.assert_array_equal(roi_problem.jacobian(point), full_problem.jacobian(point))
                    np.testing.assert_array_equal(roi_problem._state.linear, full_problem._state.linear)
                def objective(problem):
                    def call(point):
                        problem.residual(point)
                        return problem.jacobian(point)
                    return call
                full_timing = _seconds_per_call(objective(full_problem), points, args.rounds)
                roi_timing = _seconds_per_call(objective(roi_problem), points, args.rounds)
                record = dict(line=line, broad_count=count, nuisance_family=family,
                    fit_pixels=len(roi.fit_indices), free_parameters=len(roi.names),
                    positive_linear_coefficients=int(np.count_nonzero(roi_problem._state.linear[:-2] > 0)),
                    bit_identical_checked=True, full_domain=full_timing, sparse_roi=roi_timing,
                    median_speedup=full_timing["median_seconds"]/roi_timing["median_seconds"],
                    response_evaluation=roi.evaluation_card)
                records.append(record)
                print(f"{line} broad{count} {family}: {record['median_speedup']:.2f}x", flush=True)
    result = dict(schema_version=1, benchmark="synthetic_repeated_bounded_likelihood_and_reduced_jacobian",
        fits_launched=False, production_data_used=False, python=sys.version, platform=platform.platform(),
        numerical_thread_environment={name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        source_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(), records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
