# Line-peak and WS22 validation

The benchmark uses the same three spectra extracted from agnkiaa for
`validation/iron_balmer_v1/source_requests.json`. Each fit holds the archived
host fixed and uses Mg II, Hβ/[O III], and the blue optical complex. No remote
run is changed. The benchmark script is `examples/benchmark_line_peaks.py`.

| Object | Full fit, including peaks (s) | Peak recalculation (s) | WS22 diagnostic (s) |
|---|---:|---:|---:|
| 113506544049645456 | 19.93 | 0.142 | 0.00087 |
| 1480622244667403470 | 13.92 | 0.203 | 0.00056 |
| 2779599899604114953 | 22.60 | 0.170 | 0.00025 |

Peak recalculation includes covariance propagation. Times are one run on the
local machine, not a general performance guarantee. Spectral optimizer entry
points were guarded during post-processing. All three diagnostics were
available, and all adopted input redshifts were exactly preserved. The detailed
per-line corrections, exclusions, uncertainties, and timings are in
`benchmark.json`. These diagnostics have not been checked against independent
systemic-redshift ground truth.

Validation used `/Users/yuming/miniforge3/bin/python`, without a new environment.
The full regression suite passed 400 tests at the initial integration check.
Subsequent targeted checks cover peak uncertainty, generic profiles, legacy
recovery, measurement export, and run-store round trips. Tests use analytic
Gaussian centers, correlated parameter draws for asymmetric profiles,
competing/boundary/absent peaks, calibration signs and luminosity floor,
line eligibility, deterministic clipping, and existing matched bootstrap draws.
