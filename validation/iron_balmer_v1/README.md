# Iron / Balmer validation

All fits and statistics use unsmoothed flux and valid errors. The three spectra were read from agnkiaa through localhost:6000 using the exact identifiers in the saved DESI QA selection. Remote installations and run bundles were not modified. Archived flux is rest-frame, with a physical scale of 1e-17.

The real comparison holds each archived host fixed. It tests the AGN renderer/refinement and conditional errors; it does not establish host-inclusive errors for these three objects. Real pPXF/E-MILES integration and host-inclusive matched-resampling orchestration were exercised separately.

## Signed residual bias

Values are mean (data − model) / pixel error; each cell is legacy → combined.

| Object | 3400–3600 Å | 3600–4000 Å | 4000–4400 Å |
|---|---:|---:|---:|
| 113506544049645456 | +0.628 → +0.205 | +0.207 → +0.024 | -0.236 → -0.139 |
| 1480622244667403470 | +0.958 → +0.240 | +0.203 → -0.036 | -0.563 → -0.141 |
| 2779599899604114953 | +1.280 → +0.353 | +0.279 → -0.088 | -0.606 → -0.200 |

The full JSON records all four modes, RMS, lag-one residual correlation, Mg II/Hβ assessment windows, continuum/host samples, parameters, diagnostics, and elapsed time. The bridge-only change does not improve every interval. Remaining structure is visible, especially in the third spectrum; this comparison does not establish absence of template systematics.

For these default maximum kernels, the guard-adjusted handoffs are **3156.30–3306.30 Å** and **4233.19–4383.19 Å**. The bridge shares the optical additional kernel in all three fits. Its reference normalization is fixed on the template grid at a 3000 km/s kernel.

## Outputs

[Machine-readable results](real_comparison.json) and [exact source requests](source_requests.json).

- [113506544049645456 PDF](113506544049645456.pdf)
- [1480622244667403470 PDF](1480622244667403470.pdf)
- [2779599899604114953 PDF](2779599899604114953.pdf)

## Reproduction

```bash
/Users/yuming/miniforge3/bin/python examples/extract_archived_comparison.py --requests validation/iron_balmer_v1/source_requests.json --output /tmp/three-spectra.jsonl
OPENBLAS_NUM_THREADS=1 /Users/yuming/miniforge3/bin/python examples/compare_archived_iron_balmer.py --input /tmp/three-spectra.jsonl --output-dir /tmp/new-comparison
/Users/yuming/miniforge3/bin/python examples/plot_iron_balmer_comparison.py --source /tmp/new-comparison --output /tmp/new-comparison-plots
```

Use new output paths: extraction and comparison do not overwrite an existing result.

## Uncertainty interpretation

The synthetic example deliberately uses a non-adopted Hγ ratio and an omitted feature. Two bootstrap trials per mode exercise propagation and matched trial persistence, not converged tails. Host tests include a four-trial fixture with changing host and AGN flux, plus the installed pPXF and E-MILES library. No virial, bolometric-correction, template-mismatch, redshift or cosmology systematic has been added.

Fast sequential line errors condition on the global continuum; joint Hγ errors include its continuum covariance. Full-workflow resampling is opt-in and preserves best-fit estimates. Unknown cross-block covariance is not filled with zeros. Selected-profile uncertainty is conditional on explicit component selection.

Older stored covariance can support recovery; absent covariance generally requires refitting for off-pivot continuum and full host fractions. Updating the iron/Balmer model always requires new point-estimate fits.

No input spectra, stellar libraries, survey tables, or large production catalogues are included here. Only the comparison plots, compact scalar diagnostics, and exact source references are retained.

## Verification record

The final regression run passed **383 tests** (one capfit runtime warning). Eight targeted host checks passed, including actual pPXF/E-MILES hybrid-basis closure and the host-refit fixture. See [validation checks](validation_checks.json) and [the two-trial synthetic bootstrap results](synthetic_bootstrap.json). Every requested bootstrap trial completed in all four modes.

Residual panels display ±5 sigma; numerical statistics include all valid assessment pixels. The PDFs were rendered and visually checked.
