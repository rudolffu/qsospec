# Local Balmer hypotheses

`fit_balmer_local` returns a requested local model without assigning a broad-line detection. The continuum and line parameters are fitted jointly with the existing native bounded variable-projection solver. No host subtraction, global continuum, Balmer pseudocontinuum, linked Hγ fit, or MC uncertainty calculation is run.

```python
from qsospec import (
    BalmerLocalConfig, BandResolutionOperator, Spectrum,
    evaluate_balmer_local_model, fit_balmer_local,
)

# flux_rest and error_rest are F_lambda on the supplied rest-wavelength grid.
# They include the object's own redshift conversion and the caller's recorded
# foreground correction. This entry point does not repeat either correction.
spectrum = Spectrum.from_arrays(
    wave_rest, flux_rest, err=error_rest, mask=valid,
    z=own_redshift, wave_frame="rest", flux_unit="cgs", flux_scale=1e-17,
)

# The input wavelengths of an operator are always observed-frame native-band
# wavelengths. output_indices refer to rows in spectrum, not wavelength ranks.
operator = BandResolutionOperator(
    name="desi-r",
    input_wave_obs=native_band_wave_obs,
    output_indices=band_output_rows,
    matrix=native_response_csr,
    output_weights=recorded_coadd_weights,
    input_factor=foreground_transmission_on_native_input,
    output_factor=1 / foreground_transmission_on_output,
    provenance={
        "source": "exact object/coadd native DESI resolution matrix",
        "is_object_specific": True,
        "is_approximate": False,
        "input_fingerprint": recorded_input_hash,
    },
)
config = BalmerLocalConfig(
    line="hbeta", broad_count=0, nuisance_family="shared_outflow",
    require_native_response=True,
)
continuum, null = fit_balmer_local(
    spectrum, config, instrumental_response=operators,
)
intrinsic_null_on_native_band = evaluate_balmer_local_model(
    native_band_wave_obs / (1 + own_redshift), null, config,
)
```

Provide each participating native band in `operators`. The example operator is one such band. Its matrix shape is `(len(output_indices), len(input_wave_obs))`. Convert DESI diagonal storage to a dense or sparse matrix with its actual recorded offsets before calling this API. Signed native coefficients are accepted and preserved exactly. The fitter does not clip or normalize them, convert them to a Gaussian σ, interpolate the data, or replace missing resolution with a guessed curve.

For each band, the forward model is

```
output_weights * output_factor * matrix @ (input_factor * intrinsic_rest_Flambda)
```

Contributions from bands add on common output rows. The recorded nonnegative output weights must sum to one on every fitted row; missing or duplicated weight support raises an error. Validation allows an absolute roundoff tolerance of four float32 epsilons (`4.76837158203125e-7`, relative tolerance zero) because original inverse-variance coaddition and division use float32. The actual weights remain unchanged. `resolution_weight_validation` records the tolerance, fitted-row sum range, maximum deviation, and `normalization_applied=False`. Positive-weight band rows require finite positive matrix row sums. The matrix row sums themselves are recorded and left unchanged. `input_factor` and `output_factor` represent a caller-declared wavelength-dependent transformation such as `diag(1/T_output) R diag(T_input)` for an already foreground-corrected spectrum. A spatially constant `(1+z)` F_lambda factor commutes with R; record the rest-frame conversion in the spectrum provenance. No extinction law or band coaddition is inferred here.

For sparse operators, repeated likelihood and Jacobian calls use an exact suboperator: select the frozen fitted output rows and retain every stored native input column contributing to those rows. This includes input wavelengths outside the fitting window, negative coefficients, duplicate sparse entries and stored zeros. Sparse entry accumulation order, foreground factors and overlap weights remain unchanged. Bands with no fitted output rows are not evaluated during optimization. Final full-grid continuum and line components still use the original complete operators. `resolution_evaluation` records the strategy, original and evaluated dimensions, and selected row/column fingerprints. Dense or mixed dense/sparse operator inputs retain the original full-domain path to preserve their BLAS reduction rounding. No starts, parameter bounds, masks, tolerances, solver fallback, or covariance calculation change.

The reproducible synthetic benchmark measures repeated bounded-linear residual and reduced-Jacobian calls, including signed three-band responses and foreground conjugation. It checks bit-identical residuals, Jacobians and linear coefficients against the original full-domain path before reporting timings; it launches no fits and reads no production data:

```sh
PYTHONPATH=src python benchmarks/benchmark_balmer_response_roi.py \
  --repeats 25 --rounds 3 --output /tmp/balmer_response_roi_benchmark.json
```

`require_native_response=True` rejects missing or unverified operators. Verification requires provenance `source`, `is_object_specific=True`, and `is_approximate=False`. With the default `False`, fitting without a response is an explicit observed-profile diagnostic: `resolution_status="missing_not_intrinsic"`, an `instrumental_response_missing` warning, and an observed-profile width definition. A supplied but unverified matrix is labeled `unverified_operator`. It cannot satisfy the strict contract. No σ-only object is accepted by this API.

## Hypotheses and nuisance models

`BalmerLocalConfig` supports `line="hbeta"` or `"halpha"`, and `broad_count=0` or `1`. Both use a fixed line-centred log-velocity window of ±20000 km/s:

- Hβ: 4548.861–5198.149 Å in vacuum.
- Hα: the same velocity window about vacuum 6564.61 Å.

The intrinsic continuum is affine in rest wavelength, with its pivot at the window midpoint. Hβ masks 4660–4715 Å by default to exclude unmodeled HeII. Additional `mask_windows` are declared explicitly and must be identical under compared hypotheses.

`nuisance_family="core"` ties the narrow Balmer and forbidden-line core velocity and width. Hβ includes [OIII]4960/5008 with ratio2.98. Hα includes [NII]6550/6585 with ratio2.96 and either [SII] line when its core has sufficient valid support. The active [SII] selection depends only on the data and shared configuration, not on broad count.

`nuisance_family="shared_outflow"` adds a second family of forbidden-line and narrow-Balmer profiles with shared velocity and width but separate amplitudes. The doublet ratios remain fixed. Centered outflow wings are allowed: no velocity-separation threshold or automatic wing acceptance rule is imposed. Return both core and shared-outflow candidates under both broad hypotheses if profiling nuisance-model uncertainty. This is a new explicit model family and does not alter the older Hβ [OIII]-wing selection defaults.

The broad Gaussian has nonnegative full integrated flux, independent velocity within ±2000 km/s, and FWHM900–20000 km/s. Deterministic width/velocity starts are recorded; the successful start with the smallest χ² is returned, with stable ordering breaking exact ties. The caller can provide different **predeclared** start grids in the configuration. Null fits contain no broad flux, velocity, or width coordinates. They report zero broad flux and an undefined broad width/error; an unobserved or failed fit instead has no broad measurement.

Every comparison must share the data, supplied errors, response, masks, narrow/outflow model family, and continuum family. These input hashes, exact parameter count, χ², BIC, covariance coordinate names and start diagnostics are stored in `fit.metadata`. The fitted continuum coefficients differ under the hypotheses because they are refitted jointly. Do not compare BIC across different masks, data or response operators.

## Outputs and uncertainty

The return value is `(continuum, fit)`:

- `continuum.model` is the forward-convolved affine continuum on the Spectrum grid.
- `fit.model` and `fit.component_models` contain forward-convolved **lines only**.
- `fit.param_values` and `fit.covariance` retain all free line and affine-continuum coordinates; `continuum.covariance` is their two-coordinate marginal block.
- Output rows with no response coverage outside the fitting mask have `NaN` model values. Such missing support is not filled with zero. Every fitted row must have valid weight support.
- `evaluate_balmer_local_model(wave_rest, fit, config)` evaluates the fitted intrinsic line+continuum model on arbitrary rest-frame native wavelengths before R. `include_continuum=False` returns lines only; `return_components=True` returns a component dictionary. It verifies the exact stored configuration and parameter contract.

Metric prefixes are `Hb` and `Ha`. `*_broad_flux_input` is the full Gaussian area in input F_lambda times rest Å; `*_broad_flux_cgs` applies the Spectrum's physical flux scale. `*_broad_fwhm_kms` is the single intrinsic Gaussian width conditional on the supplied operator, or an explicitly unconvolved observed-profile width when no operator is used. `*_broad_ew_rest` uses the intrinsic local continuum at line centre. `*_narrow_flux_input` and `*_outflow_flux_input` preserve narrow and shared-outflow Balmer amplitudes separately. Joint propagated errors are in `fit.metric_errors`. At the zero-broad null, broad errors are undefined because no broad coordinate is estimated.

The covariance uses the existing native Jacobian implementation and its residual-variance scaling. Rank-deficiency and optimizer-bound warnings are retained. These are formal errors conditional on the continuum and nuisance family and supplied response; they do not include template, continuum-family, absorption, FeII, model-selection or instrumental-response uncertainty. ΔBIC and flux/error are diagnostics, not detection probabilities. Nonnegative flux has a boundary null with unidentified broad width/velocity, so a naive χ²/F-test significance conversion is inappropriate. Detection calibration and visual checks belong to the evidence driver.

`fit_halpha_local` is a convenience wrapper for the new Hα configuration. Existing `fit_hbeta_local(spectrum, HbetaComplexConfig(...))` retains its old 4640–5100 Å defaults and old [OIII] wing policy. Passing `BalmerLocalConfig(line="hbeta", ...)` dispatches it to the new model. Explicit instrumental response requires this new configuration. Legacy `HbetaComplexConfig(broad_fwhm_bands_kms=())` now also permits a narrow-only model; its three-component default and arithmetic remain unchanged.
