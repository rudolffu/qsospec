# Examples

The public API is designed around explicit spectrum and configuration objects:

```python
import qsospec

spectrum = qsospec.Spectrum.from_arrays(
    wavelength,
    flux,
    err=uncertainty,
    z=redshift,
    wave_frame="observed",
)

result = qsospec.fit_global_lines(spectrum)
```

See [the run-bundle guide](../docs/run_bundles.md) for single-object and batch
archive examples.
