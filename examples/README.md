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
    flux_unit="cgs",
    ra=ra,
    dec=dec,
)

result = qsospec.fit_global_lines(qsospec.prepare_spectrum(spectrum))
```

The included
[`spec_J001554.18+560257.5_LJT.csv`](data/spec_J001554.18+560257.5_LJT.csv)
is used by the
[single-object tutorial](../docs/how_to/fit_j001554.rst).

See the [run-bundle reference](../docs/reference/run_bundles.rst) for
single-object and batch archives.
