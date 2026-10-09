# Examples

## Real SDSS quickstart

[`fit_sdss_quasar.py`](fit_sdss_quasar.py) downloads and fits the SDSS DR17
quasar `spec-1198-52669-0040.fits`, or accepts a local SDSS FITS path. Configure
the Planck GNILC dust map first as described in the
[dust-map instructions](../docs/getting_started/dustmaps.rst).

```bash
python examples/fit_sdss_quasar.py --output-dir validation/sdss_quickstart
python examples/fit_sdss_quasar.py --input /path/to/spec-1198-52669-0040.fits \
    --output-dir validation/sdss_quickstart_local
```

From a source checkout, prefix these commands with `PYTHONPATH=src` if the
package is not installed. The output contains the QA PNG and `provenance.json`
with the input SHA-256, coordinates, adopted redshift, preprocessing, statuses
and warning codes. Downloads are cached beneath the output directory; local
inputs are read without copying or modifying them. No run bundle is created.

See the [quickstart](../docs/getting_started/quickstart.rst) for the same
workflow as short Python snippets.

## Array inputs

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
