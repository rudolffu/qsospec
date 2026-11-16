# qsospec

[![Documentation Status](https://readthedocs.org/projects/qsospec/badge/?version=latest)](https://qsospec.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/qsospec)](https://pypi.org/project/qsospec/)
[![Python](https://img.shields.io/pypi/pyversions/qsospec)](https://pypi.org/project/qsospec/)
[![License](https://img.shields.io/badge/license-GPLv3-green)](LICENSE)

`qsospec` helps astronomers measure quasar emission lines in UV, optical,
and near-infrared spectra. Fit the AGN continuum and covered line complexes,
inspect the model and residuals, and save measurements and models for later
analysis. Optional pPXF fitting separates the stellar host.

## Fit an SDSS quasar

After [installation](https://qsospec.readthedocs.io/en/latest/getting_started/installation.html)
and the one-time [Planck dust-map setup](https://qsospec.readthedocs.io/en/latest/getting_started/dustmaps.html),
this example downloads a real SDSS DR17 spectrum and saves its fit:

```python
from astropy.utils.data import download_file
import qsospec

url = (
    "https://sas.sdss.org/sas/dr17/sdss/spectro/redux/26/"
    "spectra/lite/1198/spec-1198-52669-0040.fits"
)
path = download_file(url, cache=True)
result = qsospec.fit_object_to_store(
    path,
    "runs/sdss-quasar",
    object_id="spec-1198-52669-0040",
    run_host_decomp=False,
    write_qa=False,
)
print(result.continuum_success)
print(result.complex_statuses)
result.show_qa()
```

The workflow reads the redshift and coordinates, applies Galactic correction,
and fits the continuum and covered lines. The
[quickstart](https://qsospec.readthedocs.io/en/latest/getting_started/quickstart.html)
explains the data and QA figure; the
[measurement tutorial](https://qsospec.readthedocs.io/en/latest/user_guide/results.html)
shows how to report broad Hβ and reload the analysis without refitting.

## Features

- Fit a shared power-law, Fe II and Balmer continuum, or fit separate local windows.
- Measure covered UV, optical and NIR line complexes with coverage diagnostics
  and statistical uncertainties.
- Subtract a stellar host with optional pPXF decomposition for `z < 1.2`.
- Inspect component models, excluded pixels and residuals in Matplotlib QA figures.
- Process samples with resumable parallel fitting and portable Parquet run bundles.

## Installation

Python 3.10 or later is required for the 0.2.0 source tree documented here.
Install the published package with:

```bash
python -m pip install qsospec
```

The published package can lag `main`. To use the code documented in this
checkout, install from source:

```bash
git clone https://github.com/rudolffu/qsospec.git
cd qsospec
python -m pip install -e .
```

Add the `host` extra for pPXF, or `docs` to build this documentation. See the
[installation guide](https://qsospec.readthedocs.io/en/latest/getting_started/installation.html)
for template requirements and version checks.

## Usage

| Your task | Start here |
| --- | --- |
| Fit your own wavelength, flux and error arrays | [Array inputs](https://qsospec.readthedocs.io/en/latest/how_to/fit_arrays.html) |
| Choose global, local or saved fitting | [Choose a workflow](https://qsospec.readthedocs.io/en/latest/getting_started/choose_workflow.html) |
| Check a fit and interpret missing measurements | [QA guide](https://qsospec.readthedocs.io/en/latest/user_guide/qa_plots.html) |
| Fit a sample and resume interrupted work | [Batch fitting](https://qsospec.readthedocs.io/en/latest/user_guide/batch_fitting.html) |
| Find a flux, width, EW or host-fraction definition | [Measurement dictionary](https://qsospec.readthedocs.io/en/latest/reference/measurement_dictionary.html) |
| Update an older analysis | [0.2 migration guide](https://qsospec.readthedocs.io/en/latest/getting_started/migration_0_2.html) |

The default science model is `global_v2`; polynomial continuum correction is
off. The [science model](https://qsospec.readthedocs.io/en/latest/science/continuum_model.html)
describes the assumptions and alternatives.

## Contributing

Source code and changes are managed on [GitHub](https://github.com/rudolffu/qsospec).
See [CONTRIBUTING.md](CONTRIBUTING.md) for development and documentation checks.

## Support

Use the [issue tracker](https://github.com/rudolffu/qsospec/issues) for questions,
bug reports and feature requests. Include your package version, input format,
configuration, and the error or warning code. The
[troubleshooting guide](https://qsospec.readthedocs.io/en/latest/getting_started/troubleshooting.html)
covers setup, missing lines and saved runs.

## License

qsospec is licensed under [GNU GPL version 3 only](LICENSE).
