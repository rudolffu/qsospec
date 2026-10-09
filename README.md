# qsospec

[![Documentation Status](https://readthedocs.org/projects/qsospec/badge/?version=latest)](https://qsospec.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/qsospec)](https://pypi.org/project/qsospec/)
[![Python](https://img.shields.io/pypi/pyversions/qsospec)](https://pypi.org/project/qsospec/)
[![License](https://img.shields.io/badge/license-GPLv3-green)](https://github.com/rudolffu/qsospec/blob/main/LICENSE)

`qsospec` fits UV, optical, and near-infrared quasar spectra. It provides
coverage-aware emission-line recipes, continuum decomposition, optional pPXF
host subtraction, QA figures, and resumable Parquet run bundles.

Polynomial continuum correction is opt-in; it is disabled by default for all
surveys, including SDSS. See the
[configuration guide](https://qsospec.readthedocs.io/en/latest/reference/configuration.html)
for optional correction settings.

## Installation

```bash
python -m pip install qsospec
```

Python 3.10 or later is required. Check the installed version with
`python -c "import qsospec; print(qsospec.__version__)"`.

For host-galaxy decomposition:

```bash
python -m pip install "qsospec[host]"
```

## Fit a real SDSS spectrum

Configure the [Planck GNILC map](https://qsospec.readthedocs.io/en/latest/getting_started/dustmaps.html)
once, then run:

```python
from astropy.utils.data import download_file
import qsospec

url = (
    "https://sas.sdss.org/sas/dr17/sdss/spectro/redux/26/"
    "spectra/lite/1198/spec-1198-52669-0040.fits"
)
path = download_file(url, cache=True)
result = qsospec.fit_global_lines_workflow(path, run_host_decomp=False)
print(result.continuum_success)
print(result.complex_statuses)
result.show_qa()
```

The workflow uses the FITS redshift, applies Planck/F99 Galactic correction,
and fits the default continuum and covered lines. The download is cached;
no host stellar templates are needed for this example.

Follow the [quickstart](https://qsospec.readthedocs.io/en/latest/getting_started/quickstart.html)
and [measurement tutorial](https://qsospec.readthedocs.io/en/latest/user_guide/results.html)
to inspect broad Hβ and save/reload the same analysis. For your own arrays,
use the [array guide](https://qsospec.readthedocs.io/en/latest/how_to/fit_arrays.html);
for a sample, use [batch fitting](https://qsospec.readthedocs.io/en/latest/user_guide/batch_fitting.html).

qsospec can consume normalized SDSS/DESI spectral Parquet inputs; see the
[survey-input guide](https://qsospec.readthedocs.io/en/latest/how_to/sdss_parquet_inputs.html).

## Features

- Single or automatically selected broken power-law continua, Fe II, and a
  continuous Balmer pseudo-continuum.
- Lyα/N V, C IV, C III], Mg II, Balmer, optical, and NIR line complexes.
- Optional pPXF host decomposition for quasars at `z < 1.2`.
- Notebook-friendly and saved QA figures.
- Shared single-object and parallel batch run format.

See the [documentation](https://qsospec.readthedocs.io/en/latest/) for setup,
workflows, model definitions, examples, and API reference.

`qsospec` is distributed under the
[GPLv3 license](https://github.com/rudolffu/qsospec/blob/main/LICENSE).

Version 0.2.0 uses the `global_v2` continuum model by default. See the
[migration guide](https://qsospec.readthedocs.io/en/latest/getting_started/migration_0_2.html)
for model provenance, the legacy preset, and API changes.
