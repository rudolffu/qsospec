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

For host-galaxy decomposition:

```bash
python -m pip install "qsospec[host]"
```

## Minimal example

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

result = qsospec.fit_object_to_store(
    spectrum,
    "runs/my-quasar",
    object_id="my-quasar",
    global_config=qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(mode="auto"),
    ),
    write_qa=True,
)

result.show_qa()
```

Uncorrected spectra are dereddened by default with Planck GNILC and the
Fitzpatrick (1999) law. This requires locally configured
[`dustmaps`](https://qsospec.readthedocs.io/en/latest/getting_started/dustmaps.html)
data; already-corrected spectra can be declared with
`galactic_extinction_corrected=True`.

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
