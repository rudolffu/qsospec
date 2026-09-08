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

## SDSS Parquet inputs

SDSS acquisition belongs in `sparcli --backend sdss`. QSOSpec consumes the
normalized Parquet through the same `scan_parquet_spectra`, `fit_batch`,
`plan_batch_resume`, run-store, and HostSED APIs used for DESI.

Provide an explicit fitting redshift (`redshift` or `z_optical_fit`) and a stable
`spectrum_key`. Catalog redshift selection remains an orchestration decision;
QSOSpec does not silently choose `Z_SYS` from an arbitrary catalog. Common
optical metadata includes `optical_survey`, `optical_object_id`, catalogid,
release, run2d, coadd, observatory, MJD, source URL/checksum, and optical-redshift
provenance. Exact integer IDs survive scalar scanning, selective-row resume,
and saved object metadata. Existing DESI/SPARCL fields remain supported.

The input adapter supplies observed-vacuum wavelength and per-pixel Gaussian
`sigma_lambda` or `lsf_sigma_angstrom`. For DR20/v6_2_1, sparcli derives this from
WRESL FWHM; WDISP is retained independently by the acquisition layer. Resolution
marked non-object-specific/invalid remains approximate for host reliability
and never silently becomes a constant-resolution estimate. Optical spectra
retain their declared flux scale (SDSS: `1e-17`). Extinction correction follows
the normal fitting configuration, not the download step.

Pass explicit spectral Parquet files to the reader, excluding scalar download
manifests. MLSpecZ's versioned SDSS workflow supplies fitted-redshift metadata,
32 immutable inputs, and the established DESI host configuration.

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

Regional VW01/Verner09/Park22 iron, soft Hγ refinement, and native covariance /
matched-bootstrap products are described in
[the iron, Balmer and uncertainty guide](docs/iron-balmer-uncertainties.md).
A compact comparison is available in `examples/iron_balmer_uncertainty.py`.
