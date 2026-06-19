# qsospec

`qsospec` is a standalone Python package for fitting UV, optical, and
near-infrared quasar spectra. It provides array-based local and global fitting,
recipe-driven emission complexes, bundled iron and Balmer templates, optional
pPXF host subtraction, and resumable Parquet batch runs.

## Installation

```bash
python -m pip install qsospec
```

Install optional pPXF host decomposition with:

```bash
python -m pip install "qsospec[host]"
```

For development:

```bash
python -m pip install -e ".[dev,host]"
pytest
```

## Quick start

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

Single spectra and large samples share the same Parquet-backed run format:

```python
run = qsospec.fit_object_to_store("spectrum.fits", "runs/object", redshift=1.2)
batch = qsospec.fit_batch(["spectra-000.parquet"], "runs/sample")
```

The project is licensed under GPLv3. The initial implementation was extracted
from `qsofitmore.neofit`; its source history is retained in this repository.

