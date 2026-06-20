# qsospec

`qsospec` is a Python package for fitting UV, optical, and
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

Lyα/N V is selected automatically when useful rest-frame coverage exists.
The default continuum then uses red-side anchors and the fit records whether
coverage is full, red-side-only, edge-truncated, or absent:

```python
result = qsospec.fit_global_lines(
    spectrum,
    lya_nv_config=qsospec.LyaNVComplexConfig(
        nv_mode="effective_blend",
    ),
)
lya = result.line_complexes.get("lya_nv")
if lya is not None:
    print(lya.metadata["lya_coverage_status"])
    print(lya.metadata["lya_fit_reliable"])
```

Pass an explicit complex list without `"lya_nv"` to disable it. Use
`nv_mode="equal_doublet"` for a shared-kinematics N V doublet, or pass
`global_config=qsospec.GlobalContinuumConfig.lya_safe()` explicitly when
customizing the continuum model.

Single spectra and large samples share the same Parquet-backed run format:

```python
run = qsospec.fit_object_to_store("spectrum.fits", "runs/object", redshift=1.2)
batch = qsospec.fit_batch(["spectra-000.parquet"], "runs/sample")
```

The project is licensed under GPLv3. The initial implementation was extracted
from `qsofitmore.neofit`; its source history is retained in this repository.
