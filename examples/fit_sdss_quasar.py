"""Fit the real SDSS DR17 quickstart spectrum using configured Planck GNILC.

Run from an installed package or a checkout with PYTHONPATH=src. Downloads,
QA and provenance default to the ignored validation directory. No dust maps
are fetched by this script; configure them as in the documentation first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

from astropy.config.paths import set_temp_cache
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
import numpy as np

import qsospec


SOURCE_URL = (
    "https://sas.sdss.org/sas/dr17/sdss/spectro/redux/26/"
    "spectra/lite/1198/spec-1198-52669-0040.fits"
)
REFERENCE_SHA256 = "8fa02a4f9b59aad58e13d31db9f4edd5b77f0496fcbe4812baedfa719d75b08c"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local SDSS FITS file; omit to download the example")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/sdss_quickstart"))
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_path = args.input.expanduser().resolve() if args.input is not None else None
    if input_path is None:
        cache_dir = output / "download_cache"
        cache_dir.mkdir(exist_ok=True)
        with set_temp_cache(cache_dir):
            cached = download_file(SOURCE_URL, cache=True, timeout=60)
            input_path = output / "spec-1198-52669-0040.fits"
            shutil.copyfile(cached, input_path)

    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    data = qsospec.read_spectrum(str(input_path))
    print(f"Input: {input_path}\nz={data.redshift}, RA={data.ra}, Dec={data.dec}")
    result = qsospec.fit_global_lines_workflow(str(input_path), run_host_decomp=False)
    figure = result.plot_qa()
    qa_path = output / "sdss_quasar_qa.png"
    figure.savefig(qa_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_sha256:
        raise RuntimeError("The input FITS file changed during fitting; provenance cannot be verified.")

    with fits.open(input_path, memmap=False) as hdul:
        names = {name.lower(): name for name in hdul[2].columns.names}
        pipeline = {
            name: str(hdul[2].data[names[name]][0]).strip()
            for name in ("class", "zwarning") if name in names
        }
    good = np.isfinite(data.wave_obs) & np.isfinite(data.flux)
    good &= np.isfinite(data.uncertainty()) & (data.uncertainty() > 0)
    if data.mask is not None:
        good &= data.mask if data.mask.dtype.kind == "b" else data.mask == 0
    report = {
        "source_url": SOURCE_URL if args.input is None or input_sha256 == REFERENCE_SHA256 else None,
        "input_file": str(input_path),
        "sha256": input_sha256,
        "python_version": platform.python_version(),
        "qsospec_version": qsospec.__version__,
        "pipeline": pipeline,
        "redshift": data.redshift,
        "fitted_redshift": result.spectrum.z,
        "ra_deg": data.ra,
        "dec_deg": data.dec,
        "flux_unit": data.metadata["flux_unit"],
        "flux_scale": data.metadata["flux_scale"],
        "pixels": len(data.wave_obs),
        "valid_pixels": int(good.sum()),
        "fitted_valid_pixels": int(result.spectrum.mask.sum()),
        "observed_wavelength_range_angstrom": [float(data.wave_obs.min()), float(data.wave_obs.max())],
        "continuum_success": result.continuum_success,
        "complex_statuses": result.complex_statuses,
        "host_decomp_enabled": result.host_decomp_enabled,
        "galactic_extinction": result.metadata["galactic_extinction"],
        "warning_codes": result.warning_codes(),
        "qa_file": str(qa_path),
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return result


if __name__ == "__main__":
    main()
