"""Fit the real SDSS DR17 quickstart spectrum using configured Planck GNILC.

Run from an installed package or a checkout with PYTHONPATH=src. Downloads,
QA and provenance default to the ignored validation directory. No dust maps
are fetched by this script; configure them as in the documentation first.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import platform
import shutil
import time

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
OBJECT_ID = "spec-1198-52669-0040"


def fit_example(path, *, run_directory=None, object_id=OBJECT_ID):
    """Fit once, optionally writing a reloadable bundle with the same defaults."""
    if run_directory is not None:
        return qsospec.fit_object_to_store(
            str(path), str(run_directory), object_id=object_id,
            run_host_decomp=False, write_qa=False,
        )
    return qsospec.fit_global_lines_workflow(
        str(path), object_id=object_id, run_host_decomp=False,
    )


# broad-hbeta-start
def read_broad_hbeta(result):
    """Read the summed broad Hβ width; None denotes an unavailable error."""
    hbeta = result.line_complexes.get("hbeta_oiii")
    if hbeta is None or not hbeta.success:
        return {"status": result.complex_statuses.get("hbeta_oiii", "unavailable")}

    value = hbeta.metrics.get("Hb_broad_fwhm_kms", np.nan)
    error = hbeta.metric_errors.get("Hb_broad_fwhm_kms", np.nan)
    return {
        "status": "available" if np.isfinite(value) else "unavailable",
        "fwhm_kms": float(value) if np.isfinite(value) else None,
        "fwhm_error_kms": float(error) if np.isfinite(error) else None,
    }
# broad-hbeta-end


# peak-preparation-start
def prepare_peak_spectrum(path):
    """Prepare an observed SDSS spectrum for the low-level numerical fitter."""
    data = qsospec.read_spectrum(str(path), reader="sdss")
    spectrum = qsospec.Spectrum.from_arrays(
        data.wave_obs, data.flux, err=data.uncertainty(), mask=data.mask,
        z=data.redshift, ra=data.ra, dec=data.dec,
        wave_frame="observed", flux_unit="cgs", flux_scale=1e-17,
        survey="sdss", source=str(path),
    )
    return qsospec.prepare_spectrum(spectrum)
# peak-preparation-end


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local SDSS FITS file; omit to download the example")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/sdss_quickstart"))
    parser.add_argument("--write-run", action="store_true", help="Fit into output-dir/run, then verify reload")
    parser.add_argument("--no-qa", action="store_true", help="Skip plotting; retain numerical provenance")
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
    object_id = OBJECT_ID if input_sha256 == REFERENCE_SHA256 else (data.object_id or input_path.stem)
    print(f"Input: {input_path}\nz={data.redshift}, RA={data.ra}, Dec={data.dec}")
    run_directory = output / "run" if args.write_run else None
    start = time.perf_counter()
    result = fit_example(input_path, run_directory=run_directory, object_id=object_id)
    fit_and_save_seconds = time.perf_counter() - start
    qa_path = None
    if not args.no_qa:
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
    hbeta = result.line_complexes.get("hbeta_oiii")
    numerical_measurements = read_broad_hbeta(result)
    if hbeta is not None:
        for key in ("Hb_broad_flux_input", "Hb_broad_flux_cgs", "Hb_broad_sigma_kms", "Hb_broad_ew_rest"):
            value = hbeta.metrics.get(key, np.nan)
            error = hbeta.metric_errors.get(key, np.nan)
            numerical_measurements[key] = float(value) if np.isfinite(value) else None
            numerical_measurements[key + "_error"] = float(error) if np.isfinite(error) else None
        numerical_measurements.update(
            component_ids=list(hbeta.metadata.get("line_peaks", {}).get("measurements", {}).get("hbeta_broad", {}).get("component_ids", [])),
            fit_success=hbeta.success,
            selected_model=hbeta.selected_model,
            line_lsf=hbeta.metadata.get("line_lsf", {}),
            warning_codes=hbeta.warning_codes(),
            parameter_bounds=[dict(warning.context) for warning in hbeta.warnings if warning.code == "parameter_at_bound"],
            covariance_available=hbeta.covariance is not None,
            covariance_warning_codes=[code for code in hbeta.warning_codes() if "covariance" in code],
            oiii_profile_adequate=hbeta.metadata.get("profile_adequate"),
            oiii_profile_quality_flags=hbeta.metadata.get("profile_quality_flags", []),
            uncertainty_method="local covariance propagation",
            uncertainty_conditioning=["selected line model", "fitted global continuum", "input redshift"],
            continuum_model_uncertainty_included=False,
        )
    reload_check = None
    if run_directory is not None:
        start = time.perf_counter()
        loaded = qsospec.load_model(str(run_directory), object_id)
        reload_seconds = time.perf_counter() - start
        assert read_broad_hbeta(loaded) == read_broad_hbeta(result)
        assert loaded.complex_statuses == result.complex_statuses
        assert loaded.spectrum.z == result.spectrum.z
        np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux, rtol=1e-10, atol=1e-12)
        reload_check = {"object_id": object_id, "measurement_and_error_equal": True,
                        "complex_statuses_equal": True, "redshift_equal": True,
                        "reload_seconds": reload_seconds}
    report = {
        "source_url": SOURCE_URL if args.input is None or input_sha256 == REFERENCE_SHA256 else None,
        "input_file": str(input_path),
        "object_id": object_id,
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
        "qa_file": str(qa_path) if qa_path is not None else None,
        "broad_hbeta": numerical_measurements,
        "run_directory": str(run_directory) if run_directory is not None else None,
        "reload_check": reload_check,
        "runtime": {"fit_and_save_seconds": fit_and_save_seconds, "machine": platform.machine(),
                    "system": platform.system(), "processor": platform.processor(),
                    "thread_environment": {name: os.environ.get(name) for name in
                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "configuration": {"global": asdict(qsospec.GlobalContinuumConfig()),
                          "hbeta": asdict(qsospec.HbetaComplexConfig()),
                          "mgii": asdict(qsospec.MgIIComplexConfig()),
                          "uncertainty": asdict(qsospec.UncertaintyConfig()),
                          "complexes": None,
                          "run_host_decomp": False,
                          "global_model_id": result.metadata["global_model_id"]},
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return result


if __name__ == "__main__":
    main()
