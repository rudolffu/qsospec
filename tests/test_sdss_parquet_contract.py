"""SDSS follows the same scanner, fitting, store, and selective-resume APIs."""
import numpy as np
import pandas as pd

import qsospec
from qsospec.io.readers import spectrum_data_from_mapping


def test_sdss_provenance_through_batch_store_and_resume(tmp_path):
    wave = np.linspace(3500., 4500., 120)
    rows = [dict(spectrum_key=f"sdss:DR20:v6_2_1:allepoch:apo:60000:{63050395803782712 + i}",
                 catalogid=63050395803782712 + i, optical_survey="sdss", z_optical_fit=0.,
                 z_optical_source="DR20Q:Z_SYS", release="DR20", run2d="v6_2_1", coadd="allepoch",
                 observatory="apo", mjd=60000, source_url=f"https://example.test/{i}.fits",
                 source_checksum="a" * 64, source_backend="sdss", input_row_index=i + 50,
                 wavelength=wave, flux=2 * (wave / 4000.)**-1.1, ivar=np.ones(120)*400,
                 mask=np.zeros(120, dtype=np.uint8), sigma_lambda=np.ones(120)*1.5,
                 resolution_is_object_specific=True, resolution_status="complete_object_specific") for i in range(3)]
    path = tmp_path / "spectra.parquet"
    pd.DataFrame(rows).to_parquet(path, row_group_size=1)
    descriptors = list(qsospec.scan_parquet_spectrum_inputs(str(path), row_indices=[0, 2]))
    spectra = list(qsospec.scan_parquet_spectra(str(path), row_indices=[0, 2]))
    assert descriptors == [d for d, _ in spectra]
    assert spectra[0][1].object_id == "63050395803782712"
    assert spectra[0][1].targetid is None
    assert spectra[0][1].metadata["survey"] == "sdss"
    assert descriptors[0].metadata["catalogid"] == 63050395803782712
    kwargs = dict(n_workers=1, show_progress=False,
                  galactic_extinction_config=qsospec.GalacticExtinctionConfig(ebv_override=0.),
                  global_config=qsospec.GlobalContinuumConfig(uv_iron=None, optical_iron=None,
                      balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False), clip_passes=0),
                  complexes=[])
    run = tmp_path / "run"
    qsospec.fit_batch(str(path), str(run), row_indices=[0, 2], **kwargs)
    result = qsospec.fit_batch(str(path), str(run), **kwargs)
    assert result.n_submitted == 1 and result.n_skipped == 2
    store = qsospec.open_run(run)
    objects = store.read_table("objects").to_pandas()
    import json
    metadata = {entry["key"]: json.loads(entry["value"]) for entry in objects.iloc[0].metadata}
    assert metadata["optical_survey"] == "sdss"
    assert metadata["source_checksum"] == "a" * 64
    assert metadata["z_optical_source"] == "DR20Q:Z_SYS"


def test_invalid_sdss_resolution_cannot_be_reliable():
    row = dict(wavelength=[4000., 4001.], flux=[1., 2.], sigma_lambda=[float("nan"), 1.],
               resolution_is_object_specific=False, resolution_status="invalid_or_missing")
    spec = spectrum_data_from_mapping(row, source="test")
    assert spec.resolution.status != "valid"
    assert not spec.resolution.is_object_specific
    assert np.isnan(spec.resolution.sigma_lambda(np.array([4000., 4001.]))[0])
