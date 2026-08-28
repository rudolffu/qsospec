import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from qsospec.euclid_rgs import normalize_rgs_grid, validate_sample_manifest
from qsospec.io.readers import SpectrumInput, scan_parquet_spectra


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _lines(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _input(path: Path, keys=("euclid_dr1_rgs:-1", "euclid_dr1_rgs:2")) -> None:
    wavelength = np.linspace(11900.0, 19002.0, 531, dtype=np.float32)[11:511]
    frame = pd.DataFrame(
        {
            "object_id": [-1, 2],
            "qsospec_object_key": list(keys),
            "wavelength": [wavelength, wavelength],
            "flux": [np.ones(500), np.ones(500)],
            "ivar": [np.ones(500), np.ones(500)],
            "mask": [np.zeros(500, dtype=np.uint8), np.zeros(500, dtype=np.uint8)],
            "redshift": [1.0, 2.0],
        }
    )
    frame.to_parquet(path, index=False)


def test_explicit_object_key_is_path_independent(tmp_path: Path):
    first = SpectrumInput("/tmp/a.parquet", 0, explicit_object_key="portable:key")
    second = SpectrumInput("/different/b.parquet", 99, explicit_object_key="portable:key")
    assert first.object_key == second.object_key == "portable:key"
    legacy = SpectrumInput("relative.parquet", 3)
    assert legacy.object_key.endswith("relative.parquet#3")


def test_scan_reads_keys_and_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "spectra.parquet"
    _input(path)
    descriptors = [descriptor for descriptor, _ in scan_parquet_spectra(str(path), batch_size=1)]
    assert [item.object_key for item in descriptors] == ["euclid_dr1_rgs:-1", "euclid_dr1_rgs:2"]
    _input(path, keys=("duplicate", "duplicate"))
    with pytest.raises(ValueError, match="Duplicate explicit object key"):
        list(scan_parquet_spectra(str(path)))


def test_empty_explicit_key_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _ = SpectrumInput("a.parquet", 0, explicit_object_key=" ").object_key


def test_generic_grid_accepts_only_established_contract():
    source = np.linspace(11900.0, 19002.0, 531, dtype=np.float32)
    assert normalize_rgs_grid(source).shape == (500,)
    assert normalize_rgs_grid(source[11:511]).shape == (500,)
    with pytest.raises(ValueError):
        normalize_rgs_grid(source[:-1])


def test_sample_manifest_exact_hash_validation(tmp_path: Path):
    path = tmp_path / "spectra.parquet"
    _input(path)
    table = pq.read_table(path, columns=["object_id", "qsospec_object_key", "wavelength"])
    grid = np.asarray(table["wavelength"][0].as_py(), dtype="<f8")
    manifest = {
        "sample_id": "synthetic",
        "sample_scope": "test",
        "input_rows": 2,
        "input_sha256": _sha256(path),
        "schema_sha256": hashlib.sha256(pq.ParquetFile(path).schema_arrow.serialize().to_pybytes()).hexdigest(),
        "ordered_object_id_sha256": _lines([-1, 2]),
        "ordered_object_key_sha256": _lines(["euclid_dr1_rgs:-1", "euclid_dr1_rgs:2"]),
        "wavelength_grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
        "flux_scale": 1e-17,
        "flux_density_unit": "erg s^-1 cm^-2 Angstrom^-1",
        "wavelength_frame": "observed",
        "flux_frame": "observed",
        "galactic_extinction_corrected": False,
        "shard_id": 0,
        "n_shards": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_sample_manifest(path, manifest_path)["input_rows"] == 2
    manifest["input_rows"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row count"):
        validate_sample_manifest(path, manifest_path)
