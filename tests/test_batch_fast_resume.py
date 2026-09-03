from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pytest
from astropy.io import fits

import qsospec
from qsospec.io import readers
from qsospec.workflows import batch as batch_module


def _write_spectra(path: Path, count: int = 10, *, prefix: str = "key") -> None:
    wave = np.linspace(3500.0, 4500.0, 120)
    rows = []
    for index in range(count):
        rows.append({
            "qsospec_object_key": f"{prefix}-{index}",
            "object_id": f"object-{index}",
            "redshift": 0.0,
            "input_row_index": index,
            "qsospec_shard_id": index % 2,
            "wavelength": wave.tolist(),
            "flux": (2.0 * (wave / 4000.0) ** -1.1).tolist(),
            "ivar": np.full_like(wave, 400.0).tolist(),
            "mask": np.zeros_like(wave, dtype=np.int16).tolist(),
            "lsf_sigma_angstrom": np.full_like(wave, 1.5).tolist(),
        })
    pd.DataFrame(rows).to_parquet(path, index=False, row_group_size=3)


def _kwargs() -> dict:
    return {
        "n_workers": 1,
        "show_progress": False,
        "galactic_extinction_config": qsospec.GalacticExtinctionConfig(ebv_override=0.0),
        "global_config": qsospec.GlobalContinuumConfig(
            uv_iron=None,
            optical_iron=None,
            balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
            clip_passes=0,
        ),
        "complexes": [],
    }


def test_identity_scanner_matches_full_scanner_across_batches_and_files(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_spectra(first, 5, prefix="first")
    _write_spectra(second, 4, prefix="second")
    rows = {str(first): [1, 4], str(second): [0, 3]}
    lightweight = list(qsospec.scan_parquet_spectrum_inputs(
        [str(first), str(second)], row_indices=rows, batch_size=2
    ))
    full = [item for item, _ in qsospec.scan_parquet_spectra(
        [str(first), str(second)], row_indices=rows, batch_size=2
    )]
    assert lightweight == full
    assert [item.row_index for item in lightweight] == [1, 4, 0, 3]


def test_completed_resume_reads_no_vectors_or_workers(tmp_path, monkeypatch):
    source = tmp_path / "spectra.parquet"
    run = tmp_path / "run"
    _write_spectra(source, 3)
    qsospec.fit_batch(str(source), str(run), **_kwargs())

    def forbidden(*args, **kwargs):
        raise AssertionError("completed resume reached a vector/worker path")

    monkeypatch.setattr(batch_module, "scan_parquet_spectra", forbidden)
    monkeypatch.setattr(batch_module, "read_spectrum", forbidden)
    monkeypatch.setattr(batch_module, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(readers, "spectrum_data_from_mapping", forbidden)
    result = qsospec.fit_batch(str(source), str(run), n_workers=2, **{
        key: value for key, value in _kwargs().items() if key != "n_workers"
    })
    assert result.n_submitted == 0
    assert result.n_skipped == 3
    assert result.timings["resume_vector_rows_loaded"] == 0
    assert result.timings["resume_vector_rows_avoided"] == 3


def test_partial_resume_materializes_only_unfinished_rows(tmp_path, monkeypatch):
    source = tmp_path / "spectra.parquet"
    run = tmp_path / "run"
    _write_spectra(source, 10)
    qsospec.fit_batch(str(source), str(run), row_indices=list(range(8)), **_kwargs())
    original = readers.spectrum_data_from_mapping
    materialized = []

    def counted(row, **kwargs):
        materialized.append(kwargs["row_index"])
        return original(row, **kwargs)

    monkeypatch.setattr(readers, "spectrum_data_from_mapping", counted)
    result = qsospec.fit_batch(str(source), str(run), **_kwargs())
    assert materialized == [8, 9]
    assert result.n_submitted == 2
    assert result.n_skipped == 8
    assert result.timings["resume_vector_rows_loaded"] == 2


def test_retry_failure_plan_preserves_terminal_semantics(tmp_path):
    source = tmp_path / "spectra.parquet"
    run = tmp_path / "run"
    _write_spectra(source, 10)
    qsospec.fit_batch(str(source), str(run), row_indices=list(range(7)), **_kwargs())
    missing = [
        qsospec.SpectrumInput(
            source=str(tmp_path / f"missing-{index}.fits"),
            explicit_object_key=f"key-{index}",
            object_id=f"object-{index}",
            redshift=0.0,
        )
        for index in (7, 8)
    ]
    qsospec.fit_batch(missing, str(run), retry_failures=True, **_kwargs())
    terminal = qsospec.plan_batch_resume(
        str(source), str(run), retry_failures=False
    )
    retry = qsospec.plan_batch_resume(
        str(source), str(run), retry_failures=True
    )
    assert (terminal.failed_terminal_count, terminal.unfinished_count) == (2, 1)
    assert (retry.retry_failed_count, retry.unfinished_count) == (2, 3)
    assert retry.unfinished_row_indices[str(source)] == (7, 8, 9)


def test_lightweight_filter_rejected_and_auto_falls_back(tmp_path):
    source = tmp_path / "spectra.parquet"
    run = tmp_path / "run"
    _write_spectra(source, 2)
    with pytest.raises(ValueError, match="filter_expression"):
        qsospec.plan_batch_resume(
            str(source), str(run), filter_expression=(pads.field("redshift") == 0)
        )


def test_scalar_scanner_rejects_duplicate_explicit_keys(tmp_path):
    source = tmp_path / "spectra.parquet"
    _write_spectra(source, 2)
    frame = pd.read_parquet(source)
    frame.loc[1, "qsospec_object_key"] = frame.loc[0, "qsospec_object_key"]
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="Duplicate explicit object key"):
        list(qsospec.scan_parquet_spectrum_inputs(str(source)))


def test_legacy_resume_mode_remains_vector_first(tmp_path, monkeypatch):
    source = tmp_path / "spectra.parquet"
    run = tmp_path / "run"
    _write_spectra(source, 2)
    qsospec.fit_batch(str(source), str(run), **_kwargs())
    original = readers.spectrum_data_from_mapping
    materialized = []

    def counted(row, **kwargs):
        materialized.append(kwargs["row_index"])
        return original(row, **kwargs)

    monkeypatch.setattr(readers, "spectrum_data_from_mapping", counted)
    result = qsospec.fit_batch(
        str(source), str(run), resume_planning="legacy", **_kwargs()
    )
    assert materialized == [0, 1]
    assert result.n_skipped == 2
    assert result.timings["resume_vector_rows_loaded"] == 2
    assert result.timings["resume_vector_rows_avoided"] == 0


def test_completed_fits_descriptor_is_checked_before_read(tmp_path, monkeypatch):
    source = tmp_path / "object.fits"
    run = tmp_path / "run"
    wave = np.linspace(3500.0, 4500.0, 120)
    header = fits.Header()
    header["CRVAL1"] = wave[0]
    header["CDELT1"] = wave[1] - wave[0]
    fits.PrimaryHDU(2.0 * (wave / 4000.0) ** -1.1, header=header).writeto(source)
    qsospec.fit_batch(str(source), str(run), **_kwargs())

    def forbidden(*args, **kwargs):
        raise AssertionError("completed FITS input was read")

    monkeypatch.setattr(batch_module, "read_spectrum", forbidden)
    result = qsospec.fit_batch(str(source), str(run), **_kwargs())
    assert result.n_submitted == 0
    assert result.n_skipped == 1
    assert result.timings["resume_vector_rows_loaded"] == 0
