from __future__ import annotations

import numpy as np
import pandas as pd

from qsospec.catalog_bridge import build_euclid_dr1_qsospec_input


def test_bridge_filters_and_preserves_integer_ids() -> None:
    identified = pd.DataFrame(
        {
            "object_id": pd.Series([1, 2], dtype="int64"),
            "class_final": ["QSO_AUTO", "GALAXY"],
            "z_final": [1.2, 0.3],
            "redshift_source": ["hybrid", "vi"],
            "ra": [10.0, 11.0],
            "dec": [-1.0, -2.0],
        }
    )
    ledger = pd.DataFrame(
        {"object_id": [1, 2], "selection_tier": ["primary", "primary"], "field": ["edfn", "edfn"]}
    )
    spectra = pd.DataFrame(
        {
            "object_id": [1],
            "external_spectrum_id": ["desi-1"],
            "wavelength": [np.array([4000.0, 5000.0])],
            "flux": [np.array([1.0, 2.0])],
            "ivar": [np.array([1.0, 1.0])],
        }
    )
    output, summary = build_euclid_dr1_qsospec_input(
        identified,
        ledger,
        spectra,
        external_survey="DESI",
        galactic_extinction_corrected=False,
    )
    assert output["object_id"].tolist() == [1]
    assert str(output["object_id"].dtype) == "int64"
    assert summary["n_output"] == 1
    assert output.loc[0, "valid_mask"].tolist() == [True, True]
