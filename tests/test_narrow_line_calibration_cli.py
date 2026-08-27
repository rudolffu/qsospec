"""Design tests for the real-pattern He I injection command."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/calibrate_euclid_dr1_hei_pgamma.py"
)
SPEC = importlib.util.spec_from_file_location(
    "calibrate_euclid_dr1_hei_pgamma", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_injection_design_is_deterministic_balanced_and_spans_dimensions():
    first = MODULE.injection_design(120, ["-1", "2", "3"], seed=17)
    second = MODULE.injection_design(120, ["-1", "2", "3"], seed=17)
    assert first.equals(second)
    assert first["injection_design_version"].eq(MODULE.DESIGN_VERSION).all()
    assert first["truth_has_broad"].sum() == 60
    assert set(first["truth_effective_resolving_power"]) == {
        320.0,
        400.0,
        480.0,
        560.0,
        640.0,
    }
    assert 0.0 in set(first["truth_pgamma_to_hei_ratio"])
    assert 0.80 in set(first["truth_broad_fraction"])
