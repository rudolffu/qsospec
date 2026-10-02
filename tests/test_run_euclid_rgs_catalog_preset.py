"""Complex-preset selection for the portable full-RGS runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import qsospec


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/run_euclid_rgs_catalog.py"
    spec = importlib.util.spec_from_file_location("run_euclid_rgs_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_preset_keeps_historical_auto_enabled_set():
    module = _load_script()
    assert module._resolve_complexes("standard") is None


def test_extended_quasar_preset_returns_versioned_recipe_set():
    module = _load_script()
    resolved = module._resolve_complexes("extended_quasar_v1")
    assert {recipe.id for recipe in resolved} == qsospec.recipes.EXTENDED_QUASAR_RECIPE_IDS
    assert len(resolved) == len(qsospec.recipes.EXTENDED_QUASAR_RECIPE_IDS)


def test_unknown_preset_is_rejected():
    module = _load_script()
    with pytest.raises(ValueError, match="Unknown complex preset"):
        module._resolve_complexes("q1_complete")
