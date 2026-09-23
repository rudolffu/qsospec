"""Immutable emission-complex recipes and built-in registry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from . import lines
from .config import LyaNVComplexConfig

Bounds = Tuple[Optional[float], Optional[float]]
Window = Tuple[float, float]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


@dataclass(frozen=True)
class ComponentRecipe:
    id: str
    line_ids: Tuple[str, ...]
    role: str
    profile: str = "gaussian"
    multiplicity: int = 1
    enabled: bool = True
    required: bool = False
    flux_bounds: Bounds = (0.0, None)
    velocity_bounds_kms: Tuple[float, float] = (-1000.0, 1000.0)
    fwhm_bands_kms: Tuple[Tuple[float, float], ...] = ((70.0, 1200.0),)
    kinematic_group: Optional[str] = None
    velocity_group: Optional[str] = None
    width_group: Optional[str] = None
    fixed_ratio_to: Optional[str] = None
    fixed_ratio: Optional[float] = None
    selection_rule: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ("broad", "narrow", "very_broad", "wing", "blend"):
            raise ValueError(f"Unsupported component role: {self.role!r}")
        if self.profile not in ("gaussian", "lorentzian"):
            raise ValueError(f"Unsupported component profile: {self.profile!r}")
        if self.multiplicity < 1:
            raise ValueError("ComponentRecipe.multiplicity must be positive.")
        for line_id in self.line_ids:
            lines.get(line_id)
        if self.fixed_ratio_to is not None and (
            self.fixed_ratio is None or self.fixed_ratio <= 0
        ):
            raise ValueError("A fixed-ratio component requires fixed_ratio > 0.")


@dataclass(frozen=True)
class ComplexRecipe:
    id: str
    aliases: Tuple[str, ...]
    label: str
    fit_window: Window
    fit_windows: Tuple[Window, ...]
    mask_windows: Tuple[Window, ...]
    components: Tuple[ComponentRecipe, ...]
    required_line_ids: Tuple[str, ...]
    coverage_mode: str = "full"
    min_coverage_fraction: float = 0.8
    min_valid_pixels: int = 30
    edge_margin_kms: float = 1000.0
    local_support: bool = False
    core_margin_kms: float = 500.0
    min_core_pixels: int = 3
    continuum_mode: str = "fixed_global"
    qa_labels: Tuple[str, ...] = ()
    auto_enabled: bool = False
    priority: int = 0
    backend: str = "generic"
    exclusive_group: Optional[str] = None

    def __post_init__(self) -> None:
        if self.coverage_mode not in ("full", "component_adaptive"):
            raise ValueError("coverage_mode must be 'full' or 'component_adaptive'.")
        if self.continuum_mode not in (
            "fixed_global", "constant", "linear", "residual_linear", "absent"
        ):
            raise ValueError("Unsupported continuum_mode.")
        if self.backend not in (
            "generic",
            "mgii_adapter",
            "hbeta_adapter",
            "halpha_adapter",
            "lya_adapter",
        ):
            raise ValueError("Unsupported recipe backend.")
        for line_id in self.required_line_ids:
            lines.get(line_id)

    def with_component(self, component_id: str, **changes: Any) -> "ComplexRecipe":
        """Return a copy with one component replaced."""

        found = False
        updated = []
        for component in self.components:
            if component.id == component_id:
                component = replace(component, **changes)
                found = True
            updated.append(component)
        if not found:
            raise ValueError(f"unknown_recipe_component: {component_id!r}")
        return replace(self, components=tuple(updated))


def _component(id, line_ids, role, **kwargs):
    return ComponentRecipe(id=id, line_ids=tuple(line_ids), role=role, **kwargs)


def _bands_for_count(
    bands: Tuple[Tuple[float, float], ...],
    count: int,
) -> Tuple[Tuple[float, float], ...]:
    if len(bands) >= count:
        return tuple(bands[:count])
    return tuple((*bands, *((bands[-1],) * (count - len(bands)))))


def lya_nv_recipe(
    config: Optional[LyaNVComplexConfig] = None,
) -> ComplexRecipe:
    """Compile the specialized Lyα/N V recipe from public configuration."""

    cfg = config or LyaNVComplexConfig()
    components = []
    paired_width_group = (
        "lya_nv_broad_width"
        if cfg.tie_nv_width_to_lya else None
    )
    if cfg.fit_lya:
        components.append(
            _component(
                "Lya_broad",
                ("lya_1216",),
                "broad",
                multiplicity=cfg.lya_num_broad_gaussians,
                kinematic_group="lya_broad",
                width_group=paired_width_group,
                velocity_bounds_kms=cfg.lya_velocity_bounds_kms,
                fwhm_bands_kms=_bands_for_count(
                    cfg.lya_fwhm_bands_kms,
                    cfg.lya_num_broad_gaussians,
                ),
            )
        )
    if cfg.fit_nv:
        components.append(
            _component(
                "NV_broad",
                (
                    ("nv_blend",)
                    if cfg.nv_mode == "effective_blend"
                    else ("nv_1239", "nv_1243")
                ),
                "broad",
                multiplicity=cfg.nv_num_broad_gaussians,
                kinematic_group="nv_broad",
                width_group=paired_width_group,
                velocity_bounds_kms=cfg.nv_velocity_bounds_kms,
                fwhm_bands_kms=_bands_for_count(
                    cfg.nv_fwhm_bands_kms,
                    cfg.nv_num_broad_gaussians,
                ),
            )
        )
    required = []
    if cfg.fit_lya:
        required.append("lya_1216")
    if cfg.fit_nv:
        required.extend(
            ("nv_blend",)
            if cfg.nv_mode == "effective_blend"
            else ("nv_1239", "nv_1243")
        )
    return ComplexRecipe(
        id="lya_nv",
        aliases=("lya", "lyalpha", "lya_nv1240"),
        label="Lyα / N V",
        fit_window=cfg.window,
        fit_windows=(cfg.window,),
        mask_windows=(),
        components=tuple(components),
        required_line_ids=tuple(required),
        coverage_mode="full",
        min_coverage_fraction=cfg.minimum_useful_overlap_fraction,
        min_valid_pixels=cfg.min_valid_pixels,
        edge_margin_kms=cfg.edge_margin_kms,
        qa_labels=("lya_1216", "nv_blend"),
        auto_enabled=True,
        priority=85,
        backend="lya_adapter",
        exclusive_group="lya",
    )


_RECIPES = (
    lya_nv_recipe(),
    ComplexRecipe(
        id="mgii", aliases=("mgii_blend",), label="Mg II",
        fit_window=(2700.0, 2900.0), fit_windows=((2700.0, 2900.0),), mask_windows=(),
        components=(
            _component(
                "MgII_broad", ("mgii_blend",), "broad", multiplicity=2,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 3500.0), (3500.0, 15000.0)),
                kinematic_group="MgII_broad",
            ),
            _component(
                "MgII_narrow",
                ("mgii_blend",),
                "narrow",
                kinematic_group="MgII_narrow",
            ),
        ), required_line_ids=("mgii_blend",), qa_labels=("mgii_blend",),
        auto_enabled=True, priority=90, backend="mgii_adapter", exclusive_group="mgii",
    ),
    ComplexRecipe(
        id="hbeta_oiii", aliases=("hbeta", "hb_oiii"), label="Hβ / [O III]",
        fit_window=(4640.0, 5100.0), fit_windows=((4640.0, 5100.0),), mask_windows=(),
        components=(
            _component(
                "Hb_broad", ("hbeta",), "broad", multiplicity=3,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 2500.0), (2500.0, 6000.0), (6000.0, 20000.0)),
                kinematic_group="Hb_broad",
            ),
            _component("Hb_narrow", ("hbeta",), "narrow", kinematic_group="hbeta_narrow"),
            _component("OIII5008_core", ("oiii_5008",), "narrow", kinematic_group="hbeta_narrow"),
            _component(
                "OIII4960_core", ("oiii_4960",), "narrow",
                kinematic_group="hbeta_narrow", fixed_ratio_to="OIII5008_core",
                fixed_ratio=2.98,
            ),
            _component(
                "OIII5008_wing", ("oiii_5008",), "wing",
                kinematic_group="oiii_wing", selection_rule="bic_and_snr",
                fwhm_bands_kms=((150.0, 2500.0),),
            ),
            _component(
                "OIII4960_wing", ("oiii_4960",), "wing",
                kinematic_group="oiii_wing", fixed_ratio_to="OIII5008_wing",
                fixed_ratio=2.98, selection_rule="bic_and_snr",
                fwhm_bands_kms=((150.0, 2500.0),),
            ),
            _component(
                "HeII_broad", ("heii_4687",), "broad", enabled=False,
                kinematic_group="HeII_broad",
            ),
        ), required_line_ids=("hbeta", "oiii_4960", "oiii_5008"),
        qa_labels=("hbeta", "oiii_4960", "oiii_5008"), auto_enabled=True,
        priority=100, backend="hbeta_adapter", exclusive_group="hbeta",
    ),
    ComplexRecipe(
        id="halpha_nii_sii", aliases=("halpha", "ha_nii_sii"), label="Hα / [N II] / [S II]",
        fit_window=(6400.0, 6800.0), fit_windows=((6400.0, 6800.0),), mask_windows=(),
        components=(
            _component(
                "Ha_broad", ("halpha",), "broad", multiplicity=3,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 2500.0), (2500.0, 6000.0), (6000.0, 20000.0)),
                kinematic_group="Ha_broad",
            ),
            _component("Ha_narrow", ("halpha",), "narrow", kinematic_group="halpha_narrow"),
            _component("NII6585", ("nii_6585",), "narrow", kinematic_group="halpha_narrow"),
            _component(
                "NII6550", ("nii_6550",), "narrow",
                kinematic_group="halpha_narrow", fixed_ratio_to="NII6585",
                fixed_ratio=2.96,
            ),
            _component("SII6718", ("sii_6718",), "narrow", kinematic_group="halpha_narrow"),
            _component("SII6733", ("sii_6733",), "narrow", kinematic_group="halpha_narrow"),
        ), required_line_ids=("halpha", "nii_6550", "nii_6585"),
        qa_labels=("halpha", "nii_6550", "nii_6585", "sii_6718", "sii_6733"),
        coverage_mode="component_adaptive", min_coverage_fraction=0.6,
        auto_enabled=True, priority=100, backend="halpha_adapter", exclusive_group="halpha",
    ),
    ComplexRecipe(
        id="oii_nev_neiii_hgamma", aliases=("optical_blue",), label="[Ne V] / [O II] / [Ne III] / Hγ",
        fit_window=(3380.0, 4425.0), fit_windows=((3380.0, 3970.0), (4050.0, 4425.0)),
        mask_windows=(),
        components=(
            _component("NeV3427", ("nev_3427",), "narrow", kinematic_group="blue_narrow"),
            _component("OII3727", ("oii_3727",), "narrow", kinematic_group="blue_narrow"),
            _component("OII3730", ("oii_3730",), "narrow", kinematic_group="blue_narrow"),
            _component("NeIII3870", ("neiii_3870",), "narrow", kinematic_group="blue_narrow"),
            _component("OIII4364", ("oiii_4364",), "narrow", kinematic_group="blue_narrow"),
            _component("Hgamma_narrow", ("hgamma",), "narrow", kinematic_group="blue_narrow"),
            _component("Hgamma_broad", ("hgamma",), "broad", required=False,
                       velocity_bounds_kms=(-2000.0, 2000.0),
                       fwhm_bands_kms=((900.0, 15000.0),), kinematic_group="hgamma_broad"),
        ),
        required_line_ids=(), coverage_mode="component_adaptive", qa_labels=("nev_3427", "oii_blend", "neiii_3870", "hgamma"),
        auto_enabled=True, priority=40, backend="generic", exclusive_group="optical_blue",
    ),
    ComplexRecipe(
        id="paschen_nir", aliases=("nir", "paschen"), label="Paschen / NIR",
        fit_window=(9900.0, 13050.0),
        fit_windows=((9900.0, 10120.0), (10720.0, 11040.0), (11180.0, 11380.0), (12650.0, 13050.0)),
        mask_windows=(),
        components=(
            _component("Padelta_broad", ("padelta",), "broad", kinematic_group="padelta_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("HeI10833_broad", ("hei_10833",), "broad", kinematic_group="hei_pgamma_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("Pagamma_broad", ("pagamma",), "broad", kinematic_group="hei_pgamma_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("OI11290_broad", ("oi_11290",), "broad", kinematic_group="oi11290_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
            _component("Pabeta_broad", ("pabeta",), "broad", kinematic_group="pabeta_broad",
                       velocity_bounds_kms=(-2000.0, 2000.0), fwhm_bands_kms=((900.0, 15000.0),)),
        ),
        required_line_ids=(), coverage_mode="component_adaptive",
        qa_labels=("padelta", "hei_10833", "pagamma", "oi_11290", "pabeta"),
        auto_enabled=True, priority=40, backend="generic", exclusive_group="paschen_nir",
    ),
    ComplexRecipe(
        id="hei5877", aliases=("hei_5877", "hei5876"), label="He I 5877",
        fit_window=(5700.0, 6050.0), fit_windows=((5700.0, 6050.0),), mask_windows=(),
        components=(
            _component(
                "HeI5877_narrow", ("hei_5877",), "narrow",
                kinematic_group="hei5877_narrow",
            ),
            _component(
                "HeI5877_broad", ("hei_5877",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="hei5877_broad",
            ),
        ),
        required_line_ids=("hei_5877",), coverage_mode="component_adaptive",
        min_coverage_fraction=0.6, min_valid_pixels=15,
        local_support=True, qa_labels=("hei_5877",),
        auto_enabled=False, priority=45, backend="generic", exclusive_group="hei5877",
    ),
    ComplexRecipe(
        id="oi8449", aliases=("oi_8449", "oi8446"), label="O I 8449",
        fit_window=(8200.0, 8700.0), fit_windows=((8200.0, 8700.0),), mask_windows=(),
        components=(
            _component(
                "OI8449_narrow", ("oi_8449",), "narrow", enabled=False,
                kinematic_group="oi8449_narrow",
            ),
            _component(
                "OI8449_broad", ("oi_8449",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="oi8449_broad",
            ),
        ),
        required_line_ids=("oi_8449",), coverage_mode="component_adaptive",
        min_coverage_fraction=0.6, min_valid_pixels=15,
        local_support=True, qa_labels=("oi_8449",),
        auto_enabled=False, priority=45, backend="generic", exclusive_group="oi8449",
    ),
    ComplexRecipe(
        id="siii_nir", aliases=("siii", "siii_doublet"), label="[S III] NIR doublet",
        fit_window=(8950.0, 9680.0),
        fit_windows=((8950.0, 9180.0), (9390.0, 9680.0)),
        mask_windows=(),
        components=(
            _component(
                "SIII9071_narrow", ("siii_9071",), "narrow",
                kinematic_group="siii_nir_narrow",
            ),
            _component(
                "SIII9533_narrow", ("siii_9533",), "narrow",
                kinematic_group="siii_nir_narrow",
            ),
            _component(
                "PaEta_narrow", ("paeta_9017",), "narrow",
                kinematic_group="siii_hydrogen_narrow",
            ),
            _component(
                "PaEpsilon_narrow", ("paepsilon_9549",), "narrow",
                kinematic_group="siii_hydrogen_narrow",
            ),
            _component(
                "PaEta_broad", ("paeta_9017",), "broad", enabled=False,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="siii_hydrogen_broad",
            ),
            _component(
                "PaEpsilon_broad", ("paepsilon_9549",), "broad", enabled=False,
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="siii_hydrogen_broad",
            ),
        ),
        required_line_ids=(), coverage_mode="component_adaptive",
        min_coverage_fraction=0.5, min_valid_pixels=20,
        local_support=True,
        qa_labels=("siii_9071", "siii_9533", "paeta_9017", "paepsilon_9549"),
        auto_enabled=False, priority=45, backend="generic", exclusive_group="siii_nir",
    ),
    ComplexRecipe(
        id="padelta", aliases=("pa_delta",), label="Paδ",
        fit_window=(9900.0, 10120.0), fit_windows=((9900.0, 10120.0),), mask_windows=(),
        components=(
            _component(
                "Padelta_broad", ("padelta",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),), kinematic_group="padelta_broad",
            ),
            _component(
                "Padelta_narrow", ("padelta",), "narrow",
                kinematic_group="padelta_narrow",
            ),
        ),
        required_line_ids=("padelta",), coverage_mode="component_adaptive",
        min_coverage_fraction=0.6, min_valid_pixels=20,
        local_support=True, qa_labels=("padelta",),
        auto_enabled=False, priority=45, backend="generic",
        exclusive_group="padelta",
    ),
    ComplexRecipe(
        id="hei10833_pgamma", aliases=("hei_pgamma",), label="He I 10833 / Paγ",
        fit_window=(10720.0, 11040.0), fit_windows=((10720.0, 11040.0),), mask_windows=(),
        components=(
            _component(
                "HeI10833_broad", ("hei_10833",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="hei_pgamma_broad",
            ),
            _component(
                "Pagamma_broad", ("pagamma",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="hei_pgamma_broad",
            ),
            _component(
                "HeI10833_narrow", ("hei_10833",), "narrow",
                kinematic_group="hei_pgamma_narrow",
            ),
            _component(
                "Pagamma_narrow", ("pagamma",), "narrow",
                kinematic_group="hei_pgamma_narrow",
            ),
        ),
        required_line_ids=(), coverage_mode="component_adaptive",
        min_coverage_fraction=0.6, min_valid_pixels=20,
        local_support=True, qa_labels=("hei_10833", "pagamma"),
        auto_enabled=False, priority=45, backend="generic",
        exclusive_group="hei10833_pgamma",
    ),
    ComplexRecipe(
        id="oi11290", aliases=("oi_11290",), label="O I 11290",
        fit_window=(11180.0, 11380.0), fit_windows=((11180.0, 11380.0),), mask_windows=(),
        components=(
            _component(
                "OI11290_broad", ("oi_11290",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),),
                kinematic_group="oi11290_broad",
            ),
            _component(
                "OI11290_narrow", ("oi_11290",), "narrow",
                kinematic_group="oi11290_narrow",
            ),
        ),
        required_line_ids=("oi_11290",), coverage_mode="component_adaptive",
        min_coverage_fraction=0.6, min_valid_pixels=20,
        local_support=True, qa_labels=("oi_11290",),
        auto_enabled=False, priority=45, backend="generic",
        exclusive_group="oi11290",
    ),
    ComplexRecipe(
        id="pabeta", aliases=("pa_beta",), label="Paβ",
        fit_window=(12650.0, 13050.0), fit_windows=((12650.0, 13050.0),), mask_windows=(),
        components=(
            _component(
                "Pabeta_broad", ("pabeta",), "broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 15000.0),), kinematic_group="pabeta_broad",
            ),
            _component(
                "Pabeta_narrow", ("pabeta",), "narrow",
                kinematic_group="pabeta_narrow",
            ),
        ),
        required_line_ids=("pabeta",), coverage_mode="component_adaptive",
        min_coverage_fraction=0.6, min_valid_pixels=20,
        local_support=True, qa_labels=("pabeta",),
        auto_enabled=False, priority=45, backend="generic",
        exclusive_group="pabeta",
    ),
    ComplexRecipe(
        id="generic_narrow_lines", aliases=("narrow_lines",), label="Generic narrow lines",
        fit_window=(0.0, 1.0), fit_windows=(), mask_windows=(), components=(),
        required_line_ids=(), coverage_mode="component_adaptive", auto_enabled=False,
        priority=0, backend="generic",
    ),
    ComplexRecipe(
        id="ciii", aliases=("ciii1909", "ciii]"), label="C III]",
        fit_window=(1700.0, 1970.0), fit_windows=((1700.0, 1970.0),), mask_windows=(),
        components=(
            _component(
                "CIII_broad", ("ciii_1909",), "broad", multiplicity=2,
                kinematic_group="ciii_broad",
                velocity_bounds_kms=(-2000.0, 2000.0),
                fwhm_bands_kms=((900.0, 3500.0), (3500.0, 15000.0)),
            ),
        ),
        required_line_ids=("ciii_1909",), qa_labels=("ciii_1909",),
        auto_enabled=True, priority=80, backend="generic", exclusive_group="ciii",
    ),
    ComplexRecipe(
        id="civ", aliases=("civ1549",), label="C IV",
        fit_window=(1450.0, 1700.0), fit_windows=((1450.0, 1700.0),), mask_windows=(),
        components=(
            _component(
                "CIV_broad", ("civ_blend",), "broad", multiplicity=3,
                kinematic_group="civ_broad",
                velocity_bounds_kms=(-5000.0, 3000.0),
                fwhm_bands_kms=(
                    (900.0, 2500.0),
                    (2500.0, 6000.0),
                    (6000.0, 20000.0),
                ),
            ),
        ),
        required_line_ids=("civ_blend",), qa_labels=("civ_blend",), auto_enabled=True,
        priority=80, backend="generic", exclusive_group="civ",
    ),
)

_BY_ID: Dict[str, ComplexRecipe] = {recipe.id: recipe for recipe in _RECIPES}
_ALIASES: Dict[str, str] = {}
for _recipe in _RECIPES:
    for _alias in (_recipe.id, _recipe.label, *_recipe.aliases):
        _ALIASES[_normalize(_alias)] = _recipe.id


def list_complexes() -> Tuple[ComplexRecipe, ...]:
    return tuple(sorted(_RECIPES, key=lambda recipe: (-recipe.priority, recipe.id)))


def resolve(value: str) -> str:
    key = _normalize(value)
    if key not in _ALIASES:
        raise ValueError(f"unknown_complex_recipe: {value!r}")
    return _ALIASES[key]


def get(value: str) -> ComplexRecipe:
    return _BY_ID[resolve(value)]


def describe(value: str) -> Dict[str, Any]:
    recipe = get(value)
    return {
        "id": recipe.id,
        "label": recipe.label,
        "aliases": recipe.aliases,
        "fit_window": recipe.fit_window,
        "fit_windows": recipe.fit_windows,
        "components": tuple(component.id for component in recipe.components),
        "required_line_ids": recipe.required_line_ids,
        "auto_enabled": recipe.auto_enabled,
        "priority": recipe.priority,
        "backend": recipe.backend,
    }


NIR_COMPLETE_PRESET_ID = "nir_complete_v1"
NIR_COMPLETE_COMPACT_IDS: Tuple[str, ...] = (
    "padelta",
    "hei10833_pgamma",
    "oi11290",
    "pabeta",
)
NIR_COMPLETE_ADDED_IDS: Tuple[str, ...] = ("hei5877", "oi8449", "siii_nir")


def nir_complete_recipes() -> Tuple[ComplexRecipe, ...]:
    """Return the versioned NIR-complete recipe set (``nir_complete_v1``).

    The preset keeps every standard auto-enabled recipe but replaces the
    umbrella ``paschen_nir`` complex with four compact local NIR recipes and
    adds the newly activated lines (He I 5877, O I 8449, [S III] NIR).
    """

    selected = [
        recipe
        for recipe in _RECIPES
        if recipe.auto_enabled and recipe.id != "paschen_nir"
    ]
    for recipe_id in (*NIR_COMPLETE_ADDED_IDS, *NIR_COMPLETE_COMPACT_IDS):
        selected.append(_BY_ID[recipe_id])
    return tuple(selected)


NIR_COMPLETE_RECIPE_IDS = frozenset(recipe.id for recipe in nir_complete_recipes())


def generic_narrow_lines(line_ids: Iterable[str], **changes: Any) -> ComplexRecipe:
    canonical = tuple(lines.resolve(line_id) for line_id in line_ids)
    if not canonical:
        raise ValueError("generic_narrow_lines requires at least one line ID.")
    wavelengths = tuple(lines.get(line_id).vacuum_wavelength for line_id in canonical)
    padding = float(changes.pop("padding_angstrom", 50.0))
    components = tuple(
        _component(
            f"{line_id}_narrow",
            (line_id,),
            "narrow",
            kinematic_group="generic_narrow",
        )
        for line_id in canonical
    )
    base = get("generic_narrow_lines")
    return replace(
        base,
        id="generic_narrow_lines_" + "_".join(canonical),
        label=" / ".join(lines.get(line_id).label for line_id in canonical),
        fit_window=(min(wavelengths) - padding, max(wavelengths) + padding),
        fit_windows=((min(wavelengths) - padding, max(wavelengths) + padding),),
        components=components,
        required_line_ids=canonical,
        qa_labels=canonical,
        **changes,
    )
