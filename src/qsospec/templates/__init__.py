"""Template loading and normalization helpers for qsospec."""

from .balmer import (
    BalmerSeriesTemplate,
    BalmerTemplateError,
    balmer_bound_free_shape,
    evaluate_balmer_pseudocontinuum,
    evaluate_balmer_pseudocontinuum_with_derivatives,
    evaluate_balmer_series,
    evaluate_balmer_series_with_derivative,
    evaluate_balmer_series_with_derivatives,
    list_balmer_templates,
    load_balmer_template,
)
from .iron import (
    IronTemplate,
    IronTemplateError,
    PreparedIronTemplate,
    evaluate_iron_basis,
    evaluate_iron_basis_with_derivative,
    prepare_iron_template,
)
from .registry import list_iron_templates, load_iron_template, resolve_iron_template_name

__all__ = [
    "BalmerSeriesTemplate",
    "BalmerTemplateError",
    "balmer_bound_free_shape",
    "evaluate_balmer_pseudocontinuum",
    "evaluate_balmer_pseudocontinuum_with_derivatives",
    "IronTemplate",
    "IronTemplateError",
    "PreparedIronTemplate",
    "evaluate_balmer_series",
    "evaluate_balmer_series_with_derivative",
    "evaluate_balmer_series_with_derivatives",
    "evaluate_iron_basis",
    "evaluate_iron_basis_with_derivative",
    "list_balmer_templates",
    "list_iron_templates",
    "load_balmer_template",
    "load_iron_template",
    "prepare_iron_template",
    "resolve_iron_template_name",
]
