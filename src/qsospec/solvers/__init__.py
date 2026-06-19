"""Numerical solvers used by qsospec fitting workflows."""

from .least_squares import run_least_squares
from .variable_projection import (
    VariableProjectionError,
    VariableProjectionResult,
    VariableProjectionState,
    evaluate_profile_chi2,
    optimizer_result_adapter,
    solve_variable_projection,
)

__all__ = [
    "VariableProjectionError",
    "VariableProjectionResult",
    "VariableProjectionState",
    "evaluate_profile_chi2",
    "optimizer_result_adapter",
    "run_least_squares",
    "solve_variable_projection",
]
