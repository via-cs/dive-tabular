"""Convex postprocessing for declarative linear tabular constraints."""

from .projector import LinearProjectionResult, evaluate_dataframe, project_dataframe
from .schema import LinearConstraint, load_constraints

__all__ = [
    "LinearConstraint",
    "LinearProjectionResult",
    "evaluate_dataframe",
    "load_constraints",
    "project_dataframe",
]
