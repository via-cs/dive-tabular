"""Equational expert-constraint repair internals."""

from typing import TYPE_CHECKING, Any

from .scoring import load_constraints

if TYPE_CHECKING:
    from .fix import (
        equational_fix,
        greedy_equational_fix,
        static_global_equational_fix,
    )

__all__ = [
    "equational_fix",
    "greedy_equational_fix",
    "load_constraints",
    "static_global_equational_fix",
]


def __getattr__(name: str) -> Any:
    """Load repair entry points lazily so ``python -m .fix`` stays clean."""
    if name in {
        "equational_fix",
        "greedy_equational_fix",
        "static_global_equational_fix",
    }:
        from . import fix

        return getattr(fix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
