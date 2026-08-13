"""Validation and loading for expert linear-constraint annotations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LinearConstraint:
    """A linear inequality in canonical form ``sum(a_j x_j) >= rhs``."""

    id: str
    description: str
    formula: str
    coefficients: dict[str, float]
    rhs: float
    mutable_columns: tuple[str, ...]
    explanation: str = ""
    source: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.coefficients)


def _require_text(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Constraint {index} requires a non-empty {key!r} string.")
    return value.strip()


def _parse_constraint(item: Any, index: int) -> LinearConstraint:
    if not isinstance(item, dict):
        raise ValueError(f"Constraint {index} must be a JSON object.")
    constraint_id = _require_text(item, "id", index)
    description = _require_text(item, "description", index)
    formula = _require_text(item, "formula", index)
    if item.get("sense", ">=") != ">=":
        raise ValueError(
            f"Constraint {constraint_id!r} must use canonical sense '>='; "
            "negate coefficients and rhs for '<=' constraints."
        )

    raw_coefficients = item.get("coefficients")
    if not isinstance(raw_coefficients, dict) or len(raw_coefficients) < 2:
        raise ValueError(
            f"Constraint {constraint_id!r} requires coefficients for at least "
            "two columns."
        )
    coefficients: dict[str, float] = {}
    for column, raw_value in raw_coefficients.items():
        if not isinstance(column, str) or not column:
            raise ValueError(f"Constraint {constraint_id!r} has an invalid column name.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Constraint {constraint_id!r} coefficient for {column!r} "
                "must be numeric."
            ) from exc
        if not math.isfinite(value) or value == 0:
            raise ValueError(
                f"Constraint {constraint_id!r} coefficient for {column!r} "
                "must be finite and non-zero."
            )
        coefficients[column] = value

    try:
        rhs = float(item.get("rhs", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Constraint {constraint_id!r} rhs must be numeric.") from exc
    if not math.isfinite(rhs):
        raise ValueError(f"Constraint {constraint_id!r} rhs must be finite.")

    raw_mutable = item.get("mutable_columns", list(coefficients))
    if not isinstance(raw_mutable, list) or not raw_mutable:
        raise ValueError(
            f"Constraint {constraint_id!r} mutable_columns must be a non-empty list."
        )
    if len(raw_mutable) != len(set(raw_mutable)):
        raise ValueError(
            f"Constraint {constraint_id!r} mutable_columns contains duplicates."
        )
    unknown_mutable = [column for column in raw_mutable if column not in coefficients]
    if unknown_mutable:
        raise ValueError(
            f"Constraint {constraint_id!r} has mutable columns absent from its "
            f"coefficients: {unknown_mutable}"
        )

    columns = item.get("columns")
    if columns is not None and columns != list(coefficients):
        raise ValueError(
            f"Constraint {constraint_id!r} columns must match coefficient order."
        )

    return LinearConstraint(
        id=constraint_id,
        description=description,
        formula=formula,
        coefficients=coefficients,
        rhs=rhs,
        mutable_columns=tuple(raw_mutable),
        explanation=str(item.get("explanation", "")),
        source=str(item.get("source", "")),
    )


def load_constraints(path: str | Path) -> list[LinearConstraint]:
    """Load a JSON list of canonical linear inequalities."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Linear constraint file not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        document = document.get("constraints")
    if not isinstance(document, list) or not document:
        raise ValueError(f"{path} must contain a non-empty constraint list.")
    constraints = [_parse_constraint(item, index) for index, item in enumerate(document)]
    ids = [constraint.id for constraint in constraints]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate constraint ids.")
    return constraints
